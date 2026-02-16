import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path

from functools import partial
from pathlib import Path
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Union
from typing import Container
import random
from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50
import pandas as pd
from utils.BENv2_dataset import BENv2DataSet
from utils.pytorch_utils import (
    get_classification_report,
    init_results,
    print_micro_macro,
    update_results,
    start_cuda
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

data_dirs = {
    "images_lmdb": "/data_read_only/BigEarthNet/BigEarthNet-V2/BENv2.lmdb",
    "metadata_parquet": "/data_read_only/BigEarthNet/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/data_read_only/BigEarthNet/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

class PreFilter:
    def __init__(self, metadata: pd.DataFrame, countries: Optional[Container] | str = None, seasons: Optional[Container] | str = None):
        metadata["month"] = metadata["patch_id"].str[15:17].astype(int)
        metadata["season"] = pd.cut(
            metadata["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Autumn"],
            right=False,
        )
        metadata.loc[metadata["month"] == 12, "season"] = "Winter"

        seasons = None if seasons is None else seasons if isinstance(seasons, Container) else [seasons]
        countries = None if countries is None else countries if isinstance(countries, Container) else [countries]

        def filter_fn(metadata_row) -> bool:
            row_country = metadata_row[3]
            row_season = metadata_row[9]
            if seasons is not None and row_season not in seasons:
                return False
            if countries is not None and row_country not in countries:
                return False
            return True

        self.filter_fn = filter_fn
        self.filtered_patches = set([x[0] for x in [x for x in metadata.values if filter_fn(x)]])
        print(f"Pre-filtered {len(self.filtered_patches)} patches based on country and season (split ignored)")

    def filter(self, patch_id: str) -> bool:
        return self.filter_fn(patch_id)

    def __call__(self, patch_id: str) -> bool:
        return patch_id in self.filtered_patches


# IMPORTANT FIX: matches .mod1/.mod2/.mod3 as well
def is_modulation_key(k: str) -> bool:
    return ".mod" in k


# FedBN helper: BN params+buffers keys kept local
def get_bn_state_dict_keys(model: nn.Module) -> set[str]:
    bn_keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for pn, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{pn}" if module_name else pn
                bn_keys.add(full)
            for bn, _ in module.named_buffers(recurse=False):
                full = f"{module_name}.{bn}" if module_name else bn
                bn_keys.add(full)
    return bn_keys


class Aggregator:
    def __init__(
        self,
        eps: float = 1e-12,
        align_power: float = 2.0,
        clip_align_min: float = 0.0,
    ) -> None:
        # Option A parameters
        self.eps = eps
        self.align_power = align_power
        self.clip_align_min = clip_align_min

    def compute_option_a_weights(self, updates_sizes_dirs: list[tuple[dict, int, torch.Tensor]]):
        """
        updates_sizes_dirs: list of (update_dict, size, mod_unit_dir)
        weight_i ∝ (size_i/total) * (max(0, cos(dir_i, ref)) + eps)^align_power
        Returns: weights(list), cosines(list)
        """
        eps = self.eps
        total = sum(sz for _, sz, _ in updates_sizes_dirs)

        # reference direction
        ref = None
        for _, _, u in updates_sizes_dirs:
            ref = u if ref is None else (ref + u)
        ref = ref / (ref.norm() + eps)

        raw_w = []
        cosines = []
        for _, sz, u in updates_sizes_dirs:
            a = float(torch.dot(u, ref).item())
            a = max(self.clip_align_min, a)
            cosines.append(a)
            gate = (a + eps) ** self.align_power
            raw_w.append((sz / total) * gate)

        s = sum(raw_w) + eps
        weights = [w / s for w in raw_w]
        return weights, cosines

    def aggregate_with_weights(self, updates_sizes_dirs: list[tuple[dict, int, torch.Tensor]], weights: list[float]):
        keys = updates_sizes_dirs[0][0].keys()
        agg = {}
        for k in keys:
            s_k = None
            for (upd, _, _), w in zip(updates_sizes_dirs, weights):
                term = upd[k] * w
                s_k = term if s_k is None else (s_k + term)
            agg[k] = s_k
        return agg


class FLCLient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_path: list[str],
        batch_size: int = 512,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device('cpu'),
        dataset_filter: str = "serbia",
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)
        self.device = device

        # FedBN local BN keys
        self.bn_keys = get_bn_state_dict_keys(self.model)

        self.dataset = BENv2DataSet(
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=[csv_path], seasons=["Summer"]),
        )
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
            pin_memory=True,
        )

        self.validation_set = BENv2DataSet(
            data_dirs=data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=[csv_path], seasons="Summer"),
        )
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
            pin_memory=True,
        )

    def set_model(self, model: torch.nn.Module):
        local_sd = self.model.state_dict()
        global_sd = model.state_dict()

        for k in global_sd.keys():
            # keep modulation local
            if is_modulation_key(k):
                continue
            # keep BN local (FedBN)
            if k in self.bn_keys:
                continue
            local_sd[k] = global_sd[k].detach().clone()

        self.model.load_state_dict(local_sd, strict=True)

    # -------- Option A support: MOD UPDATE direction (FIXED) --------
    def _get_mod_vector(self) -> torch.Tensor:
        vec = []
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if is_modulation_key(name) and (name.endswith("gamma") or name.endswith("beta")):
                    vec.append(p.detach().float().view(-1).cpu())
        if vec:
            return torch.cat(vec)
        return torch.zeros(1)

    def _unit_dir_from_delta(self, d: torch.Tensor) -> torch.Tensor:
        n = d.norm()
        if float(n.item()) < 1e-12:
            # fallback: use current mod vector direction (gamma starts at 1, so non-zero)
            v = d  # placeholder
            v = self._get_mod_vector()
            vn = v.norm()
            if float(vn.item()) < 1e-12:
                return torch.ones(1)  # last-resort non-zero vector
            return v / (vn + 1e-12)
        return d / (n + 1e-12)
    # ---------------------------------------------------------------

    def train_one_round(self, epochs: int, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())

        # capture mod BEFORE (for delta)
        mod_before = self._get_mod_vector()

        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)
            self.train_epoch()

        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)

        # capture mod AFTER (for delta)
        mod_after = self._get_mod_vector()
        dmod = (mod_after - mod_before)
        mod_unit_dir = self._unit_dir_from_delta(dmod)

        state_after = self.model.state_dict()

        model_update = {}
        for key, value_before in state_before.items():
            # do not send modulation updates
            if is_modulation_key(key):
                continue
            # do not send BN updates (FedBN)
            if key in self.bn_keys:
                continue

            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(torch.DoubleTensor)
            model_update[key] = diff

        # Option A: include mod delta direction for aggregation weighting
        return model_update, len(self.dataset), mod_unit_dir

    def train_epoch(self):
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data = batch[1]
            labels = batch[4]

            data = data.cuda()
            label_new = np.copy(labels)
            label_new = torch.from_numpy(label_new).cuda()
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()

    def validation_personalized(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"{self.dataset_filter} test"):
                data = batch[1].to(self.device)
                labels = batch[4]
                label_new = np.copy(labels)

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)
                y_true += list(label_new)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)

        y_true = np.asarray(y_true)
        report = get_classification_report(
            y_true, y_predicted, predicted_probs, self.dataset_filter
        )
        return report

    def get_validation_results(self):
        return self.results


class GlobalClient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_paths: list[str],
        batch_size: int = 512,
        num_workers: int = 0,
        num_classes: int = 19,
        dataset_filter: str = "serbia",
        state_dict_path: str = None,
        results_path: str = None,
        align_power: float = 2.0,   # Option A knob (1..3 typical)
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter

        self.aggregator = Aggregator(eps=1e-12, align_power=align_power, clip_align_min=0.0)
        self.results = init_results(self.num_classes)

        self.clients = [
            FLCLient(copy.deepcopy(self.model), lmdb_path, val_path, csv_path, num_classes=num_classes, dataset_filter=csv_path, device=self.device)
            for csv_path in csv_paths
        ]

        self.validation_set = BENv2DataSet(
            data_dirs=data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(pd.read_parquet(data_dirs["metadata_parquet"]), countries=["Finland","Ireland","Serbia","Austria", "Belgium", "Lithuania", "Portugal", "Switzerland"], seasons="Summer"),
        )
        self.val_loader = DataLoader(
            self.validation_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
            pin_memory=True,
        )

        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if state_dict_path is None:
            if isinstance(model, ConvMixer):
                self.state_dict_path = f'checkpoints/global_convmixer_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.state_dict_path = f'checkpoints/global_mlpmixer_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.state_dict_path = f'checkpoints/global_poolformer_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.state_dict_path = f'checkpoints/global_resnet18_{dt}.pkl'

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f'results/convmixer_results_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.results_path = f'results/mlpmixer_results_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.results_path = f'results/poolformer_results_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.results_path = f'results/resnet18_results_{dt}.pkl'

    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()
        last_out = None

        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            self.communication_round(epochs)

            # broadcast shared params while keeping BN+mod local
            for client in self.clients:
                client.set_model(self.model)

            # per-client eval (local BN + local mod)
            client_reports = []
            client_sizes = []
            for client in self.clients:
                rep = client.validation_personalized()
                client_reports.append(rep)
                client_sizes.append(len(client.validation_set))
                print_micro_macro(rep)

            weights = np.array(client_sizes, dtype=np.float64)
            weights = weights / weights.sum()

            summary = {
                "micro_f1": float(np.sum(weights * np.array([r["micro avg"]["f1-score"] for r in client_reports], dtype=np.float64))),
                "macro_f1": float(np.sum(weights * np.array([r["macro avg"]["f1-score"] for r in client_reports], dtype=np.float64))),
                "ap_mic":   float(np.sum(weights * np.array([r["ap_mic"] for r in client_reports], dtype=np.float64))),
                "ap_mac":   float(np.sum(weights * np.array([r["ap_mac"] for r in client_reports], dtype=np.float64))),
            }

            print("\n=== Overall Weighted Summary ===")
            print(summary)

            last_out = {"client_reports": client_reports, "summary": summary}

        self.train_time = time.perf_counter() - start

        # no checkpoint saving
        # self.save_results()
        # self.save_state_dict()

        return self.model, last_out

    def communication_round(self, epochs: int):
        # collect (update, size, mod_delta_dir)
        updates_sizes_dirs = [client.train_one_round(epochs) for client in self.clients]

        # compute weights (Option A) + print them
        weights, cosines = self.aggregator.compute_option_a_weights(updates_sizes_dirs)

        print("\nAggregation weights (Option A: size × similarity of Δmod)")
        for i, (client, w, cos_val, tup) in enumerate(zip(self.clients, weights, cosines, updates_sizes_dirs)):
            _, sz, _ = tup
            print(f"Client {i} ({client.dataset_filter}) | size={sz} | cos={cos_val:.4f} | weight={w:.4f}")
        print(f"Sum of weights: {sum(weights):.4f}\n")

        # aggregate shared updates
        update_aggregation = self.aggregator.aggregate_with_weights(updates_sizes_dirs, weights)

        global_state_dict = self.model.state_dict()
        for key, update in update_aggregation.items():
            global_state_dict[key] = global_state_dict[key] + update.to(self.device)
        self.model.load_state_dict(global_state_dict)

    def save_state_dict(self):
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)
        res = {'global': self.results, 'clients': self.client_results, 'train_time': self.train_time}
        torch.save(res, self.results_path)
