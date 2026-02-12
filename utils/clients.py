# =========================
# Plain FedBN (no modulation, no checkpoints)
# - Aggregates ONLY non-BN params/buffers (FedAvg)
# - Keeps ALL BatchNorm params + buffers local per client
# - Evaluates clients individually using their local BN + global shared params
# - NO torch.save anywhere
# =========================

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path
import random
import pandas as pd
from typing import Optional, Container

from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50
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
    def __init__(
        self,
        metadata: pd.DataFrame,
        countries: Optional[Container] | str = None,
        seasons: Optional[Container] | str = None
    ):
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


# --------------------------
# Helpers (FedBN)
# --------------------------
def get_bn_state_dict_keys(model: nn.Module) -> set[str]:
    """
    Collect full state_dict keys (params + buffers) that belong to BatchNorm modules.
    Includes:
      - weight, bias
      - running_mean, running_var, num_batches_tracked
    """
    bn_keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            # parameters
            for pn, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{pn}" if module_name else pn
                bn_keys.add(full)
            # buffers
            for bn, _ in module.named_buffers(recurse=False):
                full = f"{module_name}.{bn}" if module_name else bn
                bn_keys.add(full)
    return bn_keys


# --------------------------
# FL Client (FedBN)
# --------------------------
class FLCLient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_path: str,
        batch_size: int = 512,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 19,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "serbia",
    ) -> None:
        self.model = model.to(device)
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)
        self.device = device

        # FedBN: keep BN local (params + buffers)
        self.bn_keys = get_bn_state_dict_keys(self.model)

        self.optimizer = None
        self.criterion = None

        meta = pd.read_parquet(data_dirs["metadata_parquet"])

        self.dataset = BENv2DataSet(
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(meta, countries=[csv_path], seasons=["Summer"]),
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
            patch_prefilter=PreFilter(meta, countries=[csv_path], seasons="Summer"),
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

    # broadcast: overwrite all NON-BN keys from global; keep BN local
    def set_model(self, model: torch.nn.Module):
        local_sd = self.model.state_dict()
        global_sd = model.state_dict()

        for k in global_sd.keys():
            if k in self.bn_keys:
                continue
            local_sd[k] = global_sd[k].detach().clone()

        self.model.load_state_dict(local_sd, strict=True)

    def train_one_round(self, epochs: int):
        state_before = copy.deepcopy(self.model.state_dict())

        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for _ in range(epochs):
            self.train_epoch()

        state_after = self.model.state_dict()

        # send update ONLY for NON-BN keys
        model_update = {}
        for k, v_before in state_before.items():
            if k in self.bn_keys:
                continue
            model_update[k] = (state_after[k].double() - v_before.double())

        return model_update, len(self.dataset)

    def train_epoch(self) -> int:
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError("Optimizer/Criterion not initialized. Call train_one_round() first.")

        self.model.train()
        steps = 0
        for _, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data = batch[1].to(self.device)
            labels = torch.from_numpy(np.copy(batch[4])).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            steps += 1
        return steps

    # personalized validation (FedBN: uses this client's BN)
    def validation_personalized(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"{self.dataset_filter} test"):
                data = batch[1].to(self.device)
                labels = np.copy(batch[4])

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)
                y_true += list(labels)

        predicted_probs = np.asarray(predicted_probs)
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
        y_true = np.asarray(y_true)

        rep = get_classification_report(y_true, y_predicted, predicted_probs, self.dataset_filter)
        return rep

    def get_validation_results(self):
        return self.results


# --------------------------
# Global Client (Server) - Plain FedBN (no checkpoints)
# --------------------------
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
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.num_classes = num_classes
        self.dataset_filter = dataset_filter

        self.results = init_results(self.num_classes)
        self.personalized_history = []

        self.clients = [
            FLCLient(
                copy.deepcopy(self.model),
                lmdb_path,
                val_path,
                csv_path,
                num_classes=num_classes,
                dataset_filter=csv_path,
                device=self.device
            )
            for csv_path in csv_paths
        ]

        meta = pd.read_parquet(data_dirs["metadata_parquet"])
        self.validation_set = BENv2DataSet(
            data_dirs=data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                meta,
                countries=["Finland", "Ireland", "Serbia", "Austria", "Belgium", "Lithuania", "Portugal", "Switzerland"],
                seasons="Summer"
            ),
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

    def validation_round_personalized(self, verbose: bool = True, weighted: bool = True):
        client_reports = []
        client_sizes = []

        for i, client in enumerate(self.clients):
            rep = client.validation_personalized()
            client_reports.append(rep)
            client_sizes.append(len(client.validation_set))

            if verbose:
                print(f"\n=== Client {i} personalized ({client.dataset_filter}) ===")
                print_micro_macro(rep)

        if not client_reports:
            return {"client_reports": [], "summary": {}}

        weights = np.array(client_sizes, dtype=np.float64)
        weights = weights / weights.sum() if weighted else np.ones_like(weights) / len(weights)

        def wavg(vals):
            return float(np.sum(weights * np.array(vals, dtype=np.float64)))

        summary = {
            "micro_f1": wavg([r["micro avg"]["f1-score"] for r in client_reports]),
            "macro_f1": wavg([r["macro avg"]["f1-score"] for r in client_reports]),
            "ap_mic":   wavg([r["ap_mic"] for r in client_reports]),
            "ap_mac":   wavg([r["ap_mac"] for r in client_reports]),
        }

        if verbose:
            mode = "weighted" if weighted else "mean"
            print(f"\n=== Personalized summary ({mode} over clients) ===")
            print(
                f"micro_f1={summary['micro_f1']:.4f} | "
                f"macro_f1={summary['macro_f1']:.4f} | "
                f"mAP_micro={summary['ap_mic']:.4f} | "
                f"mAP_macro={summary['ap_mac']:.4f}"
            )

        return {"client_reports": client_reports, "summary": summary}

    def communication_round(self, epochs: int):
        updates_sizes = [client.train_one_round(epochs) for client in self.clients]
        total_n = sum(n for _, n in updates_sizes) + 1e-12

        keys = updates_sizes[0][0].keys()
        agg_update = {}

        # FedAvg over updates (weighted by local data size)
        for k in keys:
            s = None
            for (upd, n) in updates_sizes:
                w = float(n) / float(total_n)
                term = upd[k] * w
                s = term if s is None else (s + term)
            agg_update[k] = s

        # apply aggregated update to global model (NON-BN keys only)
        global_sd = self.model.state_dict()
        for k, upd in agg_update.items():
            global_sd[k] = global_sd[k] + upd.to(self.device)
        self.model.load_state_dict(global_sd, strict=True)

        print("\nFedAvg client weights (by train set size):")
        for i, (client, (_, n)) in enumerate(zip(self.clients, updates_sizes)):
            print(f"Client {i} ({client.dataset_filter}): weight={n/total_n:.4f}, n={n}")
        print(f"Sum of weights: {sum(n/total_n for _, n in updates_sizes):.4f}\n")

    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()

        for com_round in range(1, communication_rounds + 1):
            print(f"Round {com_round}/{communication_rounds}")
            print("-" * 10)

            self.communication_round(epochs)

            # broadcast shared params to clients (keep BN local)
            for client in self.clients:
                client.set_model(self.model)

            out = self.validation_round_personalized(verbose=True, weighted=True)
            self.personalized_history.append(out["summary"])

        self.train_time = time.perf_counter() - start

        self.client_results = [client.get_validation_results() for client in self.clients]
        # NO saving (per your request)
        return self.personalized_history, self.client_results
