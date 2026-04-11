import copy
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50
from utils.pytorch_utils import (
    get_classification_report,
    init_results,
    print_micro_macro,
    update_results,
)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_modulation_key(k: str) -> bool:
    return ".mod" in k


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


def dirichlet_non_iid_split(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float = 0.3,
    min_size: int = 10,
    seed: int = 42,
) -> list[list[int]]:
    """
    Non-IID split using a Dirichlet distribution over class proportions.

    Smaller alpha -> more skewed / more non-IID
    Larger alpha  -> closer to IID
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_idxs = np.where(labels == class_id)[0]
            rng.shuffle(class_idxs)

            if len(class_idxs) == 0:
                continue

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
            class_split = np.split(class_idxs, split_points)

            for client_id, idxs in enumerate(class_split):
                client_indices[client_id].extend(idxs.tolist())

        sizes = [len(idxs) for idxs in client_indices]
        if min(sizes) >= min_size:
            break

    for idxs in client_indices:
        rng.shuffle(idxs)

    return client_indices


def print_client_distribution(client_name: str, subset_indices: list[int], targets: list[int], num_classes: int):
    subset_targets = [targets[i] for i in subset_indices]
    counts = Counter(subset_targets)
    print(f"\n{client_name} class distribution:")
    for c in range(num_classes):
        print(f"  class {c}: {counts.get(c, 0)}")
    print(f"  total: {len(subset_indices)}")


class EuroSATOneHotDataset(Dataset):
    def __init__(self, subset: Subset, num_classes: int = 10):
        self.subset = subset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[label] = 1.0
        return image, one_hot


class Aggregator:
    def __init__(
        self,
        eps: float = 1e-12,
        align_power: float = 2.0,
        clip_align_min: float = 0.0,
    ) -> None:
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
        train_subset: Subset,
        test_subset: Subset,
        lmdb_path: str,
        val_path: str,
        csv_path: str,
        batch_size: int = 512,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.BCEWithLogitsLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 10,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "client",
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

        self.bn_keys = get_bn_state_dict_keys(self.model)

        self.dataset = EuroSATOneHotDataset(train_subset, num_classes=self.num_classes)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
            pin_memory=True,
        )

        self.validation_set = EuroSATOneHotDataset(test_subset, num_classes=self.num_classes)
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
            if is_modulation_key(k):
                continue
            if k in self.bn_keys:
                continue
            local_sd[k] = global_sd[k].detach().clone()

        self.model.load_state_dict(local_sd, strict=True)

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
            v = self._get_mod_vector()
            vn = v.norm()
            if float(vn.item()) < 1e-12:
                return torch.ones(1)
            return v / (vn + 1e-12)
        return d / (n + 1e-12)

    def train_one_round(self, epochs: int, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())
        mod_before = self._get_mod_vector()

        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)
            self.train_epoch()

        if validate:
            report = self.validation_personalized()
            self.results = update_results(self.results, report, self.num_classes)

        mod_after = self._get_mod_vector()
        dmod = (mod_after - mod_before)
        mod_unit_dir = self._unit_dir_from_delta(dmod)

        state_after = self.model.state_dict()

        model_update = {}
        for key, value_before in state_before.items():
            if is_modulation_key(key):
                continue
            if key in self.bn_keys:
                continue

            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(torch.DoubleTensor)
            model_update[key] = diff

        return model_update, len(self.dataset), mod_unit_dir

    def train_epoch(self):
        self.model.train()
        for data, labels in tqdm(self.train_loader, desc=f"training-{self.dataset_filter}"):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

    def validation_personalized(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc=f"{self.dataset_filter} test"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.cpu().numpy()

                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()

                predicted_probs += list(probs)
                y_true += list(labels)

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
        num_classes: int = 10,
        dataset_filter: str = "eurosat",
        state_dict_path: Optional[str] = None,
        results_path: Optional[str] = None,
        align_power: float = 2.0,
        dirichlet_alpha: float = 0.3,
        min_client_samples: int = 20,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter

        self.aggregator = Aggregator(eps=1e-12, align_power=align_power, clip_align_min=0.0)
        self.results = init_results(self.num_classes)

        transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])

        full_dataset = datasets.EuroSAT(
            root="./data",
            download=True,
            transform=transform
        )

        total_size = len(full_dataset)
        all_indices = np.arange(total_size)

        # EuroSAT labels
        # For torchvision datasets like EuroSAT, labels are in .targets
        all_targets = np.array(full_dataset.targets)

        rng = np.random.default_rng(seed)

        # Global train/test split first
        rng.shuffle(all_indices)
        train_size = int(0.8 * total_size)
        train_indices = all_indices[:train_size]
        test_indices = all_indices[train_size:]

        train_targets = all_targets[train_indices]

        # Non-IID split among clients on the TRAIN portion only
        client_relative_splits = dirichlet_non_iid_split(
            labels=train_targets,
            num_clients=len(csv_paths),
            num_classes=num_classes,
            alpha=dirichlet_alpha,
            min_size=min_client_samples,
            seed=seed,
        )

        client_train_splits = [train_indices[np.array(rel_idxs)] for rel_idxs in client_relative_splits]

        for i, csv_path in enumerate(csv_paths):
            print_client_distribution(
                client_name=csv_path,
                subset_indices=client_train_splits[i].tolist(),
                targets=full_dataset.targets,
                num_classes=num_classes,
            )

        self.clients = [
            FLCLient(
                model=copy.deepcopy(self.model),
                train_subset=Subset(full_dataset, client_train_splits[i].tolist()),
                test_subset=Subset(full_dataset, test_indices.tolist()),
                lmdb_path=lmdb_path,
                val_path=val_path,
                csv_path=csv_path,
                batch_size=batch_size,
                num_workers=num_workers,
                num_classes=num_classes,
                dataset_filter=csv_path,
                device=self.device,
            )
            for i, csv_path in enumerate(csv_paths)
        ]

        self.validation_set = EuroSATOneHotDataset(
            Subset(full_dataset, test_indices.tolist()),
            num_classes=self.num_classes
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

        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if state_dict_path is None:
            if isinstance(model, ConvMixer):
                self.state_dict_path = f"checkpoints/global_convmixer_{dt}.pkl"
            elif isinstance(model, MlpMixer):
                self.state_dict_path = f"checkpoints/global_mlpmixer_{dt}.pkl"
            elif isinstance(model, PoolFormer):
                self.state_dict_path = f"checkpoints/global_poolformer_{dt}.pkl"
            elif isinstance(model, ResNet50):
                self.state_dict_path = f"checkpoints/global_resnet50_{dt}.pkl"
            else:
                self.state_dict_path = f"checkpoints/global_model_{dt}.pkl"
        else:
            self.state_dict_path = state_dict_path

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f"results/convmixer_results_{dt}.pkl"
            elif isinstance(model, MlpMixer):
                self.results_path = f"results/mlpmixer_results_{dt}.pkl"
            elif isinstance(model, PoolFormer):
                self.results_path = f"results/poolformer_results_{dt}.pkl"
            elif isinstance(model, ResNet50):
                self.results_path = f"results/resnet50_results_{dt}.pkl"
            else:
                self.results_path = f"results/model_results_{dt}.pkl"
        else:
            self.results_path = results_path

    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()
        last_out = None

        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            self.communication_round(epochs)

            for client in self.clients:
                client.set_model(self.model)

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
                "ap_mic": float(np.sum(weights * np.array([r["ap_mic"] for r in client_reports], dtype=np.float64))),
                "ap_mac": float(np.sum(weights * np.array([r["ap_mac"] for r in client_reports], dtype=np.float64))),
            }

            print("\n=== Overall Weighted Summary ===")
            print(summary)

            last_out = {"client_reports": client_reports, "summary": summary}

        self.train_time = time.perf_counter() - start
        return self.model, last_out

    def communication_round(self, epochs: int):
        updates_sizes_dirs = [client.train_one_round(epochs) for client in self.clients]

        weights, cosines = self.aggregator.compute_option_a_weights(updates_sizes_dirs)

        print("\nAggregation weights (Option A: size × similarity of Δmod)")
        for i, (client, w, cos_val, tup) in enumerate(zip(self.clients, weights, cosines, updates_sizes_dirs)):
            _, sz, _ = tup
            print(f"Client {i} ({client.dataset_filter}) | size={sz} | cos={cos_val:.4f} | weight={w:.4f}")
        print(f"Sum of weights: {sum(weights):.4f}\n")

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
        res = {"global": self.results, "clients": self.client_results, "train_time": self.train_time}
        torch.save(res, self.results_path)
