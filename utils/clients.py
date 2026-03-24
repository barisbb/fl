import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.poolformer import PoolFormer
from utils.pytorch_models import ResNet50


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


TOP10_CLASSES = [
    "airplane",
    "bicycle",
    "bird",
    "bus",
    "car",
    "cat",
    "dog",
    "horse",
    "person",
    "train",
]


class DomainNetDataset(Dataset):
    """
    Expected folder structure:

    /path/to/domainnet/
        clipart/
            airplane/
            bicycle/
            ...
        infograph/
        painting/
        quickdraw/
        real/
        sketch/
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        domain: str,
        split: str = "train",
        train_ratio: float = 0.8,
        random_state: int = 42,
        classes: Optional[list[str]] = None,
        transform=None,
    ):
        self.root = Path(root)
        self.domain = domain
        self.domain_dir = self.root / domain
        self.split = split
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.transform = transform
        self.classes = classes if classes is not None else TOP10_CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if not self.domain_dir.exists():
            raise FileNotFoundError(
                f"Domain folder not found: {self.domain_dir}\n"
                f"Expected structure like: {self.root}/clipart/class_name/image.jpg"
            )

        all_samples = []
        all_targets = []

        for cls_name in self.classes:
            cls_dir = self.domain_dir / cls_name
            if not cls_dir.exists():
                continue

            for file_path in cls_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.IMG_EXTENSIONS:
                    all_samples.append(str(file_path))
                    all_targets.append(self.class_to_idx[cls_name])

        if len(all_samples) == 0:
            raise RuntimeError(
                f"No images found for domain '{domain}' under {self.domain_dir} "
                f"for classes {self.classes}"
            )

        # Stratified split when possible
        unique_classes = set(all_targets)
        can_stratify = len(unique_classes) > 1
        if can_stratify:
            counts = {c: all_targets.count(c) for c in unique_classes}
            if min(counts.values()) < 2:
                can_stratify = False

        train_samples, test_samples, train_targets, test_targets = train_test_split(
            all_samples,
            all_targets,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=all_targets if can_stratify else None,
        )

        if split == "train":
            self.samples = train_samples
            self.targets = train_targets
        elif split == "test":
            self.samples = test_samples
            self.targets = test_targets
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, target


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
        lmdb_path: str,
        val_path: str,
        csv_path: str,
        batch_size: int = 512,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.CrossEntropyLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 10,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "clipart",
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.device = device

        self.bn_keys = get_bn_state_dict_keys(self.model)

        transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])

        self.dataset = DomainNetDataset(
            root=lmdb_path,
            domain=csv_path,
            split="train",
            train_ratio=0.8,
            random_state=42,
            classes=TOP10_CLASSES,
            transform=transform,
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

        self.validation_set = DomainNetDataset(
            root=lmdb_path,
            domain=csv_path,
            split="test",
            train_ratio=0.8,
            random_state=42,
            classes=TOP10_CLASSES,
            transform=transform,
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

        print(
            f"Client {self.dataset_filter}: "
            f"train_samples={len(self.dataset)}, "
            f"test_samples={len(self.validation_set)}"
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

        mod_after = self._get_mod_vector()
        dmod = mod_after - mod_before
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
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc=f"{self.dataset_filter} test"):
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(data)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": float(accuracy)}


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
        dataset_filter: str = "clipart",
        state_dict_path: str = None,
        results_path: str = None,
        align_power: float = 2.0,
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter

        self.aggregator = Aggregator(eps=1e-12, align_power=align_power, clip_align_min=0.0)

        self.clients = [
            FLCLient(
                copy.deepcopy(self.model),
                lmdb_path,
                val_path,
                csv_path,
                batch_size=batch_size,
                num_workers=2,
                num_classes=num_classes,
                dataset_filter=csv_path,
                device=self.device,
            )
            for csv_path in csv_paths
        ]

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

            for client in self.clients:
                client.set_model(self.model)

            client_reports = []
            client_sizes = []
            for client in self.clients:
                rep = client.validation_personalized()
                client_reports.append(rep)
                client_sizes.append(len(client.validation_set))
                print(f"{client.dataset_filter} accuracy: {rep['accuracy']:.4f}")

            weights = np.array(client_sizes, dtype=np.float64)
            weights = weights / weights.sum()

            summary = {
                "accuracy": float(np.sum(weights * np.array([r["accuracy"] for r in client_reports], dtype=np.float64))),
            }

            print("\n=== Overall Weighted Accuracy ===")
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
            print(f"Client {i} ({client.dataset_filter}) | train_size={sz} | cos={cos_val:.4f} | weight={w:.4f}")
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
        res = {'train_time': self.train_time}
        torch.save(res, self.results_path)
