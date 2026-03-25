import copy
import random
import time
from datetime import datetime
from pathlib import Path

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


OFFICE_CALTECH10_CLASSES = [
    "backpack",
    "bike",
    "calculator",
    "headphones",
    "keyboard",
    "laptop_computer",
    "monitor",
    "mouse",
    "mug",
    "projector",
]


class OfficeCaltech10Dataset(Dataset):
    """
    Expected folder structure:

    /path/to/office_caltech_10/
        amazon/
            backpack/
            bike/
            ...
        caltech/
        dslr/
        webcam/
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        domain: str,
        classes: list[str],
        split: str = "train",
        train_ratio: float = 0.8,
        random_state: int = 42,
        transform=None,
    ):
        self.root = Path(root)
        self.domain = domain
        self.domain_dir = self.root / domain
        self.classes = classes
        self.split = split
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if not self.domain_dir.exists():
            raise FileNotFoundError(
                f"Domain folder not found: {self.domain_dir}\n"
                f"Expected structure like: {self.root}/amazon/class_name/image.jpg"
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
                f"No images found for domain '{domain}' under {self.domain_dir}"
            )

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


class FedAvgAggregator:
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def aggregate(self, updates_sizes: list[tuple[dict, int]]):
        total = sum(sz for _, sz in updates_sizes)
        weights = [sz / (total + self.eps) for _, sz in updates_sizes]

        keys = updates_sizes[0][0].keys()
        agg = {}
        for k in keys:
            s_k = None
            for (upd, _), w in zip(updates_sizes, weights):
                term = upd[k] * w
                s_k = term if s_k is None else (s_k + term)
            agg[k] = s_k
        return agg, weights


class FLCLient:
    def __init__(
        self,
        model: torch.nn.Module,
        lmdb_path: str,
        val_path: str,
        csv_path: str,
        classes: list[str],
        batch_size: int = 512,
        num_workers: int = 2,
        optimizer_constructor: callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.001, "weight_decay": 0},
        criterion_constructor: callable = torch.nn.CrossEntropyLoss,
        criterion_kwargs: dict = {"reduction": "mean"},
        num_classes: int = 10,
        device: torch.device = torch.device("cpu"),
        dataset_filter: str = "amazon",
    ) -> None:
        self.model = model
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_constructor = criterion_constructor
        self.criterion_kwargs = criterion_kwargs
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.device = device
        self.classes = classes

        transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])

        self.dataset = OfficeCaltech10Dataset(
            root=lmdb_path,
            domain=csv_path,
            classes=self.classes,
            split="train",
            train_ratio=0.8,
            random_state=42,
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

        self.validation_set = OfficeCaltech10Dataset(
            root=lmdb_path,
            domain=csv_path,
            classes=self.classes,
            split="test",
            train_ratio=0.8,
            random_state=42,
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

        # keep modulation parameters local, aggregate everything else like FedAvg
        for k in global_sd.keys():
            if is_modulation_key(k):
                continue
            local_sd[k] = global_sd[k].detach().clone()

        self.model.load_state_dict(local_sd, strict=True)

    def train_one_round(self, epochs: int):
        state_before = copy.deepcopy(self.model.state_dict())

        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)
            self.train_epoch()

        state_after = self.model.state_dict()
        model_update = {}

        # send only non-modulation updates
        for key, value_before in state_before.items():
            if is_modulation_key(key):
                continue

            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(torch.DoubleTensor)
            model_update[key] = diff

        return model_update, len(self.dataset)

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
        dataset_filter: str = "amazon",
        state_dict_path: str = None,
        results_path: str = None,
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.classes = OFFICE_CALTECH10_CLASSES

        self.aggregator = FedAvgAggregator()

        self.clients = [
            FLCLient(
                copy.deepcopy(self.model),
                lmdb_path,
                val_path,
                csv_path,
                classes=self.classes,
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
                self.state_dict_path = f'checkpoints/global_resnet50_{dt}.pkl'

        if results_path is None:
            if isinstance(model, ConvMixer):
                self.results_path = f'results/convmixer_results_{dt}.pkl'
            elif isinstance(model, MlpMixer):
                self.results_path = f'results/mlpmixer_results_{dt}.pkl'
            elif isinstance(model, PoolFormer):
                self.results_path = f'results/poolformer_results_{dt}.pkl'
            elif isinstance(model, ResNet50):
                self.results_path = f'results/resnet50_results_{dt}.pkl'

    def train(self, communication_rounds: int, epochs: int):
        start = time.perf_counter()
        last_out = None

        for com_round in range(1, communication_rounds + 1):
            print("Round {}/{}".format(com_round, communication_rounds))
            print("-" * 10)

            self.communication_round(epochs)

            # broadcast shared params while keeping modulation local
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
        updates_sizes = [client.train_one_round(epochs) for client in self.clients]

        update_aggregation, weights = self.aggregator.aggregate(updates_sizes)

        print("\nFedAvg aggregation weights (non-mod parameters only)")
        for i, (client, w, tup) in enumerate(zip(self.clients, weights, updates_sizes)):
            _, sz = tup
            print(f"Client {i} ({client.dataset_filter}) | train_size={sz} | weight={w:.4f}")
        print(f"Sum of weights: {sum(weights):.4f}\n")

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
