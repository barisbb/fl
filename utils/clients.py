import copy

import numpy as np
import torch
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
        """
        Creates a function that filters patches based on country and season.

        Args:
            metadata: The metadata DataFrame.
            countries: A country or list of countries to include. If None, all countries are included.
            seasons: A season or list of seasons to include. If None, all seasons are included.
        """
        # add season info to metadata
        # months 12, 1, 2 are winter, 3, 4, 5 are spring, 6, 7, 8 are summer, 9, 10, 11 are autumn
        # get month based on patch_id (patch_id 15:17)
        metadata["month"] = metadata["patch_id"].str[15:17].astype(int)
        metadata["season"] = pd.cut(
            metadata["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Autumn"],
            right=False,
        )
        # manually set all entries with month 12 to winter
        metadata.loc[metadata["month"] == 12, "season"] = "Winter"

        seasons = None if seasons is None else seasons if isinstance(seasons, Container) else [seasons]
        countries = None if countries is None else countries if isinstance(countries, Container) else [countries]

        def filter_fn(metadata_row) -> bool:
            # Order: 'patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name',
            #        'contains_seasonal_snow', 'contains_cloud_or_shadow', 'month',
            #        'season'
            row_country = metadata_row[3]
            row_season = metadata_row[9]
            # check if patch season is correct
            if seasons is not None and row_season not in seasons:
                return False
            # check if patch country is correct
            if countries is not None and row_country not in countries:
                return False
            return True

        self.filter_fn = filter_fn
        from tqdm import tqdm
        self.filtered_patches = set([x[0] for x in [x for x in metadata.values if filter_fn(x)]])
        print(f"Pre-filtered {len(self.filtered_patches)} patches based on country and season (split ignored)")

    def filter(self, patch_id: str) -> bool:
        return self.filter_fn(patch_id)

    def __call__(self, patch_id: str) -> bool:
        return patch_id in self.filtered_patches


class Aggregator:
    def __init__(self) -> None:
        pass

    def fed_avg_weighted(self, updates_and_sizes: list[tuple[dict, int]]):
        assert len(updates_and_sizes) > 0, "Trying to aggregate empty update list"

        total = sum(sz for _, sz in updates_and_sizes)
        keys = updates_and_sizes[0][0].keys()

        agg = {}
        for k in keys:
            # weighted sum
            s = None
            for upd, sz in updates_and_sizes:
                w = sz / total
                term = upd[k] * w
                s = term if s is None else (s + term)
            agg[k] = s
        return agg


def is_modulation_key(k: str) -> bool:
    return ".mod.gamma" in k or ".mod.beta" in k or ".mod." in k

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
        self.dataset = BENv2DataSet(
        data_dirs=data_dirs,
        # For Mars use these paths
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
        self.device = device

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
            if is_modulation_key(k):
                continue
            local_sd[k] = global_sd[k].detach().clone()

        self.model.load_state_dict(local_sd, strict=True)

    def train_one_round(self, epochs: int, validate: bool = False):
        state_before = copy.deepcopy(self.model.state_dict())

        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)

            self.train_epoch()
        
        if validate:
            report = self.validation_round()
            self.results = update_results(self.results, report, self.num_classes)

        state_after = self.model.state_dict()

        model_update = {}
        for key, value_before in state_before.items():
            if is_modulation_key(key):
                 continue  # <<< DO NOT SEND modulation updates
            value_after = state_after[key]
            diff = value_after.type(torch.DoubleTensor) - value_before.type(torch.DoubleTensor)
            model_update[key] = diff

        return model_update, len(self.dataset)

    def change_sizes(self, labels):
        new_labels=np.zeros((len(labels[0]),19))
        for i in range(len(labels[0])): #128
            for j in range(len(labels)): #19
                new_labels[i,j] =  int(labels[j][i])
        return new_labels
    
    def train_epoch(self):
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data = batch[1]
            labels = batch[4]
            
            data = data.cuda()
            label_new=np.copy(labels)
            label_new = torch.from_numpy(label_new).cuda()
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, label_new)
            loss.backward()
            self.optimizer.step()

    # --------------------------
    # NEW: per-client evaluation (BN shared, mod local)
    # --------------------------
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
        results_path: str = None
    ) -> None:
        self.model = model
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)
        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.aggregator = Aggregator()
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

            # broadcast shared params (BN shared) while keeping modulation local
            for client in self.clients:
                client.set_model(self.model)

            # ---- per-client eval (mod local) ----
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

        # IMPORTANT: no checkpoints (as you requested earlier)
        # self.save_results()
        # self.save_state_dict()

        # return compatible with your main: (global_model, global_results)
        return self.model, last_out

    def change_sizes(self, labels):
        new_labels=np.zeros((len(labels[0]),19))
        for i in range(len(labels[0])): #128
            for j in range(len(labels)): #19
                new_labels[i,j] =  int(labels[j][i])
        return new_labels
    

    def validation_round(self):
        self.model.eval()
        y_true = []
        predicted_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="test")):
                data = batch[1].to(self.device)
                labels = batch[4]
                label_new=np.copy(labels)

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

    def communication_round(self, epochs: int):
        updates_and_sizes = [client.train_one_round(epochs) for client in self.clients]
        update_aggregation = self.aggregator.fed_avg_weighted(updates_and_sizes)

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
        res = {'global':self.results, 'clients':self.client_results, 'train_time': self.train_time}
        torch.save(res, self.results_path)
