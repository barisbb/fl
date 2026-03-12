import copy
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Container, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from timm.models.convmixer import ConvMixer
from timm.models.mlp_mixer import MlpMixer

from models.poolformer import PoolFormer
from utils.BENv2_dataset import BENv2DataSet
from utils.pytorch_models import ResNet50
from utils.pytorch_utils import (
    get_classification_report,
    init_results,
    print_micro_macro,
    update_results,
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
        seasons: Optional[Container] | str = None,
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


class FedAWAAggregator:
    """
    FedAWA-style server aggregator.

    Idea from paper:
      - client vector: tau_k = theta_k - theta_g
      - optimize aggregation weights lambda instead of using fixed size
      - objective uses:
            sum_k lambda_k ||tau_k - tau_g||^2
        where tau_g = sum_k lambda_k tau_k
        plus a regularizer to keep merged model aligned with current global model.
    """

    def __init__(
        self,
        weight_opt_steps: int = 20,
        weight_opt_lr: float = 0.05,
        reg_coeff: float = 1.0,
        eps: float = 1e-12,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.weight_opt_steps = weight_opt_steps
        self.weight_opt_lr = weight_opt_lr
        self.reg_coeff = reg_coeff
        self.eps = eps
        self.device = device

    @staticmethod
    def _is_float_tensor(x: torch.Tensor) -> bool:
        return torch.is_tensor(x) and x.dtype.is_floating_point

    def _float_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in state_dict.items():
            if self._is_float_tensor(v):
                out[k] = v.detach().cpu().float().clone()
        return out

    def _client_vectors(
        self,
        local_float_states: list[dict[str, torch.Tensor]],
        global_float_state: dict[str, torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        client_vecs = []
        for local_sd in local_float_states:
            vec = {}
            for k in global_float_state.keys():
                vec[k] = local_sd[k] - global_float_state[k]
            client_vecs.append(vec)
        return client_vecs

    def _weighted_sum_float_states(
        self,
        states: list[dict[str, torch.Tensor]],
        weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {}
        for k in states[0].keys():
            s = None
            for i, sd in enumerate(states):
                term = weights[i] * sd[k]
                s = term if s is None else (s + term)
            out[k] = s
        return out

    def _squared_distance(
        self,
        a: dict[str, torch.Tensor],
        b: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = None
        for k in a.keys():
            diff = a[k] - b[k]
            val = (diff * diff).sum()
            total = val if total is None else (total + val)
        return total

    def _cosine_distance(
        self,
        a: dict[str, torch.Tensor],
        b: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        dot = None
        na = None
        nb = None
        for k in a.keys():
            av = a[k].reshape(-1)
            bv = b[k].reshape(-1)

            cur_dot = (av * bv).sum()
            cur_na = (av * av).sum()
            cur_nb = (bv * bv).sum()

            dot = cur_dot if dot is None else (dot + cur_dot)
            na = cur_na if na is None else (na + cur_na)
            nb = cur_nb if nb is None else (nb + cur_nb)

        denom = torch.sqrt(na + self.eps) * torch.sqrt(nb + self.eps)
        cos_sim = dot / (denom + self.eps)
        return 1.0 - cos_sim

    def optimize_weights(
        self,
        local_states: list[dict[str, torch.Tensor]],
        global_state: dict[str, torch.Tensor],
        client_sizes: list[int],
    ):
        """
        Returns optimized lambda and some debug info.

        Initialization is dataset-size weights (same spirit as FedAvg / paper init),
        then optimized with softmax-parameterized logits so weights stay on simplex.
        """
        local_float_states = [self._float_state_dict(sd) for sd in local_states]
        global_float_state = self._float_state_dict(global_state)
        client_vecs = self._client_vectors(local_float_states, global_float_state)

        size_weights = torch.tensor(client_sizes, dtype=torch.float32)
        size_weights = size_weights / size_weights.sum()

        logits = torch.log(size_weights + self.eps).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=self.weight_opt_lr)

        for _ in range(self.weight_opt_steps):
            optimizer.zero_grad()

            lamb = torch.softmax(logits, dim=0)

            # tau_g = sum_k lambda_k * tau_k
            tau_g = self._weighted_sum_float_states(client_vecs, lamb)

            # sum_k lambda_k ||tau_k - tau_g||^2
            align_loss = None
            for i in range(len(client_vecs)):
                term = lamb[i] * self._squared_distance(client_vecs[i], tau_g)
                align_loss = term if align_loss is None else (align_loss + term)

            # d(sum_k lambda_k theta_k, theta_g), using 1 - cosine similarity
            merged_model = self._weighted_sum_float_states(local_float_states, lamb)
            reg_loss = self._cosine_distance(merged_model, global_float_state)

            loss = align_loss + self.reg_coeff * reg_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            lamb = torch.softmax(logits, dim=0).detach().cpu()

            tau_g = self._weighted_sum_float_states(client_vecs, lamb)

            client_to_tau_g = []
            for i in range(len(client_vecs)):
                dist_i = self._squared_distance(client_vecs[i], tau_g)
                client_to_tau_g.append(float(dist_i.item()))

            merged_model = self._weighted_sum_float_states(local_float_states, lamb)
            reg_loss = self._cosine_distance(merged_model, global_float_state)

        return (
            lamb.numpy().tolist(),
            size_weights.numpy().tolist(),
            client_to_tau_g,
            float(reg_loss.item()),
        )

    def aggregate_full_models(
        self,
        local_states: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        """
        Weighted averaging of client full models.
        Float tensors are averaged.
        Non-float tensors (e.g., num_batches_tracked) are copied from the first client.
        """
        w = [float(x) for x in weights]
        out = {}
        keys = local_states[0].keys()

        for k in keys:
            first = local_states[0][k]
            if torch.is_tensor(first) and first.dtype.is_floating_point:
                agg = None
                for i, sd in enumerate(local_states):
                    term = sd[k].detach().cpu().float() * w[i]
                    agg = term if agg is None else (agg + term)
                out[k] = agg.to(dtype=first.dtype)
            else:
                out[k] = copy.deepcopy(first)

        return out


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
        device: torch.device = torch.device("cpu"),
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

        self.dataset = BENv2DataSet(
            data_dirs=data_dirs,
            split="train",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                pd.read_parquet(data_dirs["metadata_parquet"]),
                countries=[csv_path],
                seasons=["Summer"],
            ),
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
            patch_prefilter=PreFilter(
                pd.read_parquet(data_dirs["metadata_parquet"]),
                countries=[csv_path],
                seasons="Summer",
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

    def set_model(self, model: torch.nn.Module):
        self.model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)

    def train_one_round(self, epochs: int, validate: bool = False):
        self.optimizer = self.optimizer_constructor(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_constructor(**self.criterion_kwargs)

        for epoch in range(1, epochs + 1):
            print("Epoch {}/{}".format(epoch, epochs))
            print("-" * 10)
            self.train_epoch()

        if validate:
            report = self.validation_personalized()
            self.results = update_results(self.results, report, self.num_classes)

        return copy.deepcopy(self.model.state_dict()), len(self.dataset)

    def train_epoch(self):
        self.model.train()
        for _, batch in enumerate(tqdm(self.train_loader, desc="training")):
            data = batch[1].to(self.device, non_blocking=True)
            labels = batch[4]

            label_new = np.copy(labels)
            label_new = torch.from_numpy(label_new).float().to(self.device, non_blocking=True)

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
                data = batch[1].to(self.device, non_blocking=True)
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
        fedawa_weight_opt_steps: int = 20,
        fedawa_weight_opt_lr: float = 0.05,
        fedawa_reg_coeff: float = 1.0,
    ) -> None:
        self.model = model
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.num_classes = num_classes
        self.dataset_filter = dataset_filter
        self.results = init_results(self.num_classes)

        self.aggregator = FedAWAAggregator(
            weight_opt_steps=fedawa_weight_opt_steps,
            weight_opt_lr=fedawa_weight_opt_lr,
            reg_coeff=fedawa_reg_coeff,
            device=torch.device("cpu"),
        )

        self.clients = [
            FLCLient(
                copy.deepcopy(self.model),
                lmdb_path,
                val_path,
                csv_path,
                num_classes=num_classes,
                dataset_filter=csv_path,
                device=self.device,
            )
            for csv_path in csv_paths
        ]

        self.validation_set = BENv2DataSet(
            data_dirs=data_dirs,
            split="test",
            img_size=(10, 120, 120),
            include_snowy=False,
            include_cloudy=False,
            patch_prefilter=PreFilter(
                pd.read_parquet(data_dirs["metadata_parquet"]),
                countries=["Finland", "Ireland", "Serbia", "Austria", "Belgium", "Lithuania", "Portugal", "Switzerland"],
                seasons="Summer",
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

            round_info = self.communication_round(epochs)

            # broadcast updated global model to all clients
            for client in self.clients:
                client.set_model(self.model)

            # per-client evaluation
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

            last_out = {
                "client_reports": client_reports,
                "summary": summary,
                "fedawa_debug": round_info,
            }

        self.train_time = time.perf_counter() - start
        return self.model, last_out

    def communication_round(self, epochs: int):
        global_state_before = copy.deepcopy(self.model.state_dict())

        # collect local full models
        local_models_sizes = [client.train_one_round(epochs) for client in self.clients]
        local_states = [x[0] for x in local_models_sizes]
        client_sizes = [x[1] for x in local_models_sizes]

        fedawa_weights, init_size_weights, client_tau_distances, reg_loss = self.aggregator.optimize_weights(
            local_states=local_states,
            global_state=global_state_before,
            client_sizes=client_sizes,
        )

        print("\nAggregation weights (FedAWA optimized)")
        for i, (client, size_w, fed_w, tau_dist, sz) in enumerate(
            zip(self.clients, init_size_weights, fedawa_weights, client_tau_distances, client_sizes)
        ):
            print(
                f"Client {i} ({client.dataset_filter}) | "
                f"size={sz} | init_size_w={size_w:.4f} | "
                f"tau_dist={tau_dist:.6f} | fedawa_w={fed_w:.4f}"
            )
        print(f"Regularization cosine-distance: {reg_loss:.6f}")
        print(f"Sum of weights: {sum(fedawa_weights):.6f}\n")

        aggregated_state_cpu = self.aggregator.aggregate_full_models(local_states, fedawa_weights)

        # move aggregated model to actual device and load
        final_state = {}
        current_state = self.model.state_dict()
        for k in current_state.keys():
            if k in aggregated_state_cpu:
                v = aggregated_state_cpu[k]
                if torch.is_tensor(v):
                    final_state[k] = v.to(device=current_state[k].device, dtype=current_state[k].dtype)
                else:
                    final_state[k] = v
            else:
                final_state[k] = current_state[k]

        self.model.load_state_dict(final_state, strict=True)

        return {
            "weights": fedawa_weights,
            "initial_size_weights": init_size_weights,
            "client_tau_distances": client_tau_distances,
            "reg_loss": reg_loss,
        }

    def save_state_dict(self):
        if not Path(self.state_dict_path).parent.is_dir():
            Path(self.state_dict_path).parent.mkdir(parents=True)
        torch.save(self.model.state_dict(), self.state_dict_path)

    def save_results(self):
        if not Path(self.results_path).parent.is_dir():
            Path(self.results_path).parent.mkdir(parents=True)
        res = {
            "global": self.results,
            "train_time": self.train_time,
        }
        torch.save(res, self.results_path)
