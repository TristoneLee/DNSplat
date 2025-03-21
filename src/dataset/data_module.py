import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..misc.step_tracker import StepTracker
from . import DatasetCfgWrapper, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfgs: list[DatasetCfgWrapper]
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "train", self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "train")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.train.batch_size,
                    shuffle=not isinstance(dataset, IterableDataset),
                    num_workers=self.data_loader_cfg.train.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.train),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.train),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def val_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "val", self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "val")
            data_loaders.append(
                DataLoader(
                    ValidationWrapper(dataset, 1),
                    self.data_loader_cfg.val.batch_size,
                    num_workers=self.data_loader_cfg.val.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.val),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.val),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def test_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "test", self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "test")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.test.batch_size,
                    num_workers=self.data_loader_cfg.test.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.test),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.data_loader_cfg.test),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]
