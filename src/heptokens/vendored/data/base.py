"""Generic Lightning DataModule base for all map-style datasets."""

from abc import ABC, abstractmethod
from functools import partial

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from heptokens.vendored.data.collation import collate_and_transform


class BaseMapModule(LightningDataModule, ABC):
    """Base LightningDataModule with shared DataLoader construction logic.

    Modality-agnostic: handles DataLoader creation, transforms, and worker
    configuration. Subclasses implement ``setup`` to populate ``self.train_set``,
    ``self.valid_set``, and ``self.test_set``, and override ``get_data_sample``
    to return a representative tensor for model initialisation.

    ``n_classes`` is optional — only needed for classification tasks.
    """

    def __init__(
        self,
        *,
        n_classes: int | None = None,
        num_workers: int = 6,
        batch_size: int = 1000,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        multiprocessing_context: str | None = None,
        transforms: dict | None = None,
        **data_config,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        self.multiprocessing_context = multiprocessing_context
        self.transforms = transforms
        self.data_config = data_config

        self.train_set: Dataset
        self.valid_set: Dataset
        self.test_set: Dataset

    @abstractmethod
    def setup(self, stage: str) -> None:
        pass

    @abstractmethod
    def get_data_sample(self) -> torch.Tensor:
        """Return a representative input tensor for model initialisation."""

    def get_n_classes(self) -> int | None:
        return self.n_classes

    def _get_dataloader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.valid_set, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_set, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
