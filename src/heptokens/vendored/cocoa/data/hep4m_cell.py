"""Dataset and DataModule for HEP4M calorimeter cell data (patch representation).

Loads preprocessed HDF5 files produced by scripts/preprocess_hep4m_cells.py.
Each event is represented as 156 fixed-size patch tokens of dimension 64
(the maximum patch pixel count across all 6 calorimeter layers).

Batch dict returned per event:
  "energy": float32 Tensor[156, 67]  — [64 pixel energies (energy^0.3) | eta, cos_phi, sin_phi]
  "mask":   bool    Tensor[156]      — True if patch has any energy deposit (non-empty)

The mask is energy-based: patches with zero total energy are masked out.
This handles both genuinely empty spatial regions and the zero-padded
pixels within layer-5 patches (which have 16 real pixels, padded to 64).
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, random_split

from heptokens.vendored.data.base import BaseMapModule

log = logging.getLogger(__name__)


def _make_item(energy: torch.Tensor, pos: torch.Tensor) -> dict:
    indicator = (energy > 0).float()  # [156, n_pixels] binary
    combined = torch.cat([energy, indicator], dim=-1)  # [156, 2*n_pixels]
    mask = energy.sum(dim=-1) > 0  # [156] patch-level bool
    return {"energy": combined, "positions": pos, "mask": mask}


class CellDataset(Dataset):
    """Load HEP4M cell patch data from a preprocessed HDF5 file.

    The HDF5 file must contain:
      - "cell_patches": float32 array [N, 156, 64]
      - "patch_positions": float32 array [N, 156, 3]

    Args:
        file_path: Path to the HDF5 file produced by preprocess_hep4m_cells.py.
        num_samples: If set, only load the first ``num_samples`` events.
    """

    def __init__(self, file_path: str, num_samples: int | None = None) -> None:
        super().__init__()
        self.file_path = file_path

        with h5py.File(file_path, "r") as f:
            self.patches = f["cell_patches"][:num_samples]  # [N, 156, 64] float32
            self.positions = f["patch_positions"][:num_samples]  # [N, 156, 3] float32

        log.info(
            f"Loaded {len(self.patches)} events from {file_path} "
            f"(csts shape: {self.patches.shape[1:]}, positions shape: {self.positions.shape[1:]})"
        )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        energy = torch.tensor(self.patches[idx], dtype=torch.float32)  # [156, n_pixels]
        pos = torch.tensor(self.positions[idx], dtype=torch.float32)  # [156, 3]
        return _make_item(energy, pos)


class SparseCellDataset(Dataset):
    """Same events and outputs as ``CellDataset``, read from the sparse format
    written by scripts/sparsify_cell_h5.py (non-zero pixels only, ~16 GB for
    20M events instead of ~800 GB dense, plus ~37 GB of patch positions).

    Args:
        meta_path: ``<stem>.meta.json`` written by sparsify_cell_h5.py.
        num_samples: If set, only use the first ``num_samples`` events.
    """

    def __init__(self, meta_path: str, num_samples: int | None = None) -> None:
        super().__init__()
        meta_path = Path(meta_path)
        prefix = meta_path.parent / meta_path.name.removesuffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        self.n_patches, self.n_pixels = meta["n_patches"], meta["n_pixels"]

        n = meta["n_events"] if num_samples is None else min(num_samples, meta["n_events"])
        self.offsets = np.load(f"{prefix}.offsets.npy")[: n + 1]
        nnz = int(self.offsets[-1])
        self.values = np.load(f"{prefix}.values.npy", mmap_mode="r")[:nnz].copy()
        self.index = np.load(f"{prefix}.index.npy", mmap_mode="r")[:nnz].copy()
        # Read into RAM: per-event random reads from a network filesystem are ~500x slower.
        self.positions = np.load(f"{prefix}.positions.npy", mmap_mode="r")[:n].copy()
        log.info(f"Loaded {n} events ({nnz} non-zero pixels) from {prefix}")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, idx: int) -> dict:
        start, end = self.offsets[idx], self.offsets[idx + 1]
        dense = np.zeros(self.n_patches * self.n_pixels, dtype=np.float32)
        dense[self.index[start:end]] = self.values[start:end]
        energy = torch.from_numpy(dense.reshape(self.n_patches, self.n_pixels))
        pos = torch.tensor(self.positions[idx], dtype=torch.float32)
        return _make_item(energy, pos)


def _resolve_files(data_path: str | list, pattern: str = "*.h5") -> list[Path]:
    """Return a sorted list of data files from a file, directory, or list."""
    if isinstance(data_path, (list, tuple)):
        return sorted(Path(p) for p in data_path)
    p = Path(data_path)
    if p.is_dir():
        files = sorted(p.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No {pattern} files found in {p}")
        return files
    return [p]


class CellDataModule(BaseMapModule):
    """Lightning DataModule for HEP4M cell patch data.

    Two modes:
      * ``data_path``: a file, directory, or list of files, concatenated and
        randomly split into train / val / test (the original behaviour).
      * ``train_path`` / ``val_path`` / ``test_path``: each split read from its
        own files, no random split. Any of the three may be omitted (e.g. an
        eval job only needs ``test_path``).

    Args:
        data_path: Path to an HDF5 file, a directory of ``*.h5`` files, or a
            list of file paths.
        train_path, val_path, test_path: Per-split paths (same forms as
            ``data_path``); mutually exclusive with ``data_path``.
        sparse: Read the sparse format from scripts/sparsify_cell_h5.py
            (directories of ``*.meta.json``) instead of HDF5.
        train_frac: Fraction of events for training.
        val_frac: Fraction of events for validation.
        test_frac: Fraction of events for testing.
        seed: Random seed for the split.
        **kwargs: Forwarded to BaseMapModule (batch_size, num_workers, …).
    """

    def __init__(
        self,
        *,
        data_path: str | list | None = None,
        train_path: str | list | None = None,
        val_path: str | list | None = None,
        test_path: str | list | None = None,
        sparse: bool = False,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        split_paths = {"train": train_path, "val": val_path, "test": test_path}
        has_splits = any(p is not None for p in split_paths.values())
        if (data_path is None) == (not has_splits):
            raise ValueError("Provide either data_path or train_path/val_path/test_path, not both")

        self.data_path = data_path
        self.sparse = sparse
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        if has_splits:
            self.train_set, self.valid_set, self.test_set = (
                self._load(p) if p is not None else None for p in split_paths.values()
            )
            return

        full_dataset = self._load(data_path)
        total = len(full_dataset)
        train_size = int(total * train_frac)
        val_size = int(total * val_frac)
        test_size = total - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.valid_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

    def _load(self, path: str | list) -> Dataset:
        if self.sparse:
            files = _resolve_files(path, "*.meta.json")
            datasets = [SparseCellDataset(str(f), **self.data_config) for f in files]
        else:
            files = _resolve_files(path)
            datasets = [CellDataset(str(f), **self.data_config) for f in files]
        log.info(f"CellDataModule: loaded {len(files)} file(s) from {path}")
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def get_data_sample(self) -> dict:
        split = next(s for s in (self.valid_set, self.test_set, self.train_set) if s is not None)
        return next(iter(split))

    def setup(self, stage: str) -> None:
        pass
