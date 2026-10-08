"""Shared utilities for loading COCOA ROOT files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import uproot

log = logging.getLogger(__name__)

TRACK_FEATURES = ["track_pt", "track_eta", "track_phi", "track_d0", "track_z0"]

CELL_FEATURES = [
    "cell_eta", "cell_phi", "cell_e", "cell_che", "cell_nue",
    "cell_x", "cell_y", "cell_z", "cell_calo_region",
]

TOPO_FEATURES = [
    "topo_eta", "topo_phi", "topo_e", "topo_rho",
    "topo_sigma_eta", "topo_sigma_phi", "topo_ecal_e", "topo_hcal_e",
]

PARTICLE_FEATURES = [
    "particle_pt", "particle_eta", "particle_phi", "particle_e",
    "particle_dep_e", "particle_pdgid",
]

TREE_NAME = "EventTree"


def root_files_from_dir(directory: str | Path) -> list[Path]:
    """Discover and sort ROOT files in a directory."""
    d = Path(directory)
    files = sorted(d.glob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {d}")
    return files


def load_root_arrays(
    file_path: str | Path,
    branches: list[str],
    max_entries: int | None = None,
) -> dict[str, np.ndarray]:
    """Load jagged arrays from a ROOT file.

    Args:
        file_path: Path to the ROOT file.
        branches: List of branch names to read.
        max_entries: Optional limit on number of entries to read.

    Returns:
        Dictionary mapping branch name to numpy object arrays (jagged).
    """
    with uproot.open(f"{file_path}:{TREE_NAME}") as tree:
        arrays = tree.arrays(branches, library="np", entry_stop=max_entries)
    return dict(zip(branches, [arrays[b] for b in branches]))


def pad_jagged_to_fixed(
    arrays: dict[str, np.ndarray],
    features: list[str],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length arrays to fixed size and build mask.

    Args:
        arrays: Dict of branch_name -> object array of 1D arrays (one per event).
        features: Ordered list of feature names to stack.
        max_length: Maximum sequence length (pad/truncate to this).

    Returns:
        csts: float32 array of shape [n_events, max_length, n_features]
        mask: bool array of shape [n_events, max_length]
    """
    n_events = len(arrays[features[0]])
    n_features = len(features)

    csts = np.zeros((n_events, max_length, n_features), dtype=np.float32)
    mask = np.zeros((n_events, max_length), dtype=bool)

    for i in range(n_events):
        n = min(len(arrays[features[0]][i]), max_length)
        if n == 0:
            continue
        mask[i, :n] = True
        for f_idx, feat in enumerate(features):
            csts[i, :n, f_idx] = arrays[feat][i][:n]

    return csts, mask
