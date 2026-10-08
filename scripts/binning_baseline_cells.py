#!/usr/bin/env python
"""Per-pixel fixed-binning tokenizer baseline for COCOA calorimeter cells (no training).

Each of the 64 pixels of a patch is one token with K symbols: symbol 0 = empty pixel, symbols 1..K-1 = K-1
equal-width bins in the model's input space (E^0.3) between the q_lo and q_hi quantiles of non-zero training
pixels (values outside are clipped). Reconstruction is the bin centre, with occupancy = (symbol > 0). This is
log2(K) bits per pixel, i.e. an information fraction of 64 log2(K) / (32 * 64) = log2(K) / 32.

Evaluation is the cells VQ pipeline itself: compute_scan_point_metrics from scripts/vendored/ (same test files,
same first `--max_events` events, same decompression and jet energy IQR/median) with only its build_model
replaced by the binning stand-in. Bins come from the first `--fit_events` events of the first training file
of the same sparse dataset. Existing outputs are never overwritten.

  pixi run python scripts/binning_baseline_cells.py --K 1024 --data_dir <CELLS_SPARSE_DIR>
outputs: results/binning_baseline/cells/binning_uniform_K<K>/{test_metrics.json, bins.npz, config.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
REPO_B = REPO / "scripts/vendored"
DATA = None  # --data_dir: sparse cell dataset (<CELLS_SPARSE_DIR>, with train/ and test/)
SEED = 0

sys.path.insert(0, str(REPO_B))
import compute_cell_scan_resolution as csr  # noqa: E402
from heptokens.vendored.cocoa.data.hep4m_cell import SparseCellDataset  # noqa: E402


class _Encoder:
    def __init__(self, inner: torch.Tensor):
        self.inner = inner

    def __call__(self, batch):
        e = batch["energy"][..., : batch["energy"].shape[-1] // 2]
        idx = 1 + torch.bucketize(e.double().contiguous(), self.inner, right=True)
        return torch.where(e > 0, idx, torch.zeros_like(idx))  # [B, P, 64]


class _VQ:
    def __call__(self, idx):
        return idx, idx.permute(2, 0, 1), None  # one quantizer per pixel for codebook utilisation


class _Decoder:
    def __init__(self, centres: torch.Tensor):
        self.centres = centres  # [K], centres[0] = 0 (empty)

    def __call__(self, idx, batch):
        occ = torch.where(idx > 0, 10.0, -10.0)
        return torch.cat([self.centres[idx], occ], dim=-1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo: Path) -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def main():
    global DATA
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--q_lo", type=float, default=1e-4)
    p.add_argument("--q_hi", type=float, default=1 - 1e-4)
    p.add_argument("--fit_events", type=int, default=20000)
    p.add_argument("--max_events", type=int, default=100_000, help="same cap as the VQ cell evals")
    p.add_argument("--batch_size", type=int, default=256, help="same as the VQ cell evals")
    p.add_argument("--out_root", type=Path, default=REPO / "results/binning_baseline")
    args = p.parse_args()
    DATA = args.data_dir

    out = args.out_root / "cells" / f"binning_uniform_K{args.K}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already contains files; not overwriting")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    fit_meta = sorted((DATA / "train").glob("*.meta.json"))[0]
    fit = SparseCellDataset(str(fit_meta), num_samples=args.fit_events).values.astype(np.float64)
    fit = fit[fit > 0]
    lo, hi = np.quantile(fit, [args.q_lo, args.q_hi])
    edges = np.linspace(lo, hi, args.K)  # K-1 bins for symbols 1..K-1
    centres = np.concatenate([[0.0], 0.5 * (edges[:-1] + edges[1:])])
    np.savez(out / "bins.npz", edges=edges, centres=centres)

    inner = torch.as_tensor(edges[1:-1], dtype=torch.float64)
    enc, vq, dec = _Encoder(inner), _VQ(), _Decoder(torch.as_tensor(centres, dtype=torch.float32))
    csr.build_model = lambda ckpt_path: (enc, vq, dec, 1, args.K, 64)
    metrics = csr.compute_scan_point_metrics(
        "binning", None, downsample=1, batch_size=args.batch_size, max_events=args.max_events,
        label=out.name, test_path=str(DATA / "test"), sparse=True,
    )
    metrics.update({"information_fraction": float(np.log2(args.K) / 32), "bits_per_patch": 64 * float(np.log2(args.K))})

    config = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "command": " ".join(sys.argv), "seed": SEED, "test_path": str(DATA / "test"), "fit_file": str(fit_meta),
        "n_fit_nonzero_pixels": int(len(fit)), "fit_range_E0.3": [float(lo), float(hi)],
        "metric_code": str(Path(csr.__file__).resolve()), "metric_code_sha256": sha256(Path(csr.__file__)),
        "git": git_state(REPO), "git_heptokens_cocoa": git_state(REPO_B),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("energy_resolution_iqr", "energy_response_median", "n_test_events",
                                              "n_valid_events", "information_fraction")}, indent=2))


if __name__ == "__main__":
    main()
