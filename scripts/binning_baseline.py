#!/usr/bin/env python
"""Per-feature fixed-binning tokenizer baseline (no training), evaluated exactly like the VQ tokenizers.

Every constituent feature is quantized on its own into K bins in the tokenizer's preprocessed (scaled)
space and reconstructed at the bin centre: D tokens of log2(K) bits per element, i.e. an information
fraction of D log2(K) / (32 D) = log2(K) / 32.

Reproducibility: data, preprocessing, the test events (first `max_batches` test batches) and the jet
metrics all come from a reference VQ run's stored fast-path-off test config, whose datamodule and
ReconstructionMonitor are instantiated unchanged; the output test_metrics.npz has the same keys and the
same events as every other point. Bin ranges are fixed from the training split of that same datamodule
config (never from the test split), read in order without shuffling. Edges/centres, the configuration,
input file hashes and the git commit are written next to the metrics. Existing outputs are never
overwritten.

  uniform  : K equal-width bins between the q_lo and q_hi training quantiles (outside values are clipped)
  quantile : K equal-population bins from the training distribution between q_lo and q_hi

usage: python scripts/binning_baseline.py --modality tracks --K 1024 [--mode uniform]
outputs: results/binning_baseline/<modality>/binning_<mode>_K<K>/{test_metrics/test_metrics.npz,
         bins.npz, config.json, results.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
REFERENCE = {  # stored fast-path-off test configs; their datamodule/monitor settings define the evaluation
    "tracks": "results/cocoa_scan/tracks/cocoa_tracks_cb4096_cd8_nq3_lr0.001/test_nofastpath/.hydra/config.yaml",
    "topos": "results/cocoa_scan/topos/cocoa_topos_cb4096_cd8_nq4_lr0.001/test_nofastpath/.hydra/config.yaml",
    "jetset": "results/vqvae_scan/vqvae_scan/transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj10000000"
              "/test_nofastpath/.hydra/config.yaml",
}
SEED = 0


class BinningTokenizer:
    """Stand-in for a LitVqVae: encode -> per-feature bin indices, decode -> bin centres (scaled space)."""

    def __init__(self, edges: np.ndarray, centres: np.ndarray):
        self.inner = torch.as_tensor(edges[:, 1:-1], dtype=torch.float64)  # [D, K-1]
        self.centres = torch.as_tensor(centres, dtype=torch.float32)  # [D, K]

    def encode(self, batch):
        x = batch["csts"].double()
        idx = torch.stack([torch.bucketize(x[..., j].contiguous(), self.inner[j], right=True)
                           for j in range(x.shape[-1])], dim=-1)
        return (idx,)

    def decode(self, idx, batch):
        return torch.stack([self.centres[j][idx[..., j]] for j in range(idx.shape[-1])], dim=-1)

    def log(self, *args, **kwargs):
        pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def fit_values(dm, fit_batches: int) -> np.ndarray:
    """Valid scaled constituents from the training split, in file order."""
    dm.setup("fit")
    loader = dm._get_dataloader(dm.train_set, shuffle=False, drop_last=False)
    vals = []
    for i, batch in enumerate(loader):
        if i >= fit_batches:
            break
        vals.append(batch["csts"][batch["mask"]].double().numpy())
    return np.concatenate(vals)


def make_bins(x: np.ndarray, K: int, mode: str, q_lo: float, q_hi: float):
    edges, centres = [], []
    for j in range(x.shape[1]):
        lo, hi = np.quantile(x[:, j], [q_lo, q_hi])
        if mode == "uniform":
            e = np.linspace(lo, hi, K + 1)
        else:
            e = np.quantile(x[:, j], np.linspace(q_lo, q_hi, K + 1))
        edges.append(e)
        centres.append(0.5 * (e[:-1] + e[1:]))
    return np.array(edges), np.array(centres)


def normalized_mae(o: np.ndarray, r: np.ndarray) -> float:
    """Mean over features of MAE / central-68% half-width (same definition as plot_design_rules.py)."""
    lo, hi = np.percentile(o, [16, 84], axis=0)
    return float(np.mean(np.mean(np.abs(r - o), axis=0) / ((hi - lo) / 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=sorted(REFERENCE), required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--mode", choices=["uniform", "quantile"], default="uniform")
    p.add_argument("--q_lo", type=float, default=1e-4)
    p.add_argument("--q_hi", type=float, default=1 - 1e-4)
    p.add_argument("--fit_batches", type=int, default=100, help="training batches used to set the bin ranges")
    p.add_argument("--out_root", type=Path, default=REPO / "results/binning_baseline")
    args = p.parse_args()

    out = args.out_root / args.modality / f"binning_{args.mode}_K{args.K}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already contains files; not overwriting")
    out.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(SEED, workers=True)
    torch.use_deterministic_algorithms(True)
    ref = REPO / REFERENCE[args.modality]
    cfg = OmegaConf.load(ref)
    cfg.datamodule.pin_memory = False
    dm = instantiate(cfg.datamodule)
    monitor = instantiate(cfg.callbacks.reconstruction_monitor)
    monitor.save_metrics = True

    fit = fit_values(dm, args.fit_batches)
    edges, centres = make_bins(fit, args.K, args.mode, args.q_lo, args.q_hi)
    np.savez(out / "bins.npz", edges=edges, centres=centres)

    model = BinningTokenizer(edges, centres)
    trainer = SimpleNamespace(default_root_dir=str(out), logger=None, current_epoch=0, sanity_checking=False)
    dm.setup("test")
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(dm.test_dataloader()):
            if i >= monitor.max_batches:
                break
            monitor._on_batch_end(trainer, model, batch, i, stage="val")
            n_batches += 1
    monitor._on_epoch_end(trainer, model, stage="val")

    m = np.load(out / "test_metrics/test_metrics.npz")
    pr = m["pt_ratio"][np.isfinite(m["pt_ratio"]) & (m["pt_ratio"] > 0)]
    q25, q50, q75 = np.percentile(pr, [25, 50, 75])
    D = edges.shape[0]
    results = {
        "jet_iqr_over_median": float((q75 - q25) / q50), "jet_response_median": float(q50),
        "normalized_mae_mean": normalized_mae(m["cst_original"], m["cst_recon"]),
        "n_jets": int(len(m["pt_ratio"])), "n_constituents": int(len(m["cst_original"])),
        "D": int(D), "bits_per_element": float(D * np.log2(args.K)), "information_fraction": float(np.log2(args.K) / 32),
    }
    scaler = Path(cfg.datamodule.transforms.preprocess.cst_fn.filename)
    config = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "command": " ".join(sys.argv), "seed": SEED, "reference_config": str(ref.relative_to(REPO)),
        "reference_config_sha256": sha256(ref), "cst_scaler": str(scaler), "cst_scaler_sha256": sha256(scaler),
        "n_fit_constituents": int(len(fit)), "n_test_batches": n_batches, "git": git_state(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
