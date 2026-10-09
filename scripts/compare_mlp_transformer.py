"""Side-by-side heatmaps comparing MLP vs Transformer VQ-VAE encoders.

Each heatmap cell shows the global IQR/median of the jet pT response
(reco_pt / truth_pt) — the standard ATLAS jet-energy resolution metric.

Grid axes:
  rows  : model size  {small, medium, large}
  cols  : training dataset size {1 M, 10 M jets}

Usage
-----
pixi run python scripts/compare_mlp_transformer.py \
    --scan_dir  results/vqvae_scan/vqvae_scan \
    --output_dir results/vqvae_scan/plots_jetset \
    --output_name mlp_vs_transformer
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
import numpy as np

# ---------------------------------------------------------------------------
# Scan grid definition
# ---------------------------------------------------------------------------
SIZES   = ["small", "medium", "large"]
N_JETS  = [1_000_000, 10_000_000]
NJ_LABELS = ["0.7M", "7M"]  # training jets: 70% of the N_JETS loaded

# Encoder parameter counts (encoder only, not full model)
PARAMS = {
    "MLP":  {"small": "23k", "medium": "351k", "large": "1.4M"},
    "Transformer":  {"small": "204k", "medium": "1.6M", "large": "9.5M"},
}

CB, CD, NQ, CW = 32768, 8, 4, 1.0

RUNS = {
    "MLP": {
        (enc, nj): (
            f"enc_{enc}_cb{CB}_cd{CD}_nq{NQ}_lr0.001_cw{CW}_nj{nj}",
            "vqvae",
        )
        for enc in SIZES for nj in N_JETS
    },
    "Transformer": {
        (enc, nj): (
            f"transformer_vqvae_enc_{enc}_cb{CB}_cd{CD}_nq{NQ}"
            f"_lr0.0003_cw{CW}_nj{nj}",
            "transformer_vqvae",
        )
        for enc in SIZES for nj in N_JETS
    },
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def find_npz(scan_dir: Path, run_name: str) -> Path | None:
    """Locate test_metrics.npz — handles both flat and nested output layouts."""
    for candidate in [
        scan_dir / run_name / "test" / "test_metrics" / "test_metrics.npz",
        # nested: output_dir=results/vqvae_scan, project=vqvae_scan → double-dir
        scan_dir / run_name / "test" / "results" / "vqvae_scan" / run_name
        / "test" / "test_metrics" / "test_metrics.npz",
    ]:
        if candidate.exists():
            return candidate
    return None


def iqr_over_median(pt_ratio: np.ndarray) -> float:
    """Global IQR/median of the pT-response distribution."""
    valid = np.isfinite(pt_ratio) & (pt_ratio > 0)
    r = pt_ratio[valid]
    q25, q50, q75 = np.percentile(r, [25, 50, 75])
    return float((q75 - q25) / q50)


# ---------------------------------------------------------------------------
# Heatmap helper (reuses viridis style from compare_scan.py)
# ---------------------------------------------------------------------------
def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_heatmap(
    ax,
    grid: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap="viridis_r",
    show_ylabels=False,
    ylabels_side: str = "left",   # "left", "right", or "none"
):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=20)
    if show_ylabels:
        ax.set_yticklabels(row_labels, fontsize=20)
        ax.set_ylabel("Model size", fontsize=28)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Training dataset size", fontsize=28)
    ax.set_title(title, fontsize=22, pad=8)

    cmap_obj = plt.get_cmap(cmap)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = grid[i, j]
            if np.isnan(val):
                txt = "N/A"
            else:
                txt = f"{val:.4f}"
            rgba = cmap_obj(norm(val) if not np.isnan(val) else 0.5)
            color = "black" if _luminance(rgba) > 0.45 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=16, color=color)

    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir",   default="results/vqvae_scan/vqvae_scan")
    parser.add_argument("--output_dir", default="results/vqvae_scan/plots_jetset")
    parser.add_argument("--output_name", default="mlp_vs_transformer")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(hep.style.CMS)

    # ------------------------------------------------------------------ data
    grids: dict[str, np.ndarray] = {}
    csv_rows: list[dict] = []

    for arch_label, run_map in RUNS.items():
        grid = np.full((len(SIZES), len(N_JETS)), np.nan)
        for ri, enc in enumerate(SIZES):
            for ci, nj in enumerate(N_JETS):
                run_name, _ = run_map[(enc, nj)]
                npz = find_npz(scan_dir, run_name)
                if npz is None:
                    print(f"  [MISSING] {run_name}")
                    continue
                d = np.load(npz, allow_pickle=True)
                if "pt_ratio" not in d:
                    print(f"  [no pt_ratio] {run_name}")
                    continue
                val = iqr_over_median(d["pt_ratio"])
                grid[ri, ci] = val
                csv_rows.append({
                    "arch": arch_label,
                    "enc_size": enc,
                    "n_jets": nj,
                    "run": run_name,
                    "iqr_over_median": val,
                })
                print(f"  {arch_label:12s}  {enc:6s}  nj={nj:>10,}  IQR/med={val:.4f}")
        grids[arch_label] = grid

    # ------------------------------------------------------------------ plot
    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    finite   = all_vals[np.isfinite(all_vals)]
    vmin, vmax = finite.min() * 0.98, finite.max() * 1.02

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ims = []
    sides = ["left", "right"]
    for idx, (ax, (arch_label, grid)) in enumerate(zip(axes, grids.items())):
        im = plot_heatmap(
            ax, grid,
            row_labels=SIZES,
            col_labels=NJ_LABELS,
            title=arch_label,
            vmin=vmin, vmax=vmax,
            show_ylabels=(idx == 0),
        )
        ims.append(im)

    # shared colorbar
    fig.subplots_adjust(right=0.86, wspace=0.12)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.022, 0.70])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label(r"IQR / median", fontsize=36)
    cbar.ax.tick_params(labelsize=15)

    pdf_path = out_dir / f"{args.output_name}.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {pdf_path}")

    # ------------------------------------------------------------------ csv
    if csv_rows:
        csv_path = out_dir / f"{args.output_name}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arch", "enc_size", "n_jets", "run", "iqr_over_median"])
            w.writeheader(); w.writerows(csv_rows)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
