#!/usr/bin/env python
"""Preprocess HEP4M ROOT files (cell modality) → HDF5 for gold_diggers.

Converts calorimeter cell data from a ROOT TTree into a fixed-shape HDF5 array
of image patches suitable for training the cell VQ-VAE tokenizer.

Output HDF5 layout:
  cell_patches    float32 [N, 156, 64]  — patch pixel energies (energy^0.3 transformed)
  patch_positions float32 [N, 156,  3]  — patch centre (eta, cos_phi, sin_phi)
  event_number    int64   [N]           — original ROOT event numbers

Patch layout (156 total, at downsample=1 — see --downsample for coarser grids,
which shrink both the per-layer bin/patch counts *and* MAX_PATCH_PIXELS stays
fixed; the physical ROI window is always kept the same size, see
CellImageHelper's docstring):
  Layer 0: 64×64 ROI, 8×8 patches →  64 patches × 64 pixels
  Layer 1: 64×64 ROI, 8×8 patches →  64 patches × 64 pixels
  Layer 2: 32×32 ROI, 8×8 patches →  16 patches × 64 pixels
  Layer 3: 16×16 ROI, 8×8 patches →   4 patches × 64 pixels
  Layer 4: 16×16 ROI, 8×8 patches →   4 patches × 64 pixels
  Layer 5:  8×8  ROI, 4×4 patches →   4 patches × 16 pixels (zero-padded to 64)

Usage:
    python scripts/preprocess_hep4m_cells.py \\
        --input  /path/to/singlejet_3_100kseg_bw100.0.root \\
        --output /path/to/cells.h5 \\
        --max_events 100000

    # With visualizations:
    python scripts/preprocess_hep4m_cells.py \\
        --input  /path/to/singlejet_3_100kseg_bw100.0.root \\
        --output /path/to/cells.h5 \\
        --visualize 4 \\
        --plot_output /path/to/cell_preview.pdf
"""

import argparse
import logging

import awkward as ak
import h5py
import matplotlib
import numpy as np
import torch
import uproot
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from heptokens.vendored.cocoa.data.cell_image_helper import CellImageHelper

log = logging.getLogger(__name__)

# ── Patch geometry (mirrors HEP4M configs/variables/cell.yml) ───────────────
# NOTE: these are the downsample=1 (native) values, used as defaults/docs and
# for MAX_PATCH_PIXELS (patch pixel-size in bins never changes with
# downsample). Per-run, downsample-aware bins_to_keep/patches-per-layer/
# total-patches are derived from the actual CellImageHelper instance instead
# (see main()) since CellImageHelper.downsample now shrinks bins_to_keep_dict
# too (keeps the physical ROI window fixed — see its docstring).
POW2_SCALE = [3, 3, 3, 3, 3, 2]  # patch size = 2^s per layer
PATCH_SIZES = [2**s for s in POW2_SCALE]  # [8, 8, 8, 8, 8, 4]
BINS_TO_KEEP = [64, 64, 32, 16, 16, 8]  # ROI bins per layer (square), downsample=1
PATCHES_PER_LAYER = [(b // p) ** 2 for b, p in zip(BINS_TO_KEEP, PATCH_SIZES)]
# = [64, 64, 16, 4, 4, 4]
TOTAL_PATCHES = sum(PATCHES_PER_LAYER)  # 156
MAX_PATCH_PIXELS = max(p**2 for p in PATCH_SIZES)  # 64 (downsample-invariant)

ENERGY_POWER = 0.3  # per-cell transform: e → e^ENERGY_POWER

TREE_NAME = "EventTree"
BRANCHES = [
    "cell_eta",
    "cell_phi",
    "cell_e",
    "cell_calo_region",  # integer 0-5, mapped directly to layer index
    "particle_eta",
    "particle_phi",
    "eventNumber",
]


def _ref_eta(arr) -> float:
    """Arithmetic mean of eta values (matching HEP4M compute_references_etaphi)."""
    eta_np = ak.to_numpy(arr) if hasattr(arr, "to_numpy") else np.asarray(arr)
    if len(eta_np) == 0:
        return 0.0
    # TODO: this isn't exactly the same as what is being done in 4M
    return float(eta_np.mean())


def _ref_phi(arr) -> float:
    """Circular mean of phi values via arctan2(mean(sin), mean(cos)).

    Matches HEP4M compute_references_etaphi, which uses this to avoid
    wrap-around artefacts when phi straddles ±π.
    """
    phi_np = ak.to_numpy(arr) if hasattr(arr, "to_numpy") else np.asarray(arr)
    if len(phi_np) == 0:
        return 0.0
    return float(np.arctan2(np.sin(phi_np).mean(), np.cos(phi_np).mean()))


def event_to_patches(helper, event: dict, return_images: bool = False):
    """Convert one event's cell data to a [156, 64] float32 array.

    Args:
        helper: CellImageHelper instance.
        event: dict with keys cell_eta, cell_phi, cell_e, cell_layer,
               ref_eta, ref_phi (scalars).
        return_images: if True, also return the per-layer feat_dict from
            set_to_image (for visualisation).

    Returns:
        (patches, positions) or (patches, positions, feat_dict) when
        return_images=True.  patches is float32 [TOTAL_PATCHES, MAX_PATCH_PIXELS].
    """
    # Apply per-cell energy transform before histogramming
    cell_e_t = event["cell_e"] ** ENERGY_POWER

    feat_dict, _ind, _lbirbi, patch_mids = helper.set_to_image(
        ref=(event["ref_eta"], event["ref_phi"]),
        coordinates=(event["cell_eta"], event["cell_phi"], event["cell_layer"]),
        var_dict={"cell_e": cell_e_t},
    )
    # patch_mids: Tensor[156, 3] with (eta, cos_phi, sin_phi) per patch centre

    patches_list = []
    for li, ps in enumerate(PATCH_SIZES):
        img = feat_dict[f"feat0_{li}"]  # Tensor [H, W, 1]
        img = img.squeeze(-1)  # [H, W]

        # Unfold into patches: [nH, nW, ps, ps]
        patches = img.unfold(0, ps, ps).unfold(1, ps, ps)  # [nH, nW, ps, ps]
        patches = patches.reshape(-1, ps * ps)  # [n_patches, ps*ps]

        # Zero-pad to MAX_PATCH_PIXELS (only layer 5 needs this: 16 → 64)
        if ps * ps < MAX_PATCH_PIXELS:
            pad_width = MAX_PATCH_PIXELS - ps * ps
            patches = torch.nn.functional.pad(patches, (0, pad_width))

        patches_list.append(patches)

    energy_patches = torch.cat(patches_list, dim=0)  # [156, 64]
    patches_np = energy_patches.numpy().astype(np.float32)
    positions_np = patch_mids.numpy().astype(np.float32)
    if return_images:
        return patches_np, positions_np, feat_dict
    return patches_np, positions_np


_LAYER_COLORS = plt.cm.tab10.colors  # one color per layer in summary plots


def _add_patch_grid(ax, bins: int, patch_size: int) -> None:
    """Overlay white patch-boundary grid lines on an imshow axis."""
    boundaries = np.arange(patch_size, bins, patch_size) - 0.5
    for b in boundaries:
        ax.axvline(b, color="white", linewidth=0.6, alpha=0.8)
        ax.axhline(b, color="white", linewidth=0.6, alpha=0.8)


def visualize_events(vis_data: list, stats: dict, plot_output: str, bins_to_keep: list) -> None:
    """Write a multi-figure PDF with per-sample image grids and summary histograms.

    Figures:
      1. Raw cell scatter  — N_samples × 6 layers
      2. Preprocessed 2-D histograms — N_samples × 6 layers
      3. Patched (same as Fig 2 + white grid lines) — N_samples × 6 layers
      4. Summary histograms over all processed events

    Args:
        bins_to_keep: actual per-layer ROI bin counts for this run (from the
            CellImageHelper instance used, i.e. downsample-aware) — used only
            for the patch-grid line spacing in Figure 3.
    """
    n_samples = len(vis_data)
    n_layers = 6

    with PdfPages(plot_output) as pdf:
        # ── Figures 1–3: per-sample image grids ─────────────────────────────
        fig_titles = [
            "Raw cell scatter  (colour = log₁₊(energy))",
            "Preprocessed 2-D histogram  (energy^0.3)",
            "Preprocessed + patch grid overlay",
        ]

        for fig_idx in range(3):
            fig, axes = plt.subplots(
                n_samples,
                n_layers,
                figsize=(3 * n_layers, 3 * n_samples),
                squeeze=False,
            )
            fig.suptitle(fig_titles[fig_idx], fontsize=13, y=1.01)

            for row, ev in enumerate(vis_data):
                raw = ev["raw"]
                feat_dict = ev["feat_dict"]

                for li in range(n_layers):
                    ax = axes[row, li]

                    if fig_idx == 0:
                        # Raw scatter
                        mask = raw["cell_layer"] == li
                        eta_li = raw["cell_eta"][mask]
                        phi_li = raw["cell_phi"][mask]
                        e_li = raw["cell_e"][mask]
                        if len(eta_li) > 0:
                            sc = ax.scatter(
                                eta_li,
                                phi_li,
                                c=np.log1p(e_li),
                                s=4,
                                cmap="viridis",
                                linewidths=0,
                            )
                            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_xlabel("η", fontsize=7)
                        ax.set_ylabel("φ", fontsize=7)

                    else:
                        # 2-D histogram image
                        img = feat_dict[f"feat0_{li}"].squeeze(-1).numpy()  # [H, W]
                        im = ax.imshow(
                            img.T,
                            origin="lower",
                            cmap="viridis",
                            aspect="auto",
                            interpolation="nearest",
                        )
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_xlabel("η bin", fontsize=7)
                        ax.set_ylabel("φ bin", fontsize=7)

                        if fig_idx == 2:
                            _add_patch_grid(ax, bins=bins_to_keep[li], patch_size=PATCH_SIZES[li])

                    if row == 0:
                        ax.set_title(f"Layer {li}", fontsize=9)
                    if li == 0:
                        ax.set_ylabel(
                            f"Evt {ev['event_number']}\n" + ("φ" if fig_idx == 0 else "φ bin"),
                            fontsize=7,
                        )

                    ax.tick_params(labelsize=6)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Figure 4: Summary histograms ─────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("Summary statistics (all processed events)", fontsize=13)

        # 4a — Energy distribution: raw vs transformed
        ax = axes[0, 0]
        raw_e = np.array(stats["raw_energies"])
        trans_e = np.array(stats["trans_energies"])
        bins_e = np.logspace(np.log10(max(raw_e[raw_e > 0].min(), 1e-6)), np.log10(raw_e.max()), 80)
        ax.hist(raw_e[raw_e > 0], bins=bins_e, histtype="step", label="raw energy", density=True)
        bins_t = np.logspace(
            np.log10(max(trans_e[trans_e > 0].min(), 1e-6)), np.log10(trans_e.max()), 80
        )
        ax.hist(
            trans_e[trans_e > 0], bins=bins_t, histtype="step", label="energy^0.3", density=True
        )
        ax.set_xscale("log")
        ax.set_xlabel("Energy (MeV / transformed)")
        ax.set_ylabel("Density")
        ax.set_title("Cell energy: raw vs transformed")
        ax.legend(fontsize=8)

        # 4b — Cells per event per layer
        ax = axes[0, 1]
        for li in range(n_layers):
            counts = np.array(stats["cells_per_layer"][li])
            ax.hist(
                counts,
                bins=40,
                histtype="step",
                label=f"L{li}",
                color=_LAYER_COLORS[li],
                density=True,
            )
        ax.set_xlabel("Cells per event")
        ax.set_ylabel("Density")
        ax.set_title("Cells per event per layer")
        ax.legend(fontsize=7, ncol=2)

        # 4c — Total raw energy per event per layer
        ax = axes[0, 2]
        for li in range(n_layers):
            totals = np.array(stats["total_energy_per_layer"][li])
            valid = totals[totals > 0]
            if len(valid) > 0:
                bins_l = np.logspace(np.log10(valid.min()), np.log10(valid.max()), 50)
                ax.hist(
                    valid,
                    bins=bins_l,
                    histtype="step",
                    label=f"L{li}",
                    color=_LAYER_COLORS[li],
                    density=True,
                )
        ax.set_xscale("log")
        ax.set_xlabel("Total raw energy per event (MeV)")
        ax.set_ylabel("Density")
        ax.set_title("Total raw energy per event per layer")
        ax.legend(fontsize=7, ncol=2)

        # 4d — Patch occupancy per layer
        ax = axes[1, 0]
        for li in range(n_layers):
            occ = np.array(stats["patch_occupancy"][li])
            ax.hist(
                occ, bins=40, histtype="step", label=f"L{li}", color=_LAYER_COLORS[li], density=True
            )
        ax.set_xlabel("Fraction of non-zero patches")
        ax.set_ylabel("Density")
        ax.set_title("Patch occupancy per layer")
        ax.legend(fontsize=7, ncol=2)

        # 4e — Max patch pixel energy per event
        ax = axes[1, 1]
        max_vals = np.array(stats["max_patch_energy"])
        max_vals = max_vals[max_vals > 0]
        if len(max_vals) > 0:
            ax.hist(max_vals, bins=60, histtype="step", color="steelblue")
        ax.set_xlabel("Max patch pixel value (energy^0.3)")
        ax.set_ylabel("Count")
        ax.set_title("Max patch energy per event")

        # 4f — All non-zero patch pixel values
        ax = axes[1, 2]
        patch_e = np.array(stats["all_patch_energies"])
        patch_e = patch_e[patch_e > 0]
        if len(patch_e) > 0:
            bins_p = np.logspace(np.log10(patch_e.min()), np.log10(patch_e.max()), 80)
            ax.hist(patch_e, bins=bins_p, histtype="step", color="darkorange")
            ax.set_xscale("log")
        ax.set_xlabel("Patch pixel value (energy^0.3)")
        ax.set_ylabel("Count")
        ax.set_title("Non-zero patch pixel distribution (sampled)")

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ROOT file")
    parser.add_argument("--output", required=True, help="Output HDF5 file")
    parser.add_argument(
        "--max_events", type=int, default=None, help="Cap on number of events to process"
    )
    parser.add_argument("--chunk_size", type=int, default=500, help="Events per uproot batch")
    parser.add_argument("--tree", default=TREE_NAME, help=f"ROOT TTree name (default: {TREE_NAME})")
    parser.add_argument(
        "--visualize",
        type=int,
        default=5,
        metavar="N",
        help="Number of random events to include in the preview PDF (0 = skip)",
    )
    parser.add_argument(
        "--plot_output",
        default="cell_preview.pdf",
        help="Output path for the visualisation PDF (default: cell_preview.pdf)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Coarsen the pixel grid by this integer factor relative to the native "
            "per-layer cell pitch (1 = native cell granularity, the default; only 1 "
            "and 2 are usable with the current pow2_scale/bins_to_keep geometry — 4 "
            "and 8 divide evenly but leave at least one layer with zero patches, and "
            "raise ValueError). Bigger pixels mean multiple cells are genuinely summed "
            "per pixel; the ROI's *physical* eta-phi window is kept fixed (same as "
            "downsample=1), so the total patch/pixel count per event shrinks by "
            "downsample**2 instead of the window growing. See CellImageHelper's "
            "docstring for details."
        ),
    )
    args = parser.parse_args()

    helper = CellImageHelper(pow2_scale=POW2_SCALE, downsample=args.downsample)
    # Actual, downsample-aware patch geometry for this run (CellImageHelper.downsample
    # now shrinks bins_to_keep_dict too, so these differ from the module-level
    # BINS_TO_KEEP/PATCHES_PER_LAYER/TOTAL_PATCHES defaults whenever downsample > 1).
    actual_bins_to_keep = [helper.bins_to_keep_dict[i] for i in range(6)]
    actual_patches_per_layer = [n**2 for n in helper.num_patches_per_layer_1d]
    actual_total_patches = sum(actual_patches_per_layer)
    log.info(
        f"CellImageHelper initialised (downsample={args.downsample}) — "
        f"{actual_total_patches} patches × {MAX_PATCH_PIXELS} pixels "
        f"(bins_to_keep per layer: {actual_bins_to_keep})"
    )

    # ── Count events ────────────────────────────────────────────────────────
    with uproot.open(args.input) as f:
        n_total = f[args.tree].num_entries
    n_events = min(n_total, args.max_events) if args.max_events else n_total
    log.info(f"Processing {n_events}/{n_total} events from {args.input}")

    # ── Allocate output arrays ──────────────────────────────────────────────
    all_patches = np.zeros((n_events, actual_total_patches, MAX_PATCH_PIXELS), dtype=np.float32)
    # (eta, cos_phi, sin_phi) per patch centre
    all_positions = np.zeros((n_events, actual_total_patches, 3), dtype=np.float32)
    all_event_numbers = np.zeros(n_events, dtype=np.int64)

    # ── Set up visualisation bookkeeping ────────────────────────────────────
    n_vis = args.visualize
    rng = np.random.default_rng(seed=42)
    vis_indices: set = (
        set(rng.choice(n_events, size=min(n_vis, n_events), replace=False).tolist())
        if n_vis > 0
        else set()
    )
    vis_data: list = []

    # Running stats accumulators for summary histograms (capped to limit memory)
    _SAMPLE_CAP = 500_000
    stats: dict = {
        "raw_energies": [],
        "trans_energies": [],
        "cells_per_layer": [[] for _ in range(6)],  # counts per event
        "total_energy_per_layer": [[] for _ in range(6)],  # summed raw energy per event
        "patch_occupancy": [[] for _ in range(6)],  # fraction non-zero patches per event
        "all_patch_energies": [],  # non-zero pixel values (sampled)
        "max_patch_energy": [],  # max pixel per event
    }
    # Track per-layer patch offsets for occupancy calculation
    _LAYER_PATCH_STARTS = [sum(actual_patches_per_layer[:li]) for li in range(6)]

    # ── Iterate over chunks ─────────────────────────────────────────────────
    processed = 0
    with uproot.open(args.input) as f:
        tree = f[args.tree]
        for batch in tqdm(
            tree.iterate(
                BRANCHES,
                step_size=args.chunk_size,
                entry_stop=n_events,
                library="ak",
            ),
            desc="chunks",
            total=(n_events + args.chunk_size - 1) // args.chunk_size,
        ):
            batch_size = len(batch["eventNumber"])
            for i in range(batch_size):
                ref_eta = _ref_eta(batch["particle_eta"][i])
                ref_phi = _ref_phi(batch["particle_phi"][i])

                cell_e_np = ak.to_numpy(batch["cell_e"][i])
                cell_layer_np = ak.to_numpy(batch["cell_calo_region"][i])

                event = {
                    "cell_eta": batch["cell_eta"][i],
                    "cell_phi": batch["cell_phi"][i],
                    "cell_e": batch["cell_e"][i],
                    # cell_calo_region is assumed to hold layer indices 0-5
                    "cell_layer": batch["cell_calo_region"][i],
                    "ref_eta": ref_eta,
                    "ref_phi": ref_phi,
                }

                want_images = processed in vis_indices
                result = event_to_patches(helper, event, return_images=want_images)
                if want_images:
                    patches, positions, feat_dict = result
                else:
                    patches, positions = result

                all_patches[processed] = patches
                all_positions[processed] = positions
                all_event_numbers[processed] = int(batch["eventNumber"][i])

                # ── Collect stats ──────────────────────────────────────────
                if len(stats["raw_energies"]) < _SAMPLE_CAP:
                    stats["raw_energies"].extend(cell_e_np.tolist())
                    stats["trans_energies"].extend((cell_e_np**ENERGY_POWER).tolist())

                for li in range(6):
                    mask_li = cell_layer_np == li
                    stats["cells_per_layer"][li].append(int(mask_li.sum()))
                    stats["total_energy_per_layer"][li].append(float(cell_e_np[mask_li].sum()))
                    # patch occupancy for this layer
                    start = _LAYER_PATCH_STARTS[li]
                    end = start + actual_patches_per_layer[li]
                    layer_patches = patches[start:end]  # [n_patches, 64]
                    occ = float((layer_patches.sum(axis=1) > 0).mean())
                    stats["patch_occupancy"][li].append(occ)

                nonzero_pixels = patches[patches > 0]
                if len(stats["all_patch_energies"]) < _SAMPLE_CAP and len(nonzero_pixels) > 0:
                    stats["all_patch_energies"].extend(nonzero_pixels.tolist())
                stats["max_patch_energy"].append(float(patches.max()) if patches.size > 0 else 0.0)

                # ── Store vis event ────────────────────────────────────────
                if want_images:
                    vis_data.append(
                        {
                            "event_idx": processed,
                            "event_number": int(batch["eventNumber"][i]),
                            "raw": {
                                "cell_eta": ak.to_numpy(batch["cell_eta"][i]),
                                "cell_phi": ak.to_numpy(batch["cell_phi"][i]),
                                "cell_e": cell_e_np,
                                "cell_layer": cell_layer_np,
                            },
                            "feat_dict": feat_dict,
                            "patches": patches,
                        }
                    )

                processed += 1

            if processed >= n_events:
                break

    log.info(f"Processed {processed} events; saving to {args.output}")

    # ── Write HDF5 ──────────────────────────────────────────────────────────
    with h5py.File(args.output, "w") as f:
        f.create_dataset("cell_patches", data=all_patches, compression="gzip", chunks=True)
        f.create_dataset("patch_positions", data=all_positions, compression="gzip", chunks=True)
        f.create_dataset("event_number", data=all_event_numbers)
        f.attrs["downsample"] = args.downsample

    log.info(
        f"Done. Output shape: {all_patches.shape}, energy range: [{all_patches.max():.4f} max]"
    )

    if n_vis > 0 and vis_data:
        log.info(f"Generating visualisations for {len(vis_data)} events → {args.plot_output}")
        visualize_events(vis_data, stats, args.plot_output, bins_to_keep=actual_bins_to_keep)
        log.info(f"Saved visualisations to {args.plot_output}")


if __name__ == "__main__":
    main()
