#!/usr/bin/env python
"""Generate an event-display schematic of the cell -> token pipeline.

Three-step pipeline (matches the cell-tokenization slides / paper figure):
  1. Pixelize   — rasterize raw calorimeter cells onto a fixed eta-phi pixel
                  grid via np.histogram2d(weights=energy). For this dataset the
                  grid resolution is built to match the native per-layer cell
                  pitch exactly (see CellImageHelper.grans), so in practice this
                  step is a lossless 1:1 relabeling: every occupied pixel
                  contains exactly one cell (verified empirically — 0 bins with
                  >1 cell across 200 events x 6 layers). No energy summing
                  actually occurs here, even though histogram2d would sum
                  multiple cells' energy into a bin if several fell in it.
  2. Form patch — group pixels into non-overlapping 8x8 (4x4 for layer 5)
                  blocks. Also lossless: this concatenates 64 pixels into one
                  vector, it does not sum/aggregate them.
  3. Tokenize   — pass one non-empty patch through the VQ-VAE encoder + residual
                  quantizer to extract a discrete token id. This is the only
                  step in the pipeline that actually compresses information: a
                  learned nonlinear map from the 128-d patch vector down to a
                  32-d code, which is then discretized by the residual VQ.

Uses the exact same preprocessing code path as scripts/preprocess_hep4m_cells.py
(CellImageHelper + event_to_patches) so the figure is a faithful depiction of
the production pipeline, and loads a trained VQ-VAE checkpoint to show a real
token id for the highlighted patch.

Each panel (input cells, pixelize, form-patches, patch zoom, tokenize) is saved
as its own standalone figure file so they can be arranged/formatted freely
(e.g. in a paper). Pass --combined to additionally save one assembled overview.

Usage:
    pixi run python scripts/make_tokenization_schematic.py \\
        --input <COCOA_ROOT>/train_topup2_20M_cells256/singlejet_270_100kseg_bw100.0.root \\
        --ckpt  <CELL_RUN_DIR>/checkpoints/last.ckpt \\
        --layer 0 \\
        --outdir plots/cell_tokenization_schematic \\
        --combined
"""

import argparse
import sys
from pathlib import Path

import awkward as ak
import matplotlib
import numpy as np
import torch
import uproot

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("ATLAS")

# Uniform, large label sizes (matching HEP-style 2D scan plots: big x/y/z labels).
LABEL_FS = 22
TICK_FS = 16
CBAR_LABEL_FS = 22

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_hep4m_cells import (  # noqa: E402
    BRANCHES,
    PATCH_SIZES,
    TREE_NAME,
    _ref_eta,
    _ref_phi,
    event_to_patches,
)

from heptokens.vendored import b_module_names  # noqa: E402
from heptokens.vendored.models.coders import Encoder  # noqa: E402
from heptokens.vendored.cocoa.data.cell_image_helper import CellImageHelper  # noqa: E402
from vector_quantize_pytorch import ResidualVQ  # noqa: E402

def _layer_geometry(helper: CellImageHelper) -> tuple[list[int], list[int]]:
    """Per-layer (patch_start_index, n_patches_per_side) for this helper's actual
    (downsample-aware) geometry — NOT the fixed downsample=1 defaults, since
    CellImageHelper.downsample now shrinks bins_to_keep_dict to keep the
    physical ROI window fixed (see its docstring), which shrinks patch counts.
    """
    n1d = helper.num_patches_per_layer_1d
    patches_per_layer = [n**2 for n in n1d]
    starts = [sum(patches_per_layer[:li]) for li in range(6)]
    return starts, n1d


def _read_rows(root_path: str, tree: str, start: int, stop: int):
    with uproot.open(root_path) as f:
        return f[tree].arrays(BRANCHES, entry_start=start, entry_stop=stop, library="ak")


def _event_dict(batch, i: int) -> dict:
    return {
        "cell_eta": batch["cell_eta"][i],
        "cell_phi": batch["cell_phi"][i],
        "cell_e": batch["cell_e"][i],
        "cell_layer": batch["cell_calo_region"][i],
        "ref_eta": _ref_eta(batch["particle_eta"][i]),
        "ref_phi": _ref_phi(batch["particle_phi"][i]),
    }


def pick_event(
    root_path: str, tree: str, helper: CellImageHelper, layer: int, n_scan: int, energy_quantile: float
) -> tuple[int, float, int, int]:
    """Scan the first n_scan events and return the row index whose deposits in
    `layer` are most *spread out* across patches, restricted to events that are
    reasonably energetic (raw energy above the `energy_quantile` quantile of
    the scanned pool). Picking purely by total energy tends to return single
    concentrated showers (all energy in 1-2 patches); ranking by patch
    occupancy among the energetic subset gives a visually richer, less
    trivial patching example.

    Returns (row_idx, total_energy_GeV, n_active_patches, n_patches_in_layer).
    """
    with uproot.open(root_path) as f:
        n_scan = min(n_scan, f[tree].num_entries)
    batch = _read_rows(root_path, tree, 0, n_scan)

    layer_starts, layer_n1d = _layer_geometry(helper)
    start, n_patch = layer_starts[layer], layer_n1d[layer] ** 2
    energies = np.zeros(n_scan)
    occupancies = np.zeros(n_scan, dtype=int)
    for i in range(n_scan):
        layer_np = ak.to_numpy(batch["cell_calo_region"][i])
        e_np = ak.to_numpy(batch["cell_e"][i])
        energies[i] = e_np[layer_np == layer].sum()

        patches, _ = event_to_patches(helper, _event_dict(batch, i))
        occupancies[i] = int((patches[start : start + n_patch].sum(axis=1) > 0).sum())

    energy_cut = np.quantile(energies, energy_quantile)
    candidates = np.where(energies >= energy_cut)[0]
    best_idx = int(candidates[np.argmax(occupancies[candidates])])
    return best_idx, float(energies[best_idx]), int(occupancies[best_idx]), n_patch


def load_event(root_path: str, tree: str, idx: int, helper: CellImageHelper):
    batch = _read_rows(root_path, tree, idx, idx + 1)
    patches, positions, feat_dict = event_to_patches(
        helper, _event_dict(batch, 0), return_images=True
    )
    raw = {
        "cell_eta": ak.to_numpy(batch["cell_eta"][0]),
        "cell_phi": ak.to_numpy(batch["cell_phi"][0]),
        "cell_e": ak.to_numpy(batch["cell_e"][0]),
        "cell_layer": ak.to_numpy(batch["cell_calo_region"][0]),
    }
    return raw, feat_dict, patches, positions, int(batch["eventNumber"][0])


def build_encoder_and_vq(ckpt_path: str):
    """Rebuild the encoder + ResidualVQ from a checkpoint, auto-detecting
    codebook_dim/codebook_size/num_quantizers from the state_dict shapes so
    this works for any trained config (e.g. the cb=256/cd=32/nq=3 smoke
    checkpoint, or a cb=1024/cd=16/nq=3 grid-scan point) without hardcoding.
    """
    with b_module_names():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    codebook_dim = sd["encoder.coder.model.9.weight"].shape[0]
    codebook_size = sd["vector_quantization.layers.0.embed"].shape[-1]
    num_quantizers = sum(1 for k in sd if k.endswith(".embed") and "vector_quantization.layers." in k)

    encoder = Encoder(input_dim=128, output_dim=codebook_dim, input_key="energy")
    enc_state = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.eval()

    vq = ResidualVQ(
        dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, commitment=1.0
    )
    vq_state = {
        k[len("vector_quantization.") :]: v
        for k, v in sd.items()
        if k.startswith("vector_quantization.")
    }
    vq.load_state_dict(vq_state, strict=False)
    vq.eval()
    return encoder, vq


def tokenize_patch(encoder, vq, energy_pixels: np.ndarray) -> list[int]:
    """energy_pixels: [64] energy^0.3 values for one patch -> list of `num_quantizers` token ids."""
    indicator = (energy_pixels > 0).astype(np.float32)
    combined = np.concatenate([energy_pixels, indicator]).astype(np.float32)  # [128]
    x = torch.from_numpy(combined).view(1, 1, -1)  # [B=1, T=1, 128]
    with torch.no_grad():
        z_e = encoder({"energy": x})  # [1, 1, 32]
        _, indices_batched, _ = vq(z_e)  # [num_quantizers, 1, 1]
    return indices_batched[:, 0, 0].tolist()


def add_patch_grid(ax, bins: int, patch_size: int, **kwargs) -> None:
    boundaries = np.arange(patch_size, bins, patch_size) - 0.5
    style = dict(color="white", linewidth=1.4, alpha=1.0)
    style.update(kwargs)
    for b in boundaries:
        ax.axvline(b, **style)
        ax.axhline(b, **style)


def _square_window(lo: float, hi: float, full_lo: float, full_hi: float) -> tuple[float, float]:
    """Expand (lo, hi) to span max(hi-lo, ...) evenly, then shift (not shrink) to
    fit inside [full_lo, full_hi] if it overflows one side."""
    span = hi - lo
    full_span = full_hi - full_lo
    span = min(span, full_span)
    center = (lo + hi) / 2
    lo, hi = center - span / 2, center + span / 2
    if lo < full_lo:
        hi += full_lo - lo
        lo = full_lo
    if hi > full_hi:
        lo -= hi - full_hi
        hi = full_hi
    return lo, hi


def active_crop_bins(img: np.ndarray, pad: int = 3) -> tuple[float, float, float, float]:
    """Square (x0, x1, y0, y1) window (imshow pixel-edge coords) tightly bounding
    img's non-zero pixels, padded by `pad` bins per side and clipped to img's
    own extent. Used to zoom pixelize/patches panels into the active region
    instead of showing the full mostly-empty grid.
    """
    rows, cols = np.nonzero(img)
    if rows.size == 0:
        return -0.5, img.shape[0] - 0.5, -0.5, img.shape[1] - 0.5
    x0, x1 = rows.min() - pad - 0.5, rows.max() + pad + 0.5
    y0, y1 = cols.min() - pad - 0.5, cols.max() + pad + 0.5
    x0, x1 = _square_window(x0, x1, -0.5, img.shape[0] - 0.5)
    y0, y1 = _square_window(y0, y1, -0.5, img.shape[1] - 0.5)
    return x0, x1, y0, y1


def bin_crop_to_physical(
    img_crop: tuple[float, float, float, float],
    patch0_center: tuple[float, float, float],
    ps: int,
    eta_bin_width: float,
    phi_bin_width: float,
) -> tuple[float, float, float, float]:
    """Convert a bin-edge crop window (see active_crop_bins) to the *exact*
    physical (eta0, eta1, phi0, phi1) window it corresponds to, using the
    known center of patch (row=0, col=0) of this layer as a reference point.

    This keeps the raw-cell scatter panel's physical crop in lockstep with the
    pixelize/patches panels' bin-index crop — same window, not independently
    estimated from the scatter data's own bounding box (which could differ
    slightly due to isolated outlier cells skewing the padding).
    """
    eta_mid0, cosphi0, sinphi0 = patch0_center
    phi_mid0 = float(np.arctan2(sinphi0, cosphi0))
    center_bin = (ps - 1) / 2.0
    offset_eta = float(eta_mid0) - center_bin * eta_bin_width
    offset_phi = phi_mid0 - center_bin * phi_bin_width
    x0, x1, y0, y1 = img_crop
    eta0, eta1 = offset_eta + x0 * eta_bin_width, offset_eta + x1 * eta_bin_width
    phi0, phi1 = offset_phi + y0 * phi_bin_width, offset_phi + y1 * phi_bin_width
    return eta0, eta1, phi0, phi1


# ── Individual panel drawers ────────────────────────────────────────────────
# Each function draws onto a supplied Axes only (no figure-level calls), so the
# same drawing code can be reused for both the standalone per-panel figures
# and the assembled overview figure.


def draw_raw_cells(ax, eta, phi, e, crop=None, vmin=None, vmax=None):
    """Panel: raw cell scatter (input). Square aspect — same eta/phi extent.

    Colored (and sized) by E^0.3, matching the pixelize/patches/zoom panels'
    scale exactly (rather than raw linear GeV) — with a long-tailed energy
    distribution, linear-GeV color mapping makes almost every point the same
    dark shade (energy differences invisible); the power-law compression used
    everywhere else in this pipeline fixes that here too, and means all four
    spatial panels can share one colorbar. Marker size also scales with
    energy for an extra, redundant visual cue.

    crop: optional (eta0, eta1, phi0, phi1) window (see bin_crop_to_physical)
    to zoom into the active region instead of showing the full cell spread.
    vmin/vmax: fix the E^0.3 color scale (e.g. to share one colorbar).
    """
    e_pow = np.asarray(e) ** 0.3
    e_max = float(e_pow.max()) if e_pow.size else 1.0
    sizes = 6 + 90 * (e_pow / e_max if e_max > 0 else e_pow)
    sc = ax.scatter(eta, phi, c=e_pow, s=sizes, cmap="viridis", linewidths=0, vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$\eta$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\phi$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_aspect("equal", adjustable="box")
    if crop is not None:
        ax.set_xlim(crop[0], crop[1])
        ax.set_ylim(crop[2], crop[3])
    return sc


def draw_pixelized(ax, img, crop=None, vmin=None, vmax=None):
    """Panel: Step 1 — pixelize. Square: n_eta_bins == n_phi_bins for a layer.

    For this dataset, this is a lossless 1:1 rasterization of cells onto their
    native grid — no bin ever receives more than one cell (verified), so no
    energy summing actually happens here. It's a relabeling, not a compression.

    crop: optional (x0, x1, y0, y1) bin-edge window (see active_crop_bins) to
    zoom into the active region instead of showing the full mostly-empty grid.
    vmin/vmax: fix the color scale (e.g. to share one colorbar across panels)
    instead of auto-scaling to this panel's own data range.
    """
    im = ax.imshow(
        img.T,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\eta$ bin", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\phi$ bin", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    if crop is not None:
        ax.set_xlim(crop[0], crop[1])
        ax.set_ylim(crop[2], crop[3])
    return im


def draw_patches(ax, img, bins, ps, zoom_row, zoom_col, crop=None, vmin=None, vmax=None, show_dims=False):
    """Panel: Step 2 — form patches (grid overlay + highlighted patch). Square."""
    im = ax.imshow(
        img.T,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    add_patch_grid(ax, bins, ps)
    ax.add_patch(
        mpatches.Rectangle(
            (zoom_row * ps - 0.5, zoom_col * ps - 0.5),
            ps,
            ps,
            linewidth=2.2,
            edgecolor="red",
            facecolor="none",
        )
    )
    ax.set_xlabel(r"$\eta$ bin", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\phi$ bin", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    if crop is not None:
        ax.set_xlim(crop[0], crop[1])
        ax.set_ylim(crop[2], crop[3])
    if show_dims:
        n1d = bins // ps
        ax.set_title(
            f"{bins}\u00d7{bins} px image  |  {ps}\u00d7{ps} px patch  |  "
            f"{n1d}\u00d7{n1d} = {n1d * n1d} patches",
            fontsize=13,
            pad=8,
        )
    return im


def draw_patch_zoom(ax, patch_img, ps, vmin=None, vmax=None):
    """Panel: zoom of the single highlighted patch feeding into the encoder. Square."""
    im = ax.imshow(
        patch_img.T,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    for b in range(1, ps):
        ax.axvline(b - 0.5, color="white", linewidth=1.0, alpha=0.9)
        ax.axhline(b - 0.5, color="white", linewidth=1.0, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _draw_colored_sequence(ax, y: float, parts: list[tuple[str, str]], fontsize: int, **text_kwargs):
    """Draw `parts` (list of (text, color)) concatenated and horizontally
    centered at axes-fraction y, each piece in its own color. Matplotlib has
    no built-in multi-color Text, so this renders each piece once to measure
    its width, then repositions them in sequence (a standard workaround).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    origin_x = inv.transform((0, 0))[0]

    texts, widths = [], []
    for text, color in parts:
        t = ax.text(0, y, text, ha="left", va="center", fontsize=fontsize, color=color, transform=ax.transAxes, **text_kwargs)
        bbox = t.get_window_extent(renderer=renderer)
        w_axes = inv.transform((bbox.width, 0))[0] - origin_x
        texts.append(t)
        widths.append(w_axes)

    total_w = sum(widths)
    x = 0.5 - total_w / 2
    for t, w in zip(texts, widths):
        t.set_position((x, y))
        x += w
    return texts


# Distinguishable, colorblind-friendly palette for residual-VQ stages (reused
# for both the codebook-stage chips and the matching token-id digits, so a
# reader can visually trace which stage produced which integer).
STAGE_COLORS = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]


def draw_token_diagram(ax, token_ids, codebook_size=256, codebook_dim=32, num_quantizers=3):
    """Panel: Step 3 — encoder / residual-VQ schematic + the extracted token id.

    The residual VQ is drawn as `num_quantizers` separate, distinctly colored
    stage chips, and the resulting token integers are colored to match — the
    payoff of the whole figure (which stage produced which number) is explicit
    rather than left for the reader to infer from left-to-right ordering alone.

    Not a spatial panel, so it is intentionally not forced to a square aspect.
    """
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    colors = (STAGE_COLORS * 2)[:num_quantizers]

    boxes = [
        (0.5, 0.87, "flatten\n[64 $E$ | 64 indicator\n(occupied 0/1)]\n\u2192 128-d vector"),
        (0.5, 0.60, f"Encoder MLP\n128\u2192256\u2192512\u2192{codebook_dim}"),
    ]
    for x, y, label in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - 0.46, y - 0.09),
                0.92,
                0.18,
                boxstyle="round,pad=0.02",
                linewidth=1.3,
                edgecolor="black",
                facecolor="#eef3fb",
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=15)
    ax.annotate(
        "",
        xy=(0.5, boxes[1][1] + 0.09),
        xytext=(0.5, boxes[0][1] - 0.09),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )

    # Residual VQ: one colored chip per quantizer stage, linked to the token digits below.
    vq_y, vq_h, gap, total_w = 0.35, 0.14, 0.025, 0.86
    n = num_quantizers
    w = (total_w - gap * (n - 1)) / n
    x0 = 0.5 - total_w / 2
    for i in range(n):
        cx = x0 + i * (w + gap) + w / 2
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (cx - w / 2, vq_y - vq_h / 2),
                w,
                vq_h,
                boxstyle="round,pad=0.01",
                linewidth=2.0,
                edgecolor=colors[i],
                facecolor=colors[i],
                alpha=0.18,
            )
        )
        ax.text(cx, vq_y, f"Q{i + 1}", ha="center", va="center", fontsize=15, fontweight="bold", color=colors[i])
    ax.text(
        0.5,
        vq_y - vq_h / 2 - 0.05,
        f"Residual VQ: {n} \u00d7 codebook({codebook_size})",
        ha="center",
        va="top",
        fontsize=13,
    )
    ax.annotate(
        "",
        xy=(0.5, vq_y + vq_h / 2 + 0.03),
        xytext=(0.5, boxes[1][1] - 0.09),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )

    # Token result — each digit colored to match its stage chip above.
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.04, 0.02),
            0.92,
            0.15,
            boxstyle="round,pad=0.02",
            linewidth=1.6,
            edgecolor="#333333",
            facecolor="#f7f7f7",
        )
    )
    parts = [("Token = [", "#333333")]
    for i, tok in enumerate(token_ids):
        parts.append((str(tok), colors[i]))
        parts.append((", " if i < len(token_ids) - 1 else "]", "#333333"))
    _draw_colored_sequence(ax, 0.095, parts, fontsize=20, fontweight="bold")
    ax.annotate(
        "",
        xy=(0.5, 0.17),
        xytext=(0.5, vq_y - vq_h / 2 - 0.10),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Input ROOT file (single-jet sample)"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Trained cell VQ-VAE checkpoint (last.ckpt). Not required if --fake_token "
            "is set."
        ),
    )
    parser.add_argument("--tree", default=TREE_NAME)
    parser.add_argument("--layer", type=int, default=0, help="Calorimeter layer to feature (0-5)")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help=(
            "Pixel downsample factor matching CellImageHelper.downsample (1=native "
            "cell size, 2=2x coarser pixels with real cell-summing, etc.) — must match "
            "how --input's data was preprocessed"
        ),
    )
    parser.add_argument(
        "--event_idx", type=int, default=None, help="Fix a specific event row (skips auto-scan)"
    )
    parser.add_argument(
        "--n_scan",
        type=int,
        default=2000,
        help="Number of events to scan when auto-selecting",
    )
    parser.add_argument(
        "--energy_quantile",
        type=float,
        default=0.5,
        help=(
            "Only consider scanned events with raw layer energy above this quantile "
            "(0-1), then pick the one with the most active/spread-out patches among them"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="plots/cell_tokenization_schematic",
        help="Directory to write one file per panel into",
    )
    parser.add_argument(
        "--format",
        default="pdf,png",
        help="Comma-separated output format(s) for each panel, e.g. 'pdf,png'",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also save one assembled overview figure alongside the individual panels",
    )
    parser.add_argument(
        "--fake_token",
        nargs="*",
        type=int,
        default=None,
        metavar="TOKEN_ID",
        help=(
            "Skip loading --ckpt entirely and use this made-up token id (or ids, one "
            "per quantizer stage) for the Step 3 panel instead of a real encoder/VQ "
            "forward pass. Useful when no trained checkpoint exists yet. If given with "
            "no values, a placeholder [?, ?, ?] is shown. Clearly a placeholder either way "
            "— not derived from any trained model."
        ),
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=1024,
        help="Codebook size to display in the Step 3 diagram (default matches the ds=2 baseline)",
    )
    parser.add_argument(
        "--codebook_dim",
        type=int,
        default=16,
        help="Codebook dim to display in the Step 3 diagram (default matches the ds=2 baseline)",
    )
    parser.add_argument(
        "--num_quantizers",
        type=int,
        default=3,
        help="Num quantizers to display in the Step 3 diagram (default matches the ds=2 baseline)",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help=(
            "Disable auto-cropping to the active (non-empty) region — by default the "
            "input/pixelize/patches panels zoom into a padded square window around the "
            "jet instead of showing the full mostly-empty grid."
        ),
    )
    parser.add_argument(
        "--crop_pad_bins",
        type=int,
        default=3,
        help="Padding (in bins) around the active region for the pixelize/patches crop",
    )
    args = parser.parse_args()

    if args.ckpt is None and args.fake_token is None:
        parser.error("one of --ckpt or --fake_token is required")

    helper = CellImageHelper(pow2_scale=[3, 3, 3, 3, 3, 2], downsample=args.downsample)

    if args.event_idx is not None:
        idx = args.event_idx
    else:
        idx, energy, n_active, n_patch = pick_event(
            args.input, args.tree, helper, args.layer, args.n_scan, args.energy_quantile
        )
        print(
            f"Auto-selected row {idx}: layer-{args.layer} energy = {energy:.2f} GeV, "
            f"{n_active}/{n_patch} patches active"
        )
    raw, feat_dict, patches, positions, event_number = load_event(args.input, args.tree, idx, helper)
    print(f"Selected event row {idx} (eventNumber={event_number})")
    print(f"Max cell energy in layer {args.layer}: {raw['cell_e'][raw['cell_layer'] == args.layer].max():.2f} GeV")

    li = args.layer
    bins, ps = helper.bins_to_keep_dict[li], PATCH_SIZES[li]
    layer_starts, layer_n1d = _layer_geometry(helper)
    n1d = layer_n1d[li]
    img = feat_dict[f"feat0_{li}"].squeeze(-1).numpy()  # [n_eta_bins, n_phi_bins]
    vmax = float(img.max()) if img.max() > 0 else 1.0

    layer_patches = patches[layer_starts[li] : layer_starts[li] + n1d * n1d]  # [n1d*n1d, 64]
    zoom_flat_idx = int(np.argmax(layer_patches.sum(axis=1)))
    zoom_row, zoom_col = divmod(zoom_flat_idx, n1d)  # (eta block, phi block)
    patch_img = img[zoom_row * ps : (zoom_row + 1) * ps, zoom_col * ps : (zoom_col + 1) * ps]

    mask_li = raw["cell_layer"] == li
    eta_li, phi_li, e_li = raw["cell_eta"][mask_li], raw["cell_phi"][mask_li], raw["cell_e"][mask_li]

    if args.no_crop:
        img_crop = phys_crop = None
    else:
        img_crop = active_crop_bins(img, pad=args.crop_pad_bins)
        eta_bin_width = 6.0 / helper.grans[li]
        phi_bin_width = 2 * np.pi / helper.grans[li]
        phys_crop = bin_crop_to_physical(
            img_crop, positions[layer_starts[li]], ps, eta_bin_width, phi_bin_width
        )

    if args.fake_token is not None:
        token_ids = args.fake_token if args.fake_token else ["?", "?", "?"]
        print(f"Using FAKE (made-up, not from a trained model) token ids: {token_ids}")
    else:
        encoder, vq = build_encoder_and_vq(args.ckpt)
        token_ids = tokenize_patch(encoder, vq, layer_patches[zoom_flat_idx])
        print(f"Zoomed patch (eta_block={zoom_row}, phi_block={zoom_col}) -> token ids: {token_ids}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lstrip(".") for fmt in args.format.split(",") if fmt.strip()]

    # ── Individual panels, each its own figure/file ─────────────────────────
    # All spatial panels share one E^0.3 color scale (0, vmax) — including the raw-cell
    # scatter now (see draw_raw_cells) — so a reader can compare energies across panels.
    panels = [
        (
            "00_input_cells",
            (5, 5),
            lambda ax: draw_raw_cells(ax, eta_li, phi_li, e_li, crop=phys_crop, vmin=0, vmax=vmax),
            r"$E^{0.3}$",
        ),
        (
            "01_step1_pixelize",
            (5, 5),
            lambda ax: draw_pixelized(ax, img, crop=img_crop, vmin=0, vmax=vmax),
            r"$E^{0.3}$",
        ),
        (
            "02_step2_patches",
            (5, 5),
            lambda ax: draw_patches(
                ax, img, bins, ps, zoom_row, zoom_col, crop=img_crop, vmin=0, vmax=vmax, show_dims=True
            ),
            r"$E^{0.3}$",
        ),
        (
            "03_patch_zoom",
            (4, 4),
            lambda ax: draw_patch_zoom(ax, patch_img, ps, vmin=0, vmax=vmax),
            r"$E^{0.3}$",
        ),
        (
            "04_step3_tokenize",
            (5, 5),
            lambda ax: draw_token_diagram(
                ax, token_ids, args.codebook_size, args.codebook_dim, args.num_quantizers
            ),
            None,
        ),
    ]

    for name, figsize, draw_fn, cbar_label in panels:
        fig, ax = plt.subplots(figsize=figsize)
        mappable = draw_fn(ax)
        if cbar_label is not None:
            cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=CBAR_LABEL_FS)
            cbar.ax.tick_params(labelsize=TICK_FS)
        fig.tight_layout()
        for fmt in formats:
            out_path = outdir / f"{name}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            print(f"Saved {out_path}")
        plt.close(fig)

    # ── Optional assembled overview ──────────────────────────────────────────
    # 4 panels, not 5: "pixelize" and "form patches" look identical except the grid
    # overlay, so they're merged into one panel here (grid + highlighted patch),
    # freeing horizontal space for the flowchart. The standalone exports above
    # still keep them separate for anyone who wants the intermediate no-grid step.
    if args.combined:
        fig = plt.figure(figsize=(19, 5.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.85, 1.5], wspace=0.5)
        ax_raw, ax_patch, ax_zoom, ax_token = (fig.add_subplot(gs[i]) for i in range(4))

        # All spatial panels share one E^0.3 scale (see draw_raw_cells docstring).
        sc = draw_raw_cells(ax_raw, eta_li, phi_li, e_li, crop=phys_crop, vmin=0, vmax=vmax)
        draw_patches(
            ax_patch, img, bins, ps, zoom_row, zoom_col, crop=img_crop, vmin=0, vmax=vmax, show_dims=True
        )
        draw_patch_zoom(ax_zoom, patch_img, ps, vmin=0, vmax=vmax)
        cbar = fig.colorbar(sc, ax=[ax_raw, ax_patch, ax_zoom], fraction=0.022, pad=0.015, aspect=45)
        cbar.set_label(r"$E^{0.3}$", fontsize=CBAR_LABEL_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)

        draw_token_diagram(ax_token, token_ids, args.codebook_size, args.codebook_dim, args.num_quantizers)

        # Finalize layout before reading back axes positions (needed for both the
        # inter-panel arrows and the zoom-lens connector lines below).
        fig.canvas.draw()

        # Pipeline arrows: raw cells -> patches, colorbar -> flowchart. (patches -> zoom
        # already has an explicit "zoom lens" connector instead, drawn next.) The second
        # arrow starts from the colorbar's right edge, not the zoom axes' edge, since the
        # shared colorbar sits directly to the right of the zoom panel and a straight
        # zoom -> token arrow would otherwise be drawn on top of it. Its tick labels
        # (e.g. "0.8") extend past cbar.ax's own bounding box, so give it extra left padding.
        for a_from, a_to, left_pad in [(ax_raw, ax_patch, 0.006), (cbar.ax, ax_token, 0.05)]:
            p_from, p_to = a_from.get_position(), a_to.get_position()
            y = (p_from.y0 + p_from.y1) / 2
            fig.add_artist(
                mpatches.FancyArrowPatch(
                    (p_from.x1 + left_pad, y),
                    (p_to.x0 - 0.006, y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=22,
                    linewidth=1.8,
                    color="black",
                )
            )

        # Zoom-lens connector: dotted red lines from the highlighted patch's corners
        # (in ax_patch data coords) to the zoom panel's corners (in ax_zoom data coords).
        patch_corners = [
            (zoom_row * ps - 0.5, zoom_col * ps + ps - 0.5),  # top-left of red box
            (zoom_row * ps - 0.5, zoom_col * ps - 0.5),  # bottom-left of red box
        ]
        zoom_corners = [(-0.5, ps - 0.5), (-0.5, -0.5)]  # top-left, bottom-left of zoom axes
        for (xa, ya), (xb, yb) in zip(patch_corners, zoom_corners):
            con = mpatches.ConnectionPatch(
                xyA=(xa, ya),
                coordsA=ax_patch.transData,
                xyB=(xb, yb),
                coordsB=ax_zoom.transData,
                linestyle=":",
                linewidth=1.6,
                color="red",
            )
            fig.add_artist(con)

        for fmt in formats:
            out_path = outdir / f"overview.{fmt}"
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
