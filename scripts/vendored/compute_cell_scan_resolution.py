#!/usr/bin/env python
"""Compute patch/pixel/jet(event)-level reconstruction resolution & response
for one or more cell VQ-VAE codebook scans, and plot cb x cd heatmaps of them
— reproducing the metrics/plots that used to live at:
  <CELLS_RESULTS>/scan_default/cells_scan_default_2M/plots_cb16384/

Apples-to-apples ds comparison (floor / total_vs_raw)
------------------------------------------------------
`total_energy_response`/`total_energy_resolution` (and the patch/pixel
metrics) compare a scan's RECO pixels to that SAME scan's own TRUE pixels --
but the true-pixel value is sum_cell(E_cell^0.3), compressed per CELL before
being summed into a pixel. Since sum_i(E_i^0.3) != (sum_i E_i)^0.3 (x^0.3 is
concave), decompressing a pixel that merges multiple cells does NOT recover
sum_i(E_i) -- it's a systematically different quantity, and how different
depends on how many cells got merged, i.e. on downsample. At ds=1 a pixel is
~1 cell (verified empirically: 0 bins ever contain >1 cell), so decompression
is nearly a no-op; at ds=2 a pixel merges ~4 cells, so it isn't -- and this
turns out to be a LARGE effect in practice, not a small correction: checked
directly on one event, ds=2's own decompressed truth pixel sum was ~14x the
true raw cell-energy sum for that event (two ~19 GeV cells merging into one
pixel alone contributes a ~5x error, before even summing over the rest of
the event). So naively comparing ds=1's and ds=2's total_energy_resolution
compares two very different definitions of "truth" -- the exact same
confound the pixel/patch binning was flagged for, just reintroduced (and
amplified by the 1/0.3 ~ 3.33 inverse power) through the compression step.

The fix: build_raw_truth_energy() computes ONE truth number per event that
never touches ANY preprocessed/pixelized representation at all -- it reads
the per-cell energies directly from the ORIGINAL ROOT files and sums them
per event (see that function's docstring), shared by every scan regardless
of its own downsample. Every scan's compute_scan_point_metrics() call can
then also report two metrics against this single shared denominator instead
of its own ds-specific truth:
  - floor:        (this ds's TRUE pixels, decompressed, no VQ) / raw truth
                   -- isolates the binning+compression bias alone, per ds.
  - total_vs_raw:  (this ds's RECO pixels, decompressed)       / raw truth
                   -- binning+compression+quantization combined.
Comparing `floor` vs `total_vs_raw` for a given ds separates "coarser bins
lose information" from "the VQ-VAE reconstructs badly"; comparing either
one ACROSS ds is now valid, since the denominator no longer depends on ds.
Pass --raw_truth_root_dir (the directory of original ROOT files) to enable
these; without it they're reported as NaN and only the ds-specific-truth
metrics are computed (as before this feature existed).

Background
----------
The ds=1 scan (.../cells_scan_default_2M/default_cb*_cd*_nq*/) already has a
``test_metrics.json`` per scan point, written by a ``CellTestMetrics`` Lightning
callback that exists on another git branch (feat/cell-geometry-config-and-scan-prep,
commit b47882d) but was never merged into this branch, so it isn't importable
here anymore. Rather than resurrecting that callback (which also depends on a
``heptokens_cocoa.data.cell_geometry`` registry module that no longer exists on
this branch), this script reimplements the same metric definitions as a plain,
Trainer-free post-hoc analysis: load a checkpoint's encoder/VQ/decoder directly
from its state_dict (same technique as make_tokenization_schematic.py and
compare_cell_energy_resolution.py), run it over that scan point's own held-out
test split (CellDataModule's test_frac=0.1, seed=42 — the exact same split
workflow/hep4m_cell_pixel_scan.smk's test_ds2_scan rule evaluates on), and
compute the response/resolution numbers directly from the (small, since
occupancy is sparse) per-pixel/per-patch/per-event ratio arrays using exact
quantiles — simpler than the original callback's streaming log-histograms, at
the cost of holding O(10-30M) float32 ratios in memory (a few hundred MB),
which is fine given these jobs already run with --mem=200G.

This means the script works identically whether or not diagnostic files exist:
  - ds=1 scan points: test_metrics.json already exists -> reused as-is (fast).
  - ds=2 scan points: missing -> computed fresh here, written in the same
    schema, so the two scans are directly comparable.
Pass --force to recompute every point fresh with this script's own formula
(median/exact-quantile IQR) uniformly, if perfect internal consistency across
scans matters more than reusing the (histogram-interpolated) legacy numbers.
The two conventions agree closely in practice (ratio histograms use 800 bins
over [0, 4], i.e. bin width 0.005, versus typical medians around 0.8-1.0).

Metrics per scan point (matching plots_cb16384's plot names):
  - jet/event-level:  total_energy_response (mean), energy_response_median,
                      total_energy_resolution (population std of response),
                      energy_resolution_iqr ((q75-q25)/median of response)
  - patch-level:      patch_response_median(_thr), patch_response_iqr(_thr)
  - pixel-level:      pixel_response_median(_thr), pixel_response_iqr(_thr)
  - codebook_util_avg, codebook_util_per_quantizer
  - longitudinal_profile_mae (per-event per-layer energy-fraction MAE)
  - position_resolution (RMS eta-phi residual of the energy-weighted patch
    centroid, truth vs reco -- see compute_scan_point_metrics' docstring),
    plus position_eta/phi_bias and position_eta/phi_resolution components.
    NOTE: this metric did not exist when the ds=1 test_metrics.json files
    were originally written, so REUSED ds=1 points will show it as NaN in
    the heatmap/CSV unless you pass --force to recompute them.
"_thr" variants only include patches/pixels whose *parent patch's* truth
energy exceeds --patch_energy_threshold (default 0.5 GeV) — same convention
as the original callback (excludes noise-level deposits from the "real
physics" resolution number).

Both truth and reco per-pixel energies use the ground-truth pixel occupancy
indicator for TRUTH, but the model's OWN predicted occupancy
(sigmoid(indicator_logit) > 0.5) for RECO — so a model that reconstructs the
right energy in the wrong place is correctly penalized, not silently graded
against ground-truth occupancy.

Usage
-----
  # ds=1 (reuses existing test_metrics.json; instant) + ds=2 (computed fresh)
  # + --raw_truth_root_dir to also get the apples-to-apples floor/total_vs_raw
  # metrics (requires --force to backfill points whose json predates this).
  pixi run python scripts/compute_cell_scan_resolution.py \\
      --scan ds1 1 <CELLS_RESULTS>/scan_default/cells_scan_default_2M \\
                   <CELLS_H5_2M_DIR> \\
      --scan ds2 2 results/hep4m_cell_pixel_scan/ds2_scan \\
                   data/cocoa/calo/ds2 \\
      --raw_truth_root_dir <COCOA_ROOT>/train_topup2_20M_cells256 \\
      --force \\
      --outdir plots/cell_scan_resolution

  # Quick smoke test on a handful of events per scan point:
  # (add --max_events 2000)

Each --scan is a repeatable (label, downsample, scan_root, data_path) tuple.
scan_root is searched for subdirectories matching cb<N>_cd<N>_nq<N> (any
prefix is fine, e.g. "default_cb256_cd4_nq3" or "cb256_cd4_nq3").
"""

import argparse
import json
import re
from functools import partial
from pathlib import Path

import awkward as ak
import matplotlib
import mplhep as hep
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from vector_quantize_pytorch import ResidualVQ

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heptokens.vendored import b_module_names
from heptokens.vendored.models.coders import Decoder, Encoder
from heptokens.vendored.models.transformer import Transformer
from heptokens.vendored.cocoa.callbacks.cell_recon import _layer_geometry, _N_LAYERS
from heptokens.vendored.cocoa.data.hep4m_cell import CellDataModule

plt.style.use(hep.style.CMS)

ENERGY_POWER = 0.3
DEFAULT_THR_GEV = 0.5
SCAN_DIR_RE = re.compile(r"cb(\d+)_cd(\d+)_nq(\d+)")
LABEL_FS, TICK_FS, CBAR_LABEL_FS = 15, 12, 13

# Transformer coder hyperparameters that are fixed across the whole
# hep4m_cell_transformer_vqvae.yaml scan family and NOT recoverable from a
# state_dict's tensor shapes alone (n_heads only changes how attention splits
# the existing in_proj_weight, so shape-based auto-detection can't recover it).
TRANSFORMER_N_HEADS = 8
TRANSFORMER_ACTIVATION = "gelu"
TRANSFORMER_DROPOUT = 0.0


# ── Model loading (mirrors make_tokenization_schematic.py's build_encoder_and_vq) ──


def _build_transformer_coder(prefix: str, sd: dict, input_dim: int, output_dim: int, coder_cls) -> torch.nn.Module:
    """Build+load an Encoder/Decoder whose inner model is a
    heptokens.models.transformer.Transformer (configs/model/
    hep4m_cell_transformer_vqvae.yaml or hep4m_cell_transformer_masked_posenc_
    vqvae.yaml), auto-detecting d_model, num_layers, and dim_feedforward from
    the checkpoint's own tensor shapes (n_heads/activation/dropout are NOT
    shape-recoverable -- see TRANSFORMER_* consts).

    Masking (mask_key) and positional encoding (positions_key/pos_dim) are
    NOT recoverable from tensor shapes alone in general (masking adds no
    separate params) -- but presence of a `pos_proj.weight` tensor uniquely
    identifies hep4m_cell_transformer_masked_posenc_vqvae.yaml, the only
    transformer variant in this codebase that combines positional encoding
    with masking empty patches out of attention/loss -- so it's used as a
    reliable proxy for both. Plain hep4m_cell_transformer_vqvae.yaml
    checkpoints (mask_key: null, no posenc) have no pos_proj.weight and are
    handled exactly as before (mask_key=None, positions_key=None).
    """
    d_model = sd[f"{prefix}.coder.input_proj.weight"].shape[0]
    dim_feedforward = sd[f"{prefix}.coder.encoder.layers.0.linear1.weight"].shape[0]
    layer_re = re.compile(rf"^{re.escape(prefix)}\.coder\.encoder\.layers\.(\d+)\.self_attn\.in_proj_weight$")
    num_layers = max((int(m.group(1)) for k in sd if (m := layer_re.match(k))), default=-1) + 1

    pos_proj_key = f"{prefix}.coder.pos_proj.weight"
    has_posenc = pos_proj_key in sd
    pos_dim = sd[pos_proj_key].shape[1] if has_posenc else None
    mask_key = "mask" if has_posenc else None
    positions_key = "positions" if has_posenc else None

    model_partial = partial(
        Transformer,
        d_model=d_model,
        n_heads=TRANSFORMER_N_HEADS,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=TRANSFORMER_DROPOUT,
        activation=TRANSFORMER_ACTIVATION,
        pos_dim=pos_dim,
    )
    coder = coder_cls(
        input_dim=input_dim,
        output_dim=output_dim,
        model=model_partial,
        input_key="energy",
        mask_key=mask_key,
        positions_key=positions_key,
    )
    coder.load_state_dict({k[len(prefix) + 1 :]: v for k, v in sd.items() if k.startswith(prefix + ".")})
    coder.eval()
    return coder


def build_model(ckpt_path: str):
    """Rebuild encoder + ResidualVQ + decoder directly from a checkpoint's
    state_dict, auto-detecting codebook_dim/codebook_size/num_quantizers, and
    the coder architecture itself: the default per-patch MLP (CoderModel,
    hep4m_cell_vqvae.yaml) or a self-attention Transformer over all patches
    (hep4m_cell_transformer_vqvae.yaml) -- detected from which keys are
    present in encoder.coder.*.
    """
    with b_module_names():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    is_transformer = "encoder.coder.input_proj.weight" in sd

    codebook_dim = (
        sd["encoder.coder.output_proj.weight"].shape[0] if is_transformer else sd["encoder.coder.model.9.weight"].shape[0]
    )
    codebook_size = sd["vector_quantization.layers.0.embed"].shape[-1]
    num_quantizers = sum(
        1 for k in sd if k.endswith(".embed") and "vector_quantization.layers." in k
    )

    if is_transformer:
        encoder = _build_transformer_coder("encoder", sd, input_dim=128, output_dim=codebook_dim, coder_cls=Encoder)
        decoder = _build_transformer_coder("decoder", sd, input_dim=codebook_dim, output_dim=128, coder_cls=Decoder)
    else:
        encoder = Encoder(input_dim=128, output_dim=codebook_dim, input_key="energy")
        encoder.load_state_dict({k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")})
        encoder.eval()

        decoder = Decoder(input_dim=codebook_dim, output_dim=128, input_key="energy")
        decoder.load_state_dict({k[len("decoder.") :]: v for k, v in sd.items() if k.startswith("decoder.")})
        decoder.eval()

    vq = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, commitment=1.0)
    vq.load_state_dict(
        {k[len("vector_quantization.") :]: v for k, v in sd.items() if k.startswith("vector_quantization.")},
        strict=False,
    )
    vq.eval()
    return encoder, vq, decoder, codebook_dim, codebook_size, num_quantizers


def _median_iqr_over_median(x: torch.Tensor) -> tuple[float, float]:
    """(median, IQR/median) of a 1D tensor via exact quantiles.

    Uses numpy.percentile rather than torch.quantile: torch.quantile has a
    hard ~16.7M-element (2^24) limit ("quantile() input tensor is too
    large"), which pixel-level ratio arrays over a full ~200k-event
    held-out test set (10s of millions of occupied pixels) routinely exceed.
    """
    if x.numel() == 0:
        return float("nan"), float("nan")
    x_np = x.float().numpy()
    med = float(np.percentile(x_np, 50))
    q25, q75 = np.percentile(x_np, [25, 75])
    return med, (float(q75 - q25) / med if med > 0 else float("nan"))


# ── Shared, downsample-independent truth energy ─────────────────────────────

DEFAULT_ROOT_PATTERN = "{stem}_100kseg_bw100.0.root"
DEFAULT_ROOT_TREE = "EventTree"


def build_raw_truth_energy(
    ref_data_path: str,
    root_dir: str,
    root_pattern: str = DEFAULT_ROOT_PATTERN,
    tree_name: str = DEFAULT_ROOT_TREE,
) -> np.ndarray:
    """Per-event total raw cell energy, read directly from the ORIGINAL ROOT
    files (bypassing every preprocessed pixel representation entirely, at
    any downsample) -- the single truth number ANY scan can be compared
    against on equal footing, since it never touches per-ds binning or the
    per-cell E^0.3 compression at all. See the module docstring's
    "Apples-to-apples ds comparison" section for why decompressing an
    already-preprocessed (and, at downsample>1, cell-merged) pixel value is
    NOT an adequate substitute for this (verified empirically: for one event,
    ds=2's own decompressed pixel sum was ~14x the true raw cell-energy sum).

    `ref_data_path` is any ONE scan's preprocessed HDF5 directory (e.g. the
    ds=1 scan's data_path) -- used ONLY to get the exact list/order of source
    file stems ("singlejet_270" from "singlejet_270.h5", sorted the same way
    CellDataModule's `_resolve_files` does), NOT to read any pixel data from
    it. This matters because `root_dir` is typically a much larger shared
    ROOT-file directory (e.g. a 20M-event dataset shard directory) containing
    far more files than the ~20 actually used by any given scan -- naively
    globbing root_dir for ALL matching ROOT files silently pulls in those
    extra, unrelated events too (this exact bug: a full-dataset glob returned
    19,981,852 events for a 1,999,383-event scan, ~10x too many). Building
    the file list from ref_data_path's own (already-scoped-down) HDF5
    directory instead guarantees exactly the right ROOT files, in the right
    order, with a per-file existence check as a safety net.
    preprocess_hep4m_cells.py reads each ROOT file's tree sequentially with
    no shuffling (entry i -> HDF5 row i), so this per-file, per-row alignment
    is exact.
    """
    import uproot

    h5_files = sorted(Path(ref_data_path).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No *.h5 files found in {ref_data_path}")

    per_file_sums = []
    for h5f in h5_files:
        root_path = Path(root_dir) / root_pattern.format(stem=h5f.stem)
        if not root_path.exists():
            raise FileNotFoundError(
                f"Expected ROOT file {root_path} (matching HDF5 file {h5f}) not found -- "
                "check --raw_truth_root_dir/--raw_truth_root_pattern."
            )
        with uproot.open(root_path) as fh:
            cell_e = fh[tree_name]["cell_e"].array(library="ak")
        per_file_sums.append(ak.to_numpy(ak.sum(cell_e, axis=1)).astype(np.float64))
    return np.concatenate(per_file_sums)


@torch.no_grad()
def compute_scan_point_metrics(
    ckpt_path: str,
    data_path: str,
    downsample: int,
    batch_size: int = 256,
    max_events: int | None = None,
    thr_gev: float = DEFAULT_THR_GEV,
    label: str | None = None,
    raw_truth_full: np.ndarray | None = None,
    test_path: str | None = None,
    sparse: bool = False,
) -> dict:
    """Run one checkpoint over its own held-out test split and compute all
    patch/pixel/jet response+resolution metrics, codebook utilization, the
    longitudinal-profile MAE, and a position-residual (energy-weighted patch
    centroid) metric.

    The energy-only metrics above (patch/pixel/jet response) are blind to
    WHERE the energy landed within an event -- two reconstructions with the
    same total energy but a different spatial distribution score identically.
    The position residual complements them: per event, compare the
    energy-weighted (eta, phi) centroid of the truth patches to that of the
    reco patches (using each patch's own position, already stored in the
    data -- no sub-patch/pixel geometry needed). phi uses circular
    (cos, sin)-weighted averaging and a wrapped residual so it is well
    defined across the +-pi boundary.

    If `raw_truth_full` is given (see build_raw_truth_energy), this also
    computes two ds-comparable "apples-to-apples" metrics against that single
    shared, downsample-independent truth number instead of this scan's own
    (ds-specific, compression-biased) truth pixels:
      - floor:  (this ds's TRUE pixels, decompressed, no VQ) / raw_truth --
                isolates the binning+compression bias alone, per ds.
      - total_vs_raw: (this ds's RECO pixels, decompressed) / raw_truth --
                binning+compression+quantization combined, directly
                comparable across ds since the denominator no longer depends
                on ds-specific compression.
    raw_truth_full must be aligned to the SAME global event indexing as this
    scan's own dataset (same files, same order, same total event count) --
    see build_raw_truth_energy's docstring.

    If `test_path` is given, every event under it is the test set (no random
    split of `data_path`); `sparse` reads the sparsify_cell_h5.py format.
    """
    encoder, vq, decoder, codebook_dim, codebook_size, num_quantizers = build_model(ckpt_path)
    bins_to_keep, patches_per_layer, layer_starts = _layer_geometry(downsample)

    if test_path is not None and raw_truth_full is not None:
        raise ValueError("raw_truth_full alignment assumes the random split of data_path, not test_path")
    split = {"test_path": test_path} if test_path is not None else {"data_path": data_path}
    dm = CellDataModule(**split, sparse=sparse, batch_size=batch_size, num_workers=0)
    loader = DataLoader(dm.test_set, batch_size=batch_size, shuffle=False)

    raw_truth_test = None
    if raw_truth_full is not None:
        total_len = len(dm.test_set.dataset)
        if total_len != len(raw_truth_full):
            raise ValueError(
                f"raw_truth_full has {len(raw_truth_full)} events but this scan's dataset has "
                f"{total_len} -- they must come from the same (# files, # events/file) source "
                "for the shared-truth event alignment to be valid."
            )
        raw_truth_test = torch.from_numpy(raw_truth_full[np.asarray(dm.test_set.indices)]).float()

    pixel_all, pixel_thr, patch_all, patch_thr = [], [], [], []
    evt_true, evt_reco = [], []
    floor_resp_chunks, total_raw_resp_chunks = [], []
    frac_true_chunks, frac_reco_chunks = [], []
    eta_resid_chunks, phi_resid_chunks = [], []
    used_codes = [torch.zeros(codebook_size, dtype=torch.bool) for _ in range(num_quantizers)]

    n_events = 0
    for batch in loader:
        batch_start = n_events
        energy, mask = batch["energy"], batch["mask"]
        # Include positions (if present in the batch) so masked_posenc
        # transformer checkpoints (positions_key="positions") get the same
        # positional encoding at eval time that they were trained with;
        # Encoder/Decoder simply ignore this key when positions_key=None.
        model_batch = {"energy": energy, "mask": mask}
        if "positions" in batch:
            model_batch["positions"] = batch["positions"]
        z_e = encoder(model_batch)
        z_q, indices, _ = vq(z_e)  # indices: [num_quantizers, B, P]
        recon = decoder(z_q, model_batch)

        n_pix = energy.shape[-1] // 2
        e_true = energy[..., :n_pix]
        ind_true = energy[..., n_pix:] > 0.5
        e_recon = recon[..., :n_pix]
        reco_occ = torch.sigmoid(recon[..., n_pix:]) > 0.5

        true_pix = (ind_true & mask.unsqueeze(-1)).to(e_true.dtype)
        reco_pix = (reco_occ & mask.unsqueeze(-1)).to(e_recon.dtype)

        e_true_phys = e_true.clamp(min=0.0).pow(1.0 / ENERGY_POWER) * true_pix  # [B, P, 64]
        e_reco_phys = e_recon.clamp(min=0.0).pow(1.0 / ENERGY_POWER) * reco_pix

        # ── patch level ──
        e_true_patch = e_true_phys.sum(-1)  # [B, P]
        e_reco_patch = e_reco_phys.sum(-1)

        patch_valid_all = mask & (e_true_patch > 0)
        if patch_valid_all.any():
            patch_all.append(e_reco_patch[patch_valid_all] / e_true_patch[patch_valid_all])
        patch_valid_thr = mask & (e_true_patch > thr_gev)
        if patch_valid_thr.any():
            patch_thr.append(e_reco_patch[patch_valid_thr] / e_true_patch[patch_valid_thr])

        # ── pixel level ──
        pix_active = mask.unsqueeze(-1).expand_as(e_true_phys)
        pix_valid_all = pix_active & (e_true_phys > 0)
        if pix_valid_all.any():
            pixel_all.append(e_reco_phys[pix_valid_all] / e_true_phys[pix_valid_all])
        parent_patch_e = e_true_patch.unsqueeze(-1).expand_as(e_true_phys)
        pix_valid_thr = pix_valid_all & (parent_patch_e > thr_gev)
        if pix_valid_thr.any():
            pixel_thr.append(e_reco_phys[pix_valid_thr] / e_true_phys[pix_valid_thr])

        # ── event (jet) level ──
        e_true_evt = e_true_patch.sum(-1)  # [B]
        e_reco_evt = e_reco_patch.sum(-1)

        # Apples-to-apples vs the shared, ds-independent raw truth (if given):
        # floor = ds-specific TRUE pixels (no VQ) / raw truth; total_vs_raw =
        # ds-specific RECO pixels (with VQ) / raw truth. Both use the SAME
        # denominator regardless of downsample, so the binning+compression
        # bias (floor) and the full pipeline error (total_vs_raw) are directly
        # comparable across ds -- unlike total_energy_response/resolution
        # below, whose denominator (e_true_evt) is itself ds-dependent.
        if raw_truth_test is not None:
            raw_truth_batch = raw_truth_test[batch_start : batch_start + energy.shape[0]]
            raw_valid = raw_truth_batch > 0
            if raw_valid.any():
                floor_resp_chunks.append(e_true_evt[raw_valid] / raw_truth_batch[raw_valid])
                total_raw_resp_chunks.append(e_reco_evt[raw_valid] / raw_truth_batch[raw_valid])

        valid_evt = e_true_evt > 0
        if valid_evt.any():
            et = e_true_evt[valid_evt]
            er = e_reco_evt[valid_evt]
            evt_true.append(et)
            evt_reco.append(er)

            # Longitudinal profile: per-event per-layer truth/reco energy fraction.
            er_safe = torch.where(er > 0, er, et)  # guard zero-reco events
            frac_true = torch.zeros(int(valid_evt.sum()), _N_LAYERS)
            frac_reco = torch.zeros(int(valid_evt.sum()), _N_LAYERS)
            for li in range(_N_LAYERS):
                start, end = layer_starts[li], layer_starts[li] + patches_per_layer[li]
                frac_true[:, li] = e_true_phys[valid_evt, start:end, :].sum(dim=(-2, -1)) / et
                frac_reco[:, li] = e_reco_phys[valid_evt, start:end, :].sum(dim=(-2, -1)) / er_safe
            frac_true_chunks.append(frac_true)
            frac_reco_chunks.append(frac_reco)

            # Position residual: energy-weighted patch centroid, truth vs reco.
            valid_pos = valid_evt & (e_reco_evt > 0)
            if valid_pos.any():
                positions = batch["positions"]  # [B, P, 3] = (eta, cos_phi, sin_phi)
                eta_p, cosphi_p, sinphi_p = positions[..., 0], positions[..., 1], positions[..., 2]

                eta_true_c = (eta_p * e_true_patch).sum(-1)[valid_pos] / e_true_evt[valid_pos]
                eta_reco_c = (eta_p * e_reco_patch).sum(-1)[valid_pos] / e_reco_evt[valid_pos]

                cos_true_c = (cosphi_p * e_true_patch).sum(-1)[valid_pos]
                sin_true_c = (sinphi_p * e_true_patch).sum(-1)[valid_pos]
                cos_reco_c = (cosphi_p * e_reco_patch).sum(-1)[valid_pos]
                sin_reco_c = (sinphi_p * e_reco_patch).sum(-1)[valid_pos]
                phi_true_c = torch.atan2(sin_true_c, cos_true_c)
                phi_reco_c = torch.atan2(sin_reco_c, cos_reco_c)

                eta_resid_chunks.append(eta_reco_c - eta_true_c)
                phi_resid_chunks.append(
                    torch.atan2(torch.sin(phi_reco_c - phi_true_c), torch.cos(phi_reco_c - phi_true_c))
                )

        for q in range(num_quantizers):
            idx_q = indices[q][mask]
            used_codes[q][idx_q.unique()] = True

        n_events += energy.shape[0]
        if max_events is not None and n_events >= max_events:
            break

    patch_response_median, patch_response_iqr = _median_iqr_over_median(
        torch.cat(patch_all) if patch_all else torch.tensor([])
    )
    patch_response_median_thr, patch_response_iqr_thr = _median_iqr_over_median(
        torch.cat(patch_thr) if patch_thr else torch.tensor([])
    )
    pixel_response_median, pixel_response_iqr = _median_iqr_over_median(
        torch.cat(pixel_all) if pixel_all else torch.tensor([])
    )
    pixel_response_median_thr, pixel_response_iqr_thr = _median_iqr_over_median(
        torch.cat(pixel_thr) if pixel_thr else torch.tensor([])
    )

    evt_true_cat = torch.cat(evt_true)
    evt_reco_cat = torch.cat(evt_reco)
    response_evt = evt_reco_cat / evt_true_cat
    total_energy_response = float(response_evt.mean())
    total_energy_resolution = float(response_evt.std(unbiased=False))
    energy_response_median, energy_resolution_iqr = _median_iqr_over_median(response_evt)

    # Apples-to-apples vs the shared raw truth (NaN if raw_truth_full wasn't given).
    floor_response_median, floor_resolution_iqr = _median_iqr_over_median(
        torch.cat(floor_resp_chunks) if floor_resp_chunks else torch.tensor([])
    )
    total_response_raw_median, total_resolution_raw_iqr = _median_iqr_over_median(
        torch.cat(total_raw_resp_chunks) if total_raw_resp_chunks else torch.tensor([])
    )

    frac_true_all = torch.cat(frac_true_chunks)
    frac_reco_all = torch.cat(frac_reco_chunks)
    longitudinal_profile_mae = float(torch.abs(frac_true_all - frac_reco_all).mean())

    eta_resid_all = torch.cat(eta_resid_chunks) if eta_resid_chunks else torch.tensor([])
    phi_resid_all = torch.cat(phi_resid_chunks) if phi_resid_chunks else torch.tensor([])
    position_eta_bias = float(eta_resid_all.mean()) if eta_resid_all.numel() else float("nan")
    position_eta_resolution = float(eta_resid_all.std(unbiased=False)) if eta_resid_all.numel() else float("nan")
    position_phi_bias = float(phi_resid_all.mean()) if phi_resid_all.numel() else float("nan")
    position_phi_resolution = float(phi_resid_all.std(unbiased=False)) if phi_resid_all.numel() else float("nan")
    # Combined 2D (eta-phi) RMS residual -- a single-number "position resolution"
    # radius, analogous to how total_energy_resolution is a single energy number.
    position_resolution = (
        float(torch.sqrt((eta_resid_all**2 + phi_resid_all**2).mean()))
        if eta_resid_all.numel()
        else float("nan")
    )

    utils = [float(used_codes[q].float().mean()) for q in range(num_quantizers)]

    return {
        "label": label,
        "cb": codebook_size,
        "cd": codebook_dim,
        "nq": num_quantizers,
        "downsample": downsample,
        "total_energy_response": total_energy_response,
        "energy_response_median": energy_response_median,
        "total_energy_resolution": total_energy_resolution,
        "energy_resolution_iqr": energy_resolution_iqr,
        "floor_response_median": floor_response_median,
        "floor_resolution_iqr": floor_resolution_iqr,
        "total_response_raw_median": total_response_raw_median,
        "total_resolution_raw_iqr": total_resolution_raw_iqr,
        "patch_response_iqr": patch_response_iqr,
        "patch_response_iqr_thr": patch_response_iqr_thr,
        "patch_response_median": patch_response_median,
        "patch_response_median_thr": patch_response_median_thr,
        "pixel_response_iqr": pixel_response_iqr,
        "pixel_response_iqr_thr": pixel_response_iqr_thr,
        "pixel_response_median": pixel_response_median,
        "pixel_response_median_thr": pixel_response_median_thr,
        "longitudinal_profile_mae": longitudinal_profile_mae,
        "position_eta_bias": position_eta_bias,
        "position_eta_resolution": position_eta_resolution,
        "position_phi_bias": position_phi_bias,
        "position_phi_resolution": position_phi_resolution,
        "position_resolution": position_resolution,
        "codebook_util_avg": float(np.mean(utils)),
        "codebook_util_per_quantizer": utils,
        "n_test_events": n_events,
        "n_valid_events": int(evt_true_cat.numel()),
        "codebook_size": codebook_size,
        "num_quantizers": num_quantizers,
    }


def find_checkpoint(scan_point_dir: Path, prefer: str = "last") -> Path | None:
    # prefer="best" tries best.ckpt (min-val-loss epoch) first; default "last"
    # keeps the original last-then-best order.
    names = ("best.ckpt", "last.ckpt") if prefer == "best" else ("last.ckpt", "best.ckpt")
    for name in names:
        ckpt = scan_point_dir / "checkpoints" / name
        if ckpt.exists():
            return ckpt
    return None


def collect_scan(
    label: str,
    downsample: int,
    scan_root: str,
    data_path: str,
    force: bool,
    batch_size: int,
    max_events: int | None,
    thr_gev: float,
    raw_truth_full: np.ndarray | None = None,
    ckpt_prefer: str = "last",
) -> list[dict]:
    """Discover cb/cd/nq scan-point subdirectories under scan_root, and for
    each: reuse its test_metrics.json if present (unless force), else compute
    fresh from its checkpoint and write test_metrics.json in the same schema.
    """
    scan_root = Path(scan_root)
    rows = []
    for sub in sorted(p for p in scan_root.iterdir() if p.is_dir()):
        m = SCAN_DIR_RE.search(sub.name)
        if not m:
            continue
        cb, cd, nq = (int(g) for g in m.groups())

        json_path = sub / "test_metrics.json"
        if json_path.exists() and not force:
            with open(json_path) as f:
                metrics = json.load(f)
            print(f"[{label}] {sub.name}: reused existing {json_path}")
        else:
            ckpt = find_checkpoint(sub, prefer=ckpt_prefer)
            if ckpt is None:
                print(f"[{label}] {sub.name}: SKIPPED (no checkpoint found)")
                continue
            print(f"[{label}] {sub.name}: computing from {ckpt} ...")
            metrics = compute_scan_point_metrics(
                str(ckpt),
                data_path,
                downsample=downsample,
                batch_size=batch_size,
                max_events=max_events,
                thr_gev=thr_gev,
                label=sub.name,
                raw_truth_full=raw_truth_full,
            )
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"[{label}] {sub.name}: response={metrics['total_energy_response']:.4f} "
                f"resolution={metrics['total_energy_resolution']:.4f} "
                f"cb_util={metrics['codebook_util_avg']:.4f} -> {json_path}"
            )

        row = {"scan_label": label, "downsample": downsample, "cb": cb, "cd": cd, "nq": nq}
        row.update({k: v for k, v in metrics.items() if k not in ("codebook_util_per_quantizer",)})
        rows.append(row)
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────

# (out_name_stem, metric, cbar_label, cmap, fmt) -- cbar_label/cmap/fmt match
# the legacy scripts/compare_cell_scan.py (git commit d41229a, deleted from
# this branch) exactly for the metrics that script plotted; new metrics that
# didn't exist there get a similarly-styled label/cmap/fmt.
HEATMAPS = [
    ("cell_scan_resolution_heatmap", "energy_resolution_iqr", "IQR / median", "viridis", "{:.4f}"),
    ("cell_scan_response_heatmap", "total_energy_response", "Energy response (median reco/true)", "viridis_r", "{:.3f}"),
    ("patch_resolution_heatmap", "patch_response_iqr", "IQR / median (patch)", "viridis", "{:.3f}"),
    ("patch_resolution_thr0p5_heatmap", "patch_response_iqr_thr", "IQR / median (patch)", "viridis", "{:.3f}"),
    ("pixel_resolution_heatmap", "pixel_response_iqr", "IQR / median (pixel)", "viridis", "{:.3f}"),
    ("pixel_resolution_thr0p5_heatmap", "pixel_response_iqr_thr", "IQR / median (pixel)", "viridis", "{:.3f}"),
    ("codebook_util_heatmap", "codebook_util_avg", "codebook utilization", "viridis_r", "{:.3f}"),
    ("position_resolution_heatmap", "position_resolution", "RMS eta-phi residual", "viridis", "{:.4f}"),
    ("floor_resolution_heatmap", "floor_resolution_iqr", "IQR / median (floor)", "viridis", "{:.3f}"),
    ("floor_response_heatmap", "floor_response_median", "Response (median, floor)", "viridis_r", "{:.3f}"),
    ("total_resolution_vs_raw_heatmap", "total_resolution_raw_iqr", "IQR / median (vs raw)", "viridis", "{:.3f}"),
    ("total_response_vs_raw_heatmap", "total_response_raw_median", "Response (median, vs raw)", "viridis_r", "{:.3f}"),
]


def plot_heatmaps(df: pd.DataFrame, outdir: Path, formats: list[str]) -> None:
    """Plot cb x cd heatmaps faceted by nq (columns) and scan_label (rows),
    styled to match the legacy scripts/compare_cell_scan.py::plot_metric_heatmap
    (git commit d41229a) pixel-for-pixel where this script's grid reduces to
    its single-scan case: cd on the y-axis (raw ints, origin="lower"), cb on
    the x-axis (as 2^N), one colorbar PER panel (but all panels for a given
    metric still share one global vmin/vmax), un-scanned cb/cd combinations
    hatched instead of blank, luminance-adaptive black/white value labels,
    and no figure suptitle (omitted "per paper style" in the original).
    """
    scan_labels = list(dict.fromkeys(df["scan_label"]))
    nqs = sorted(df["nq"].unique())
    multi_scan = len(scan_labels) > 1

    for fname, metric, cbar_label, cmap, fmt in HEATMAPS:
        n_rows, n_cols = len(scan_labels), len(nqs)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows), squeeze=False
        )
        vals = df[metric].replace([np.inf, -np.inf], np.nan).dropna() if metric in df else pd.Series(dtype=float)
        vmin, vmax = (float(vals.min()), float(vals.max())) if len(vals) else (None, None)
        norm = plt.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax > vmin) else None
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#e6e6e6")

        any_plotted = False
        for i, scan_label in enumerate(scan_labels):
            for j, nq in enumerate(nqs):
                ax = axes[i, j]
                sub = df[(df["scan_label"] == scan_label) & (df["nq"] == nq)]
                if sub.empty or metric not in sub or vmin is None:
                    ax.axis("off")
                    continue
                any_plotted = True
                cbs = sorted(df["cb"].unique())
                cds = sorted(df["cd"].unique())
                pivot = sub.pivot_table(index="cd", columns="cb", values=metric)
                grid = pivot.reindex(index=cds, columns=cbs).to_numpy(dtype=float)
                # A combo can be NaN for two different reasons: (a) it was never
                # scanned at all (no row for this cb/cd/nq/scan_label -- hatched
                # below), or (b) it WAS trained but the metric itself is
                # undefined (e.g. a collapsed/degenerate VQ-VAE where the median
                # response is 0, so IQR/median is a 0/0 NaN). Distinguish via a
                # presence grid (row count, NaN-agnostic) and render (b) as an
                # explicit 0.0 (red-outlined) instead of hatching it like (a).
                presence = sub.pivot_table(index="cd", columns="cb", values=metric, aggfunc="size")
                presence = presence.reindex(index=cds, columns=cbs).fillna(0).to_numpy()
                failed = (presence > 0) & np.isnan(grid)
                grid[failed] = 0.0
                masked = np.ma.masked_invalid(grid)
                im = ax.imshow(masked, cmap=cmap_obj, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
                for r in range(len(cds)):
                    for c in range(len(cbs)):
                        if np.isnan(grid[r, c]):
                            ax.add_patch(
                                Rectangle(
                                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                                    hatch="///", edgecolor="#b3b3b3", linewidth=0.0,
                                )
                            )
                        elif failed[r, c]:
                            ax.add_patch(
                                Rectangle(
                                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                                    edgecolor="red", linewidth=1.5,
                                )
                            )
                ax.set_xticks(range(len(cbs)))
                ax.set_xticklabels([rf"$2^{{{int(round(np.log2(cb)))}}}$" for cb in cbs])
                ax.set_yticks(range(len(cds)))
                ax.set_yticklabels(cds)
                ax.set_xlabel("Codebook size", fontsize=18)
                ax.set_ylabel("Codebook dim" if not multi_scan else f"{scan_label}\nCodebook dim", fontsize=18)
                ax.set_title(f"nq = {nq}" if not multi_scan else f"{scan_label}, nq = {nq}", fontsize=13)
                for r in range(len(cds)):
                    for c in range(len(cbs)):
                        v = grid[r, c]
                        if np.isnan(v):
                            continue
                        lum = 1.0
                        if norm is not None:
                            rr, gg, bb, _ = cmap_obj(norm(v))
                            lum = 0.299 * rr + 0.587 * gg + 0.114 * bb
                        text_color = "white" if lum < 0.5 else "black"
                        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=9, color=text_color)
                cbar = plt.colorbar(im, ax=ax, shrink=0.75)
                cbar.set_label(cbar_label, labelpad=6, fontsize=28)

        if not any_plotted:
            plt.close(fig)
            continue
        # Per-figure suptitle intentionally omitted (titles removed per paper style,
        # matching the legacy compare_cell_scan.py this reproduces).
        fig.tight_layout()
        for fmt_ext in formats:
            out_path = outdir / f"{fname}.{fmt_ext}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
            print(f"Saved {out_path}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scan",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "DOWNSAMPLE", "SCAN_ROOT", "DATA_PATH"),
        help="Repeatable: a (label, downsample, scan_root_dir, preprocessed_data_dir) tuple.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics for every scan point (ignores existing test_metrics.json).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_events", type=int, default=None, help="Cap events per scan point (default: all)")
    parser.add_argument(
        "--patch_energy_threshold",
        type=float,
        default=DEFAULT_THR_GEV,
        help="GeV threshold (on parent-patch truth energy) for the *_thr metric variants",
    )
    parser.add_argument(
        "--raw_truth_root_dir",
        default=None,
        help=(
            "Directory of ORIGINAL ROOT files used to build the single shared, "
            "ds-independent truth-energy reference for the apples-to-apples floor/"
            "total_vs_raw metrics (see build_raw_truth_energy) -- read directly, bypassing "
            "every preprocessed pixel representation. If omitted, floor/total_vs_raw are "
            "skipped (NaN) and only the ds-specific-truth metrics are computed, as before."
        ),
    )
    parser.add_argument(
        "--raw_truth_ref_data_path",
        default=None,
        help=(
            "Any ONE scan's preprocessed HDF5 directory, used ONLY to get the exact list/"
            "order of source file stems for --raw_truth_root_dir (see build_raw_truth_energy) "
            "-- root_dir is typically a much bigger shared directory with far more files than "
            "any one scan uses, so this avoids silently pulling in unrelated events. Defaults "
            "to the first --scan's DATA_PATH if omitted."
        ),
    )
    parser.add_argument("--raw_truth_root_pattern", default=DEFAULT_ROOT_PATTERN)
    parser.add_argument("--raw_truth_tree", default=DEFAULT_ROOT_TREE)
    parser.add_argument("--outdir", default="plots/cell_scan_resolution")
    parser.add_argument("--format", default="pdf,png")
    parser.add_argument(
        "--checkpoint",
        choices=("last", "best"),
        default="last",
        help="Which checkpoint to compute fresh metrics from (best = min-val-loss epoch).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lstrip(".") for f in args.format.split(",") if f.strip()]

    raw_truth_full = None
    if args.raw_truth_root_dir:
        ref_data_path = args.raw_truth_ref_data_path or args.scan[0][3]
        print(
            f"Building shared raw-truth energy reference from ROOT files in {args.raw_truth_root_dir} "
            f"(file list/order from {ref_data_path}) ..."
        )
        raw_truth_full = build_raw_truth_energy(
            ref_data_path,
            args.raw_truth_root_dir,
            root_pattern=args.raw_truth_root_pattern,
            tree_name=args.raw_truth_tree,
        )
        print(f"Raw truth reference: {len(raw_truth_full)} events")

    rows = []
    for label, downsample, scan_root, data_path in args.scan:
        rows.extend(
            collect_scan(
                label=label,
                downsample=int(downsample),
                scan_root=scan_root,
                data_path=data_path,
                force=args.force,
                batch_size=args.batch_size,
                max_events=args.max_events,
                thr_gev=args.patch_energy_threshold,
                raw_truth_full=raw_truth_full,
                ckpt_prefer=args.checkpoint,
            )
        )

    if not rows:
        raise RuntimeError("No scan points found across the given --scan directories.")

    df = pd.DataFrame(rows)
    csv_path = outdir / "scan_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df)} scan points)")

    plot_heatmaps(df, outdir, formats)


if __name__ == "__main__":
    main()
