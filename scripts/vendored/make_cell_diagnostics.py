#!/usr/bin/env python
"""Paper-ready cell-reconstruction diagnostics for a single cell VQ-VAE checkpoint.

Reproduces the six-panel "cell diagnostics" figure (truth vs predicted
distributions + reco-minus-true residuals, at pixel / patch / event level)
that the old ``CellReconstructionMonitor.energy_diagnostics`` used to emit,
but as a standalone, publication-styled figure for an arbitrary checkpoint
(MLP or Transformer).

It reuses ``compute_cell_scan_resolution.build_model`` (auto-detects the
architecture from the state_dict and handles the masked+posenc transformer)
and the exact same truth/reco definitions as that script:
  * per-pixel physical energy  = (energy^0.3 stored value)^(1/0.3), kept only
    where the patch is non-empty AND the pixel is occupied -- truth uses the
    ground-truth occupancy indicator, reco uses the model's OWN predicted
    occupancy (sigmoid(indicator_logit) > 0.5).
  * per-patch energy = sum of pixel energies in the patch.
  * active pixels / event = count of occupied pixels over all valid patches.

Panels (rows = pixel / patch / event; left = distribution, right = residual):
  (0,0) pixel energy          (0,1) pixel  reco - true
  (1,0) patch energy          (1,1) patch  reco - true
  (2,0) active pixels/event   (2,1) active pixels reco - true

Usage:
  pixi run python scripts/make_cell_diagnostics.py \\
      --ckpt <CELLS_RESULTS>/.../cb16384_cd16_nq4/checkpoints/best.ckpt \\
      --data_path <CELLS_H5_2M_DIR> \\
      --outdir plots/cell_diagnostics_transformer_cb16384_cd16_nq4 \\
      --max_events 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from torch.utils.data import DataLoader

# Reuse the (architecture-auto-detecting) model loader + physics constant and
# the dataloading from the scan-resolution script (same directory).
from compute_cell_scan_resolution import ENERGY_POWER, build_model
from heptokens.vendored.cocoa.data.hep4m_cell import CellDataModule

plt.style.use(hep.style.ATLAS)

# Consistent palette: truth is a filled light band, prediction an accent line,
# residual a distinct warm accent.
C_TRUTH = "#9fb8d6"      # muted steel-blue fill
C_TRUTH_EDGE = "#3b6ea5"
C_PRED = "#e8802b"       # orange
C_RESID = "#c0392b"      # crimson


def _superscript_pow2(n: int) -> str:
    """Render a power-of-two as 2^{k} for embedding INSIDE a math span (no $)."""
    if n > 0 and (n & (n - 1)) == 0:
        return rf"2^{{{n.bit_length() - 1}}}"
    return str(n)


@torch.no_grad()
def collect_diagnostics(ckpt: str, data_path: str | None, batch_size: int, max_events: int | None,
                        test_path: str | None = None, sparse: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, vq, decoder, cd, cb, nq = build_model(ckpt)
    for m in (encoder, vq, decoder):
        m.to(device).eval()

    split = {"test_path": test_path} if test_path is not None else {"data_path": data_path}
    dm = CellDataModule(**split, sparse=sparse, batch_size=batch_size, num_workers=0)
    loader = DataLoader(dm.test_set, batch_size=batch_size, shuffle=False)

    pix_e_true, pix_e_pred, pix_resid = [], [], []
    patch_e_true, patch_e_pred, patch_resid = [], [], []
    evt_true_cnt, evt_pred_cnt = [], []

    seen = 0
    for batch in loader:
        energy, mask = batch["energy"].to(device), batch["mask"].to(device)
        model_batch = {"energy": energy, "mask": mask}
        if "positions" in batch:
            model_batch["positions"] = batch["positions"].to(device)

        z_e = encoder(model_batch)
        z_q, _, _ = vq(z_e)
        recon = decoder(z_q, model_batch)

        n_pix = energy.shape[-1] // 2
        e_true = energy[..., :n_pix]
        ind_true = energy[..., n_pix:] > 0.5
        e_recon = recon[..., :n_pix]
        reco_occ = torch.sigmoid(recon[..., n_pix:]) > 0.5

        m3 = mask.unsqueeze(-1)
        true_pix = ind_true & m3
        reco_pix = reco_occ & m3

        e_true_phys = e_true.clamp(min=0.0).pow(1.0 / ENERGY_POWER) * true_pix
        e_reco_phys = e_recon.clamp(min=0.0).pow(1.0 / ENERGY_POWER) * reco_pix

        # pixel-level distributions (occupied pixels only)
        pix_e_true.append(e_true_phys[true_pix].cpu().numpy())
        pix_e_pred.append(e_reco_phys[reco_pix].cpu().numpy())
        # pixel residual over pixels occupied in truth OR reco (within valid patches)
        pix_union = m3 & (true_pix | reco_pix)
        pix_resid.append((e_reco_phys - e_true_phys)[pix_union].cpu().numpy())

        # patch-level
        e_true_patch = e_true_phys.sum(-1)
        e_reco_patch = e_reco_phys.sum(-1)
        pt = mask & (e_true_patch > 0)
        pp = mask & (e_reco_patch > 0)
        patch_e_true.append(e_true_patch[pt].cpu().numpy())
        patch_e_pred.append(e_reco_patch[pp].cpu().numpy())
        patch_union = mask & ((e_true_patch > 0) | (e_reco_patch > 0))
        patch_resid.append((e_reco_patch - e_true_patch)[patch_union].cpu().numpy())

        # event-level active-pixel counts
        evt_true_cnt.append(true_pix.sum(dim=(1, 2)).cpu().numpy())
        evt_pred_cnt.append(reco_pix.sum(dim=(1, 2)).cpu().numpy())

        seen += energy.shape[0]
        if max_events is not None and seen >= max_events:
            break

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return {
        "pix_e_true": cat(pix_e_true), "pix_e_pred": cat(pix_e_pred), "pix_resid": cat(pix_resid),
        "patch_e_true": cat(patch_e_true), "patch_e_pred": cat(patch_e_pred), "patch_resid": cat(patch_resid),
        "evt_true_cnt": cat(evt_true_cnt), "evt_pred_cnt": cat(evt_pred_cnt),
        "cb": cb, "cd": cd, "nq": nq, "n_events": seen,
    }


def _log_bins(*arrays, n=70, lo_q=0.1, hi_q=99.95):
    vals = np.concatenate([a[a > 0] for a in arrays if a.size])
    if vals.size == 0:
        return np.logspace(-4, 2, n)
    lo = max(np.percentile(vals, lo_q), vals.min(), 1e-6)
    hi = np.percentile(vals, hi_q)
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _sym_bins(arr, n=81, q=99.5):
    if arr.size == 0:
        return np.linspace(-1, 1, n)
    m = np.percentile(np.abs(arr), q)
    m = max(m, 1e-6)
    return np.linspace(-m, m, n)


def _dist_panel(ax, truth, pred, bins, xlabel, logx, logy):
    ax.hist(truth, bins=bins, density=True, histtype="stepfilled",
            color=C_TRUTH, edgecolor=C_TRUTH_EDGE, linewidth=1.0, alpha=0.85, label="Truth")
    ax.hist(pred, bins=bins, density=True, histtype="step",
            color=C_PRED, linewidth=1.8, label="Predicted")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=13, loc="upper right")


def _resid_panel(ax, resid, bins, xlabel, logy):
    ax.hist(resid, bins=bins, density=True, histtype="stepfilled",
            color=C_RESID, edgecolor="#7b241c", linewidth=1.0, alpha=0.85)
    ax.axvline(0.0, color="0.35", lw=1.0, ls="--")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")


def make_figure(d: dict, outdir: Path, formats: list[str], title_extra: str = ""):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 14.5))

    # Row 0 — pixel level
    pb = _log_bins(d["pix_e_true"], d["pix_e_pred"])
    _dist_panel(axes[0, 0], d["pix_e_true"], d["pix_e_pred"], pb, "pixel energy [GeV]", logx=True, logy=True)
    _resid_panel(axes[0, 1], d["pix_resid"], _sym_bins(d["pix_resid"]), "pixel  (reco $-$ true) [GeV]", logy=True)

    # Row 1 — patch level
    qb = _log_bins(d["patch_e_true"], d["patch_e_pred"])
    _dist_panel(axes[1, 0], d["patch_e_true"], d["patch_e_pred"], qb, "patch energy [GeV]", logx=True, logy=True)
    _resid_panel(axes[1, 1], d["patch_resid"], _sym_bins(d["patch_resid"]), "patch  (reco $-$ true) [GeV]", logy=True)

    # Row 2 — event level (active pixels)
    hi = max(d["evt_true_cnt"].max() if d["evt_true_cnt"].size else 1,
             d["evt_pred_cnt"].max() if d["evt_pred_cnt"].size else 1)
    eb = np.linspace(0, hi * 1.02, 70)
    _dist_panel(axes[2, 0], d["evt_true_cnt"], d["evt_pred_cnt"], eb, "active pixels per event", logx=False, logy=False)
    evt_resid = d["evt_pred_cnt"] - d["evt_true_cnt"]
    m = max(int(np.ceil(np.percentile(np.abs(evt_resid), 99.7))) if evt_resid.size else 5, 1)
    int_bins = np.arange(-m - 0.5, m + 1.5, 1.0)  # integer-centered bars
    _resid_panel(axes[2, 1], evt_resid, int_bins, "active pixels  (reco $-$ true)", logy=False)

    cb_str = _superscript_pow2(int(d["cb"]))
    header = (rf"Transformer VQ-VAE   $(Q, K, d) = ({int(d['nq'])},\ {cb_str},\ {int(d['cd'])})$")
    if title_extra:
        header += f"   {title_extra}"
    fig.suptitle(header, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = outdir / f"cell_diagnostics.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to a checkpoint (best.ckpt).")
    ap.add_argument("--data_path", help="Preprocessed HDF5 cell data dir (uses its random test split).")
    ap.add_argument("--test_path", help="Use every event under this dir instead of data_path's test split.")
    ap.add_argument("--sparse", action="store_true", help="Test data is in the sparsify_cell_h5.py format.")
    ap.add_argument("--outdir", default="plots/cell_diagnostics")
    ap.add_argument("--max_events", type=int, default=50000, help="Cap events for the figure (default 50000; None=all).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--format", default="pdf,png")
    ap.add_argument("--title_extra", default="")
    ap.add_argument("--from_npz", default=None,
                    help="Skip the forward pass and replot from a previously-saved arrays npz (style tweaks).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.from_npz:
        z = np.load(args.from_npz)
        d = {k: z[k] for k in z.files}
        d["cb"], d["cd"], d["nq"] = int(d["cb"]), int(d["cd"]), int(d["nq"])
        print(f"Replotting from {args.from_npz}")
    else:
        if not (args.data_path or args.test_path):
            ap.error("--data_path or --test_path is required unless --from_npz")
        d = collect_diagnostics(args.ckpt, args.data_path, args.batch_size,
                                None if args.max_events is not None and args.max_events <= 0 else args.max_events,
                                test_path=args.test_path, sparse=args.sparse)
        print(f"Collected over {d['n_events']} events: "
              f"{d['pix_e_true'].size} truth pixels, {d['pix_e_pred'].size} pred pixels, "
              f"{d['patch_e_true'].size} patches")
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outdir / "diagnostics_arrays.npz", **d)
    make_figure(d, outdir, [f.strip() for f in args.format.split(",") if f.strip()], args.title_extra)


if __name__ == "__main__":
    main()
