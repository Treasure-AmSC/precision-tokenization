#!/usr/bin/env python
"""Export frozen-RVQ-tokenizer z_q (or, in continuous-baseline mode, the raw
preprocessed 5-feature vector) per COCOA track, plus the per-jet constituent
validity mask and truth jet pT, for downstream jet-pT regression.

Sibling of scripts/export_topo_tokens.py (see that file + repo memory notes
for the general pipeline design) -- this is the "COCOA tracks" variant, Step
1 of the same "downstream jet-pT regression from RVQ tokens" ladder.

Writes one npz per (config, split) containing:
    z         : [N, max_csts, dim] float32  (dim = codebook_dim if tokenized,
                                              or 5 if --ckpt_path is omitted)
    mask      : [N, max_csts]      bool
    truth_pt  : [N]                float32  (GeV, physical units)

Truth jet pT is built by summing the tracks' OWN track_pt vectors --
px = track_pt*cos(track_phi), py = track_pt*sin(track_phi) -- over all valid
tracks in the event. This matches heptokens.callbacks.recon
.ReconstructionMonitor._compute_jet_from_constituents (pt_idx=0, deta_idx=1,
dphi_idx=2, energy_idx=None -- tracks store pt directly, unlike topoclusters
which store energy), i.e. the same definition already backing the pt_ratio
numbers in the original tracks scan's test_metrics.npz (the decode-and-sum
baseline reused by scripts/eval_topo_regressor.py's --decode_sum_npz).

Usage:
    python scripts/export_track_tokens.py \\
        --root_dir <COCOA_ROOT>/train_topup2_20M_cells256 \\
        --num_events 1000000 \\
        --ckpt_path <A_RESULTS>/cocoa_scan/tracks/cocoa_tracks_cb4096_cd4_nq4_lr0.001/checkpoints/best.ckpt \\
        --out_path results/track_token_regression/tokens/cb4096_cd4_nq4/train.npz

    # Continuous-feature baseline (omit --ckpt_path):
    python scripts/export_track_tokens.py \\
        --root_dir .../train_topup2_20M_cells256 --num_events 1000000 \\
        --out_path results/track_token_regression/tokens/continuous/train.npz
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from heptokens.vendored.cocoa.data.cocoa_base import load_root_arrays, pad_jagged_to_fixed, root_files_from_dir

log = logging.getLogger(__name__)

# Exact feature order the referenced tracks checkpoints were trained with
# (datamodule.features in their full_config.yaml) -- D=5.
FEATURES = ["track_pt", "track_eta", "track_phi", "track_d0", "track_z0"]
PT_IDX = FEATURES.index("track_pt")
PHI_IDX = FEATURES.index("track_phi")

# The exact preprocessing transform these checkpoints were trained with.
DEFAULT_CST_FN = str(Path(__file__).resolve().parents[2] / "resources/cocoa_cst_log_standard.joblib")

# COCOATrackModule's own max_csts (see full_config.yaml of any tracks
# checkpoint) -- tracks are truncated to the first 15 per event during
# training, so we must match that here for consistency with the trained
# tokenizer and with the decode-and-sum baseline's own truth-pt definition.
DEFAULT_MAX_CSTS = 15


def compute_truth_jet_pt(csts_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum valid tracks' own track_pt vectors into one jet pT per event.

    Args:
        csts_raw: [N, max_csts, 5] RAW (untransformed, physical-unit) features.
        mask: [N, max_csts] bool validity mask.

    Returns:
        [N] float32 truth jet pT in GeV.
    """
    pt = csts_raw[:, :, PT_IDX]
    phi = csts_raw[:, :, PHI_IDX]
    px = np.where(mask, pt * np.cos(phi), 0.0)
    py = np.where(mask, pt * np.sin(phi), 0.0)
    jet_px = px.sum(axis=1)
    jet_py = py.sum(axis=1)
    return np.sqrt(jet_px**2 + jet_py**2).astype(np.float32)


def load_events(root_dir: str, num_events: int | None, max_csts: int) -> tuple[np.ndarray, np.ndarray]:
    """Read up to num_events events (in sorted-file order) from root_dir."""
    files = root_files_from_dir(root_dir)
    csts_parts, mask_parts = [], []
    loaded = 0
    for fpath in files:
        remaining = None if num_events is None else num_events - loaded
        if remaining is not None and remaining <= 0:
            break
        arrays = load_root_arrays(fpath, FEATURES, max_entries=remaining)
        csts, mask = pad_jagged_to_fixed(arrays, FEATURES, max_csts)
        csts_parts.append(csts)
        mask_parts.append(mask)
        loaded += len(mask)
        log.info(f"  read {fpath.name}: {len(mask):,} events (total {loaded:,})")
    csts = np.concatenate(csts_parts, axis=0)
    mask = np.concatenate(mask_parts, axis=0)
    if num_events is not None and len(csts) > num_events:
        csts = csts[:num_events]
        mask = mask[:num_events]
    return csts, mask


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root_dir", required=True, help="ROOT files directory (one split)")
    parser.add_argument("--num_events", type=int, default=None)
    parser.add_argument("--max_csts", type=int, default=DEFAULT_MAX_CSTS)
    parser.add_argument(
        "--ckpt_path", default=None,
        help="Frozen VQ-VAE checkpoint (best.ckpt). Omit for continuous-feature baseline mode.",
    )
    parser.add_argument("--cst_fn", default=DEFAULT_CST_FN)
    parser.add_argument("--batch_size", type=int, default=4096, help="Encoding batch size (tokenized mode only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info(f"Loading up to {args.num_events} events from {args.root_dir}")
    csts_raw, mask = load_events(args.root_dir, args.num_events, args.max_csts)
    log.info(f"Loaded {csts_raw.shape[0]:,} events, {int(mask.sum()):,} valid tracks total")

    truth_pt = compute_truth_jet_pt(csts_raw, mask)
    n_zero = int((truth_pt <= 0).sum())
    if n_zero:
        log.warning(f"{n_zero:,} events have truth_pt <= 0 (no valid tracks?) -- kept as-is")

    log.info(f"Applying preprocessing transform {args.cst_fn}")
    cst_fn = joblib.load(args.cst_fn)
    csts_t = csts_raw.copy()
    valid = csts_raw[mask]
    csts_t[mask] = cst_fn.transform(valid)

    if args.ckpt_path is None:
        log.info("Continuous-baseline mode: exporting raw (preprocessed) 5 features, no tokenizer")
        z = csts_t.astype(np.float32)
    else:
        log.info(f"Loading frozen VQ-VAE checkpoint: {args.ckpt_path} (device={args.device})")
        # Local import: heptokens is a heavy optional dependency only needed
        # in tokenized mode.
        from heptokens.vendored import b_module_names
        from heptokens.vendored.models.vq_vae import LitVqVae

        with b_module_names():
            model = LitVqVae.load_from_checkpoint(args.ckpt_path, map_location=args.device)
        model.eval()
        model.to(args.device)

        n = csts_t.shape[0]
        codebook_dim = model.hparams["codebook_dim"]
        z = np.zeros((n, args.max_csts, codebook_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                batch = {
                    "csts": torch.from_numpy(csts_t[start:end]).float().to(args.device),
                    "mask": torch.from_numpy(mask[start:end]).to(args.device),
                }
                z_q, _, _ = model.encode(batch)
                z[start:end] = z_q.cpu().numpy()
                if (start // args.batch_size) % 20 == 0:
                    log.info(f"  encoded {end:,}/{n:,}")
        log.info(f"Finished encoding: z {z.shape}, codebook_size={model.codebook_size}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, z=z, mask=mask, truth_pt=truth_pt)
    log.info(
        f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB): "
        f"z {z.shape}, mask {mask.shape}, truth_pt {truth_pt.shape}"
    )


if __name__ == "__main__":
    main()
