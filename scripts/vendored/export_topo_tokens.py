#!/usr/bin/env python
"""Export frozen-RVQ-tokenizer z_q (or, in continuous-baseline mode, the raw
preprocessed 8-feature vector) per COCOA topocluster, plus the per-jet
constituent validity mask and truth jet pT, for downstream jet-pT regression.

This is Step 1 of the "downstream jet-pT regression from RVQ tokens" ladder:
loads a frozen VQ-VAE checkpoint (no re-training), encodes topoclusters read
directly from ROOT files (bypassing the Lightning datamodule entirely, same
pattern as scripts/compute_cell_scan_resolution.py), and writes one npz per
(config, split) containing:

    z         : [N, max_csts, dim] float32  (dim = codebook_dim if tokenized,
                                              or 8 if --ckpt_path is omitted)
    mask      : [N, max_csts]      bool
    truth_pt  : [N]                float32  (GeV, physical units)

Truth jet pT is built by summing the topoclusters' OWN (truth-level) 4-vectors
-- pt = topo_e / cosh(topo_eta), px = pt*cos(topo_phi), py = pt*sin(topo_phi) --
over all valid clusters in the event. This is the exact recipe used by
heptokens_cocoa.callbacks.topo_recon.TopoReconstructionMonitor
._compute_jet_from_clusters, i.e. the same definition already backing the
IQR/median numbers in plot_rate_distortion.py's SCANS["COCOA topoclusters"].
NOT the "particle_pt" ROOT branch (that's per truth PARTICLE, i.e. individual
jet constituents like photons/pions -- a different quantity).

Usage:
    python scripts/export_topo_tokens.py \\
        --root_dir <COCOA_ROOT>/train_topup2_20M_cells256 \\
        --num_events 1000000 \\
        --ckpt_path <A_RESULTS>/cocoa_scan/topos/cocoa_topos_cb4096_cd8_nq4_lr0.001/checkpoints/best.ckpt \\
        --out_path results/topo_token_regression/tokens/cb4096_cd8_nq4/train.npz

    # Continuous-feature baseline (omit --ckpt_path):
    python scripts/export_topo_tokens.py \\
        --root_dir .../train_topup2_20M_cells256 --num_events 1000000 \\
        --out_path results/topo_token_regression/tokens/continuous/train.npz
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from heptokens.vendored.cocoa.data.cocoa_base import load_root_arrays, pad_jagged_to_fixed, root_files_from_dir

log = logging.getLogger(__name__)

# Exact feature order the referenced topos checkpoints were trained with
# (datamodule.features in their full_config.yaml) -- "positions-in" design,
# eta/phi folded directly into the 8-dim content vector (D=8).
FEATURES = [
    "topo_eta", "topo_phi", "topo_e", "topo_rho",
    "topo_sigma_eta", "topo_sigma_phi", "topo_ecal_e", "topo_hcal_e",
]
ETA_IDX = FEATURES.index("topo_eta")
PHI_IDX = FEATURES.index("topo_phi")
ENERGY_IDX = FEATURES.index("topo_e")

# The exact preprocessing transform these checkpoints were trained with --
# lives in the OTHER heptokens repo's resources/ dir, NOT this workspace's
# top-level resources/cocoa_topo_log_standard_8features.joblib (different
# file, different md5 -- do not substitute).
DEFAULT_CST_FN = str(Path(__file__).resolve().parents[2] / "resources/cocoa_topo_log_standard.joblib")


def compute_truth_jet_pt(csts_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum valid topoclusters' own (truth) 4-vectors into one jet pT per event.

    Args:
        csts_raw: [N, max_csts, 8] RAW (untransformed, physical-unit) features.
        mask: [N, max_csts] bool validity mask.

    Returns:
        [N] float32 truth jet pT in GeV.
    """
    eta = csts_raw[:, :, ETA_IDX]
    phi = csts_raw[:, :, PHI_IDX]
    e = csts_raw[:, :, ENERGY_IDX]
    pt = e / np.cosh(eta)
    px = np.where(mask, pt * np.cos(phi), 0.0)
    py = np.where(mask, pt * np.sin(phi), 0.0)
    jet_px = px.sum(axis=1)
    jet_py = py.sum(axis=1)
    return np.sqrt(jet_px**2 + jet_py**2).astype(np.float32)


def load_events(root_dir: str, num_events: int | None, max_csts: int) -> tuple[np.ndarray, np.ndarray]:
    """Read up to num_events events (in sorted-file order) from root_dir.

    Stops reading further files once enough events are loaded (and caps the
    last file's read via max_entries), matching COCOATopoDataset's own
    efficient partial-load behavior -- no need to load all 20M train events
    to get the first 1M.
    """
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
    parser.add_argument("--max_csts", type=int, default=50)
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
    log.info(f"Loaded {csts_raw.shape[0]:,} events, {int(mask.sum()):,} valid clusters total")

    truth_pt = compute_truth_jet_pt(csts_raw, mask)
    n_zero = int((truth_pt <= 0).sum())
    if n_zero:
        log.warning(f"{n_zero:,} events have truth_pt <= 0 (no valid clusters?) -- kept as-is")

    log.info(f"Applying preprocessing transform {args.cst_fn}")
    cst_fn = joblib.load(args.cst_fn)
    csts_t = csts_raw.copy()
    valid = csts_raw[mask]
    csts_t[mask] = cst_fn.transform(valid)

    if args.ckpt_path is None:
        log.info("Continuous-baseline mode: exporting raw (preprocessed) 8 features, no tokenizer")
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
