#!/usr/bin/env python
"""Step 3 (per-config): evaluate a trained jet-pT regressor on the held-out
test set exported by scripts/export_topo_tokens.py.

For one (config, trained regressor) pair, on the common held-out test set
(test_1M_cells256_isInfFalse):
  - predicted pT = exp(un-standardize(model(z_test, mask_test)))
  - r = pT_pred / truth_pT; report IQR/median overall and in ~5 truth-pT
    quantile bins (flatness check -- the single-jet pT spectrum falls
    steeply, so an overall number alone can hide the regressor just
    learning the prior).
  - "decode-and-sum" sanity check: reuse the SAME config's already-computed
    reconstruction-only test_metrics.npz pt_ratio (same truth-jet-pT
    definition as here, see export_topo_tokens.py's docstring) as the naive
    constituent-summing baseline to beat.

Writes exactly two files (no appending -- safe to run in parallel / rerun):
  {out_dir}/{config_name}__{variant}.summary.csv  (1 header + 1 row)
  {out_dir}/{config_name}__{variant}.binned.csv   (1 header + up to n_bins rows)

A separate collection step (scripts/make_topo_regression_plot.py) globs these
per-config files into the final master table + figure.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from heptokens.vendored.models.transformer import SetToVectorTransformer

log = logging.getLogger(__name__)

D_DEFAULT = 8  # input features per constituent, topocluster "positions-in" design default


def compression_factor(Q: float, K: float, D: float = D_DEFAULT) -> float:
    return 32.0 * D / (Q * np.log2(K))


def iqr_over_median(x) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return float((q75 - q25) / q50)


def quantile_bin_edges(truth_pt: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(truth_pt, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def load_regressor(meta_path: Path, ckpt_path: Path, device: str):
    with open(meta_path) as f:
        meta = json.load(f)
    model = SetToVectorTransformer(
        input_dim=meta["input_dim"],
        output_dim=1,
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        num_layers=meta["num_layers"],
        dim_feedforward=meta["dim_feedforward"],
        dropout=meta["dropout"],
        activation="gelu",
        pooling=meta["pooling"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, meta


def predict_pt(model, meta, z, mask, device, batch_size=4096) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            end = min(start + batch_size, len(z))
            z_b = torch.from_numpy(z[start:end]).float().to(device)
            mask_b = torch.from_numpy(mask[start:end]).bool().to(device)
            pred = model(z_b, mask_b).reshape(-1).cpu().numpy()
            preds.append(pred)
    pred_std = np.concatenate(preds)
    log_pt = pred_std * meta["target_std"] + meta["target_mean"]
    return np.exp(log_pt)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--regressor_dir", required=True, help="Dir with best.pt + meta.json")
    parser.add_argument("--config_name", required=True)
    parser.add_argument(
        "--variant", default="default", choices=["default", "continuous", "capacity_check"]
    )
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--Q", type=int, default=None)
    parser.add_argument(
        "--D", type=float, default=D_DEFAULT,
        help="Input features per constituent (8 for topoclusters, 5 for tracks) -- only affects compression_factor",
    )
    parser.add_argument(
        "--decode_sum_npz", default=None,
        help="Original reconstruction test_metrics.npz (has pt_ratio/truth_pt) for the same config",
    )
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    d_test = np.load(args.test_npz)
    z, mask, truth_pt = d_test["z"], d_test["mask"], d_test["truth_pt"]
    valid = truth_pt > 0
    z, mask, truth_pt = z[valid], mask[valid], truth_pt[valid]
    log.info(f"Loaded test set: {len(truth_pt):,} events")

    model, meta = load_regressor(
        Path(args.regressor_dir) / "meta.json", Path(args.regressor_dir) / "best.pt", args.device
    )
    pt_pred = predict_pt(model, meta, z, mask, args.device)
    r = pt_pred / truth_pt
    iqr_med_overall = iqr_over_median(r)
    log.info(f"Regression IQR/median (overall, n={len(r):,}): {iqr_med_overall:.5f}")

    # --- Binned flatness check (regression side) ---
    edges = quantile_bin_edges(truth_pt, args.n_bins)
    binned_rows = []
    for i in range(args.n_bins):
        in_bin = (truth_pt >= edges[i]) & (truth_pt < edges[i + 1])
        if in_bin.sum() < 10:
            continue
        binned_rows.append(
            {
                "config_name": args.config_name,
                "variant": args.variant,
                "bin_idx": i,
                "pt_lo": float(edges[i]),
                "pt_hi": float(edges[i + 1]),
                "n_in_bin": int(in_bin.sum()),
                "iqr_median_regression": iqr_over_median(r[in_bin]),
                "iqr_median_decode_sum": "",
            }
        )

    # --- Decode-and-sum sanity-check baseline (same config, same truth-pt definition) ---
    iqr_med_decode_sum = ""
    if args.decode_sum_npz is not None and Path(args.decode_sum_npz).exists():
        ref = np.load(args.decode_sum_npz)
        iqr_med_decode_sum = iqr_over_median(ref["pt_ratio"])
        log.info(f"Decode-and-sum IQR/median (same config, n={len(ref['pt_ratio']):,}): {iqr_med_decode_sum:.5f}")
        ref_truth_pt = ref.get("truth_pt")
        ref_pt_ratio = ref.get("pt_ratio")
        if ref_truth_pt is not None:
            for row in binned_rows:
                in_bin = (ref_truth_pt >= row["pt_lo"]) & (ref_truth_pt < row["pt_hi"])
                if in_bin.sum() >= 10:
                    row["iqr_median_decode_sum"] = iqr_over_median(ref_pt_ratio[in_bin])
    else:
        log.info("No --decode_sum_npz provided (or file missing) -- skipping decode-and-sum baseline")

    if args.K is not None and args.Q is not None:
        bits_per_constituent = args.Q * float(np.log2(args.K))
        comp_factor = compression_factor(args.Q, args.K, args.D)
    else:
        # Continuous baseline: fp32 features, no compression at all.
        bits_per_constituent = 32.0 * args.D
        comp_factor = 1.0

    summary_row = {
        "config_name": args.config_name,
        "variant": args.variant,
        "K": args.K if args.K is not None else "",
        "d": args.d if args.d is not None else "",
        "Q": args.Q if args.Q is not None else "",
        "n_params": meta["n_params"],
        "d_model": meta["d_model"],
        "num_layers": meta["num_layers"],
        "bits_per_constituent": bits_per_constituent,
        "compression_factor": comp_factor,
        "iqr_median_overall": iqr_med_overall,
        "iqr_median_decode_sum": iqr_med_decode_sum,
        "n_test_events": len(truth_pt),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.config_name}__{args.variant}"

    summary_path = out_dir / f"{stem}.summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    log.info(f"Wrote {summary_path}")

    binned_path = out_dir / f"{stem}.binned.csv"
    if binned_rows:
        with open(binned_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(binned_rows[0].keys()))
            writer.writeheader()
            writer.writerows(binned_rows)
        log.info(f"Wrote {binned_path} ({len(binned_rows)} bins)")

    # Raw per-event ratio + truth_pt, so downstream plotting can bootstrap a
    # sampling-variance uncertainty band for any single run (no repeated
    # seeds needed) -- used for the continuous-baseline's shaded band in
    # scripts/make_topo_regression_plot.py.
    ratios_path = out_dir / f"{stem}.ratios.npz"
    np.savez_compressed(ratios_path, r=r, truth_pt=truth_pt)
    log.info(f"Wrote {ratios_path}")


if __name__ == "__main__":
    main()
