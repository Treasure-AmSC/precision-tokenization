"""Table C.4: jet resolution (IQR/median, %) versus training-set size, collected from stored run metrics.

Set-based modalities: IQR/median of pt_ratio in each run's test_nofastpath test_metrics.npz. The COCOA
topocluster 5M entry is the mean over the three seeds of (4096, 8, 4). Cells: energy_resolution_iqr from each
run's test_metrics.json.

Usage:
    python scripts/table_c4.py --a_results results --cells_results <CELLS_RESULTS> --out table_c4.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

JETSET_RUN = "transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj{n}"
TOPOS_RUN = "cocoa_topos_cb4096_cd8_nq4_lr0.001"


def rows(a: Path, cells: Path):
    """(modality, (K, d, Q), training events, [metric sources])."""
    tracks = a / "cocoa_scan/tracks_datascan"
    topos = a / "cocoa_scan/topos_datascan"
    jetset = a / "vqvae_scan/vqvae_scan"
    cells_2m = cells / "cells_scan_2M_a0.4_frontier/transformer"
    cells_20m = cells / "cells_20M/full"
    return [
        ("COCOA tracks", (4096, 4, 4), "1M", [tracks / "cocoa_tracks_datascan_n1000000"]),
        ("COCOA tracks", (4096, 4, 4), "5M", [tracks / "cocoa_tracks_datascan_n5000000"]),
        ("COCOA tracks", (4096, 4, 4), "10M", [tracks / "cocoa_tracks_datascan_n10000000"]),
        ("COCOA tracks", (4096, 4, 4), "20M", [a / "cocoa_scan/tracks/cocoa_tracks_cb4096_cd4_nq4_lr0.001"]),
        ("COCOA topoclusters", (4096, 8, 4), "1M", [topos / "cocoa_topos_datascan_n1000000"]),
        ("COCOA topoclusters", (4096, 8, 4), "5M", [a / "cocoa_scan/topos" / TOPOS_RUN,
                                                    a / "cocoa_scan/topos_seed" / f"{TOPOS_RUN}_seed1234",
                                                    a / "cocoa_scan/topos_seed" / f"{TOPOS_RUN}_seed5678"]),
        ("COCOA topoclusters", (4096, 8, 4), "10M", [topos / "cocoa_topos_datascan_n10000000"]),
        ("JetSet", (32768, 8, 4), "1M", [jetset / JETSET_RUN.format(n=1_000_000)]),
        ("JetSet", (32768, 8, 4), "10M", [jetset / JETSET_RUN.format(n=10_000_000)]),
        ("COCOA cells", (4096, 8, 8), "2M", [cells_2m / "cb4096_cd8_nq8_a0.4_large"]),
        ("COCOA cells", (4096, 8, 8), "20M", [cells_20m / "cb4096_cd8_nq8_a0.4_large"]),
        ("COCOA cells", (4096, 8, 16), "2M", [cells_2m / "cb4096_cd8_nq16_a0.4_large"]),
        ("COCOA cells", (4096, 8, 16), "20M", [cells_20m / "cb4096_cd8_nq16_a0.4_large"]),
    ]


def resolution(run: Path) -> float:
    if (run / "test_metrics.json").exists():
        return float(json.loads((run / "test_metrics.json").read_text())["energy_resolution_iqr"])
    (npz,) = (run / "test_nofastpath").rglob("test_metrics.npz")
    pr = np.load(npz)["pt_ratio"]
    pr = pr[np.isfinite(pr)]
    q25, q50, q75 = np.percentile(pr, [25, 50, 75])
    return float((q75 - q25) / q50)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a_results", type=Path, default=Path("results"))
    p.add_argument("--cells_results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    out = []
    for modality, (K, d, Q), n, runs in rows(args.a_results, args.cells_results):
        vals = [resolution(r) for r in runs]
        out.append({"modality": modality, "K": K, "d": d, "Q": Q, "training_events": n,
                    "iqr_over_median_pct": 100 * float(np.mean(vals)), "n_runs": len(vals),
                    "runs": ";".join(r.name for r in runs)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        v = r["iqr_over_median_pct"]
        shown = f"{v:.3f}" if v < 0.1 else f"{v:.2f}"  # paper precision
        print(f"{r['modality']:20s} ({r['K']}, {r['d']}, {r['Q']}) {r['training_events']:>4s}  {v:.4f}  {shown}")


if __name__ == "__main__":
    main()
