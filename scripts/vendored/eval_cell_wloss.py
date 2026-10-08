#!/usr/bin/env python
"""Evaluate the cell energy-loss-weighting round and make a rate-distortion plot.

All runs share the fixed reference point (K=codebook_size=4096, d=codebook_dim=8)
and vary only two knobs relative to the cb4096_cd8_nq4 reference:
  * num_quantizers Q  (the *rate* axis — bits/token = Q*log2(K)); Q in {4,6,8}
    at energy_loss_alpha=0.
  * energy_loss_alpha  (the per-pixel energy-MSE weighting w_i=E_i^alpha); a
    single alpha=0.7 point at Q=4 (same bitrate as the Q=4 baseline).

Metric plotted (the "IQR"): jet/event-level energy resolution = IQR/median of
E^reco/E^true per event (``energy_resolution_iqr`` from
compute_cell_scan_resolution.compute_scan_point_metrics). Lower is better.

Reuses compute_scan_point_metrics (same model loading + metric definitions as
the cb x cd heatmap script) so the numbers are directly comparable to the rest
of the cell scan. Writes a per-run test_metrics.json, a summary CSV, and a
rate-distortion figure that keeps the Q-scan (alpha=0) and the alpha ablation
visually distinct.

Usage (run on a GPU node, --mem=200G — CellDataModule eager-loads the dataset):
    python scripts/eval_cell_wloss.py \
        --scan_dir <CELLS_RESULTS>/scan_default/cells_scan_2M_nq4_deadcode_warmup_wloss/transformer \
        --data_path <CELLS_H5_2M_DIR> \
        --outdir plots/cell_scan_wloss
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cell_scan_resolution import compute_scan_point_metrics  # noqa: E402

plt.style.use(hep.style.CMS)

# cb<K>_cd<d>_nq<Q>[_a<alpha>][_l1][_{ev,pt}sum<lambda>][_large]
# (alpha absent == 0, loss absent == mse, sum term absent == off).
NAME_RE = re.compile(
    r"^cb(\d+)_cd(\d+)_nq(\d+)(?:_a([0-9.]+))?(_l1)?(?:_(ev|pt)sum([0-9.]+))?(_large)?$"
)


def parse_name(name: str):
    m = NAME_RE.match(name)
    if not m:
        return None
    return {
        "cb": int(m.group(1)),
        "cd": int(m.group(2)),
        "nq": int(m.group(3)),
        "alpha": float(m.group(4)) if m.group(4) else 0.0,
        "loss": "l1" if m.group(5) else "mse",
        "sum_level": {"ev": "event", "pt": "patch"}.get(m.group(6)),
        "sum_w": float(m.group(7)) if m.group(7) else 0.0,
        "arch": "large" if m.group(8) else "base",
    }


def variant_label(r: dict) -> str:
    parts = ["L1" if r["loss"] == "l1" else "MSE", rf"$\alpha={r['alpha']:g}$"]
    if r["sum_w"]:
        parts.append(rf"{r['sum_level']}-sum $\lambda={r['sum_w']:g}$")
    parts.append("Large enc/dec" if r["arch"] == "large" else "base enc/dec")
    return ", ".join(parts)


def _is_baseline(r: dict) -> bool:
    return r["alpha"] == 0.0 and r["loss"] == "mse" and r["sum_w"] == 0.0 and r["arch"] == "base"


def make_plot(rows: list[dict], outdir: Path) -> None:
    """Left: rate-distortion (jet IQR/median vs information fraction). The
    baseline (MSE, alpha=0, base enc/dec) Q scan is a line; every other variant
    is a single marker at its own rate. Right: all variants at the Q=4 rate,
    side by side, with values printed, so same-rate points don't overlap."""
    base = sorted((r for r in rows if _is_baseline(r)), key=lambda r: r["nq"])
    others = [r for r in rows if not _is_baseline(r)]
    alpha_colors = {0.7: "#ff7f0e", 1.0: "#d62728", 1.4: "#9467bd"}

    def style(r):
        color = "#2ca02c" if r["arch"] == "large" else alpha_colors.get(r["alpha"], "#8c564b")
        marker = "*" if r["arch"] == "large" else ("s" if r["loss"] == "l1" else "D")
        return color, marker

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})

    if base:
        ax.plot([r["info_frac"] for r in base], [r["jet_iqr"] for r in base], "-o",
                color="#1f77b4", ms=9, lw=2, label=r"baseline (MSE, $\alpha=0$, base), $Q$ scan")
        for r in base:
            ax.annotate(f"Q={r['nq']}", (r["info_frac"], r["jet_iqr"]),
                        textcoords="offset points", xytext=(10, -4), fontsize=12, color="#1f77b4")
    for r in others:
        color, marker = style(r)
        ax.plot(r["info_frac"], r["jet_iqr"], marker, color=color,
                ms=16 if marker == "*" else 11, label=variant_label(r) + f", $Q={r['nq']}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"Information fraction $= Q\log_2 K\,/\,(32\cdot D)$,  $D=64$", fontsize=15)
    ax.set_ylabel(r"Jet IQR/median of $E^{\mathrm{reco}}/E^{\mathrm{true}}$", fontsize=15)
    ax.set_title("Rate–distortion ($K$=4096, $d$=8)", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    q4 = sorted((r for r in rows if r["nq"] == 4), key=lambda r: -r["jet_iqr"])
    for i, r in enumerate(q4):
        color, marker = ("#1f77b4", "o") if _is_baseline(r) else style(r)
        ax2.plot(r["jet_iqr"], i, marker, color=color, ms=16 if marker == "*" else 11)
        ax2.annotate(f"{r['jet_iqr']:.4f}  (resp. median {r['jet_response_median']:.3f})",
                     (r["jet_iqr"], i), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=11)
    ax2.set_yticks(range(len(q4)))
    ax2.set_yticklabels([variant_label(r) for r in q4], fontsize=11)
    ax2.set_ylim(-0.6, len(q4) - 0.2)
    lo, hi = min(r["jet_iqr"] for r in q4), max(r["jet_iqr"] for r in q4)
    pad = 0.35 * (hi - lo) + 3e-3
    ax2.set_xlim(lo - pad, hi + pad)
    ax2.set_xlabel(r"Jet IQR/median (lower is better)", fontsize=15)
    ax2.set_title(r"All variants at the same rate: $Q$=4 (info frac 0.023)", fontsize=14)
    ax2.grid(True, axis="x", alpha=0.3)

    n_evts = sorted({r["n_test_events"] for r in rows})
    n_str = f"{n_evts[0] / 1000:.0f}k" if len(n_evts) == 1 else "mixed-size"
    fig.suptitle(f"Cell tokenizer, trained on ~2M events; evaluated on {n_str} held-out test events",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        p = outdir / f"cell_wloss_rate_distortion.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan_dir", help="Dir holding cb*_cd*_nq*[_aX] run subdirs (aggregate/plot)")
    ap.add_argument("--run_dir", help="A single cb*_cd*_nq*[_aX] run dir (compute one point)")
    ap.add_argument("--runs", nargs="+", help="Explicit run dirs to aggregate (instead of --scan_dir)")
    ap.add_argument("--data_path", help="Preprocessed HDF5 dir (ds=1); required unless --plot_only")
    ap.add_argument("--test_path", help="Evaluate on every event under this dir instead of data_path's random test split")
    ap.add_argument("--sparse", action="store_true", help="Test data is in the sparsify_cell_h5.py format")
    ap.add_argument("--outdir", default="plots/cell_scan_wloss")
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_events", type=int, default=None, help="Cap test events (default: all)")
    ap.add_argument("--plot_only", action="store_true",
                    help="Skip compute; read each run's existing test_metrics.json and plot.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Single-run compute mode: eval one checkpoint, write its json, exit. ──
    if args.run_dir:
        sub = Path(args.run_dir)
        parsed = parse_name(sub.name)
        if parsed is None:
            raise ValueError(f"--run_dir name {sub.name!r} is not cb*_cd*_nq*[_aX][_l1][_evsumX][_large]")
        cb, cd, nq, alpha = parsed["cb"], parsed["cd"], parsed["nq"], parsed["alpha"]
        ckpt = sub / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            ckpt = sub / "checkpoints" / "last.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint under {sub}")
        if not (args.data_path or args.test_path):
            raise ValueError("--data_path or --test_path is required for --run_dir compute mode")
        print(f"[eval] {sub.name}  (K={cb} d={cd} Q={nq} alpha={alpha}) ...", flush=True)
        metrics = compute_scan_point_metrics(
            str(ckpt), args.data_path, downsample=args.downsample,
            batch_size=args.batch_size, max_events=args.max_events, label=sub.name,
            test_path=args.test_path, sparse=args.sparse,
        )
        with open(sub / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[done] {sub.name}: jet IQR/median = {metrics['energy_resolution_iqr']:.4f} "
              f"-> {sub / 'test_metrics.json'}")
        return

    if args.runs:
        candidates = [Path(p) for p in args.runs]
    elif args.scan_dir:
        candidates = sorted(Path(args.scan_dir).iterdir())
    else:
        raise ValueError("Provide --run_dir (compute one), --runs, or --scan_dir")

    rows = []
    for sub in candidates:
        if not sub.is_dir():
            continue
        parsed = parse_name(sub.name)
        if parsed is None:
            continue
        cb, cd, nq, alpha = parsed["cb"], parsed["cd"], parsed["nq"], parsed["alpha"]

        if args.plot_only:
            # Data-free: read each run's already-written test_metrics.json.
            mpath = sub / "test_metrics.json"
            if not mpath.exists():
                print(f"[skip] {sub.name}: no test_metrics.json (run its eval job first)")
                continue
            with open(mpath) as f:
                metrics = json.load(f)
        else:
            ckpt = sub / "checkpoints" / "best.ckpt"
            if not ckpt.exists():
                ckpt = sub / "checkpoints" / "last.ckpt"
            if not ckpt.exists():
                print(f"[skip] {sub.name}: no checkpoint")
                continue
            if not (args.data_path or args.test_path):
                raise ValueError("--data_path or --test_path is required unless --plot_only")
            print(f"[eval] {sub.name}  (K={cb} d={cd} Q={nq} alpha={alpha}) ...", flush=True)
            metrics = compute_scan_point_metrics(
                str(ckpt), args.data_path, downsample=args.downsample,
                batch_size=args.batch_size, max_events=args.max_events, label=sub.name,
                test_path=args.test_path, sparse=args.sparse,
            )
            with open(sub / "test_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        rows.append({
            "name": sub.name, "cb": cb, "cd": cd, "nq": nq, "alpha": alpha,
            "loss": parsed["loss"], "arch": parsed["arch"], "path": str(sub),
            "sum_w": parsed["sum_w"], "sum_level": parsed["sum_level"],
            "bits_per_token": nq * float(np.log2(cb)),
            "info_frac": nq * float(np.log2(cb)) / (32 * 64),
            "jet_iqr": metrics["energy_resolution_iqr"],
            "jet_response_median": metrics["energy_response_median"],
            "patch_iqr_thr": metrics["patch_response_iqr_thr"],
            "pixel_iqr_thr": metrics["pixel_response_iqr_thr"],
            "codebook_util_avg": metrics["codebook_util_avg"],
            "n_test_events": metrics["n_test_events"],
        })

    if not rows:
        raise RuntimeError("No usable run dirs found")

    rows.sort(key=lambda r: (r["arch"], r["loss"], r["alpha"], r["nq"]))
    csv_path = outdir / "cell_wloss_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    print("\n=== Jet energy resolution (IQR/median of E_reco/E_true) ===")
    print(f"{'run':<26} {'Q':>2} {'alpha':>5} {'loss':>4} {'arch':>5} {'info_frac':>9} {'jet_IQR':>9} "
          f"{'resp_med':>9} {'patch_IQR_thr':>13} {'cb_util':>8} {'n_test':>7}")
    for r in rows:
        print(f"{r['name']:<26} {r['nq']:>2} {r['alpha']:>5.2g} {r['loss']:>4} {r['arch']:>5} "
              f"{r['info_frac']:>9.4f} {r['jet_iqr']:>9.4f} {r['jet_response_median']:>9.4f} "
              f"{r['patch_iqr_thr']:>13.4f} {r['codebook_util_avg']:>8.4f} {r['n_test_events']:>7}")

    make_plot(rows, outdir)


if __name__ == "__main__":
    main()
