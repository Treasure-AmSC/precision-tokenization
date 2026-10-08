#!/usr/bin/env python
"""Step 3 (collection): gather all per-config eval_topo_regressor.py outputs
into one master results table + a rate-distortion-style summary plot.

Master CSV: results/topo_token_regression/topo_regression_results.csv
  (config_name, variant, K, d, Q, bits_per_constituent, compression_factor,
   iqr_median_overall, iqr_median_decode_sum, n_test_events, ...) -- one row
  per (config, seed) eval run (config_name includes the "__seed{N}" suffix
  here, same as the underlying per-run summary.csv files).

Each ladder point (the 12 codebook-dim-8 configs + the continuous baseline)
is trained with 3 random seeds (42, 1, 2) -- single-seed point estimates
turned out to be quite seed-sensitive (see repo memory notes: weak
scalar-only supervision + sum-pooling + no regularization -> a fairly wide
spread of achievable outcomes per config). This script aggregates across
seeds (mean +/- SEM, i.e. std/sqrt(n_seeds)) for the actual error bars on
the main ladder + the continuous baseline; capacity_check has no repeated
seeds (one-off diagnostic), so it keeps a bootstrap (resampled test events)
error bar instead, same as before.

Plot: single panel, IQR/median vs compression factor (log y):
  - All 12 configs connected by a solid line (sorted by compression factor;
    NOT Pareto-frontier-filtered -- several (K, Q) pairs can share the same
    x since compression factor depends only on the product Q*log2(K), and
    keeping every point visible is useful for spotting non-monotonic
    behavior, which this ladder has turned out to have).
  - decode-and-sum sanity-check baseline (same configs, same truth-pt
    definition) as a dotted line for direct comparison (deterministic --
    identical across seeds, so no error bar needed on this line itself).
  - "capacity_check" point (2-4x larger model, same tokenizer config as the
    optimal point) shown as a distinct marker, NOT connected into the main
    line, with a bootstrap error bar.
  - "continuous" (raw, uncompressed features) baseline as a horizontal
    dashed line with a shaded band = std across its 3 seeds.

Usage:
    python scripts/make_topo_regression_plot.py \\
        --eval_dir results/topo_token_regression/eval \\
        --out_csv results/topo_token_regression/topo_regression_results.csv \\
        --out_plot_stem results/topo_token_regression/topo_regression_ladder
"""

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

log = logging.getLogger(__name__)


def load_summaries(eval_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(eval_dir.glob("*.summary.csv")):
        with open(path) as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    return rows


def to_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_config_seed(config_name: str) -> tuple[str, int | None]:
    """Split 'cb1024_cd8_nq4__seed42' -> ('cb1024_cd8_nq4', 42).

    Returns (config_name, None) unchanged if there's no "__seed{N}" suffix
    (e.g. capacity_check, which has no repeated seeds).
    """
    if "__seed" in config_name:
        base, seed_str = config_name.rsplit("__seed", 1)
        return base, int(seed_str)
    return config_name, None


def aggregate_by_seed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Group rows by (base_config_name, variant), average across seeds.

    Returns {(base_config, variant): {K, d, Q, compression_factor, mean,
    std, n_seeds, decode_sum}}.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        base, seed = parse_config_seed(r["config_name"])
        key = (base, r["variant"])
        groups.setdefault(key, []).append(r)

    agg = {}
    for key, group_rows in groups.items():
        ys = np.array([to_float(r["iqr_median_overall"]) for r in group_rows])
        decode_sums = [to_float(r["iqr_median_decode_sum"]) for r in group_rows]
        decode_sums = [d for d in decode_sums if np.isfinite(d)]
        std = float(ys.std()) if len(ys) > 1 else 0.0
        agg[key] = dict(
            K=group_rows[0]["K"],
            d=group_rows[0]["d"],
            Q=group_rows[0]["Q"],
            compression_factor=to_float(group_rows[0]["compression_factor"]),
            mean=float(ys.mean()),
            std=std,
            sem=std / np.sqrt(len(ys)) if len(ys) > 1 else 0.0,
            n_seeds=len(group_rows),
            decode_sum=float(np.mean(decode_sums)) if decode_sums else None,
        )
    return agg


def bootstrap_iqr_median_from_ratios(r: np.ndarray, n_boot: int = 200, seed: int = 42) -> tuple[float, float]:
    """Bootstrap (mean, std) of IQR/median over resampled events.

    Used only for `capacity_check`, which has no repeated seeds -- resampling
    test EVENTS (not re-training) estimates the sampling variance of the
    IQR/median statistic itself as a stand-in error bar.
    """
    rng = np.random.default_rng(seed)
    n = len(r)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = r[idx]
        q25, q50, q75 = np.percentile(sample, [25, 50, 75])
        if q50 != 0:
            vals.append((q75 - q25) / q50)
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_plot_stem", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eval_dir = Path(args.eval_dir)
    rows = load_summaries(eval_dir)
    if not rows:
        raise SystemExit(f"No *.summary.csv files found in {eval_dir}")
    log.info(f"Loaded {len(rows)} per-(config,seed) summaries from {eval_dir}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote master results table: {out_csv}")

    agg = aggregate_by_seed(rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.style.use(hep.style.CMS)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    color_reg = "#2ca02c"
    color_decode = "#7f7f7f"
    color_cap = "#d62728"
    color_cont = "#1f77b4"

    # --- Main ladder: all "default"-variant configs, aggregated over seeds ---
    default_points = [(base, v) for (base, variant), v in agg.items() if variant == "default"]
    default_points.sort(key=lambda item: item[1]["compression_factor"])

    if default_points:
        x = np.array([v["compression_factor"] for _, v in default_points])
        y = np.array([v["mean"] for _, v in default_points])
        yerr = np.array([v["sem"] for _, v in default_points])
        n_seeds = {v["n_seeds"] for _, v in default_points}
        seeds_label = (
            f"{min(n_seeds)}-{max(n_seeds)} seeds" if len(n_seeds) > 1
            else ("1 seed" if n_seeds == {1} else f"{n_seeds.pop()} seeds")
        )
        ax.errorbar(
            x, y, yerr=yerr, fmt="o-", color=color_reg, linewidth=1.8, markersize=6,
            capsize=3, zorder=3,
            label=f"regressor (tokenized, {seeds_label}, SEM)",
        )

        y_decode = np.array([v["decode_sum"] for _, v in default_points])
        if all(d is not None for d in y_decode):
            ax.plot(
                x, y_decode.astype(float), "s:", color=color_decode, linewidth=1.5,
                markersize=5, zorder=2, label="decode-and-sum (sanity check)",
            )

    # --- Capacity check: single run, no repeated seeds -> bootstrap error bar ---
    cap_key = next(((base, variant) for (base, variant) in agg if variant == "capacity_check"), None)
    if cap_key is not None:
        base, variant = cap_key
        v = agg[cap_key]
        cap_rows = [r for r in rows if r["variant"] == "capacity_check"]
        cx, cy = v["compression_factor"], v["mean"]
        cyerr = 0.0
        if cap_rows:
            ratios_path = eval_dir / f"{cap_rows[0]['config_name']}__{variant}.ratios.npz"
            if ratios_path.exists():
                _, cyerr = bootstrap_iqr_median_from_ratios(np.load(ratios_path)["r"])
        ax.errorbar(
            [cx], [cy], yerr=[cyerr], fmt="D", color=color_cap, markersize=9,
            markeredgecolor="black", markeredgewidth=0.7, capsize=3, zorder=5,
            label="capacity check (larger model)",
        )

    # --- Continuous baseline: aggregated over its own 3 seeds ---
    cont_key = next(((base, variant) for (base, variant) in agg if variant == "continuous"), None)
    if cont_key is not None:
        v = agg[cont_key]
        base_val, base_sem = v["mean"], v["sem"]
        ax.axhline(
            base_val, color=color_cont, linestyle="--", linewidth=1.5, zorder=1,
            label="continuous features (fp32)",
        )
        if base_sem > 0:
            ax.axhspan(base_val - base_sem, base_val + base_sem, color=color_cont, alpha=0.15, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel(r"Compression factor  ($32\,D$ bits / $Q\,\log_2 K$)")
    ax.set_ylabel(r"Jet $p_T$ IQR / median")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, bottom=True, left=True)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = Path(f"{args.out_plot_stem}.{ext}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

