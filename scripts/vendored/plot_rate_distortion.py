#!/usr/bin/env python
"""Rate-distortion summary figures for the RVQ tokenization paper.

Produces three standalone figures (all sharing the same information-fraction
x-axis convention and the larger, standalone hep.style.CMS-matched fonts):

  rate_distortion_pareto : cross-modality jet-pT IQR/median Pareto scan
      (JetSet, COCOA tracks, COCOA topoclusters) -- the front-page "money"
      plot. Faint per-d scan cloud + per-modality Pareto frontier + optimal-
      config stars, codebook dimension d encoded as marker SHAPE via one
      shared ABSOLUTE-value mapping (MARKER_BY_D).
  rate_distortion_jetset : JetSet classifier light-jet rejection vs
      information fraction (single panel).
  rate_distortion_cocoa  : COCOA downstream jet-pT resolution (IQR/median),
      two stacked panels sharing the x-axis -- (top) topocluster + track
      regression Pareto frontiers + continuous baselines, (bottom) particle-flow
      jet-pT (positions-in vs positions-out) in its own panel (linear y); each
      curve's star marks its most performant config.

X-axis: information fraction (fp32 baseline, per modality) -- the fraction
of the continuous/fp32 feature's bit budget actually used by the RVQ token:

    information_fraction = Q * log2(K) / (32 * D) = 1 / compression_factor

where D is the number of input features per token, K is the codebook size,
Q is the number of residual-VQ quantizers, and d (codebook_dim) does NOT
affect the bitrate (only K and Q do) -- it is scanned but marginalized out
of the x-axis (encoded as marker shape instead, see above). Continuous
(uncompressed, fp32) features sit at exactly 1.0; more compression means a
smaller fraction. `compression_factor()` (its reciprocal) is kept internally
to decide Pareto-frontier dominance ("more compression + lower distortion is
better"), but every plotted x-value is `information_fraction()`.

The raw scan data below (SCANS[modality]["data"]) was extracted once from the
underlying test_metrics.npz / test_metrics.json files on disk (see
scripts/_extract_rate_distortion_data.py for the exact paths/formula used)
and is kept here as plain literals so it's trivial to hand-edit or append to
when new trainings land, without needing to re-read multi-GB result
directories every time this figure is regenerated.

Usage:
    python scripts/plot_rate_distortion.py [--outdir plots/rate_distortion]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

# =============================================================================
# Scan data: {(K, d, Q): iqr_over_median} per modality. IQR/median of the
# per-jet reco_pt / truth_pt ratio ("pt_ratio" in test_metrics.npz).
# D = number of raw input features per token (does not vary within a
# modality); d (codebook_dim) is scanned but does not affect the bitrate --
# it's encoded as marker shape instead (see MARKER_BY_D below).
# =============================================================================
SCANS = {
    "JetSet": dict(
        D=20,
        color="#1f77b4",
        data={
            (1024, 4, 3): 0.050278,
            (1024, 4, 4): 0.035964,
            (1024, 8, 3): 0.061766,
            (1024, 8, 4): 0.032194,
            (1024, 16, 3): 0.066305,
            (1024, 16, 4): 0.044585,
            (8192, 4, 3): 0.035335,
            (8192, 4, 4): 0.023328,
            (8192, 8, 3): 0.036286,
            (8192, 8, 4): 0.014427,
            (8192, 16, 3): 0.046705,
            (8192, 16, 4): 0.028122,
            (16384, 4, 3): 0.035826,
            (16384, 4, 4): 0.026337,
            (16384, 8, 3): 0.032675,
            (16384, 8, 4): 0.010830,
            (16384, 16, 3): 0.042768,
            (16384, 16, 4): 0.024618,
            (32768, 4, 3): 0.034092,
            (32768, 4, 4): 0.044031,
            (32768, 8, 3): 0.026885,
            (32768, 8, 4): 0.008132,
            (32768, 16, 3): 0.040569,
            (32768, 16, 4): 0.022004,
        },
    ),
    "COCOA tracks": dict(
        D=5,
        color="#ff7f0e",
        data={
            (64, 2, 2): 0.032501,
            (64, 2, 3): 0.011921,
            (64, 2, 4): 0.005302,
            (64, 4, 2): 0.063162,
            (64, 4, 3): 0.031508,
            (64, 4, 4): 0.015031,
            (64, 8, 2): 0.055001,
            (64, 8, 3): 0.036810,
            (64, 8, 4): 0.024489,
            (256, 2, 2): 0.011695,
            (256, 2, 3): 0.007139,
            (256, 2, 4): 0.002843,
            (256, 4, 2): 0.033002,
            (256, 4, 3): 0.011047,
            (256, 4, 4): 0.005690,
            (256, 8, 2): 0.036976,
            (256, 8, 3): 0.014932,
            (256, 8, 4): 0.010749,
            (1024, 2, 2): 0.005503,
            (1024, 2, 3): 0.005149,
            (1024, 2, 4): 0.004507,
            (1024, 4, 2): 0.015297,
            (1024, 4, 3): 0.006169,
            (1024, 4, 4): 0.002090,
            (1024, 8, 2): 0.026142,
            (1024, 8, 3): 0.010432,
            (1024, 8, 4): 0.004487,
            (4096, 2, 2): 0.007873,
            (4096, 2, 3): 0.007097,
            (4096, 2, 4): 0.004464,
            (4096, 4, 2): 0.009823,
            (4096, 4, 3): 0.003281,
            (4096, 4, 4): 0.000890,
            (4096, 8, 2): 0.021397,
            (4096, 8, 3): 0.006800,
            (4096, 8, 4): 0.003090,
        },
    ),
    "COCOA topoclusters": dict(
        D=8,
        color="#2ca02c",
        data={
            (64, 2, 4): 0.047071,
            (64, 4, 4): 0.045822,
            (64, 8, 4): 0.055237,
            (64, 12, 4): 0.079050,
            (256, 2, 4): 0.046921,
            (256, 4, 4): 0.031561,
            (256, 8, 4): 0.030512,
            (256, 12, 4): 0.041431,
            (1024, 2, 4): 0.045888,
            (1024, 4, 4): 0.021977,
            (1024, 8, 4): 0.016761,
            (1024, 12, 4): 0.026675,
            (4096, 2, 4): 0.046603,
            (4096, 4, 4): 0.023907,
            (4096, 8, 4): 0.010507,
            (4096, 12, 4): 0.018291,
        },
    ),
}

# Codebook dimension d -> (marker shape, legend label), one SHARED mapping
# across all modalities, keyed by the ABSOLUTE d value (not "smallest in this
# modality"/"largest in this modality"). d=12 (topoclusters) and d=16
# (JetSet) never co-occur within one modality, so they share a single "large
# d" triangle and a combined legend label -- everything else gets its own
# shape so the same shape always means the same d across every color.
MARKER_BY_D = {
    2: ("v", "d = 2"),
    4: ("o", "d = 4"),
    8: ("s", "d = 8"),
    12: ("^", "d = 12/16"),
    16: ("^", "d = 12/16"),
}

# Optimal (K, d, Q) config to star per modality.
OPTIMAL_CONFIGS = {
    "JetSet": (2**15, 8, 4),
    "COCOA tracks": (2**12, 4, 4),
    "COCOA topoclusters": (2**12, 8, 4),
}

# JetSet classifier ladder: light-jet rejection at eps_b = 0.70, mean +/- std.
LADDER = [
    dict(bits=30, K=2**10, Q=3, mean=584.8, std=3.7),
    dict(bits=40, K=2**10, Q=4, mean=859.6, std=6.7),
    dict(bits=52, K=2**13, Q=4, mean=1104.5, std=10.3),
    dict(bits=60, K=2**15, Q=4, mean=1166.3, std=11.1),
]
LADDER_HEADLINE_BITS = 60  # which ladder point gets a star

CONTINUOUS_BASELINE = dict(mean=1371.6, std=12.7)

RESOLUTION_GOAL = 0.01

# ─────────────────────────────────────────────────────────────────────────
# Middle panel: downstream jet-pT REGRESSION ladder (COCOA topoclusters).
# mean/sem = mean/standard-error-of-the-mean across 3 independently-trained
# random seeds (42, 1, 2) of IQR/median of pT_pred/pT_truth on the held-out
# test set -- extracted once from results/topo_token_regression/{eval,
# topo_regression_results.csv} (see scripts/make_topo_regression_plot.py's
# aggregate_by_seed(), sample std / sqrt(n)) and kept here as plain
# literals, same convention as SCANS/LADDER above. All 4 points use cd=8,
# Q=4, varying K (single-seed point estimates turned out to be
# seed-sensitive; see repo memory notes).
# ─────────────────────────────────────────────────────────────────────────
TOPO_REGRESSION_LADDER = [
    dict(K=64, d=8, Q=4, mean=0.063381, sem=0.000400),
    dict(K=256, d=8, Q=4, mean=0.045308, sem=0.006313),
    dict(K=1024, d=8, Q=4, mean=0.021277, sem=0.001325),
    dict(K=4096, d=8, Q=4, mean=0.024479, sem=0.004247),
]
TOPO_REGRESSION_OPTIMAL = (4096, 8, 4)  # matches OPTIMAL_CONFIGS["COCOA topoclusters"]

TOPO_REGRESSION_CONTINUOUS_BASELINE = dict(mean=0.016292, sem=0.003967)

# Q=3 companion ladder (same cd=8, same 4 K values), shown as a dotted line
# alongside the Q=4 solid line in the same color -- both trained/evaluated
# as part of the same 12-config (K x Q) topo_token_regression scan.
TOPO_REGRESSION_LADDER_Q3 = [
    dict(K=64, d=8, Q=3, mean=0.099477, sem=0.001360),
    dict(K=256, d=8, Q=3, mean=0.058007, sem=0.000461),
    dict(K=1024, d=8, Q=3, mean=0.036389, sem=0.000837),
    dict(K=4096, d=8, Q=3, mean=0.031720, sem=0.002348),
]

# Combined Q=3+Q=4 point cloud for the middle panel's Pareto-frontier style
# (all scanned points shown faint, only the non-dominated ones connected).
TOPO_REGRESSION_ALL = TOPO_REGRESSION_LADDER + TOPO_REGRESSION_LADDER_Q3

# ─────────────────────────────────────────────────────────────────────────
# 3rd panel: downstream jet-pT REGRESSION ladder (COCOA tracks). mean/sem =
# mean/SEM across 3 seeds (42, 1, 2), same convention as the topo ladder
# above -- extracted from results/track_token_regression/{eval,
# track_regression_results.csv} (see scripts/make_topo_regression_plot.py's
# aggregate_by_seed()). Q FIXED at 4 (matches the topo ladder's Q=4-only
# design above), varying K only, at fixed cd=4 (matches
# OPTIMAL_CONFIGS["COCOA tracks"]) -- the full 12-config (K x Q) grid is
# still trained/evaluated (see results/track_token_regression/
# track_regression_results.csv + the standalone rate_distortion_track_
# regression.{pdf,png}), just not all shown on this shared panel, to avoid
# multiple (K, Q) pairs colliding at the same compression factor here.
# ─────────────────────────────────────────────────────────────────────────
TRACK_REGRESSION_LADDER = [
    dict(K=64, d=4, Q=4, mean=0.026100, sem=0.004853),
    dict(K=256, d=4, Q=4, mean=0.022524, sem=0.002173),
    dict(K=1024, d=4, Q=4, mean=0.018286, sem=0.003555),
    dict(K=4096, d=4, Q=4, mean=0.015645, sem=0.001964),
]
TRACK_REGRESSION_OPTIMAL = (4096, 4, 4)  # matches OPTIMAL_CONFIGS["COCOA tracks"]

TRACK_REGRESSION_CONTINUOUS_BASELINE = dict(mean=0.020364, sem=0.006512)

# Q=3 companion ladder (same cd=4, same 4 K values), shown as a dotted line
# alongside the Q=4 solid line in the same color -- both trained/evaluated
# as part of the same 12-config (K x Q) track_token_regression scan.
TRACK_REGRESSION_LADDER_Q3 = [
    dict(K=64, d=4, Q=3, mean=0.034294, sem=0.001595),
    dict(K=256, d=4, Q=3, mean=0.034679, sem=0.003587),
    dict(K=1024, d=4, Q=3, mean=0.018970, sem=0.001060),
    dict(K=4096, d=4, Q=3, mean=0.019164, sem=0.002006),
]

# Combined Q=3+Q=4 point cloud for the middle panel's Pareto-frontier style
# (all scanned points shown faint, only the non-dominated ones connected).
TRACK_REGRESSION_ALL = TRACK_REGRESSION_LADDER + TRACK_REGRESSION_LADDER_Q3

# ─────────────────────────────────────────────────────────────────────────
# Bottom panel: positions-in vs positions-out jet-pT IQR/median ("own count"),
# i.e. jets rebuilt from the model's OWN reco multiplicity, on 500k held-out
# test events; Q=4, codebook_dim=8 (does NOT affect bitrate), K scanned.
# positions-in folds all 7.345 features (incl. positions) into the content
# token, so its bitrate is Q*log2 K. positions-out tokenizes only the 5.345
# content features but transmits the 3 position coords as a separate stream
# (3 coords * 10 bits = 30 extra bits), so its TOTAL bitrate is Q*log2 K + 30.
# Both variants reconstruct the same 7.345-feature object, so both use the same
# fp32 denominator 32*7.345 in the information-fraction x-axis:
#     positions-in : Q*log2 K        / (32 * 7.345)
#     positions-out: (Q*log2 K + 30) / (32 * 7.345)
# err = statistical error on the 500k-event test set.
# ─────────────────────────────────────────────────────────────────────────
PARTICLE_FLOW_D = 7.345  # fp32 feature count of the reconstructed object
POSITIONS_OUT_EXTRA_BITS = 30  # 3 position coords * 10 bits, sent separately
PARTICLE_FLOW_COLOR = "#9467bd"  # purple -- distinct from the 3 single modalities
PARTICLE_FLOW_CONTINUOUS_BASELINE = dict(mean=0.08461, err=0.00015)
POSITIONS_IN_LADDER = [
    dict(K=256, Q=4, mean=0.09423, err=0.00017),
    dict(K=1024, Q=4, mean=0.09201, err=0.00017),
    dict(K=4096, Q=4, mean=0.08983, err=0.00017),
]
POSITIONS_OUT_LADDER = [
    dict(K=256, Q=4, mean=0.08777, err=0.00015),
    dict(K=1024, Q=4, mean=0.08605, err=0.00015),
    dict(K=4096, Q=4, mean=0.08632, err=0.00017),
]


def compression_factor(D: float, Q: int, K: int) -> float:
    """32 * D / (Q * log2(K)) -- fp32 baseline bits / RVQ token bits.

    Used internally to decide Pareto-frontier dominance ("higher compression
    + lower distortion is better"); the x-axis actually DISPLAYED is its
    reciprocal, information_fraction() below.
    """
    return 32.0 * D / (Q * np.log2(K))


def information_fraction(D: float, Q: int, K: int) -> float:
    """Q * log2(K) / (32 * D) -- fraction of the fp32/continuous-feature
    bit budget actually used by the RVQ token, i.e. 1 / compression_factor().
    This is the x-axis quantity actually plotted (continuous features sit at
    exactly 1.0; more compression -> smaller fraction).
    """
    return 1.0 / compression_factor(D, Q, K)


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated (x, y) points where higher x and lower y are "better".

    A point is dominated if some other point has >= its x AND <= its y.
    Multiple points can share the same x (d varies at fixed K, Q); at equal
    x only the lowest y matters. Sort by descending x, then ascending y, and
    keep a running minimum of y.
    """
    ordered = sorted(points, key=lambda p: (-p[0], p[1]))
    frontier = []
    running_min = np.inf
    for x, y in ordered:
        if y < running_min:
            frontier.append((x, y))
            running_min = y
    # Return in ascending-x order for easy line plotting.
    return sorted(frontier, key=lambda p: p[0])


def pareto_frontier_points(points: list[dict]) -> list[dict]:
    """Like pareto_frontier(), but operates on/returns full point dicts
    (each must have 'x' and 'y' keys) instead of bare (x, y) tuples, so
    other fields (e.g. 'sem', 'K', 'Q') survive for error bars/labels."""
    ordered = sorted(points, key=lambda p: (-p["x"], p["y"]))
    frontier = []
    running_min = np.inf
    for p in ordered:
        if p["y"] < running_min:
            frontier.append(p)
            running_min = p["y"]
    return sorted(frontier, key=lambda p: p["x"])


def build_points(modality: str) -> list[tuple[int, int, int, float, float]]:
    """List of (K, d, Q, compression_factor, iqr_over_median) for a modality."""
    spec = SCANS[modality]
    D = spec["D"]
    return [
        (K, d, Q, compression_factor(D, Q, K), y)
        for (K, d, Q), y in spec["data"].items()
    ]


def draw_pareto_panel(
    ax,
    *,
    legend_size=10.5,
    goal_size=9,
    label_size=None,
    tick_size=None,
    star_size=16,
    scatter_size=26,
    line_width=1.8,
    xlim=(0.015, 0.33),
    ylim=(3e-4, 0.20),
):
    """Draw the cross-modality jet-pT rate-distortion Pareto scan on `ax`.

    Faint per-d scan cloud + per-modality Pareto frontier + optimal-config
    stars + the 1% resolution goal line, plus both legends (modality colors
    and shared codebook-dim marker shapes). Shared by the full stacked figure
    (as its bottom panel, all size defaults) and the standalone teaser figure
    (larger fonts/markers + a slightly tighter zoom, passed explicitly).
    """
    for modality, spec in SCANS.items():
        color = spec["color"]
        pts = build_points(modality)
        xs = np.array([p[3] for p in pts])  # compression_factor, for Pareto dominance only
        ys = np.array([p[4] for p in pts])

        # Scatter grouped by d so each point gets its shared, absolute-d
        # marker shape (see MARKER_BY_D) -- color still encodes modality.
        # Display x = information_fraction = 1/compression_factor.
        for d_val in sorted(set(p[1] for p in pts)):
            marker, _ = MARKER_BY_D[d_val]
            sel_x = [1.0 / p[3] for p in pts if p[1] == d_val]
            sel_y = [p[4] for p in pts if p[1] == d_val]
            ax.scatter(
                sel_x,
                sel_y,
                marker=marker,
                s=scatter_size,
                facecolor=color,
                edgecolor="none",
                alpha=0.35,
                zorder=2,
            )

        fx, fy = zip(*pareto_frontier(list(zip(xs, ys))))
        fx_disp = [1.0 / x for x in fx]
        ax.plot(
            fx_disp,
            fy,
            "-",
            color=color,
            linewidth=line_width,
            zorder=3,
            label=modality,
        )

        if modality in OPTIMAL_CONFIGS:
            K, d, Q = OPTIMAL_CONFIGS[modality]
            ox = information_fraction(spec["D"], Q, K)
            oy = spec["data"][(K, d, Q)]
            ax.plot(
                ox,
                oy,
                marker="*",
                markersize=star_size,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.7,
                linestyle="none",
                zorder=5,
            )

    ax.set_yscale("log")
    # Pad limits so the two legends (upper-right + lower-left, the empty
    # corners of a rate-distortion plot) clear the data.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(RESOLUTION_GOAL, color="0.3", linestyle=":", linewidth=1.3, zorder=1)
    ax.text(
        0.62,
        RESOLUTION_GOAL,
        "1% resolution goal",
        color="0.3",
        fontsize=goal_size,
        va="bottom",
        ha="left",
        transform=ax.get_yaxis_transform(),
    )

    # Legend 1: modality colors (as lines already labeled above) + a generic
    # "optimal config" star entry. Upper-right = empty corner (frontiers dive
    # low there, so no data to collide with).
    legend_marker_scale = legend_size / 10.5
    star_handle = plt.Line2D(
        [],
        [],
        marker="*",
        markersize=13 * legend_marker_scale,
        markerfacecolor="0.5",
        markeredgecolor="black",
        markeredgewidth=0.7,
        linestyle="none",
        label="optimal config",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(star_handle)
    labels.append("optimal config")
    legend_color = ax.legend(
        handles, labels, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none",
        fontsize=legend_size, ncol=1,
    )
    ax.add_artist(legend_color)

    # Legend 2: shared marker-shape -> absolute d mapping (neutral gray
    # markers, since shape here is deliberately decoupled from color/modality).
    # Lower-left = empty corner (no config reaches low distortion at low rate).
    shape_handles = []
    seen_labels = set()
    for d_val in sorted(MARKER_BY_D):
        marker, label = MARKER_BY_D[d_val]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        shape_handles.append(
            plt.Line2D(
                [],
                [],
                marker=marker,
                markersize=7 * legend_marker_scale,
                markerfacecolor="0.4",
                markeredgecolor="none",
                linestyle="none",
                label=label,
            )
        )
    ax.legend(
        handles=shape_handles, loc="lower left", frameon=True, framealpha=0.9, edgecolor="none",
        fontsize=legend_size, title="codebook dim.", title_fontsize=legend_size,
    )

    label_kw = {"fontsize": label_size} if label_size is not None else {}
    ax.set_ylabel(r"Jet $p_T$ IQR / median", **label_kw)
    ax.set_xlabel(r"Information fraction  ($Q\,\log_2 K \,/\, 32\,D$)", **label_kw)
    tick_kw = {"labelsize": tick_size} if tick_size is not None else {}
    ax.tick_params(
        axis="both", which="both", direction="in",
        top=True, right=True, bottom=True, left=True, **tick_kw,
    )


INFO_FRAC_XLABEL = r"Information fraction  ($Q\,\log_2 K \,/\, 32\,D$)"


def _finish_panel(ax, *, label_size, tick_size, xlabel):
    """Shared per-panel styling: x-label (optional), inward ticks, tick size."""
    if xlabel:
        ax.set_xlabel(INFO_FRAC_XLABEL, fontsize=label_size)
    ax.tick_params(
        axis="both", which="both", direction="in",
        top=True, right=True, bottom=True, left=True, labelsize=tick_size,
    )


def draw_classifier_panel(
    ax, *, legend_size=15, label_size=20, tick_size=16, star_size=22,
    marker_size=7, line_width=2.0, xlabel=True,
):
    """JetSet classifier light-jet-rejection ladder + continuous baseline."""
    jetset_color = SCANS["JetSet"]["color"]
    jetset_D = SCANS["JetSet"]["D"]

    ladder_x = np.array([information_fraction(jetset_D, pt["Q"], pt["K"]) for pt in LADDER])
    ladder_y = np.array([pt["mean"] for pt in LADDER])
    ladder_yerr = np.array([pt["std"] for pt in LADDER])

    ax.errorbar(
        ladder_x, ladder_y, yerr=ladder_yerr, fmt="o-", color=jetset_color,
        markersize=marker_size, linewidth=line_width, capsize=4, zorder=3,
        label="JetSet classifier ladder",
    )

    headline = next(pt for pt in LADDER if pt["bits"] == LADDER_HEADLINE_BITS)
    hx = information_fraction(jetset_D, headline["Q"], headline["K"])
    ax.plot(
        hx, headline["mean"], marker="*", markersize=star_size,
        markerfacecolor=jetset_color, markeredgecolor="black", markeredgewidth=0.7,
        linestyle="none", zorder=5,
    )

    base_mean = CONTINUOUS_BASELINE["mean"]
    base_std = CONTINUOUS_BASELINE["std"]
    ax.axhline(base_mean, color="0.2", linestyle="--", linewidth=1.3, zorder=1, label="continuous features")
    ax.axhspan(base_mean - base_std, base_mean + base_std, color="0.2", alpha=0.15, zorder=0)

    ax.set_ylim(450, 1550)
    ax.set_ylabel(r"Light-jet rej. @ $\varepsilon_b=0.70$", fontsize=label_size)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=legend_size)
    _finish_panel(ax, label_size=label_size, tick_size=tick_size, xlabel=xlabel)


def draw_regression_panel(
    ax, *, legend_size=15, label_size=20, tick_size=16, star_size=22,
    marker_size=7, line_width=2.0, scatter_size=44, xlim=None, ylim=None, xlabel=True,
):
    """COCOA topocluster + track downstream jet-pT regression Pareto scans
    (IQR/median of pT_pred/pT_truth). Each curve's star marks its MOST
    PERFORMANT (lowest IQR/median) config; continuous baselines dotted.
    """
    topo_color = SCANS["COCOA topoclusters"]["color"]
    topo_D = SCANS["COCOA topoclusters"]["D"]
    track_color = SCANS["COCOA tracks"]["color"]
    track_D = SCANS["COCOA tracks"]["D"]

    for color, D, all_points, base in (
        (topo_color, topo_D, TOPO_REGRESSION_ALL, TOPO_REGRESSION_CONTINUOUS_BASELINE),
        (track_color, track_D, TRACK_REGRESSION_ALL, TRACK_REGRESSION_CONTINUOUS_BASELINE),
    ):
        labeled_points = [
            dict(pt, x=compression_factor(D, pt["Q"], pt["K"]), y=pt["mean"]) for pt in all_points
        ]
        for p in labeled_points:
            p["x_disp"] = 1.0 / p["x"]  # information_fraction, for display only

        ax.scatter(
            [p["x_disp"] for p in labeled_points], [p["y"] for p in labeled_points],
            marker="o", s=scatter_size, facecolor=color, edgecolor="none", alpha=0.35, zorder=2,
        )

        frontier = pareto_frontier_points(labeled_points)
        modality_label = "topocluster" if color == topo_color else "track"
        ax.errorbar(
            [p["x_disp"] for p in frontier], [p["y"] for p in frontier],
            yerr=[p["sem"] for p in frontier], fmt="o-", color=color,
            markersize=marker_size, linewidth=line_width, capsize=4, capthick=1.3,
            elinewidth=1.3, ecolor=color, zorder=3,
            label=f"{modality_label} jet-$p_T$ regressor",
        )

        # Star = most performant (lowest IQR/median) config on the frontier.
        best = min(frontier, key=lambda p: p["y"])
        ax.plot(
            best["x_disp"], best["y"], marker="*", markersize=star_size,
            markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.7,
            linestyle="none", zorder=5,
        )

        ax.axhline(base["mean"], color=color, linestyle=":", linewidth=1.6, zorder=1,
                   label=f"{modality_label} continuous")
        ax.axhspan(base["mean"] - base["sem"], base["mean"] + base["sem"], color=color, alpha=0.15, zorder=0)

    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    ax.set_ylabel("Regressed $p_T$ IQR / median", fontsize=label_size)
    # Upper-right = empty corner (frontiers descend toward lower-right).
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=legend_size)
    _finish_panel(ax, label_size=label_size, tick_size=tick_size, xlabel=xlabel)


def draw_particle_flow_panel(
    ax, *, legend_size=15, label_size=20, tick_size=16, star_size=22,
    marker_size=7, line_width=2.0, ylim=None, xlabel=True,
):
    """Particle-flow jet-pT IQR/median ("own count"): positions-in (solid) vs
    positions-out (dotted), Q=4, K scanned, with a continuous-feature baseline.

    Both series share one color; the line style distinguishes the two tokenizer
    variants. x = information fraction, with a SHARED fp32 denominator
    (32 * 7.345) since both reconstruct the same object: positions-in spends
    Q*log2 K bits, positions-out spends Q*log2 K + 30 (its 3 position coords go
    out as a separate 30-bit stream). Linear y: the resolution varies by only
    ~10% across the scan.
    """
    color = PARTICLE_FLOW_COLOR
    D = PARTICLE_FLOW_D

    for ladder, style, label, extra_bits in (
        (POSITIONS_IN_LADDER, "-", "positions-in", 0),
        (POSITIONS_OUT_LADDER, ":", "positions-out", POSITIONS_OUT_EXTRA_BITS),
    ):
        xs = [(pt["Q"] * np.log2(pt["K"]) + extra_bits) / (32.0 * D) for pt in ladder]
        ys = [pt["mean"] for pt in ladder]
        yerr = [pt["err"] for pt in ladder]
        ax.errorbar(
            xs, ys, yerr=yerr, fmt="o", linestyle=style, color=color,
            markersize=marker_size, linewidth=line_width, capsize=4, capthick=1.3,
            elinewidth=1.3, ecolor=color, zorder=3, label=label,
        )

    base = PARTICLE_FLOW_CONTINUOUS_BASELINE
    ax.axhline(base["mean"], color="0.2", linestyle="--", linewidth=1.3, zorder=1,
               label="continuous features")
    ax.axhspan(base["mean"] - base["err"], base["mean"] + base["err"], color="0.2", alpha=0.15, zorder=0)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(r"Jet $p_T$ IQR / median", fontsize=label_size)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=legend_size)
    _finish_panel(ax, label_size=label_size, tick_size=tick_size, xlabel=False)
    if xlabel:
        ax.set_xlabel("Information fraction", fontsize=label_size)


def save_figure(fig, outdir, stem):
    """Save a figure to both PDF and PNG at 300 dpi under `outdir/stem`."""
    for ext in ("pdf", "png"):
        out_path = outdir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "plots" / "rate_distortion",
    )
    parser.add_argument("--stem", default="rate_distortion")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plt.style.use(hep.style.CMS)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10.5,
            "legend.handlelength": 2.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    # ==================================================================
    # Three separate publication figures, all rendered with the larger,
    # standalone hep.style.CMS-matched fonts (via the draw_* helpers' size
    # args) rather than the old cramped 4-panel stack:
    #   1. rate_distortion_pareto  -- cross-modality jet-pT Pareto scan
    #   2. rate_distortion_jetset  -- light-jet rejection + particle flow
    #   3. rate_distortion_cocoa   -- COCOA topo+track regressed jet pT
    # ==================================================================
    big = dict(legend_size=15, label_size=20, tick_size=16, star_size=22)

    # 1. Standalone Pareto teaser (cross-modality "money plot").
    fig_p, ax_p = plt.subplots(figsize=(6.4, 6.4))
    draw_pareto_panel(
        ax_p, goal_size=13, scatter_size=44, line_width=2.4,
        xlim=(0.025, 0.315), ylim=(5e-4, 0.16), **big,
    )
    ax_p.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    fig_p.tight_layout()
    save_figure(fig_p, args.outdir, f"{args.stem}_pareto")

    # 2. JetSet figure: light-jet rejection (single panel).
    fig_j, ax_cls = plt.subplots(figsize=(7.0, 5.6))
    draw_classifier_panel(ax_cls, xlabel=True, **big)
    fig_j.tight_layout()
    save_figure(fig_j, args.outdir, f"{args.stem}_jetset")

    # 3. COCOA figure: two stacked panels sharing the info-fraction x-axis --
    #    (top) topocluster+track regressed jet pT, (bottom) positions-in vs
    #    positions-out particle-flow jet pT (linear y).
    fig_c, (ax_reg, ax_pf) = plt.subplots(
        2, 1, figsize=(7.2, 9.4), sharex=True, gridspec_kw={"hspace": 0.12},
    )
    draw_regression_panel(
        ax_reg, xlabel=False, scatter_size=44,
        xlim=(0.05, 0.345), ylim=(8e-3, 0.14), **big,
    )
    draw_particle_flow_panel(ax_pf, xlabel=True, ylim=(0.082, 0.097), **big)
    fig_c.align_ylabels([ax_reg, ax_pf])
    save_figure(fig_c, args.outdir, f"{args.stem}_cocoa")




if __name__ == "__main__":
    main()
