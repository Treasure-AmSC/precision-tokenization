#!/usr/bin/env python
"""Design-rules figure (four panels), d-crossing appendix plots, and Fig. 1 (rate-distortion Pareto).

All values are read from the test_nofastpath/ test_metrics.npz files of repo A at run time (fused MHA
fast path off, precision=highest), so new runs are picked up automatically; missing runs are simply
absent from the plot. Metric: IQR/median of per-jet pt_ratio. For COCOA tracks the 50-epoch Q = 3
retrains (tracks_e50/) replace the original Q = 3 runs.

  (a) resolution vs Q at the reference (K, d); COCOA tracks up to Q = 4
  (b) resolution vs log2 K at the reference (d, Q)
  (c) d of the Pareto-optimal configuration vs bits per element (Q log2 K)
  (d) positions in (solid) vs positions out (dashed, hollow) vs total bits per element: Q log2 K, plus
      for positions out the fixed global position binning (pos_bits: 10 bits per coordinate,
      20 for JetSet (deta, dphi), 30 for COCOA (eta, cos phi, sin phi)); best d at each K, Q = 4

Where several seeds exist for one configuration the mean is plotted with a min-max error bar.
JetSet d = 2 is excluded from the main figures and shown (marked) in the appendix d-crossing plot.
Fig. 1 is also drawn with normalized MAE (per-feature MAE / central-68% half-width of the feature;
mean over features and worst feature) on the y-axis.

usage: python scripts/plot_design_rules.py [--outdir paper/figures]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"

MODALITIES = {
    "JetSet": dict(
        color="#1f77b4", D=20, pos_idx=[1, 2], pos_bits=20, ref=(32768, 8, 4), selected=(32768, 8, 4), slug="jetset",
        posout_grid=([8192, 16384, 32768], [4, 8, 16]),
        dirs=[RES / "vqvae_scan/vqvae_scan"],
        rx=r"^transformer_vqvae_enc_large_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.0003_cw1\.0_nj10000000"
           r"(?:_(posout)|_seed(\d+))?$",
    ),
    "COCOA tracks": dict(
        color="#ff7f0e", D=5, pos_idx=[1, 2], pos_bits=30, ref=(4096, 4, 4), selected=(4096, 8, 3), slug="tracks",
        q_max_a=4,  # Q = 6, 8 runs early-stopped at worse loss than Q = 4
        posout_grid=([256, 1024, 4096], [4, 8]),  # positions-out d = 2 reconstructs z0 as a constant
        dirs=[RES / "cocoa_scan/tracks", RES / "cocoa_scan/tracks_posout", RES / "cocoa_scan/tracks_e50"],
        rx=r"^cocoa_tracks(?:_(posout))?_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.001(?:_seed(\d+))?(_e50)?$",
        e100=dict(ref=(4096, 8, 4), q_max_a=8, posout_grid=([256, 1024, 4096], [8]),
                  dirs=[RES / "cocoa_scan/tracks_e100"],
                  rx=r"^cocoa_tracks(?:_(posout))?_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.001(?:_seed(\d+))?(_e100)$"),
    ),
    "COCOA topoclusters": dict(
        color="#2ca02c", D=8, pos_idx=[0, 1], pos_bits=30, ref=(4096, 8, 4), selected=(4096, 8, 6), slug="topos",
        posout_grid=([256, 1024, 4096], [4, 8, 12]),
        dirs=[RES / "cocoa_scan/topos", RES / "cocoa_scan/topos_posout", RES / "cocoa_scan/topos_seed"],
        rx=r"^cocoa_topos(?:_(posout))?_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.001(?:_seed(\d+))?(_e50)?$",
        e100=dict(ref=(4096, 8, 4), posout_grid=([256, 1024, 4096], [8]),
                  dirs=[RES / "cocoa_scan/topos_e100"],
                  rx=r"^cocoa_topos(?:_(posout))?_cb(\d+)_cd(\d+)_nq(\d+)_lr0\.001(?:_seed(\d+))?(_e100)$"),
    ),
}
Q_VALUES = [2, 3, 4, 6, 8]

# Calorimeter cells (Fig. 1 resolution only): jet energy IQR/median from each run's test_metrics.json,
# same 100,096 test events; Large model, MSE alpha=0.4, all trained on 2M events.
CELLS = dict(
    color="#9467bd", D=64, slug="cells",
    dir=Path("/fs/ddn/sdf/group/atlas/d/jkrupa/cocoa_results/cells_scan_2M_a0.4_frontier/transformer"),
)
CELLS_Q_MAX = 16  # cell runs with Q > 16 are not part of the paper


def load_cells():
    runs = {}
    for run in sorted(CELLS["dir"].glob("cb*_cd*_nq*_a0.4_large")):
        if not (run / "test_metrics.json").exists():
            print(f"  cells: no test_metrics.json yet: {run.name}")
            continue
        m = json.loads((run / "test_metrics.json").read_text())
        if int(m["nq"]) > CELLS_Q_MAX:
            continue
        runs[(int(m["cb"]), int(m["cd"]), int(m["nq"]), "in")] = [float(m["energy_resolution_iqr"])]
    return runs


# Per-feature fixed uniform binning (scripts/binning_baseline.py, binning_baseline_cells.py): information fraction log2(K)/32
BINNING_K_MIN = 16


def load_binning(metric):
    """{modality name: [(information fraction, metric value)]} from results/binning_baseline."""
    out = {}
    for name, spec in MODALITIES.items():
        pts = []
        for run in sorted((RES / "binning_baseline" / spec["slug"]).glob("binning_uniform_K*")):
            K = int(run.name.rsplit("K", 1)[1])
            npz = run / "test_metrics/test_metrics.npz"
            if npz.exists() and K >= BINNING_K_MIN:
                pts.append((np.log2(K) / 32, metric(npz)))
        if pts:
            out[name] = sorted(pts)
    return out


def load_cells_binning():
    pts = []
    for run in sorted((RES / "binning_baseline/cells").glob("binning_uniform_K*")):
        if (run / "test_metrics.json").exists() and int(run.name.rsplit("K", 1)[1]) >= BINNING_K_MIN:
            m = json.loads((run / "test_metrics.json").read_text())
            pts.append((m["information_fraction"], m["energy_resolution_iqr"]))
    return sorted(pts)


def load_cells_mae():
    """Pixel-energy normalized MAE (scripts/cells_pixel_mae.py): (VQ runs dict, binning [(x, y)])."""
    runs, binning = {}, []
    for f in sorted((RES / "cells_mae").glob("*/pixel_mae.json")):
        m = json.loads(f.read_text())
        if f.parent.name.startswith("binning_uniform_K"):
            if m["cb"] >= BINNING_K_MIN:
                binning.append((m["information_fraction"], m["normalized_mae"]))
        elif m["nq"] <= CELLS_Q_MAX:
            runs[(m["cb"], m["cd"], m["nq"], "in")] = [m["normalized_mae"]]
    return runs, sorted(binning)


def iqr_over_median(path):
    pr = np.load(path)["pt_ratio"]
    pr = pr[np.isfinite(pr) & (pr > 0)]
    q25, q50, q75 = np.percentile(pr, [25, 50, 75])
    return (q75 - q25) / q50


def normalized_mae(path):
    """Per-feature MAE / central-68% half-width of the feature, as an array over features."""
    d = np.load(path)
    o, r = d["cst_original"], d["cst_recon"]
    lo, hi = np.percentile(o, [16, 84], axis=0)
    return np.mean(np.abs(r - o), axis=0) / ((hi - lo) / 2)


def parse(name, spec):
    m = re.match(spec["rx"], name)
    if m is None:
        return None
    g = m.groups()
    if spec["slug"] == "jetset":
        K, d, Q, posout, seed = g
        e50 = None
    else:
        posout, K, d, Q, seed, e50 = g
    return int(K), int(d), int(Q), ("out" if posout else "in"), int(seed) if seed else 42, bool(e50)


def load(spec, metric=iqr_over_median, sides=("in", "out")):
    """{(K, d, Q, 'in'|'out'): [metric values over seeds]}"""
    runs, retrained = {}, {}
    for base in spec["dirs"]:
        for run in sorted(base.glob("*")):
            key = parse(run.name, spec)
            if key is None or key[3] not in sides:
                continue
            files = sorted(set(glob.glob(str(run / "test_nofastpath/**/test_metrics.npz"), recursive=True)))
            if len(files) > 1:
                raise RuntimeError(f"several test_metrics.npz under {run}: {files}")
            if not files:
                print(f"  no test_nofastpath result: {run.name}")
                continue
            (retrained if key[5] else runs).setdefault(key[:4], []).append(metric(files[0]))
    runs.update(retrained)
    return runs


def central(vals):
    v = np.asarray(vals)
    return v.mean(), (v.mean() - v.min(), v.max() - v.mean())


def plot_series(ax, xs, vals, color, hollow=False, ls="-", label=None, marker="o"):
    """Line through the per-configuration means; min-max error bars where several seeds exist."""
    if not xs:
        return
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    cv = [central(vals[i]) for i in order]
    y = np.array([c[0] for c in cv])
    err = np.array([c[1] for c in cv]).T
    ax.errorbar(xs, y, yerr=err if err.any() else None, fmt=marker, ls=ls, color=color, ms=6,
                mfc="white" if hollow else color, mec=color, lw=1.6, capsize=3, label=label, zorder=3)


def pareto(points):
    """points: [(bits, y, cfg)], lower y and fewer bits better. Best per bits, then running min."""
    best = {}
    for b, y, cfg in points:
        if b not in best or y < best[b][0]:
            best[b] = (y, cfg)
    out, run_min = [], np.inf
    for b in sorted(best):
        y, cfg = best[b]
        if y < run_min:
            out.append((b, y, cfg))
            run_min = y
    return out


def main_figure(data, outdir, data_d=None, ylabel=r"Jet $p_T$ IQR / median", what="Resolution",
                stem="design_rules", include_c=True, data_c=None, specs=None, d_legend_loc="lower left"):
    """data feeds (a), (b); data_c (default data) feeds (c); data_d (default data) feeds (d).
    Without (c) the panels are laid out in one row."""
    data_d = data if data_d is None else data_d
    data_c = data if data_c is None else data_c
    specs = MODALITIES if specs is None else specs
    if include_c:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.6))
        (ax_a, ax_b), (ax_c, ax_d) = axes
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
        (ax_a, ax_b, ax_d), ax_c = axes, None
    c_offsets = {"JetSet": 1.05, "COCOA tracks": 1.0, "COCOA topoclusters": 0.95}  # separate overlapping steps

    for name, spec in specs.items():
        runs, col = data[name], spec["color"]
        K0, d0, Q0 = spec["ref"]
        inn = {k[:3]: v for k, v in runs.items() if k[3] == "in" and not (spec["slug"] == "jetset" and k[1] == 2)}

        # (a) resolution vs Q
        qs = [Q for Q in Q_VALUES if (K0, d0, Q) in inn and Q <= spec.get("q_max_a", max(Q_VALUES))]
        plot_series(ax_a, qs, [inn[(K0, d0, Q)] for Q in qs], col, label=name)

        # (b) resolution vs log2 K
        ks = sorted(K for (K, d, Q) in inn if d == d0 and Q == Q0)
        plot_series(ax_b, [np.log2(K) for K in ks], [inn[(K, d0, Q0)] for K in ks], col,
                    label=f"{name} (d={d0}, Q={Q0})")

        # (c) d of the Pareto-optimal configuration vs bits
        inn_c = {k[:3]: v for k, v in data_c[name].items()
                 if k[3] == "in" and not (spec["slug"] == "jetset" and k[1] == 2)}
        front = pareto([(Q * np.log2(K), central(v)[0], (K, d, Q)) for (K, d, Q), v in inn_c.items()])
        if front and ax_c is not None:
            b = [p[0] for p in front]
            dstar = [p[2][1] * c_offsets[name] for p in front]
            ax_c.step(b, dstar, where="mid", color=col, lw=1.6)
            ax_c.plot(b, dstar, "o", color=col, ms=4)

        # (d) positions in vs out, Q=4, best d at each K over the positions-out d grid;
        # a point is drawn only once every d of the grid exists at that K
        inn_d = {k[:3]: v for k, v in data_d[name].items() if k[3] == "in"}
        out = {k[:3]: v for k, v in data_d[name].items() if k[3] == "out"}
        K_grid, d_grid = spec["posout_grid"]
        for table, hollow, ls, extra_bits in ((inn_d, False, "-", 0), (out, True, "--", spec["pos_bits"])):
            xs, vals = [], []
            for K in K_grid:
                if all((K, d, 4) in table for d in d_grid):
                    xs.append(4 * np.log2(K) + extra_bits)
                    vals.append(min((table[(K, d, 4)] for d in d_grid), key=lambda v: central(v)[0]))
            plot_series(ax_d, xs, vals, col, hollow=hollow, ls=ls)

    for ax in (ax_a, ax_b, ax_d):
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    ax_a.set_xlabel(r"Number of quantizers $Q$")
    ax_a.set_xticks(Q_VALUES)
    ax_a.set_title(f"(a) {what} vs $Q$", loc="left", fontsize=12)
    fig.legend(*ax_a.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=13, frameon=False)
    ax_b.set_xlabel(r"$\log_2 K$")
    ax_b.set_title(f"(b) {what} vs codebook size", loc="left", fontsize=12)
    if ax_c is not None:
        ax_c.set_yscale("log", base=2)
        ax_c.yaxis.set_major_locator(mticker.FixedLocator([2, 4, 8, 16]))
        ax_c.yaxis.set_major_formatter(mticker.FixedFormatter(["2", "4", "8", "16"]))
        ax_c.yaxis.set_minor_locator(mticker.NullLocator())
        ax_c.set_ylim(1.6, 20)
        ax_c.set_xlabel(r"Bits per element $Q\,\log_2 K$")
        ax_c.set_ylabel(r"Pareto-optimal $d^\ast$")
        ax_c.set_title(r"(c) Optimal codebook dimension for jet resolution", loc="left", fontsize=12)
    ax_d.set_xlabel(r"Total bits per element (RVQ + positions)")
    ax_d.set_title("(d) Positions in vs out ($Q=4$)", loc="left", fontsize=12)
    ax_d.legend(handles=[plt.Line2D([], [], color="0.3", marker="o", ls="-", label="positions in"),
                         plt.Line2D([], [], color="0.3", marker="o", mfc="white", ls="--", label="positions out")],
                fontsize=13, loc=d_legend_loc, frameon=False)
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
    fig.tight_layout()
    save(fig, outdir, stem)


def d_crossing(data, outdir):
    for name, spec in MODALITIES.items():
        inn = {k[:3]: v for k, v in data[name].items() if k[3] == "in"}
        Ks = sorted({K for K, _, _ in inn})
        Qs = sorted({Q for _, _, Q in inn})
        cmap = plt.get_cmap("viridis")
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        for i, K in enumerate(Ks):
            for Q, ls in zip(Qs, ["-", "--", ":", "-.", (0, (5, 1)), (0, (1, 3))]):
                ds = sorted(d for (k, d, q) in inn if k == K and q == Q)
                if not ds:
                    continue
                c = cmap(i / max(len(Ks) - 1, 1))
                y = [central(inn[(K, d, Q)])[0] for d in ds]
                ax.plot(ds, y, ls=ls, color=c, marker="o", ms=5, label=f"K={K}, Q={Q}")
                if spec["slug"] == "jetset" and 2 in ds:
                    ax.plot(2, y[0], "o", ms=11, mfc="none", mec="red", mew=1.4, zorder=4)
        if spec["slug"] == "jetset" and any(d == 2 for _, d, _ in inn):
            ax.plot([], [], "o", ms=9, mfc="none", mec="red", mew=1.4, label="d = 2 (appendix only)")
        ax.set_xscale("log", base=2)
        ds_all = sorted({d for _, d, _ in inn})
        ax.xaxis.set_major_locator(mticker.FixedLocator(ds_all))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(d) for d in ds_all]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yscale("log")
        ax.set_xlabel("Codebook dimension $d$")
        ax.set_ylabel(r"Jet $p_T$ IQR / median")
        ax.set_title(name, loc="left", fontsize=13)
        ax.legend(fontsize=7.5, ncol=2, frameon=False)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.tight_layout()
        save(fig, outdir, f"design_rules_d_crossing_{spec['slug']}")


def fig1(data, outdir, ylabel=r"Jet $p_T$ IQR / median", stem="rate_distortion_pareto", goal=0.01, specs=None,
         binning=None):
    """Rate-distortion Pareto (style of heptokens-cocoa scripts/plot_rate_distortion.py) from repo A.
    binning: optional {modality name: [(x, y)]} drawn as dashed per-feature binning curves."""
    marker_by_d = {2: "v", 4: "o", 8: "s", 12: "^", 16: "^", 24: "D", 32: "D"}
    ds_plotted = set()
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    all_pts = []
    for name, spec in (MODALITIES if specs is None else specs).items():
        col = spec["color"]
        inn = {k[:3]: central(v)[0] for k, v in data[name].items() if k[3] == "in"}
        # Fig. 1 grid: JetSet without d = 2, topoclusters Q >= 4
        if spec["slug"] == "jetset":
            inn = {k: v for k, v in inn.items() if k[1] != 2}
        if spec["slug"] == "topos":
            inn = {k: v for k, v in inn.items() if k[2] >= 4}
        pts = [(Q * np.log2(K) / (32 * spec["D"]), y, (K, d, Q)) for (K, d, Q), y in inn.items()]
        all_pts += pts
        ds_plotted |= {d for _, _, (_, d, _) in pts}
        for x, y, (K, d, Q) in pts:
            ax.scatter(x, y, marker=marker_by_d[d], s=44, facecolor=col, edgecolor="none", alpha=0.35, zorder=2)
        front = pareto(pts)  # fewer bits (smaller information fraction) and lower y are better
        ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=col, lw=2.4, zorder=3, label=name)
    for name, pts in (binning or {}).items():
        col = (MODALITIES if specs is None else specs)[name]["color"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", marker="X", ms=9, color=col, lw=1.6, zorder=4)
        all_pts += [(x, y, None) for x, y in pts]
    xmax = 0.4
    ys = np.array([p[1] for p in all_pts if p[0] <= xmax])
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ys.min() / (6 if binning else 1.6), ys.max() * (40 if binning else 1.6))  # room for legends
    if goal is not None:
        ax.axhline(goal, color="0.3", ls=":", lw=1.3, zorder=1)
        ax.text(0.62, goal, "1% resolution goal", color="0.3", fontsize=13, va="bottom", ha="left",
                transform=ax.get_yaxis_transform())
    handles, labels = ax.get_legend_handles_labels()
    if binning:
        handles.append(plt.Line2D([], [], color="0.3", ls="--", marker="X", ms=9, lw=1.6))
        labels.append("fixed binning (uniform)")
    leg = ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=15)
    ax.add_artist(leg)
    shapes = [plt.Line2D([], [], marker=m, ms=10, mfc="0.4", mec="none", ls="none", label=l)
              for m, l in (("v", "d = 2"), ("o", "d = 4"), ("s", "d = 8"), ("^", "d = 12/16"))
              + ((("D", "d = 24/32"),) if ds_plotted & {24, 32} else ())]
    if binning:
        short = [plt.Line2D([], [], marker=h.get_marker(), ms=9, mfc="0.4", mec="none", ls="none",
                            label=h.get_label().removeprefix("d = ")) for h in shapes]
        ax.legend(handles=short, loc="lower center", ncol=len(short), frameon=True, framealpha=0.9,
                  edgecolor="none", fontsize=13, title="codebook dim. d", title_fontsize=13, handletextpad=0.1,
                  columnspacing=1.0)
    else:
        ax.legend(handles=shapes, loc="lower left", frameon=True, framealpha=0.9,
                  edgecolor="none", fontsize=15, title="codebook dim.", title_fontsize=15)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlabel(r"Information fraction  ($Q\,\log_2 K \,/\, 32\,D$)", fontsize=20)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=16)
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
    fig.tight_layout()
    save(fig, outdir, stem)


def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {outdir / stem}.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=REPO / "paper/figures")
    p.add_argument("--e100", action="store_true",
                   help="COCOA panels (a), (b), (d) from the 100-epoch d = 8 runs; (c) and Fig. 1 keep the full scan")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 11, "legend.handlelength": 2.8})
    data = {name: load(spec) for name, spec in MODALITIES.items()}
    for name, runs in data.items():
        n_in = sum(k[3] == "in" for k in runs)
        n_out = sum(k[3] == "out" for k in runs)
        seeds = {k: len(v) for k, v in runs.items() if len(v) > 1}
        print(f"{name}: {n_in} positions-in, {n_out} positions-out configurations; multi-seed: {seeds}")
    specs = {n: {**s, **(s.get("e100", {}) if args.e100 else {})} for n, s in MODALITIES.items()}
    swap = lambda loaded, metric=iqr_over_median: {  # noqa: E731
        n: load(specs[n], metric) if args.e100 and "e100" in MODALITIES[n] else loaded[n] for n in MODALITIES}
    main_figure(swap(data), args.outdir, data_c=data, specs=specs)
    d_crossing(data, args.outdir)
    fig1(data, args.outdir)
    cells_name = "COCOA cells (jet energy)"
    fig1({**data, cells_name: load_cells()}, args.outdir, stem="rate_distortion_pareto_with_cells",
         specs={**MODALITIES, cells_name: CELLS})
    fig1({**data, cells_name: load_cells()}, args.outdir, stem="rate_distortion_pareto_with_cells_binning",
         specs={**MODALITIES, cells_name: CELLS},
         binning={**load_binning(iqr_over_median), cells_name: load_cells_binning()})
    mae = {name: load(spec, normalized_mae) for name, spec in MODALITIES.items()}
    agg = lambda src, f: {n: {k: [f(v) for v in vs] for k, vs in r.items()} for n, r in src.items()}  # noqa: E731
    for f, what, stem in ((np.mean, "mean over features", "mean"), (np.max, "worst feature", "worst")):
        fig1(agg(mae, f), args.outdir, ylabel=f"Normalized MAE ({what})", stem=f"rate_distortion_pareto_mae_{stem}",
             goal=None)
    fig1(data, args.outdir, stem="rate_distortion_pareto_with_binning", binning=load_binning(iqr_over_median))
    fig1(agg(mae, np.mean), args.outdir, ylabel="Normalized MAE (mean over features)", goal=None,
         stem="rate_distortion_pareto_mae_mean_with_binning",
         binning=load_binning(lambda f: float(np.mean(normalized_mae(f)))))
    cells_mae, cells_mae_bin = load_cells_mae()
    cells_mae_name = "COCOA cells (pixel energy)"
    fig1({**agg(mae, np.mean), cells_mae_name: cells_mae}, args.outdir, ylabel="Normalized MAE (mean over features)",
         goal=None, stem="rate_distortion_pareto_mae_mean_with_cells_binning",
         specs={**MODALITIES, cells_mae_name: CELLS},
         binning={**load_binning(lambda f: float(np.mean(normalized_mae(f)))), cells_mae_name: cells_mae_bin})
    mae_abd = swap(mae, normalized_mae)
    # (d) compares only the features both sides tokenize: positions-in drops its position columns
    mae_d = {n: {k: [np.delete(v, MODALITIES[n]["pos_idx"]).mean() if k[3] == "in" else v.mean() for v in vs]
                 for k, vs in r.items()} for n, r in mae_abd.items()}
    main_figure(agg(mae_abd, np.mean), args.outdir, data_d=mae_d, ylabel="Normalized MAE (feature mean)",
                what="MAE", stem="design_rules_mae", include_c=False, specs=specs, d_legend_loc="upper right")


if __name__ == "__main__":
    main()
