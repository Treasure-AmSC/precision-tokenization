#!/usr/bin/env python
"""COCOA and JetSet scan heatmaps from the fast-path-off tests (test_nofastpath/).

Same grids, plotting functions and styling as scripts/compare_cocoa_scan.py and
scripts/compare_scan.py, but reading {run}/test_nofastpath/ explicitly, keeping the COCOA
panels to Q = 2, 3, 4, and using the 50-epoch tracks Q = 3 retrains (tracks_e50/) as in Fig. 1.
Writes only into new directories; refuses to run if any of them already contains files.

  results/cocoa_scan/plots_nofp/{tracks,topos}/   compare_cocoa_scan.py outputs
  results/vqvae_scan/plots_nofp/                  JetSet heatmaps + resolution vs pT
  paper/figures_nofp/                             copies under the paper's figure names

usage: python scripts/plot_heatmaps_nofp.py
"""
from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import compare_cocoa_scan as ccs  # noqa: E402
import compare_scan as cs  # noqa: E402

COCOA = REPO / "results/cocoa_scan"
COCOA_GRID = {"tracks": ([64, 256, 1024, 4096], [2, 4, 8]), "topos": ([64, 256, 1024, 4096], [2, 4, 8, 12])}
COCOA_Q = [2, 3, 4]
OUT_COCOA = COCOA / "plots_nofp"
OUT_JETSET = REPO / "results/vqvae_scan/plots_nofp"
OUT_PAPER = REPO / "paper/figures_nofp"
PAPER_NAMES = {  # source file -> paper name ({m} = tracks / topos)
    "cocoa_scan_resolution_heatmap.pdf": "cocoa_{m}_resolution_heatmap.pdf",
    "cocoa_scan_heatmap.pdf": "cocoa_{m}_mae_heatmap.pdf",
    "cocoa_jet_pt_resolution_vs_pt.pdf": "cocoa_{m}_resolution_vs_pt.pdf",
    "cocoa_mae_per_feature.pdf": "cocoa_{m}_mae_per_feature.pdf",
}


def nofp_npz(run_dir: Path) -> Path:
    files = glob.glob(str(run_dir / "test_nofastpath/**/test_metrics.npz"), recursive=True)
    if len(files) != 1:
        raise RuntimeError(f"expected one test_nofastpath/test_metrics.npz under {run_dir}, found {files}")
    return Path(files[0])


def cocoa_entry(path: Path, name: str, cb: int, cd: int, nq: int) -> dict:
    """Same fields as compare_cocoa_scan.load_scan_results."""
    data = np.load(path)
    mae_per_feat = np.mean(np.abs(data["cst_original"] - data["cst_recon"]), axis=0)
    entry = {"name": name, "cb": cb, "cd": cd, "nq": nq, "mae": np.mean(mae_per_feat), "mae_per_feat": mae_per_feat}
    pr = data["pt_ratio"].ravel()
    entry["truth_pt"], entry["pt_ratio"] = data["truth_pt"].ravel(), pr
    valid = pr[np.isfinite(pr) & (pr > 0)]
    q25, med, q75 = np.percentile(valid, [25, 50, 75])
    entry["jet_res"] = (q75 - q25) / med
    return entry


def cocoa_results(mod: str) -> list[dict]:
    cbs, cds = COCOA_GRID[mod]
    out = []
    for cb in cbs:
        for cd in cds:
            for nq in COCOA_Q:
                name = f"cocoa_{mod}_cb{cb}_cd{cd}_nq{nq}_lr0.001"
                run = COCOA / mod / name
                if mod == "tracks" and (COCOA / "tracks_e50" / f"{name}_e50").is_dir():
                    run = COCOA / "tracks_e50" / f"{name}_e50"
                out.append(cocoa_entry(nofp_npz(run), run.name, cb, cd, nq))
    out.sort(key=lambda r: r["mae"])
    return out


def jetset_load_metrics(cb, cd, nq):
    run = cs.SCAN_ROOT / cs.TPL.format(cb=cb, cd=cd, nq=nq)
    return dict(np.load(nofp_npz(run))) if run.is_dir() else None


def main():
    targets = [OUT_COCOA / "tracks", OUT_COCOA / "topos", OUT_JETSET, OUT_PAPER]
    for d in targets:
        if d.exists() and any(d.iterdir()):
            sys.exit(f"{d} already contains files; not overwriting")
    for d in targets:
        d.mkdir(parents=True, exist_ok=True)

    for mod, modality in (("tracks", "Track"), ("topos", "Topo")):
        results = cocoa_results(mod)
        out = OUT_COCOA / mod
        print(f"{mod}: {len(results)} runs ({sum('_e50' in r['name'] for r in results)} from tracks_e50)")
        ccs.plot_resolution_vs_pt(results, out, modality=modality)
        ccs.plot_heatmap(results, out, modality=modality)
        ccs.plot_resolution_heatmap(results, out, modality=modality)
        ccs.plot_mae_vs_cb(results, out)
        ccs.plot_mae_per_feature(results, out, modality=modality)
        for src, dst in PAPER_NAMES.items():
            shutil.copy2(out / src, OUT_PAPER / dst.format(m=mod))

    cs.load_metrics = jetset_load_metrics
    results = cs.build_results()
    print(f"jetset: {len(results)} runs")
    cs.plot_resolution_vs_pt(results, OUT_JETSET)
    cs._plot_grid_heatmap(results, OUT_JETSET, "mae", "MAE", "JetSet Reconstruction MAE", "jetset_mae_heatmap.pdf")
    cs._plot_grid_heatmap(results, OUT_JETSET, "jet_res", r"IQR / median", r"JetSet Jet $p_T$ Resolution",
                          "jetset_resolution_heatmap.pdf")
    for name in ("jetset_mae_heatmap.pdf", "jetset_resolution_heatmap.pdf", "jetset_jet_pt_resolution_vs_pt.pdf"):
        shutil.copy2(OUT_JETSET / name, OUT_PAPER / name)
    for d in targets:
        print(d, sorted(p.name for p in d.iterdir()))


if __name__ == "__main__":
    main()
