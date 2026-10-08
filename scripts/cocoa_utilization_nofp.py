#!/usr/bin/env python
"""COCOA codebook utilization with the torch MHA fast path off, for the heatmap run set.

Same method as scripts/cocoa_codebook_utilization.py (unique codes per RVQ stage over the full
COCOA test split), on the runs used by scripts/plot_heatmaps_nofp.py: the (K, d, Q) grid with
Q = 2, 3, 4 and the 50-epoch tracks Q = 3 retrains (tracks_e50/). Never overwrites a file.

  compute (GPU): python scripts/cocoa_utilization_nofp.py compute
                 -> results/codebook_utilization/cocoa_utilization_nofp.csv
  compute_q6:    python scripts/cocoa_utilization_nofp.py compute_q6   (the Q = 6 runs, separate CSV)
                 -> results/codebook_utilization/cocoa_utilization_nofp_q6.csv
  plot:          python scripts/cocoa_utilization_nofp.py plot [--overwrite]
                 -> results/cocoa_scan/plots_nofp/{tracks,topos}/cocoa_utilization_heatmap.pdf,
                    results/vqvae_scan/plots_nofp/jetset_utilization_heatmap.pdf (existing JetSet CSV),
                    paper/figures_nofp/cocoa_{tracks,topos}_utilization_heatmap.pdf
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import cocoa_codebook_utilization as ccu  # noqa: E402
import plot_heatmaps_nofp as phn  # noqa: E402
import plot_utilization_heatmaps as puh  # noqa: E402

log = logging.getLogger(__name__)
CSV_OUT = REPO / "results/codebook_utilization/cocoa_utilization_nofp.csv"
CSV_Q6 = REPO / "results/codebook_utilization/cocoa_utilization_nofp_q6.csv"
Q6_RUNS = {"tracks": [(4096, 4, 6)], "topos": [(4096, 8, 6)]}  # the only 50-epoch Q = 6 runs


def fresh(path: Path, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        sys.exit(f"{path} exists; not overwriting")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def runs(mod: str):
    cbs, cds = phn.COCOA_GRID[mod]
    for cb in cbs:
        for cd in cds:
            for nq in phn.COCOA_Q:
                name = f"cocoa_{mod}_cb{cb}_cd{cd}_nq{nq}_lr0.001"
                run = phn.COCOA / mod / name
                if mod == "tracks" and (phn.COCOA / "tracks_e50" / f"{name}_e50").is_dir():
                    run = phn.COCOA / "tracks_e50" / f"{name}_e50"
                yield cb, cd, nq, run


def q6_runs(mod: str):
    for cb, cd, nq in Q6_RUNS[mod]:
        yield cb, cd, nq, phn.COCOA / mod / f"cocoa_{mod}_cb{cb}_cd{cd}_nq{nq}_lr0.001"


def compute(run_fn=runs, csv_path=CSV_OUT):
    out = fresh(csv_path)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, max_nq = [], 0
    for mod in ("tracks", "topos"):
        batches = ccu.cache_test_batches(mod, None)
        for cb, cd, nq, run in run_fn(mod):
            model = ccu.load_model(run / "checkpoints/best.ckpt", device)
            res = ccu.utilization_for_run(model, batches, device)
            max_nq = max(max_nq, res["num_quantizers"])
            row = {"modality": mod, "run": run.name, "source": str(run.relative_to(REPO)), "cb": cb, "cd": cd,
                   "nq": res["num_quantizers"], "total_tokens": res["total_tokens"],
                   "avg_utilization": round(res["avg_utilization"], 6)}
            for q in range(res["num_quantizers"]):
                row[f"util_q{q}"] = round(res["per_q"][q], 6)
                row[f"unique_q{q}"] = res["counts"][q]
            rows.append(row)
            log.info("%s: avg=%.4f per_q=%s", run.name, res["avg_utilization"], [round(x, 3) for x in res["per_q"]])
            del model
            torch.cuda.empty_cache()
    cols = ["modality", "run", "source", "cb", "cd", "nq", "total_tokens", "avg_utilization"]
    cols += [f"util_q{q}" for q in range(max_nq)] + [f"unique_q{q}" for q in range(max_nq)]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows to %s", len(rows), out)


def plot(overwrite=False):
    rows = []
    for path in (CSV_OUT, CSV_Q6):
        if path.exists():
            with open(path) as f:
                rows += list(csv.DictReader(f))
    nqs = sorted({int(r["nq"]) for r in rows})
    for mod in ("tracks", "topos"):
        res = [{"cb": int(r["cb"]), "cd": int(r["cd"]), "nq": int(r["nq"]), "value": float(r["avg_utilization"])}
               for r in rows if r["modality"] == mod]
        out = fresh(phn.OUT_COCOA / mod / "cocoa_utilization_heatmap.pdf", overwrite)
        puh._plot_grid_heatmap(res, out, puh.COCOA_CBS, puh.COCOA_CDS[mod], nqs, puh.CBAR_LABEL)
        puh._print_grid(f"COCOA {mod}", res, puh.COCOA_CBS, puh.COCOA_CDS[mod], nqs)
        shutil.copy2(out, fresh(phn.OUT_PAPER / f"cocoa_{mod}_utilization_heatmap.pdf", overwrite))
    puh.JETSET_CSV = REPO / puh.JETSET_CSV
    puh._plot_grid_heatmap(puh.load_jetset(), fresh(phn.OUT_JETSET / "jetset_utilization_heatmap.pdf", overwrite),
                           puh.JETSET_CBS, puh.JETSET_CDS, puh.JETSET_NQS, puh.CBAR_LABEL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["compute", "compute_q6", "plot"])
    p.add_argument("--overwrite", action="store_true", help="plot: replace existing heatmaps")
    a = p.parse_args()
    if a.step == "compute":
        compute()
    elif a.step == "compute_q6":
        compute(q6_runs, CSV_Q6)
    else:
        plot(a.overwrite)
