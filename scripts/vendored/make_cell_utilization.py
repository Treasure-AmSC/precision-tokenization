"""Codebook-utilization plots for the 2M-event cell frontier scan, made with
heptokens' scripts/plot_codebook_utilization.py (imported, not modified).

Builds that script's input CSV from each run's test_metrics.json (per-quantizer
code usage on the 100,096 held-out test events), then calls its plot functions.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
from unittest import mock

import matplotlib.axes
import matplotlib.figure

SCAN = None  # --scan_dir: <CELLS_RESULTS>/cells_scan_2M_a0.4_frontier/transformer
PLOTTER = Path(__file__).resolve().parents[1] / "plot_codebook_utilization.py"
HEATMAP = Path(__file__).resolve().parents[1] / "plot_utilization_heatmaps.py"
OUT = None  # --outdir
N_TRAIN = 2_000_000
LR = 0.0003
Q_MAX = 16  # cell runs with Q > 16 are not part of the paper


def build_csv(path: Path) -> int:
    rows = []
    for m_path in sorted(SCAN.glob("*/test_metrics.json")):
        m = json.loads(m_path.read_text())
        cb, cd, nq = int(m["cb"]), int(m["cd"]), int(m["nq"])
        if nq > Q_MAX:
            continue
        row = {
            # Name follows the plotter's parse_run_details() conventions.
            "run": f"transformer_vqvae_enc_large_cb{cb}_cd{cd}_nq{nq}_lr{LR}_cw1.0_nj{N_TRAIN}",
            "cb": cb, "cd": cd, "nq": nq, "lr": LR, "codebook_size": cb, "num_quantizers": nq,
            "total_tokens": "", "avg_utilization": m["codebook_util_avg"],
        }
        for q, u in enumerate(m["codebook_util_per_quantizer"]):
            # Zero-padded: the plotter sorts util_q* columns as strings.
            row[f"util_q{q:02d}"] = u
            row[f"unique_q{q:02d}"] = round(u * cb)
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    global SCAN, OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan_dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    SCAN, OUT = args.scan_dir, args.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "utilization_cells.csv"
    print(f"{build_csv(csv_path)} runs -> {csv_path}")

    mod = _load_module(PLOTTER)

    orig_savefig = matplotlib.figure.Figure.savefig

    def savefig_with_png(self, fname, *args, **kwargs):
        orig_savefig(self, fname, *args, **kwargs)
        orig_savefig(self, Path(fname).with_suffix(".png"), *args, **{**kwargs, "dpi": 150})

    matplotlib.figure.Figure.savefig = savefig_with_png

    df = mod.load_data(csv_path)
    # The plotter's filters keep only 10M-jet JetSet runs; the cell runs are 2M
    # events (kept in the CSV run names), so match the filter value here only.
    df["nj"] = 10_000_000
    plot_dir = OUT / "plots"
    plot_dir.mkdir(exist_ok=True)
    for fn in (mod.plot_util_vs_codebook_size, mod.plot_util_vs_nq, mod.plot_util_vs_cd):
        with mock.patch.object(matplotlib.axes.Axes, "set_title", lambda *a, **k: None):
            fn(df, plot_dir)
    mod.plot_per_quantizer_breakdown(df, plot_dir)

    heat = _load_module(HEATMAP)
    results = [{"cb": int(r.cb), "cd": int(r.cd), "nq": int(r.nq), "value": float(r.avg_utilization)}
               for r in df.itertuples()]
    cbs, cds, nqs = (sorted({r[k] for r in results}) for k in ("cb", "cd", "nq"))
    heat._plot_grid_heatmap(results, plot_dir / "cell_utilization_heatmap.pdf", cbs, cds, nqs, heat.CBAR_LABEL)
    heat._print_grid("COCOA cells (2M, MSE alpha=0.4)", results, cbs, cds, nqs)


if __name__ == "__main__":
    main()
