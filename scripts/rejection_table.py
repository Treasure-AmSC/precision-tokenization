"""Fill the background-rejection LaTeX table (tab:classification).

Loads each run's predictions.npz once and computes the background rejection
(1 / mis-ID rate) of light, c and tau jets at fixed b-jet signal-efficiency
working points, with bootstrap uncertainties.  Emits both a plain-text table
and the LaTeX body for the paper table.

Run order maps to table columns:
    feature_clf_full_v3  -> Continuous (640 bits)
    cb1024_nq3           -> 30 bits (K=2^10, Q=3)
    cb1024_nq4           -> 40 bits (K=2^10, Q=4)
    cb8192_nq4           -> 52 bits (K=2^13, Q=4)
    token_clf_full_v2    -> 60 bits (K=2^15, Q=4)

Usage
-----
pixi run python scripts/rejection_table.py \\
    --run_dirs results/vqvae_scan/classifiers/feature_clf_full_v3/test_full/test_predictions \\
               results/vqvae_scan/classifiers/token_clf_full_cb1024_nq3 \\
               results/vqvae_scan/classifiers/token_clf_full_cb1024_nq4 \\
               results/vqvae_scan/classifiers/token_clf_full_cb8192_nq4 \\
               results/vqvae_scan/classifiers/token_clf_full_v2/test_full/test_predictions \\
    --run_labels continuous cb1024_nq3 cb1024_nq4 cb8192_nq4 token_clf_full_v2 \\
    --targets 0.50 0.70 --n_boot 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_roc import DEFAULT_CLASS_NAMES  # noqa: E402

BKG_CLASSES = ["light", "c", "tau"]


def rejection_at_eff(sig_scores: np.ndarray, bkg_scores: np.ndarray, target_eff: float) -> float:
    """Background rejection (1 / mis-ID rate) at a fixed signal efficiency."""
    thr = np.quantile(sig_scores, 1.0 - target_eff)
    misid = (bkg_scores >= thr).mean()
    return 1.0 / misid if misid > 0 else np.inf


def bootstrap_std(
    sig_sorted: np.ndarray,
    bkg_sorted: np.ndarray,
    target: float,
    n_boot: int,
    rng: np.random.Generator,
) -> float:
    """Bootstrap std of rejection@target using the order-statistic representation.

    Both score arrays must be pre-sorted ascending. For each resample the
    signal threshold is the ``C``-th signal order statistic with
    ``C ~ Binomial(n_sig, 1 - target)`` (the bootstrap law of a sample
    quantile for large N), and the surviving-background count is
    ``Binomial(n_bkg, p)`` with ``p`` the tail fraction above that threshold.
    This is exact in the large-N limit and O(n_boot log n) rather than
    O(n_boot * n).
    """
    n_sig, n_bkg = len(sig_sorted), len(bkg_sorted)
    # Bootstrap signal-quantile rank C, then threshold = sig_sorted[C].
    c = rng.binomial(n_sig, 1.0 - target, size=n_boot)
    np.clip(c, 0, n_sig - 1, out=c)
    thr = sig_sorted[c]
    # Tail fraction of background above each threshold (binary search).
    n_above = n_bkg - np.searchsorted(bkg_sorted, thr, side="left")
    p = n_above / n_bkg
    # Bootstrap surviving-background count, then rejection = n_bkg / n_pass.
    n_pass = rng.binomial(n_bkg, p)
    with np.errstate(divide="ignore"):
        vals = np.where(n_pass > 0, n_bkg / np.maximum(n_pass, 1), np.inf)
    vals = vals[np.isfinite(vals)]
    return float(np.std(vals)) if vals.size else np.inf


def load_npz(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = run_dir / "test" / "test_predictions" / "predictions.npz"
    if not npz_path.exists():
        npz_path = run_dir / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No predictions.npz found in {run_dir}")
    data = np.load(npz_path)
    return data["probs"], data["labels"]


def _fmt(val: float, err: float) -> str:
    """Format value +/- error in the paper's style (\\, thousands separators)."""
    if not np.isfinite(val):
        return r"$\infty$"
    if val >= 1000:
        v = int(round(val, -1))
        e = int(round(err, -1))
        vs = f"{v:,}".replace(",", r"\,")
        es = f"{e:,}".replace(",", r"\,")
        return rf"${vs} \pm {es}$"
    if val >= 100:
        return rf"${int(round(val))} \pm {int(round(err))}$"
    if val >= 10:
        return rf"${val:.1f} \pm {err:.1f}$"
    return rf"${val:.2f} \pm {err:.2f}$"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--run_labels", nargs="+", required=True)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.50, 0.70])
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--max_jets", type=int, default=0)
    parser.add_argument("--signal_class", type=str, default="b")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.run_labels):
        raise SystemExit("--run_dirs and --run_labels must have the same length.")

    class_names = dict(DEFAULT_CLASS_NAMES)
    name_to_idx = {v.replace("$", "").replace(r"\tau", "tau").lower(): k for k, v in class_names.items()}
    signal_idx = name_to_idx[args.signal_class.lower()]
    bkg_idx = {c: name_to_idx[c] for c in BKG_CLASSES}

    # results[label][target][bkg] = (nominal, std)
    results: dict[str, dict[float, dict[str, tuple[float, float]]]] = {}

    for run_dir, label in zip(args.run_dirs, args.run_labels):
        run_dir = run_dir.resolve()
        probs, labels = load_npz(run_dir)
        if args.max_jets and args.max_jets > 0:
            probs, labels = probs[: args.max_jets], labels[: args.max_jets]

        sig_scores = probs[labels == signal_idx, signal_idx]
        sig_sorted = np.sort(sig_scores)
        results[label] = {t: {} for t in args.targets}
        print(f"=== {label}  ({len(labels):,} jets) ===", flush=True)
        for c in BKG_CLASSES:
            bkg_scores = probs[labels == bkg_idx[c], signal_idx]
            bkg_sorted = np.sort(bkg_scores)
            rng = np.random.default_rng(args.seed)  # same resampling per class
            for t in args.targets:
                nom = rejection_at_eff(sig_scores, bkg_scores, t)
                std = bootstrap_std(sig_sorted, bkg_sorted, t, args.n_boot, rng)
                results[label][t][c] = (nom, std)
                print(f"  {c:<6s} @ {t:.0%}: {nom:10.1f} +/- {std:8.1f}", flush=True)
        del probs, labels, sig_scores, sig_sorted

    # ---- LaTeX table body ------------------------------------------------
    print("\n" + "=" * 70)
    print("LaTeX table rows (order: {})".format(", ".join(args.run_labels)))
    print("=" * 70)
    row_names = {"light": "light", "c": "$c$", "tau": r"$\tau$"}
    for t in args.targets:
        print(f"% --- eps_b = {t:.2f} ---")
        for c in BKG_CLASSES:
            cells = " & ".join(_fmt(*results[lab][t][c]) for lab in args.run_labels)
            print(f" & {row_names[c]:<7s} & {cells} \\\\")


if __name__ == "__main__":
    main()
