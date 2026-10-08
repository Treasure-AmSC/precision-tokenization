"""Compare ROC curves across arbitrary classifier runs.

Each run is specified by a directory containing ``full_config.yaml`` and a
``checkpoints/`` folder.  Any number of runs can be compared on a single plot.

Usage
-----
# From saved predictions (no GPU needed):
pixi run python scripts/compare_roc.py \\
    --from_predictions \\
    --run_dirs results/vqvae_scan/classifiers/feature_clf \\
              results/vqvae_scan/classifiers/vector_clf_model_A \\
    --run_labels "feature" "vector (model A)" \\
    --output_dir results/roc_comparison \\
    --output_name my_comparison

# Re-running inference from checkpoints (requires GPU):
pixi run python scripts/compare_roc.py \\
    --run_dirs results/long_run/classifier/feature_cf_10000000 \\
               results/preprocessing_study/classifier/quantile_100 \\
    --run_labels "feature (10M)" "token (quantile_100)" \\
    --data_path /sdf/scratch/users/s/samklein/data/atlas/mc-flavtag-ttbar-small.h5 \\
    --output_dir results/roc_comparison \\
    --output_name my_comparison

Outputs:
  {output_dir}/{output_name}_btag_roc.pdf  — HEP-style b-tagging ROC plot
  {output_dir}/{output_name}_ovr_roc.pdf   — One-vs-rest ROC plot
  {output_dir}/{output_name}_auc.csv       — per-class and macro-average AUC
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import re
from pathlib import Path

import h5py
import hydra.utils as hu
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch as T
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, roc_curve

# Register custom OmegaConf resolvers (int_div, min, max, if, …) used in saved configs
import heptokens.utils.hydra as _qhep_hydra  # noqa: F401  (side-effect import)
from heptokens.utils.hydra import reload_original_config

plt.style.use(hep.style.CMS)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "light",
    1: "c",
    2: "b",
    3: r"$\tau$",
}


def get_class_names(
    data_path: str, label_key: str = "HadronConeExclTruthLabelID"
) -> dict[int, str]:
    """Derive the class-index-to-name mapping from the raw HDF5 label column."""
    pdg_to_name = {0: "light", 4: "c", 5: "b", 15: r"$\tau$"}
    try:
        with h5py.File(data_path, "r") as f:
            raw = f["jets"][label_key][:10_000]
        unique_pdg = np.unique(raw).tolist()
        return {i: pdg_to_name.get(int(v), str(int(v))) for i, v in enumerate(unique_pdg)}
    except Exception as exc:
        log.warning("Could not derive class names from data: %s. Using defaults.", exc)
        return DEFAULT_CLASS_NAMES


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def override_data_path(
    cfg: DictConfig,
    data_path: str,
    max_jets: int | None = None,
    batch_size: int | None = None,
) -> DictConfig:
    """Return a copy of *cfg* with the datamodule data_path overridden.

    Optionally cap the number of jets loaded and override the batch size
    to keep memory usage under control during inference.
    """
    from copy import deepcopy

    cfg = deepcopy(cfg)
    OmegaConf.update(cfg, "datamodule.data_path", data_path)
    if max_jets is not None:
        OmegaConf.update(cfg, "datamodule.num_jets", max_jets)
    if batch_size is not None:
        OmegaConf.update(cfg, "datamodule.batch_size", batch_size)
    return cfg


# ---------------------------------------------------------------------------
# Transform instantiation helpers
# ---------------------------------------------------------------------------


def _instantiate_transforms(raw_transforms) -> list:
    """Convert raw transform configs (plain dicts/lists as returned by
    ``OmegaConf.to_container``) into a list of callables.

    * A ``dict`` whose values are transform configs.
    * A ``list`` of transform dicts.
    * Items that are already callable (safety net in case Hydra has already
      instantiated them in a prior pass).
    * Items that are dicts with ``_target_``/``_partial_`` keys.
    """
    import hydra.utils as hu

    if raw_transforms is None:
        return []

    # Normalise to a flat list of config-or-callable items
    if isinstance(raw_transforms, dict) and all(not isinstance(k, int) for k in raw_transforms):
        # Named dict of transforms (live-config style)
        items = list(raw_transforms.values())
    elif isinstance(raw_transforms, list):
        items = raw_transforms
    else:
        items = [raw_transforms]

    callables: list = []
    for item in items:
        if callable(item):
            # Already a callable (e.g. functools.partial from a prior
            # Hydra recursive instantiation pass)
            callables.append(item)
        elif isinstance(item, dict) and "_target_" in item:
            callables.append(hu.instantiate(item))
        else:
            log.warning(
                "Skipping transform item of type %s that is neither callable "
                "nor an instantiable config dict: %r",
                type(item).__name__,
                item,
            )
    return callables


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    cfg: DictConfig,
    data_path: str,
    device: str = "cpu",
    max_jets: int | None = None,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Instantiate datamodule + model from saved config, load weights, run
    inference over the full validation split.

    Parameters
    ----------
    max_jets : int, optional
        Cap the total number of jets loaded by the datamodule.  The
        validation split will be ``val_frac * max_jets``.  500k is more
        than enough for smooth ROC curves and uses ~360 MB.
    batch_size : int, optional
        Override the dataloader batch size for inference.

    Returns
    -------
    probs : np.ndarray, shape (N, n_classes)  — softmax probabilities
    labels : np.ndarray, shape (N,)            — integer class indices
    """

    cfg = override_data_path(cfg, data_path, max_jets=max_jets, batch_size=batch_size)

    # ---- Datamodule -------------------------------------------------------
    # Convert the entire datamodule sub-config to a plain Python dict so we
    # can freely mutate it (no OmegaConf struct/read-only limitations).
    # We then:
    #   1. Remap all old gdig.* _target_ strings to heptokens.* in-place.
    #   2. Pop the transforms key so Hydra does NOT attempt to recursively
    #      instantiate them (which fails for _partial_: true nodes and old
    #      gdig.* target paths).
    #   3. Instantiate the datamodule from the cleaned plain dict.
    #   4. Manually instantiate the transforms and assign them back.
    dm_dict: dict = OmegaConf.to_container(cfg.datamodule, resolve=False)
    raw_transforms = dm_dict.pop("transforms", None)

    datamodule = hu.instantiate(dm_dict)

    if raw_transforms is not None:
        datamodule.transforms = _instantiate_transforms(raw_transforms)
        log.info("  Instantiated %d transform(s).", len(datamodule.transforms))

    datamodule.setup("fit")

    data_sample = datamodule.get_data_sample()
    n_classes = datamodule.get_n_classes()

    # Model — instantiate with fresh weights then load saved state_dict
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model: T.nn.Module = hu.instantiate(
        model_cfg,
        data_sample=data_sample,
        n_classes=n_classes,
    )

    ckpt_path = cfg.ckpt_path
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint path found in config (cfg.ckpt_path is None).")
    ckpt = T.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    del ckpt  # free checkpoint memory immediately
    gc.collect()
    model.eval()
    model = model.to(device)

    # Inference loop over validation DataLoader
    val_loader = datamodule.val_dataloader()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with T.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, T.Tensor) else v for k, v in batch.items()}
            logits = model(batch)  # [B, n_classes]
            probs = T.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    # Free model + datamodule before returning
    del model, datamodule, val_loader
    gc.collect()
    if device != "cpu":
        T.cuda.empty_cache()

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# AUC computation & reporting
# ---------------------------------------------------------------------------

# Rows: (run_label, class_idx, class_name, auc_value)
AUCRows = list[tuple[str, int | str, str, float]]


def compute_auc_table(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    class_names: dict[int, str],
) -> AUCRows:
    """Compute per-class and macro-average AUC for every run."""
    rows: AUCRows = []
    for run_label, (probs, labels) in results.items():
        class_aucs: list[float] = []
        for class_idx in range(n_classes):
            y_true = (labels == class_idx).astype(int)
            y_score = probs[:, class_idx]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            class_aucs.append(roc_auc)
            cname = class_names.get(class_idx, str(class_idx))
            rows.append((run_label, class_idx, cname, roc_auc))
        macro = float(np.mean(class_aucs))
        rows.append((run_label, "macro_avg", "macro_avg", macro))
    return rows


def print_auc_table(rows: AUCRows) -> None:
    """Pretty-print the AUC table to stdout."""
    header = f"{'Run':<40s}  {'Class':<12s}  {'AUC':>8s}"
    print(header)
    print("-" * len(header))
    for run_label, _cidx, cname, auc_val in rows:
        print(f"{run_label:<40s}  {cname:<12s}  {auc_val:>8.5f}")


def save_auc_csv(rows: AUCRows, path: Path) -> None:
    """Write the AUC table to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_label", "class_idx", "class_name", "auc"])
        for run_label, cidx, cname, auc_val in rows:
            writer.writerow([run_label, cidx, cname, f"{auc_val:.6f}"])
    log.info("Saved AUC table → %s", path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Line styles ordered from most dashed (most compressed / smallest codebook)
# to fully solid (least compressed / continuous features). Runs are assumed
# to be passed in --run_labels in order from most to least compressed, so the
# *last* run always gets the solid line.
LINE_STYLES = [
    (0, (1, 1)),
    (0, (2, 1)),
    (0, (4, 1, 1, 1)),
    ":",
    "-.",
    "--",
    "-",
]

# Explicit translations for run labels that don't follow the "cbK_nqQ" pattern
_EXPLICIT_RUN_LABELS = {
    "token_clf_full_v2": r"60 bits ($K=2^{15}$, $Q=4$)",
    "feature_clf_full_v3": "continuous (640 bits)",
}


def _build_style_map(run_labels: list[str]) -> dict[str, object]:
    """Map each run label to a linestyle, ordered from most dashed to solid.

    The *last* label in ``run_labels`` (assumed to be the least-compressed /
    continuous-feature reference) always gets the solid line "-"; earlier
    labels get progressively more dashed styles.
    """
    n = len(run_labels)
    if n <= len(LINE_STYLES):
        styles = LINE_STYLES[-n:]
    else:
        # More runs than distinct styles: pad with the most-dashed style.
        styles = [LINE_STYLES[0]] * (n - len(LINE_STYLES)) + LINE_STYLES
    return {label: styles[i] for i, label in enumerate(run_labels)}


def _two_column_legend(
    ax,
    color_handles: list[Line2D],
    style_handles: list[Line2D],
    col1_title: str,
    col2_title: str,
    fontsize: float = 13,
) -> None:
    """Build a single 2-column legend: column 1 = color_handles (own header),
    column 2 = style_handles (own header). A single legend with a header row
    avoids the overlap that comes from stacking two separate ax.legend() boxes.

    matplotlib fills multi-column legends column-major (down column 1 first,
    then column 2), so each column's handles/labels must be laid out as a
    contiguous run of equal length (header + entries + padding).
    """
    n_rows = max(len(color_handles), len(style_handles))
    blank = Line2D([0], [0], alpha=0)

    col1_handles = [blank] + color_handles + [blank] * (n_rows - len(color_handles))
    col1_labels = (
        [rf"$\mathbf{{{col1_title}}}$"]
        + [h.get_label() for h in color_handles]
        + [""] * (n_rows - len(color_handles))
    )
    col2_handles = [blank] + style_handles + [blank] * (n_rows - len(style_handles))
    col2_labels = (
        [rf"$\mathbf{{{col2_title}}}$"]
        + [h.get_label() for h in style_handles]
        + [""] * (n_rows - len(style_handles))
    )

    ax.legend(
        col1_handles + col2_handles, col1_labels + col2_labels, ncol=2,
        loc="upper right", fontsize=fontsize, framealpha=0.9, handlelength=2.2,
        columnspacing=1.2, handletextpad=0.6,
    )


def _pretty_run_label(label: str) -> str:
    """Translate a raw --run_labels entry into a legible legend string.

    Recognizes the "cb{K}_nq{Q}" convention (e.g. "cb1024_nq3" -> "(K=2^10, Q=3)")
    plus a small table of explicit overrides. Falls back to the raw label.
    """
    if label in _EXPLICIT_RUN_LABELS:
        return _EXPLICIT_RUN_LABELS[label]
    m = re.match(r"^cb(\d+)_nq(\d+)$", label)
    if m:
        k, q = int(m.group(1)), int(m.group(2))
        log2k = k.bit_length() - 1
        k_str = f"2^{{{log2k}}}" if 2 ** log2k == k else str(k)
        bits = q * log2k
        return f"{bits} bits ($K={k_str}$, $Q={q}$)"
    return label


def plot_comparison(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    n_classes: int,
    class_names: dict[int, str] | None = None,
) -> None:
    """Produce a single HEP-style ROC plot comparing any number of classifiers.

    Parameters
    ----------
    results : dict
        ``{run_label: (probs, labels), ...}``
    output_path : Path
        Where to write the PDF.
    n_classes : int
        Number of jet classes.
    class_names : dict[int, str], optional
        Maps class index to physics label string.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Build a linestyle mapping for each run (most compressed -> most dashed)
    run_labels = list(results.keys())
    style_map = _build_style_map(run_labels)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for class_idx in range(n_classes):
        color = colors[class_idx % len(colors)]
        class_name = class_names.get(class_idx, str(class_idx))

        for run_label, (probs, labels) in results.items():
            y_true = (labels == class_idx).astype(int)
            y_score = probs[:, class_idx]

            fpr, tpr, _ = roc_curve(y_true, y_score)

            # HEP convention: signal efficiency (TPR) vs background rejection (1/FPR)
            with np.errstate(divide="ignore", invalid="ignore"):
                rejection = np.where(fpr > 0, 1.0 / fpr, np.nan)

            valid = np.isfinite(rejection) & (rejection < 1e4) & (tpr > 0)

            ax.plot(
                tpr[valid],
                rejection[valid],
                color=color,
                linestyle=style_map[run_label],
                linewidth=2,
            )

    ax.set_xlabel("Signal Efficiency", fontsize=27)
    ax.set_ylabel("Background Rejection (1/FPR)", fontsize=27)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_yscale("log")
    ax.set_xlim([0., 1.0])
    ax.set_ylim([1, 1e4])
    ax.grid(True, alpha=0.3)

    # Single 2-column legend: column 1 = flavor (color), column 2 = model
    # (linestyle), each with its own header.
    color_handles = [
        Line2D([0], [0], color=colors[class_idx % len(colors)], lw=2,
               label=class_names.get(class_idx, str(class_idx)))
        for class_idx in range(n_classes)
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=style_map[run_label],
               label=_pretty_run_label(run_label))
        for run_label in run_labels
    ]
    _two_column_legend(ax, color_handles, style_handles, "Flavor", "Model", fontsize=17)
    # hep.cms.label(ax=ax, label="Simulation", data=False, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Saved one-vs-rest ROC plot → %s", output_path)


def _signal_eff_vs_misid(
    probs: np.ndarray,
    labels: np.ndarray,
    signal_idx: int,
    bkg_idx: int,
    n_points: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute background mis-identification rate as a function of signal efficiency.

    Parameters
    ----------
    probs : (N, C) softmax probabilities
    labels : (N,) integer class indices
    signal_idx : class index treated as signal (e.g. b-jet = 2)
    bkg_idx : class index for background flavour
    n_points : number of threshold steps

    Returns
    -------
    sig_eff : (n_points,) signal efficiency values (descending threshold)
    bkg_misid : (n_points,) mis-identification rate for *bkg_idx* jets
    """
    sig_scores = probs[labels == signal_idx, signal_idx]
    bkg_scores = probs[labels == bkg_idx, signal_idx]

    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        return np.array([]), np.array([])

    # Vectorized over all thresholds: fraction of scores >= t equals
    # (n - searchsorted(sorted_scores, t)) / n. O((n+m) log n) instead of the
    # O(n*m) that a per-threshold comparison loop would cost.
    sig_sorted = np.sort(sig_scores)
    bkg_sorted = np.sort(bkg_scores)
    thresholds = np.linspace(0, 1, n_points)
    sig_eff = (len(sig_sorted) - np.searchsorted(sig_sorted, thresholds, side="left")) / len(sig_sorted)
    bkg_misid = (len(bkg_sorted) - np.searchsorted(bkg_sorted, thresholds, side="left")) / len(bkg_sorted)
    return sig_eff, bkg_misid


def plot_btag_roc(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    n_classes: int,
    class_names: dict[int, str] | None = None,
    signal_class: str = "b",
) -> None:
    """Produce a standard HEP b-tagging ROC plot.

    X-axis : b-jet (signal) efficiency
    Y-axis : rejection (1 / mis-ID rate) for each non-signal class

    This gives ``n_classes - 1`` rejection curves per run.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Find the signal class index from the name mapping
    signal_idx: int | None = None
    for idx, name in class_names.items():
        # Strip LaTeX markup for comparison
        clean = name.replace("$", "").replace(r"\tau", "tau")
        if clean.lower() == signal_class.lower():
            signal_idx = idx
            break
    if signal_idx is None:
        log.warning(
            "Signal class '%s' not found in class_names %s; defaulting to index 2.",
            signal_class,
            class_names,
        )
        signal_idx = 2

    bkg_indices = [i for i in range(n_classes) if i != signal_idx]

    run_labels = list(results.keys())
    style_map = _build_style_map(run_labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for bkg_plot_idx, bkg_idx in enumerate(bkg_indices):
        color = colors[bkg_plot_idx % len(colors)]
        bkg_name = class_names.get(bkg_idx, str(bkg_idx))

        for run_label, (probs, labels) in results.items():
            sig_eff, bkg_misid = _signal_eff_vs_misid(probs, labels, signal_idx, bkg_idx)
            if len(sig_eff) == 0:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                rejection = np.where(bkg_misid > 0, 1.0 / bkg_misid, np.nan)

            valid = np.isfinite(rejection) & (rejection < 1e6) & (sig_eff > 0)

            ax.plot(
                sig_eff[valid],
                rejection[valid],
                color=color,
                linestyle=style_map[run_label],
                linewidth=2,
            )

    sig_name = class_names.get(signal_idx, signal_class)
    ax.set_xlabel(f"{sig_name}-jet Efficiency", fontsize=27, loc="right")
    ax.set_ylabel("Background Rejection", fontsize=27, loc="top")
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_yscale("log")
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([1, 6e5])
    ax.grid(True, alpha=0.3)

    # Single 2-column legend: column 1 = background flavor (color), column 2
    # = model (linestyle), each with its own header.
    color_handles = [
        Line2D([0], [0], color=colors[bkg_plot_idx % len(colors)], lw=2,
               label=class_names.get(bkg_idx, str(bkg_idx)))
        for bkg_plot_idx, bkg_idx in enumerate(bkg_indices)
    ]
    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=style_map[run_label],
               label=_pretty_run_label(run_label))
        for run_label in run_labels
    ]
    _two_column_legend(ax, color_handles, style_handles, "Background", "Model", fontsize=17)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Saved b-tagging ROC plot → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ROC curves across arbitrary classifier runs."
    )
    parser.add_argument(
        "--run_dirs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Paths to run directories, each containing full_config.yaml "
            "and checkpoints/. At least two required."
        ),
    )
    parser.add_argument(
        "--run_labels",
        nargs="+",
        required=True,
        help="Human-readable label for each run (same order as --run_dirs).",
    )
    parser.add_argument(
        "--from_predictions",
        action="store_true",
        help=(
            "Load saved predictions.npz files instead of re-running inference. "
            "Each run_dir should contain test/test_predictions/predictions.npz "
            "with 'probs' and 'labels' arrays. No GPU required."
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to the ATLAS HDF5 file used for inference (validation split). "
            "Required when not using --from_predictions."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory in which to write the PDF and CSV.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="comparison",
        help="Stem for output files (default: 'comparison' → comparison.pdf, comparison_auc.csv).",
    )
    parser.add_argument(
        "--max_jets",
        type=int,
        default=0,
        help=(
            "Maximum number of jets to load from the HDF5 file. "
            "The validation split will be val_frac * max_jets. "
            "Default is 0 (load all jets). "
            "Set to e.g. 500000 to cap memory usage (~360 MB)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override dataloader batch size for inference (default: use config value).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if T.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if len(args.run_dirs) != len(args.run_labels):
        raise SystemExit(
            f"--run_dirs ({len(args.run_dirs)}) and --run_labels "
            f"({len(args.run_labels)}) must have the same length."
        )
    for d in args.run_dirs:
        if not d.exists():
            raise SystemExit(f"Run directory does not exist: {d}")
    if not args.from_predictions and args.data_path is None:
        raise SystemExit("--data_path is required when not using --from_predictions.")

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Derive class labels from data (or use defaults)
    # ------------------------------------------------------------------
    if args.data_path is not None:
        class_names = get_class_names(args.data_path)
    else:
        class_names = dict(DEFAULT_CLASS_NAMES)
    log.info("Class names: %s", class_names)

    # ------------------------------------------------------------------
    # Load results: either from saved predictions or via inference
    # ------------------------------------------------------------------
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_classes: int | None = None

    if args.from_predictions:
        for run_dir, run_label in zip(args.run_dirs, args.run_labels):
            run_dir = run_dir.resolve()
            npz_path = run_dir / "test" / "test_predictions" / "predictions.npz"
            if not npz_path.exists():
                # Also try the directory itself as a direct npz path
                npz_path = run_dir / "predictions.npz"
            if not npz_path.exists():
                log.warning("Skipping %s: no predictions.npz found in %s", run_label, run_dir)
                continue
            log.info("=== %s  (loading %s) ===", run_label, npz_path)
            data = np.load(npz_path)
            if args.max_jets and args.max_jets > 0:
                probs, labels = data["probs"][:args.max_jets], data["labels"][:args.max_jets]
            else:
                probs, labels = data["probs"], data["labels"]
            results[run_label] = (probs, labels)
            if n_classes is None:
                n_classes = probs.shape[1]
            log.info("  %d jets, %d classes", len(labels), probs.shape[1])
    else:
        for run_dir, run_label in zip(args.run_dirs, args.run_labels):
            run_dir = run_dir.resolve()
            log.info("=== %s  (%s) ===", run_label, run_dir)

            cfg = reload_original_config(
                path=str(run_dir),
                file_name="full_config.yaml",
                set_ckpt_path=True,
                ckpt_flag="last*",
                set_wandb_resume=False,
            )
            if cfg is None:
                log.warning("Skipping %s: full_config.yaml not found in %s.", run_label, run_dir)
                continue
            if cfg.ckpt_path is None:
                log.warning("No checkpoint found for %s, skipping.", run_label)
                continue

            if n_classes is None:
                n_classes = int(cfg.datamodule.n_classes)

            max_jets = args.max_jets if args.max_jets > 0 else None
            log.info("  running inference on %s ...", cfg.ckpt_path)
            results[run_label] = run_inference(
                cfg,
                args.data_path,
                args.device,
                max_jets=max_jets,
                batch_size=args.batch_size,
            )
            gc.collect()  # reclaim memory between runs

    if not results or n_classes is None:
        raise SystemExit("No valid runs loaded — nothing to plot.")

    # ------------------------------------------------------------------
    # AUC table
    # ------------------------------------------------------------------
    log.info("  computing AUC table ...")
    auc_rows = compute_auc_table(results, n_classes, class_names)
    print_auc_table(auc_rows)

    csv_path = output_dir / f"{args.output_name}_auc.csv"
    save_auc_csv(auc_rows, csv_path)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    log.info("  generating plots ...")
    # 1. Standard HEP b-tagging ROC: b-eff vs rejection for each other class
    btag_path = output_dir / f"{args.output_name}_btag_roc.pdf"
    plot_btag_roc(results, btag_path, n_classes, class_names)

    # 2. One-vs-rest ROC (original style)
    ovr_path = output_dir / f"{args.output_name}_ovr_roc.pdf"
    plot_comparison(results, ovr_path, n_classes, class_names)

    log.info("All done. Outputs written to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
