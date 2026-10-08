"""Plot 2D histogram correlation of original vs reconstructed constituent features."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import mplhep as hep
import numpy as np

FEATURE_NAME_SETS = {
    "jetset": [
        r"$p_T$", r"$\Delta\eta$", r"$\Delta\phi$",
        r"$d_0$", r"$d_0$ (beamspot)", r"$\sigma(d_0)$", r"$\sigma(d_0^{\mathrm{BS}})$",
        r"$z_0^{\mathrm{BS}}$", r"$\sigma(z_0^{\mathrm{BS}})$",
        r"$z_0 \sin\theta$", r"$\sigma(z_0 \sin\theta)$",
        r"signed $d_0$", r"signed $d_0$ signif.",
        r"signed $z_0 \sin\theta$", r"signed $z_0 \sin\theta$ signif.",
        r"$\theta$", r"$\sigma(\theta)$",
        r"$q/p$", r"$\sigma(q/p)$",
        r"$p_T$ frac",
    ],
    "cocoa_tracks": [
        r"$p_T$ [GeV]", r"$\eta$", r"$\phi$", r"$d_0$ [mm]", r"$z_0$ [mm]",
    ],
    "cocoa_topos": [
        r"$\eta$", r"$\phi$", r"$E$ [GeV]", r"$\rho$",
        r"$\sigma_\eta$", r"$\sigma_\phi$", r"$E_{\mathrm{ECAL}}$", r"$E_{\mathrm{HCAL}}$",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--output_dir", default="results/recon_comparison")
    parser.add_argument("--output_name", default="orig_vs_reco_2d")
    parser.add_argument("--feature_set", default="jetset", choices=list(FEATURE_NAME_SETS),
                        help="Which feature-name set to use for axis labels.")
    args = parser.parse_args()

    cst_feature_names = FEATURE_NAME_SETS[args.feature_set]

    plt.style.use(hep.style.CMS)

    metrics_path = Path(args.run_dir) / "test_metrics" / "test_metrics.npz"
    d = np.load(metrics_path)
    orig = d["cst_original"]
    reco = d["cst_recon"]

    n_features = orig.shape[1]
    n_pairs_per_row = 3
    n_rows = int(np.ceil(n_features / n_pairs_per_row))

    # GridSpec layout: [scatter|residual|SPACER|scatter|residual|SPACER|...]
    # Derived from n_pairs_per_row so changing that value is the only edit needed.
    panel_size  = 5.0   # inches per panel (= col width inside margins)
    spacer_frac = 0.38  # spacer column width relative to panel (room for axis labels)
    hspace      = 0.45  # row gap as fraction of row height (room for axis labels)
    wspace      = 0.02  # tiny intra-pair gap (shows border line without separating panels)
    label_fontsize = 24
    tick_labelsize  = 18

    # pair i → GridSpec cols (i*3, i*3+1); spacer at i*3+2 (except after last pair)
    n_grid_cols = n_pairs_per_row * 3 - 1
    PAIR_COLS   = [(i * 3, i * 3 + 1) for i in range(n_pairs_per_row)]
    # width_ratios: [1, 1, spacer_frac, 1, 1, spacer_frac, ...]
    width_ratios: list[float] = []
    for i in range(n_pairs_per_row):
        width_ratios += [1.0, 1.0]
        if i < n_pairs_per_row - 1:
            width_ratios.append(spacer_frac)

    # fig_w keeps each content column exactly panel_size wide (margin factors cancel)
    fig_w = panel_size * (n_pairs_per_row * 2 + (n_pairs_per_row - 1) * spacer_frac)
    fig_h = panel_size * (n_rows + (n_rows - 1) * hspace)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_rows, n_grid_cols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=wspace,
        hspace=hspace,
        left=0.08, right=0.97, top=0.97, bottom=0.08,
    )

    for i in range(n_features):
        row  = i // n_pairs_per_row
        pair = i %  n_pairs_per_row
        sc_col, re_col = PAIR_COLS[pair]

        ax_scatter = fig.add_subplot(gs[row, sc_col])
        ax_resid   = fig.add_subplot(gs[row, re_col])

        o = orig[:, i]
        r = reco[:, i]
        valid = np.isfinite(o) & np.isfinite(r)
        o, r = o[valid], r[valid]

        # --- 2D scatter (left of pair) ---
        lo, hi = np.percentile(np.concatenate([o, r]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 101)
        h, xedges, yedges = np.histogram2d(o, r, bins=[bins, bins])
        h = np.ma.masked_where(h == 0, h)
        ax_scatter.pcolormesh(xedges, yedges, h.T, cmap="viridis", norm=mcolors.LogNorm())
        ax_scatter.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.7)
        ax_scatter.set_xlim(lo, hi)
        ax_scatter.set_ylim(lo, hi)
        ax_scatter.set_box_aspect(1)
        ax_scatter.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
        ax_scatter.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
        ax_scatter.ticklabel_format(style="sci", axis="both", scilimits=(-2, 3), useMathText=True)
        ax_scatter.xaxis.get_offset_text().set_visible(False)  # hide ×10ⁿ — avoids overlap with x-label
        ax_scatter.yaxis.get_offset_text().set_visible(False)
        ax_scatter.tick_params(axis="both", labelsize=tick_labelsize)

        name = cst_feature_names[i] if i < len(cst_feature_names) else f"feature {i}"
        ax_scatter.set_xlabel(f"Original {name}", fontsize=label_fontsize)
        ax_scatter.set_ylabel(f"Reconstructed {name}", fontsize=label_fontsize)

        # --- marginal distributions (right of pair) ---
        # x-range matches the scatter; truth = dashed step, reco = solid fill
        mbins = np.linspace(lo, hi, 80)
        ax_resid.hist(o, bins=mbins, density=True,
                      histtype="step", linestyle="dashed", linewidth=1.8,
                      color="black")
        ax_resid.hist(r, bins=mbins, density=True,
                      histtype="stepfilled", alpha=0.55,
                      color="steelblue")
        ax_resid.set_xlim(lo, hi)
        if name != r"$\phi$":
            ax_resid.set_yscale("log")
        ax_resid.set_box_aspect(1)
        # Density label on the RIGHT so the intra-pair gap stays clean; no y-ticks
        ax_resid.yaxis.tick_right()
        ax_resid.yaxis.set_label_position("right")
        # NullLocator kills both major AND minor ticks (set_yticks only removes major)
        ax_resid.yaxis.set_major_locator(mticker.NullLocator())
        ax_resid.yaxis.set_minor_locator(mticker.NullLocator())
        ax_resid.tick_params(axis="y", which="both", left=False, right=False)
        ax_resid.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
        ax_resid.ticklabel_format(style="sci", axis="x", scilimits=(-2, 3), useMathText=True)
        ax_resid.xaxis.get_offset_text().set_visible(False)
        ax_resid.tick_params(axis="x", labelsize=tick_labelsize)
        ax_resid.set_xlabel(name, fontsize=label_fontsize)
        ax_resid.set_ylabel("Density", fontsize=label_fontsize)

    # hide unused axes if n_features is odd
    for i in range(n_features, n_rows * n_pairs_per_row):
        row  = i // n_pairs_per_row
        pair = i %  n_pairs_per_row
        sc_col, re_col = PAIR_COLS[pair]
        fig.add_subplot(gs[row, sc_col]).axis("off")
        fig.add_subplot(gs[row, re_col]).axis("off")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outpath = out / f"{args.output_name}.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
