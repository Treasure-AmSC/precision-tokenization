"""One rule per paper figure/table whose producing script exists (see README.md, "Reproducing the paper").

The set-based figure scripts read repo-A results from <repo>/results (results/ must hold, or link to, the
tokenizer runs); plot_design_rules.py additionally reads the cell frontier from a hard-coded path (README).
Figures go to <figures_dir>, except where a script has fixed output paths (noted per rule).

Usage:
  snakemake -s workflow/vendored/paper_figures.smk all_figures -n
"""


include: "common.smk"


FIG = cfg_path("figures_dir")
A_RES = cfg_path("a_results")
CELLS = cfg_path("cells_results")
SPARSE = cfg_path("cells_sparse_dir")
CELL_RUN = CELLS / "cells_scan_2M_a0.4_frontier/transformer/cb4096_cd16_nq16_a0.4_large"
CLF = A_RES / "vqvae_scan/classifiers"
CLF_RUNS = {  # Table 2 / Fig. 6a column order
    "continuous": CLF / "feature_clf_full_v3/test_full/test_predictions",
    "cb1024_nq3": CLF / "token_clf_full_cb1024_nq3",
    "cb1024_nq4": CLF / "token_clf_full_cb1024_nq4",
    "cb8192_nq4": CLF / "token_clf_full_cb8192_nq4",
    "token_clf_full_v2": CLF / "token_clf_full_v2/test_full/test_predictions",
}
NOFP = "test_nofastpath/results/cocoa_scan/{mod}/{name}/test_nofastpath"
E11_RUNS = {
    "cocoa_tracks_cb4096_cd8_nq3": ("tracks", "cocoa_tracks_cb4096_cd8_nq3_lr0.001", "cocoa_tracks"),
    "cocoa_topos_cb4096_cd8_nq6": ("topos", "cocoa_topos_cb4096_cd8_nq6_lr0.001", "cocoa_topos"),
}
CELL_DIAG_TITLE = r"MSE, $\\alpha=0.4$, 2M train"


rule all_figures:
    input:
        FIG / "design_rules/rate_distortion_pareto_with_cells_binning.pdf",  # Fig. 1
        FIG / "design_rules/design_rules.pdf",  # Fig. 4
        FIG / "design_rules/rate_distortion_pareto_mae_mean_with_binning.pdf",  # Fig. E.10
        FIG / "rate_distortion/rate_distortion_cocoa.pdf",  # Fig. 5
        FIG / "rate_distortion/rate_distortion_jetset.pdf",  # Fig. 6b
        FIG / "roc/token_clf_scan_vs_refs_btag_roc.pdf",  # Fig. 6a
        FIG / "tables/rejection_table.txt",  # Table 2
        FIG / "tables/table_c4.csv",  # Table C.4
        FIG / "mlp_vs_transformer/mlp_vs_transformer.pdf",  # Fig. C.8
        FIG / "cell_utilization/plots/util_vs_nq_transformer_large.pdf",  # Fig. D.9c
        expand(FIG / "selected_2d_correlation/{name}.pdf", name=E11_RUNS),  # Fig. E.11
        FIG / "cell_diagnostics/cell_diagnostics.pdf",  # Fig. E.12


rule fig1_fig4_figE10:
    """plot_design_rules.py --e100 (as for the published figures): Fig. 1, Fig. 4, Fig. E.10 and variants."""
    output:
        FIG / "design_rules/rate_distortion_pareto_with_cells_binning.pdf",
        FIG / "design_rules/design_rules.pdf",
        FIG / "design_rules/rate_distortion_pareto_mae_mean_with_binning.pdf",
    params:
        outdir=FIG / "design_rules",
    shell:
        "{PYTHON_BIN} scripts/plot_design_rules.py --e100 --outdir {params.outdir}"


rule binning_baseline:
    """Fixed uniform binning baseline (Fig. 1, Fig. E.10 dashed lines) for tracks, topos and JetSet."""
    output:
        A_RES / "binning_baseline/{modality}/binning_uniform_K{K}/test_metrics/test_metrics.npz",
    wildcard_constraints:
        modality="tracks|topos|jetset",
    shell:
        "{PYTHON_BIN} scripts/binning_baseline.py --modality {wildcards.modality} --K {wildcards.K} --mode uniform"


rule binning_baseline_cells:
    """Fixed uniform binning baseline for the cells (Fig. 1 dashed purple line)."""
    output:
        A_RES / "binning_baseline/cells/binning_uniform_K{K}/test_metrics.json",
    shell:
        "{PYTHON_BIN} scripts/binning_baseline_cells.py --K {wildcards.K} --data_dir {SPARSE}"


rule fig5_fig6b:
    """Fig. 5 (regression + particle flow) and Fig. 6b (light-jet rejection); values are literals in the script."""
    output:
        FIG / "rate_distortion/rate_distortion_cocoa.pdf",
        FIG / "rate_distortion/rate_distortion_jetset.pdf",
    params:
        outdir=FIG / "rate_distortion",
    shell:
        "{PYTHON_BIN} {SCRIPTS}/plot_rate_distortion.py --outdir {params.outdir}"


rule fig6a:
    """Same command as rule compare_roc_scan in workflow/Snakefile, with the output directory moved."""
    input:
        [p / "predictions.npz" if p.name == "test_predictions" else p / "test/test_predictions/predictions.npz"
         for p in CLF_RUNS.values()],
    output:
        FIG / "roc/token_clf_scan_vs_refs_btag_roc.pdf",
    params:
        run_dirs=" ".join(str(CLF_RUNS[k]) for k in ("cb1024_nq3", "cb1024_nq4", "cb8192_nq4",
                                                      "token_clf_full_v2", "continuous")),
        outdir=FIG / "roc",
    shell:
        """
        {PYTHON_BIN} scripts/compare_roc.py --from_predictions \
            --run_dirs {params.run_dirs} \
            --run_labels cb1024_nq3 cb1024_nq4 cb8192_nq4 token_clf_full_v2 feature_clf_full_v3 \
            --output_dir {params.outdir} --output_name token_clf_scan_vs_refs
        """


rule table2:
    output:
        FIG / "tables/rejection_table.txt",
    params:
        run_dirs=" ".join(str(p) for p in CLF_RUNS.values()),
        labels=" ".join(CLF_RUNS),
    shell:
        """
        mkdir -p $(dirname {output})
        {PYTHON_BIN} scripts/rejection_table.py --run_dirs {params.run_dirs} --run_labels {params.labels} \
            --targets 0.50 0.70 --n_boot 200 > {output}
        """


rule tableC4:
    """Jet resolution versus training-set size, collected from the stored run metrics."""
    output:
        FIG / "tables/table_c4.csv",
    shell:
        "{PYTHON_BIN} scripts/table_c4.py --a_results {A_RES} --cells_results {CELLS} --out {output}"


rule figC8:
    output:
        FIG / "mlp_vs_transformer/mlp_vs_transformer.pdf",
    params:
        outdir=FIG / "mlp_vs_transformer",
    shell:
        """
        {PYTHON_BIN} scripts/compare_mlp_transformer.py --scan_dir {A_RES}/vqvae_scan/vqvae_scan \
            --output_dir {params.outdir} --output_name mlp_vs_transformer
        """


rule figD9ab_d:
    """Fig. D.9a, b, d. Fixed output paths: results/cocoa_scan/plots_nofp/{tracks,topos}/cocoa_utilization_heatmap.pdf,
    results/vqvae_scan/plots_nofp/jetset_utilization_heatmap.pdf. The published (a), (b) have Q = 2, 3, 4 panels:
    the plot step adds a Q = 6 panel if results/codebook_utilization/cocoa_utilization_nofp_q6.csv exists."""
    output:
        "results/cocoa_scan/plots_nofp/tracks/cocoa_utilization_heatmap.pdf",
        "results/cocoa_scan/plots_nofp/topos/cocoa_utilization_heatmap.pdf",
        "results/vqvae_scan/plots_nofp/jetset_utilization_heatmap.pdf",
    shell:
        """
        {PYTHON_BIN} scripts/cocoa_utilization_nofp.py compute
        {PYTHON_BIN} scripts/cocoa_utilization_nofp.py plot
        """


rule figD9c:
    output:
        FIG / "cell_utilization/plots/util_vs_nq_transformer_large.pdf",
    params:
        outdir=FIG / "cell_utilization",
    shell:
        "{PYTHON_BIN} {SCRIPTS}/make_cell_utilization.py --scan_dir {CELLS}/cells_scan_2M_a0.4_frontier/transformer"
        " --outdir {params.outdir}"


rule figE11:
    output:
        FIG / "selected_2d_correlation/{name}.pdf",
    params:
        run_dir=lambda wc: A_RES / "cocoa_scan" / E11_RUNS[wc.name][0] / E11_RUNS[wc.name][1]
        / NOFP.format(mod=E11_RUNS[wc.name][0], name=E11_RUNS[wc.name][1]),
        feature_set=lambda wc: E11_RUNS[wc.name][2],
        outdir=FIG / "selected_2d_correlation",
    shell:
        """
        {PYTHON_BIN} scripts/plot_2d_correlation.py --run_dir {params.run_dir} --feature_set {params.feature_set} \
            --output_dir {params.outdir} --output_name {wildcards.name}
        """


rule figE12:
    """Same command as the published figure, including its --title_extra."""
    output:
        FIG / "cell_diagnostics/cell_diagnostics.pdf",
    params:
        outdir=FIG / "cell_diagnostics",
        slurm=f"{job_prefix('gpu')} -t 02:00:00 --mem=48G -c 8 -D {REPO_DIR}",
    shell:
        """
        {params.slurm} -J diag_cd16_nq16 -o {params.outdir}/slurm_%j.log \
            --wrap="{PYTHON_BIN} {SCRIPTS}/make_cell_diagnostics.py --ckpt {CELL_RUN}/checkpoints/best.ckpt \
                --test_path {SPARSE}/test --sparse --max_events 50000 --outdir {params.outdir} \
                --title_extra '{CELL_DIAG_TITLE}'"
        """
