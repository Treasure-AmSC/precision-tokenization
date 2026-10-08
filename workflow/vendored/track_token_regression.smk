"""Snakemake workflow: downstream jet-pT regression from frozen RVQ tokens
(COCOA tracks).

Sibling of workflow/topo_token_regression.smk -- same 3-step pipeline (see
scripts/export_track_tokens.py, train_topo_regressor.py [reused as-is,
modality-agnostic], eval_topo_regressor.py [reused, --D 5 instead of the
topo default 8], make_topo_regression_plot.py [reused as-is, modality-
agnostic]) applied to COCOA tracks instead of topoclusters.

Deliberately a SMALL scan (not the full 4x3x3=36-checkpoint tracks grid):
12 (K, Q) configs at FIXED codebook_dim d=4 (matches OPTIMAL_CONFIGS["COCOA
tracks"] in plot_rate_distortion.py) -- the full K x Q grid, same design as
the topo pipeline's 12-config cd=8 grid (originally only 4 points were run,
expanded after finding a large gap in the curve with just 4) -- each with 3
seeds (42, 1, 2) for training-variance error bars (same convention as the
topo ladder) -- plus the continuous (raw feature, no tokenizer) baseline,
also 3 seeds. No capacity_check variant here (that was a one-off
architecture diagnostic on the topo ladder, not repeated here).

  1. export_tokens        -- frozen VQ-VAE checkpoint -> z_q per track (+
                              mask, truth jet pT) for train/val/test.
     export_continuous    -- same, but raw (preprocessed) 5 features instead
                              of a tokenizer (the uncompressed baseline).
  2. train_regressor       -- identical small-transformer recipe per config,
                              pooling="sum" (jet pT scales with the SUM of
                              constituent momenta, not their average).
  3. eval_regressor        -- IQR/median (overall + binned) on the held-out
                              test set, + decode-and-sum sanity-check
                              baseline (reused from the original VQ-VAE
                              scan's own test_metrics.npz).
     make_plot             -- collects everything into one CSV + figure.

Checkpoints are the SAME ones backing plot_rate_distortion.py's "COCOA
tracks" scan (cocoa_tracks_cb{K}_cd{d}_nq{Q}_lr0.001 under CKPT_ROOT below).

Ported from heptokens-cocoa workflow/track_token_regression.smk @ cc20cbc; only paths, the job prefix
and the script locations differ (workflow/vendored/config.yaml, scripts/vendored/).

Usage:
  snakemake -s workflow/vendored/track_token_regression.smk all -n         # dry-run
  snakemake -s workflow/vendored/track_token_regression.smk all --cores 16 # submit
  # Override scale: --config n_train=50000 n_val=10000 n_test=20000
"""
from pathlib import Path


include: "common.smk"


# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = cfg_path("regression_results") / "track_token_regression"
TOKENS_DIR = OUTPUT_DIR / "tokens"
REGRESSORS_DIR = OUTPUT_DIR / "regressors"
EVAL_DIR = OUTPUT_DIR / "eval"
PLOTS_DIR = cfg_path("figures_dir") / "rate_distortion"

DATA_ROOT = cfg_path("cocoa_root")
SPLIT_DIRS = {
    "train": DATA_ROOT / "train_topup2_20M_cells256",
    "val": DATA_ROOT / "val_100K_cells256",
    "test": DATA_ROOT / "test_1M_cells256_isInfFalse",
}

CKPT_ROOT = cfg_path("a_results") / "cocoa_scan/tracks"

D_TRACKS = 5  # input features per track (track_pt, track_eta, track_phi, track_d0, track_z0)

# ── Job prefixes (time/memory as in the paper runs) ───────────────────────────
GPU_SLURM = f"{job_prefix('gpu')} -t 04:00:00 --mem=32G -D {REPO_DIR}"
CPU_SLURM = f"{job_prefix('cpu')} -t 01:00:00 --mem=16G -D {REPO_DIR}"

# ── Event budget (overridable, same scale as the topo pipeline) ─────────────
N_EVENTS = {
    "train": config.get("n_train", 1_000_000),
    "val": config.get("n_val", 100_000),
    "test": config.get("n_test", 100_000),
}

# ── 12 configs: codebook_dim d FIXED at 4 (matches OPTIMAL_CONFIGS["COCOA
# tracks"] = (4096, 4, 4) in plot_rate_distortion.py), full K x Q grid --
# same design as the topo pipeline's 12-config cd=8 grid. Originally only 4
# (K, Q) points were run; expanded to the full grid after finding a large
# gap ("cliff") between compression 13.33 and 8.00 with only 4 points --
# all 12 checkpoints + their decode-and-sum test_metrics.npz verified to
# exist on disk. Compression factor = 32*D/(Q*log2(K)), D=5 (note several
# (K, Q) pairs share the same x since it only depends on Q*log2(K), e.g.
# cb64_nq4/cb256_nq3/cb4096_nq2 all give 6.667 -- same as the topo ladder,
# not a bug):
#   cb64_cd4_nq2   -> 13.33  (highest compression)
#   cb256_cd4_nq2  -> 10.00
#   cb64_cd4_nq3   ->  8.89
#   cb1024_cd4_nq2 ->  8.00
#   cb64_cd4_nq4   ->  6.67
#   cb256_cd4_nq3  ->  6.67
#   cb4096_cd4_nq2 ->  6.67
#   cb1024_cd4_nq3 ->  5.33
#   cb256_cd4_nq4  ->  5.00
#   cb4096_cd4_nq3 ->  4.44
#   cb1024_cd4_nq4 ->  4.00
#   cb4096_cd4_nq4 ->  3.33  (lowest compression -- matches paper's starred
#                             optimal COCOA-tracks config)
CODEBOOK_SIZES = [64, 256, 1024, 4096]
NUM_QUANTIZERS = [2, 3, 4]
CONFIGS = {
    f"cb{K}_cd4_nq{Q}": dict(K=K, d=4, Q=Q)
    for K in CODEBOOK_SIZES
    for Q in NUM_QUANTIZERS
}
SPLITS = ["train", "val", "test"]

# 3 seeds per ladder point (config + continuous baseline) for training-
# variance error bars, same convention as the topo ladder.
SEEDS = [42, 1, 2]



def ckpt_path(config_name: str) -> Path:
    c = CONFIGS[config_name]
    return CKPT_ROOT / f"cocoa_tracks_cb{c['K']}_cd{c['d']}_nq{c['Q']}_lr0.001" / "checkpoints" / "best.ckpt"


def decode_sum_npz(config_name: str) -> Path:
    """Original VQ-VAE scan's own reconstruction-only test_metrics.npz for
    this config (has pt_ratio/truth_pt already, same truth-pt definition)."""
    c = CONFIGS[config_name]
    name = f"cocoa_tracks_cb{c['K']}_cd{c['d']}_nq{c['Q']}_lr0.001"
    return (
        CKPT_ROOT / name / "test" / "results" / "cocoa_scan" / "tracks" / name
        / "test" / "test_metrics" / "test_metrics.npz"
    )


wildcard_constraints:
    config_name="|".join(CONFIGS.keys()),
    split="|".join(SPLITS),
    seed="|".join(str(s) for s in SEEDS),


rule all:
    input:
        str(PLOTS_DIR / "rate_distortion_track_regression.pdf"),


# ──────────────────────────────────────────────────────────────────────────
# Step 1: token export
# ──────────────────────────────────────────────────────────────────────────
rule export_tokens:
    input:
        ckpt=lambda wc: str(ckpt_path(wc.config_name)),
    output:
        npz=str(TOKENS_DIR / "{config_name}/{split}.npz"),
    params:
        root_dir=lambda wc: str(SPLIT_DIRS[wc.split]),
        num_events=lambda wc: N_EVENTS[wc.split],
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
        log_dir=lambda wc: str(TOKENS_DIR / wc.config_name),
    resources:
        mem_mb=32_000,
        gpu=1,
    shell:
        """
        mkdir -p {params.log_dir}
        {params.slurm} -J export_{wildcards.config_name}_{wildcards.split} \
            -o {params.log_dir}/slurm_export_{wildcards.split}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/export_track_tokens.py \
                --root_dir {params.root_dir} \
                --num_events {params.num_events} \
                --ckpt_path {input.ckpt} \
                --out_path {output.npz}"
        """


rule export_continuous:
    output:
        npz=str(TOKENS_DIR / "continuous/{split}.npz"),
    params:
        root_dir=lambda wc: str(SPLIT_DIRS[wc.split]),
        num_events=lambda wc: N_EVENTS[wc.split],
        python_bin=PYTHON_BIN,
        slurm=CPU_SLURM,
        log_dir=str(TOKENS_DIR / "continuous"),
    resources:
        mem_mb=16_000,
    shell:
        """
        mkdir -p {params.log_dir}
        {params.slurm} -J export_continuous_{wildcards.split} \
            -o {params.log_dir}/slurm_export_{wildcards.split}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/export_track_tokens.py \
                --root_dir {params.root_dir} \
                --num_events {params.num_events} \
                --out_path {output.npz}"
        """


# ──────────────────────────────────────────────────────────────────────────
# Step 2: regressor training (train_topo_regressor.py is modality-agnostic --
# reused unchanged, only the exported tokens differ)
# ──────────────────────────────────────────────────────────────────────────
rule train_regressor:
    input:
        train_npz=str(TOKENS_DIR / "{config_name}/train.npz"),
        val_npz=str(TOKENS_DIR / "{config_name}/val.npz"),
    output:
        ckpt=str(REGRESSORS_DIR / "{config_name}/seed{seed}/best.pt"),
        meta=str(REGRESSORS_DIR / "{config_name}/seed{seed}/meta.json"),
    params:
        out_dir=lambda wc: str(REGRESSORS_DIR / wc.config_name / f"seed{wc.seed}"),
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
    resources:
        mem_mb=32_000,
        gpu=1,
    shell:
        """
        mkdir -p {params.out_dir}
        {params.slurm} -J train_{wildcards.config_name}_seed{wildcards.seed} \
            -o {params.out_dir}/slurm_train_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/train_topo_regressor.py \
                --train_npz {input.train_npz} \
                --val_npz {input.val_npz} \
                --out_dir {params.out_dir} \
                --pooling sum \
                --seed {wildcards.seed}"
        """


rule train_regressor_continuous:
    input:
        train_npz=str(TOKENS_DIR / "continuous/train.npz"),
        val_npz=str(TOKENS_DIR / "continuous/val.npz"),
    output:
        ckpt=str(REGRESSORS_DIR / "continuous/seed{seed}/best.pt"),
        meta=str(REGRESSORS_DIR / "continuous/seed{seed}/meta.json"),
    params:
        out_dir=lambda wc: str(REGRESSORS_DIR / "continuous" / f"seed{wc.seed}"),
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
    resources:
        mem_mb=32_000,
        gpu=1,
    shell:
        """
        mkdir -p {params.out_dir}
        {params.slurm} -J train_continuous_seed{wildcards.seed} \
            -o {params.out_dir}/slurm_train_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/train_topo_regressor.py \
                --train_npz {input.train_npz} \
                --val_npz {input.val_npz} \
                --out_dir {params.out_dir} \
                --pooling sum \
                --seed {wildcards.seed}"
        """


# ──────────────────────────────────────────────────────────────────────────
# Step 3: evaluation + final plot (eval_topo_regressor.py / make_topo_
# regression_plot.py are modality-agnostic -- reused unchanged, just with
# --D 5 instead of the topo default 8)
# ──────────────────────────────────────────────────────────────────────────
rule eval_regressor:
    input:
        test_npz=str(TOKENS_DIR / "{config_name}/test.npz"),
        ckpt=str(REGRESSORS_DIR / "{config_name}/seed{seed}/best.pt"),
        meta=str(REGRESSORS_DIR / "{config_name}/seed{seed}/meta.json"),
    output:
        summary=str(EVAL_DIR / "{config_name}__seed{seed}__default.summary.csv"),
    params:
        regressor_dir=lambda wc: str(REGRESSORS_DIR / wc.config_name / f"seed{wc.seed}"),
        decode_sum=lambda wc: str(decode_sum_npz(wc.config_name)),
        K=lambda wc: CONFIGS[wc.config_name]["K"],
        d=lambda wc: CONFIGS[wc.config_name]["d"],
        Q=lambda wc: CONFIGS[wc.config_name]["Q"],
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
    resources:
        mem_mb=16_000,
        gpu=1,
    shell:
        """
        mkdir -p {EVAL_DIR}
        {params.slurm} -J eval_{wildcards.config_name}_seed{wildcards.seed} \
            -o {EVAL_DIR}/slurm_eval_{wildcards.config_name}_seed{wildcards.seed}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/eval_topo_regressor.py \
                --test_npz {input.test_npz} \
                --regressor_dir {params.regressor_dir} \
                --config_name {wildcards.config_name}__seed{wildcards.seed} \
                --variant default \
                --K {params.K} --d {params.d} --Q {params.Q} --D {D_TRACKS} \
                --decode_sum_npz {params.decode_sum} \
                --out_dir {EVAL_DIR}"
        """


rule eval_continuous:
    input:
        test_npz=str(TOKENS_DIR / "continuous/test.npz"),
        ckpt=str(REGRESSORS_DIR / "continuous/seed{seed}/best.pt"),
        meta=str(REGRESSORS_DIR / "continuous/seed{seed}/meta.json"),
    output:
        summary=str(EVAL_DIR / "continuous__seed{seed}__continuous.summary.csv"),
    params:
        regressor_dir=lambda wc: str(REGRESSORS_DIR / "continuous" / f"seed{wc.seed}"),
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
    resources:
        mem_mb=16_000,
        gpu=1,
    shell:
        """
        mkdir -p {EVAL_DIR}
        {params.slurm} -J eval_continuous_seed{wildcards.seed} \
            -o {EVAL_DIR}/slurm_eval_continuous_seed{wildcards.seed}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/eval_topo_regressor.py \
                --test_npz {input.test_npz} \
                --regressor_dir {params.regressor_dir} \
                --config_name continuous__seed{wildcards.seed} \
                --variant continuous \
                --D {D_TRACKS} \
                --out_dir {EVAL_DIR}"
        """


rule make_plot:
    input:
        [str(EVAL_DIR / f"{name}__seed{seed}__default.summary.csv") for name in CONFIGS for seed in SEEDS],
        [str(EVAL_DIR / f"continuous__seed{seed}__continuous.summary.csv") for seed in SEEDS],
    output:
        csv=str(OUTPUT_DIR / "track_regression_results.csv"),
        pdf=str(PLOTS_DIR / "rate_distortion_track_regression.pdf"),
        png=str(PLOTS_DIR / "rate_distortion_track_regression.png"),
    params:
        python_bin=PYTHON_BIN,
        plot_stem=str(PLOTS_DIR / "rate_distortion_track_regression"),
    shell:
        """
        mkdir -p {PLOTS_DIR}
        {params.python_bin} {SCRIPTS}/make_topo_regression_plot.py \
            --eval_dir {EVAL_DIR} \
            --out_csv {output.csv} \
            --out_plot_stem {params.plot_stem}
        """
