"""Snakemake workflow: downstream jet-pT regression from frozen RVQ tokens
(COCOA topoclusters, "positions-in" design).

Pipeline (see scripts/export_topo_tokens.py, train_topo_regressor.py,
eval_topo_regressor.py, make_topo_regression_plot.py for the actual logic):

  1. export_tokens        -- frozen VQ-VAE checkpoint -> z_q per cluster (+
                              mask, truth jet pT) for train/val/test, for each
                              of 12 (K, Q) configs at FIXED codebook_dim d=8
                              (varying d was confounding the ladder with a
                              second free parameter -- d is now held fixed so
                              compression factor is driven purely by K and Q).
     export_continuous    -- same, but raw (preprocessed) 8 features instead
                              of a tokenizer (the uncompressed baseline).
  2. train_regressor       -- identical small-transformer recipe per config
                              (only the tokens differ), pooling="sum" (not
                              "mean" -- jet pT scales with the SUM of
                              constituent momenta, not their average; see
                              heptokens.models.transformer.SetToVectorTransformer).
     train_regressor_capacity_check -- same recipe, 2-4x larger model, at the
                              optimal config only. Uses a lower peak LR +
                              longer warmup than the default recipe (a first
                              attempt with the default LR destabilized after
                              a good early epoch and never recovered -- see
                              repo memory notes).
  3. eval_regressor        -- IQR/median (overall + binned) on the held-out
                              test set, + decode-and-sum sanity-check
                              baseline (reused from the original VQ-VAE scan's
                              own test_metrics.npz).
     make_plot             -- collects everything into one CSV + figure.

Checkpoints are the SAME ones backing plot_rate_distortion.py's "COCOA
topoclusters" scan (cocoa_topos_cb{K}_cd{d}_nq{Q}_lr0.001 under CKPT_ROOT below).

Ported from heptokens-cocoa workflow/topo_token_regression.smk @ cc20cbc; only paths, the job prefix
and the script locations differ (workflow/vendored/config.yaml, scripts/vendored/).

Usage:
  snakemake -s workflow/vendored/topo_token_regression.smk all -n         # dry-run
  snakemake -s workflow/vendored/topo_token_regression.smk all --cores 1  # submit
  # Override scale: --config n_train=50000 n_val=10000 n_test=20000
"""
from pathlib import Path


include: "common.smk"


# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = cfg_path("regression_results") / "topo_token_regression"
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

CKPT_ROOT = cfg_path("a_results") / "cocoa_scan/topos"

# ── Job prefixes (time/memory as in the paper runs) ───────────────────────────
GPU_SLURM = f"{job_prefix('gpu')} -t 04:00:00 --mem=32G -D {REPO_DIR}"
CPU_SLURM = f"{job_prefix('cpu')} -t 01:00:00 --mem=16G -D {REPO_DIR}"

# ── Event budget (overridable) ────────────────────────────────────────────────
N_EVENTS = {
    "train": config.get("n_train", 1_000_000),
    "val": config.get("n_val", 100_000),
    "test": config.get("n_test", 100_000),
}

# ── 12 configs: codebook_dim d FIXED at 8, varying K (codebook size) x Q
# (num_quantizers) -- removes d as a hidden/confounding variable so the
# ladder is driven purely by K and Q. All 12 checkpoints verified to exist
# on disk (cocoa_topos_cb{K}_cd8_nq{Q}_lr0.001). ──────────────────────────
CODEBOOK_SIZES = [64, 256, 1024, 4096]
NUM_QUANTIZERS = [2, 3, 4]
CONFIGS = {
    f"cb{K}_cd8_nq{Q}": dict(K=K, d=8, Q=Q)
    for K in CODEBOOK_SIZES
    for Q in NUM_QUANTIZERS
}
OPTIMAL_CONFIG = "cb4096_cd8_nq4"  # highest compression among the 12
SPLITS = ["train", "val", "test"]

# 3 seeds per ladder point (config + continuous baseline) for proper
# training-variance error bars -- single-seed point estimates turned out to
# be quite seed-sensitive (see repo memory notes: weak scalar-only
# supervision + sum-pooling + no regularization -> wide spread of possible
# outcomes per config). seed=42 reuses the already-completed first-pass
# runs; 1 and 2 are new. capacity_check stays single-seed (one-off
# architecture-capacity diagnostic, not a ladder point needing error bars).
SEEDS = [42, 1, 2]


def ckpt_path(config_name: str) -> Path:
    c = CONFIGS[config_name]
    return CKPT_ROOT / f"cocoa_topos_cb{c['K']}_cd{c['d']}_nq{c['Q']}_lr0.001" / "checkpoints" / "best.ckpt"


def decode_sum_npz(config_name: str) -> Path:
    """Original VQ-VAE scan's own reconstruction-only test_metrics.npz for
    this config (has pt_ratio/truth_pt already, same truth-pt definition)."""
    c = CONFIGS[config_name]
    name = f"cocoa_topos_cb{c['K']}_cd{c['d']}_nq{c['Q']}_lr0.001"
    return (
        CKPT_ROOT / name / "test" / "results" / "cocoa_scan" / "topos" / name
        / "test" / "test_metrics" / "test_metrics.npz"
    )


wildcard_constraints:
    config_name="|".join(CONFIGS.keys()),
    split="|".join(SPLITS),
    seed="|".join(str(s) for s in SEEDS),


rule all:
    input:
        str(PLOTS_DIR / "rate_distortion_topo_regression.pdf"),


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
            --wrap="{params.python_bin} {SCRIPTS}/export_topo_tokens.py \
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
            --wrap="{params.python_bin} {SCRIPTS}/export_topo_tokens.py \
                --root_dir {params.root_dir} \
                --num_events {params.num_events} \
                --out_path {output.npz}"
        """


# ──────────────────────────────────────────────────────────────────────────
# Step 2: regressor training
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


rule train_regressor_capacity_check:
    """Same optimal-config tokens, 2-4x larger model (d_model=256, 6 layers).
    If IQR/median is ~unchanged from the default-sized model, the small
    model was not capacity-limited."""
    input:
        train_npz=str(TOKENS_DIR / f"{OPTIMAL_CONFIG}/train.npz"),
        val_npz=str(TOKENS_DIR / f"{OPTIMAL_CONFIG}/val.npz"),
    output:
        ckpt=str(REGRESSORS_DIR / "capacity_check/best.pt"),
        meta=str(REGRESSORS_DIR / "capacity_check/meta.json"),
    params:
        out_dir=str(REGRESSORS_DIR / "capacity_check"),
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
    resources:
        mem_mb=32_000,
        gpu=1,
    shell:
        """
        mkdir -p {params.out_dir}
        {params.slurm} -J train_capacity_check \
            -o {params.out_dir}/slurm_train_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/train_topo_regressor.py \
                --train_npz {input.train_npz} \
                --val_npz {input.val_npz} \
                --out_dir {params.out_dir} \
                --pooling sum \
                --d_model 256 --num_layers 6 \
                --lr 0.0003 --warmup_epochs 8"
        """


# ──────────────────────────────────────────────────────────────────────────
# Step 3: evaluation + final plot
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
                --K {params.K} --d {params.d} --Q {params.Q} \
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
                --out_dir {EVAL_DIR}"
        """


rule eval_capacity_check:
    input:
        test_npz=str(TOKENS_DIR / f"{OPTIMAL_CONFIG}/test.npz"),
        ckpt=str(REGRESSORS_DIR / "capacity_check/best.pt"),
        meta=str(REGRESSORS_DIR / "capacity_check/meta.json"),
    output:
        summary=str(EVAL_DIR / f"{OPTIMAL_CONFIG}__capacity_check.summary.csv"),
    params:
        regressor_dir=str(REGRESSORS_DIR / "capacity_check"),
        decode_sum=str(decode_sum_npz(OPTIMAL_CONFIG)),
        K=CONFIGS[OPTIMAL_CONFIG]["K"],
        d=CONFIGS[OPTIMAL_CONFIG]["d"],
        Q=CONFIGS[OPTIMAL_CONFIG]["Q"],
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
    resources:
        mem_mb=16_000,
        gpu=1,
    shell:
        """
        mkdir -p {EVAL_DIR}
        {params.slurm} -J eval_capacity_check \
            -o {EVAL_DIR}/slurm_eval_capacity_check_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/eval_topo_regressor.py \
                --test_npz {input.test_npz} \
                --regressor_dir {params.regressor_dir} \
                --config_name {OPTIMAL_CONFIG} \
                --variant capacity_check \
                --K {params.K} --d {params.d} --Q {params.Q} \
                --decode_sum_npz {params.decode_sum} \
                --out_dir {EVAL_DIR}"
        """


rule make_plot:
    input:
        [str(EVAL_DIR / f"{name}__seed{seed}__default.summary.csv") for name in CONFIGS for seed in SEEDS],
        [str(EVAL_DIR / f"continuous__seed{seed}__continuous.summary.csv") for seed in SEEDS],
        str(EVAL_DIR / f"{OPTIMAL_CONFIG}__capacity_check.summary.csv"),
    output:
        csv=str(OUTPUT_DIR / "topo_regression_results.csv"),
        pdf=str(PLOTS_DIR / "rate_distortion_topo_regression.pdf"),
        png=str(PLOTS_DIR / "rate_distortion_topo_regression.png"),
    params:
        python_bin=PYTHON_BIN,
        plot_stem=str(PLOTS_DIR / "rate_distortion_topo_regression"),
    shell:
        """
        mkdir -p {PLOTS_DIR}
        {params.python_bin} {SCRIPTS}/make_topo_regression_plot.py \
            --eval_dir {EVAL_DIR} \
            --out_csv {output.csv} \
            --out_plot_stem {params.plot_stem}
        """
