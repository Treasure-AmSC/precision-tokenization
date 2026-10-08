"""Cell tokenizer on the full 20M-event dataset (Table C.4, cells 20M rows).
Ported from heptokens-cocoa workflow/hep4m_cell_20M.smk @ cc20cbc ("full" variant only; the "smoke"
variant is not used in the paper). eval_cells_20M is not in the original workflow: it is the command
the paper's 20M evaluations were submitted with by hand (sacct jobs 39906461/39906462).

Large Transformer encoder/decoder (d256/6L/ff1024), lr 3e-4 with a per-step linear-warmup + cosine
schedule (configs/model/hep4m_cell_transformer_masked_posenc_steplr_large_vqvae.yaml, which sets
scheduler_interval: step), dead-code revival on. Data: configs/datamodule/hep4m_cell_20M.yaml.
Q scan (paper): the 2M loss-round winner (MSE, alpha=0.4) at Q in {8, 16}, early stopping off;
best.ckpt is still kept by val/total_loss.

Usage:
  snakemake -s workflow/vendored/cells_20M.smk full_qscan_eval -n
"""


include: "common.smk"


OUTPUT_DIR = cfg_path("cells_results") / "cells_20M"
SPARSE_ROOT = cfg_path("cells_sparse_dir")

MODEL_CONFIG = "hep4m_cell_transformer_masked_posenc_steplr_large_vqvae"
CB, CD = 4096, 8
MAX_EPOCHS = config.get("max_epochs", 6)
VARIANT = "full"
LENGTH = f"trainer.max_epochs={MAX_EPOCHS}"
EVAL_EVENTS = 100_000

# run name -> settings. The first entry is the alpha=0 base run (not in the paper).
RUNS = {
    f"cb{CB}_cd{CD}_nq4_large": {"nq": 4, "alpha": 0.0, "early_stop": True},
}
QSCAN_ALPHA = 0.4
QSCAN_NQ = [8, 16]
for _nq in QSCAN_NQ:
    RUNS[f"cb{CB}_cd{CD}_nq{_nq}_a{QSCAN_ALPHA}_large"] = {
        "nq": _nq, "alpha": QSCAN_ALPHA, "early_stop": False,
    }

GPU_SLURM = f"{job_prefix('gpu')} -t 4-20:00:00 --mem=128G -c 8 -D {REPO_DIR}"
CPU_SLURM = f"{job_prefix('cpu')} -t 06:00:00 --mem=32G -c 8 -D {REPO_DIR}"


wildcard_constraints:
    run_name="|".join(RUNS),


rule full_qscan_eval:
    input:
        [str(OUTPUT_DIR / f"{VARIANT}/{n}/test_metrics.json") for n, r in RUNS.items() if r["nq"] in QSCAN_NQ],


rule train_cells_20M:
    input:
        # Every split must be fully converted (meta.json is written last).
        sparse=[str(SPARSE_ROOT / f"{s}/singlejet_{i}.meta.json")
                for s, idxs in {"train": range(270, 470), "val": [10], "test": range(10)}.items()
                for i in idxs],
    output:
        success=str(OUTPUT_DIR / VARIANT / "{run_name}" / "SUCCESS.txt"),
        ckpt=str(OUTPUT_DIR / VARIANT / "{run_name}" / "checkpoints" / "best.ckpt"),
    params:
        python_bin=PYTHON_BIN,
        slurm=GPU_SLURM,
        length=LENGTH,
        # Hydra full_path = output_dir/project_name/network_name = OUTPUT_DIR/full/{run_name}
        hydra_output_dir=OUTPUT_DIR,
        model_cfg=MODEL_CONFIG,
        sparse_root=SPARSE_ROOT,
        cb=CB,
        cd=CD,
        nq=lambda wc: RUNS[wc.run_name]["nq"],
        alpha=lambda wc: RUNS[wc.run_name]["alpha"],
        early_stop=lambda wc: "" if RUNS[wc.run_name]["early_stop"] else "~callbacks.early_stopping",
    resources:
        mem_mb=128_000,
        gpu=1,
    shell:
        """
        mkdir -p {params.hydra_output_dir}/{VARIANT}/{wildcards.run_name}
        {params.slurm} -J cells20M_{VARIANT}_{wildcards.run_name} \
            -o {params.hydra_output_dir}/{VARIANT}/{wildcards.run_name}/slurm_%j.log \
            --wrap="{params.python_bin} scripts/train.py -cn train_cell \
                output_dir={params.hydra_output_dir} \
                project_name={VARIANT} \
                network_name={wildcards.run_name} \
                callbacks=hep4m_cell_scan_default {params.early_stop} \
                datamodule=hep4m_cell_20M \
                datamodule.train_path={params.sparse_root}/train \
                datamodule.val_path={params.sparse_root}/val \
                datamodule.test_path={params.sparse_root}/test \
                model={params.model_cfg} \
                model.codebook_size={params.cb} \
                model.codebook_dim={params.cd} \
                model.num_quantizers={params.nq} \
                +model.threshold_ema_dead_code=1.0 \
                model.decoder.energy_loss_alpha={params.alpha} \
                {params.length} \
                trainer.enable_progress_bar=false \
                trainer.val_check_interval=500 \
                trainer.limit_val_batches=1.0 \
                trainer.gradient_clip_val=1.0" \
        && touch {output.success}
        """


rule eval_cells_20M:
    input:
        success=str(OUTPUT_DIR / VARIANT / "{run_name}" / "SUCCESS.txt"),
    output:
        str(OUTPUT_DIR / VARIANT / "{run_name}" / "test_metrics.json"),
    params:
        python_bin=PYTHON_BIN,
        run_dir=lambda wc: str(OUTPUT_DIR / VARIANT / wc.run_name),
        test_dir=SPARSE_ROOT / "test",
        n=EVAL_EVENTS,
        slurm=CPU_SLURM,
    shell:
        """
        {params.slurm} -J final20M_{wildcards.run_name} \
            -o {params.run_dir}/eval_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/eval_cell_wloss.py --run_dir {params.run_dir} \
                --test_path {params.test_dir} --sparse --max_events {params.n}"
        """
