"""Rate-distortion frontier of the cell tokenizer (Fig. 1 cells, Table 1 cells row, Fig. D.9c, Table C.4 2M rows).
Ported from heptokens-cocoa workflow/hep4m_cell_a04_frontier_2M.smk @ cc20cbc.

Every point uses the same recipe: Large Transformer enc/dec
(hep4m_cell_transformer_masked_posenc_warmup_large_vqvae), MSE energy loss with energy_loss_alpha=0.4,
dead-code revival, seed 42, 30 epochs, 2M events. Only K, d and Q vary. Every run is evaluated on the
same 100,096 held-out test events (cells_sparse_dir/test).

Grid (info fraction = Q*log2(K) / (32*64)):
  * d=8, K in {256, 1024, 4096, 16384} x Q in {2, 4, 8}
  * d=8, Q=16 at K in {1024, 4096}
  * d scan at K=4096: Q4 d{4,16}, Q8 d{4,16,32}, Q16 d{16,32}
  * round 2: (4096, 16, 24), (4096, 16, 32), (4096, 24, 16)

In the original run set, cb4096_cd8_nq4_a0.4_large was not trained by this workflow but symlinked from
the earlier 2M loss round (cells_scan_2M_nq4_deadcode_warmup), which used the same command.

Usage:
  snakemake -s workflow/vendored/cells_frontier_2M.smk all_frontier -n
"""


include: "common.smk"


DATA_DIR = cfg_path("cells_h5_2M_dir")
TEST_DIR = cfg_path("cells_sparse_dir") / "test"
OUTPUT_DIR = cfg_path("cells_results") / "cells_scan_2M_a0.4_frontier"

MODEL_CONFIG = "hep4m_cell_transformer_masked_posenc_warmup_large_vqvae"
ALPHA = 0.4
MAX_EPOCHS = config.get("max_epochs", 30)
EVAL_EVENTS = 100_000

GPU_SLURM = job_prefix("gpu") + " -t {time} --mem=200G -D " + str(REPO_DIR)
CPU_SLURM = f"{job_prefix('cpu')} -t 08:00:00 --mem=32G -c 8 -D {REPO_DIR}"

# (K, d, Q)
POINTS = [(K, 8, Q) for K in (256, 1024, 4096, 16384) for Q in (2, 4, 8)]
POINTS += [(1024, 8, 16), (4096, 8, 16)]
POINTS += [(4096, 4, 4), (4096, 16, 4),
           (4096, 4, 8), (4096, 16, 8), (4096, 32, 8),
           (4096, 16, 16), (4096, 32, 16)]
POINTS_ROUND2 = [(4096, 16, 24), (4096, 16, 32), (4096, 24, 16)]
POINTS += POINTS_ROUND2

RUNS = {f"cb{K}_cd{d}_nq{Q}_a{ALPHA}_large": dict(cb=K, cd=d, nq=Q) for K, d, Q in POINTS}
for _r in RUNS.values():
    _r["time"] = "3-00:00:00" if _r["nq"] >= 24 else "2-00:00:00"


wildcard_constraints:
    scan_name="|".join(RUNS),


rule all_frontier:
    input:
        expand(str(OUTPUT_DIR / "transformer/{scan_name}/test_metrics.json"), scan_name=RUNS),


rule train_frontier:
    output:
        success=str(OUTPUT_DIR / "transformer/{scan_name}/SUCCESS.txt"),
        ckpt=str(OUTPUT_DIR / "transformer/{scan_name}/checkpoints/best.ckpt"),
    params:
        # Hydra full_path = output_dir/project_name/network_name
        hydra_output_dir=OUTPUT_DIR.parent,
        project_name=OUTPUT_DIR.name,
        data_dir=DATA_DIR,
        python_bin=PYTHON_BIN,
        model_cfg=MODEL_CONFIG,
        cb=lambda wc: RUNS[wc.scan_name]["cb"],
        cd=lambda wc: RUNS[wc.scan_name]["cd"],
        nq=lambda wc: RUNS[wc.scan_name]["nq"],
        alpha=ALPHA,
        max_epochs=MAX_EPOCHS,
        slurm=lambda wc: GPU_SLURM.format(time=RUNS[wc.scan_name]["time"]),
    resources:
        mem_mb=200_000,
        gpu=1,
    shell:
        """
        mkdir -p {OUTPUT_DIR}/transformer/{wildcards.scan_name}
        {params.slurm} -J a04_frontier_{wildcards.scan_name} \
            -o {OUTPUT_DIR}/transformer/{wildcards.scan_name}/slurm_%j.log \
            --wrap="{params.python_bin} scripts/train.py -cn train_cell \
                output_dir={params.hydra_output_dir} \
                project_name={params.project_name} \
                network_name=transformer/{wildcards.scan_name} \
                callbacks=hep4m_cell_scan_default \
                model={params.model_cfg} \
                datamodule.data_path={params.data_dir} \
                model.codebook_size={params.cb} \
                model.codebook_dim={params.cd} \
                model.num_quantizers={params.nq} \
                +model.threshold_ema_dead_code=1.0 \
                model.decoder.energy_loss_alpha={params.alpha} \
                model.decoder.energy_loss_type=mse \
                model.decoder.energy_sum_loss_weight=0.0 \
                trainer.max_epochs={params.max_epochs} \
                trainer.enable_progress_bar=false \
                trainer.val_check_interval=500 \
                trainer.limit_val_batches=1.0 \
                trainer.gradient_clip_val=1.0" \
        && touch {output.success}
        """


rule eval_frontier:
    input:
        success=str(OUTPUT_DIR / "transformer/{scan_name}/SUCCESS.txt"),
    output:
        str(OUTPUT_DIR / "transformer/{scan_name}/test_metrics.json"),
    params:
        python_bin=PYTHON_BIN,
        run_dir=lambda wc: str(OUTPUT_DIR / "transformer" / wc.scan_name),
        test_dir=TEST_DIR,
        n=EVAL_EVENTS,
        slurm=CPU_SLURM,
    shell:
        """
        {params.slurm} -J a04_frontier_eval_{wildcards.scan_name} \
            -o {params.run_dir}/eval_%j.log \
            --wrap="export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; \
                {params.python_bin} {SCRIPTS}/eval_cell_wloss.py --run_dir {params.run_dir} \
                --test_path {params.test_dir} --sparse --max_events {params.n}"
        """
