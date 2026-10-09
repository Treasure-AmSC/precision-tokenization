# Precision Tokenization for Multi-Modal Data in High Energy Physics

Code for the paper *Precision Tokenization for Multi-Modal Data in High Energy Physics* (arXiv:XXXX.XXXXX).
This is a frozen release; for new work use the [heptokens library](https://github.com/treasure-AmSC/heptokens).

- **JetSet** is public: [doi:10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P).
- **COCOA:** we are still figuring out how to host the data.
- **Trained models:** not hosted yet.

## Setup

```bash
pixi install                  # environment pinned by pixi.lock
export WANDB_MODE=disabled    # training logs to Weights & Biases otherwise
```

Training needs a GPU. Evaluation and plotting run on CPU, but slowly (a JetSet test takes a few hours on CPU).

Download the JetSet files from the [record page](https://opendata.cern.ch/record/93940) or with the CERN Open
Data client. Tokenizer training uses `mc-flavtag-ttbar-medium.h5` and the test sample is
`mc-flavtag-ttbar-small.h5`. The classifiers (Fig. 6, Table 2) use all three files, including the 85 GB
`mc-flavtag-ttbar-large.h5`.

```bash
pip install cernopendata-client
cernopendata-client download-files --recid 93940 --filter-name mc-flavtag-ttbar-medium.h5
cernopendata-client download-files --recid 93940 --filter-name mc-flavtag-ttbar-small.h5
```

## Site-specific paths

The code was run on one cluster and some paths are hard-coded. Path roots:

- `/sdf/data/atlas/u/jkrupa/`: the repository (results, scalers) and the complete `mc-flavtag-ttbar-large.h5`
- `/sdf/scratch/users/j/jkrupa/data/atlas/`: the JetSet HDF5 files
- `/fs/ddn/sdf/group/atlas/d/jkrupa/`: COCOA ROOT files, calorimeter-cell data and runs, token exports
- `/sdf/scratch/users/s/samklein/`, `/sdf/group/mli/samklein/`: old defaults
- `/lscratch/`: node-local staging; `/global/cfs/`: NERSC

What to change:

- **JetSet scans:** in `workflow/Snakefile`, `WORKSPACE` (top of the file) and the JetSet file paths
  (`/sdf/scratch/...`). Or pass `datamodule.data_path` and `output_dir` on the command line, as in the examples.
- **Training defaults:** `configs/train.yaml` (`output_dir`), `configs/datamodule/atlas_mappable.yaml` and
  `atlas_iterable.yaml` (`data_path`).
- **COCOA tracks and topoclusters:** `configs/datamodule/cocoa_{tracks,topos,truthpart}.yaml` (`data_dir`,
  scaler path) and `COCOA_DATA_DIR` in `workflow/Snakefile`.
- **Calorimeter cells, regression and figures:** copy `workflow/vendored/config.yaml` and fill in its `<...>`
  placeholders.
- **Figure scripts:** `scripts/plot_design_rules.py` (`CELLS["dir"]`, the cell runs),
  `scripts/codebook_vqvae_utilization.py` (`DATA_PATH`), `scripts/compare_roc.py`.
- **Classifiers:** `configs/datamodule/{atlas_iterable_multi,token_npz}.yaml`, `configs/model/token_classifier.yaml`.
- **Not needed for the paper commands:** `workflow/preprocessing.smk`, `scripts/{get_preprocessing,simple_demo,
  build_hep4m_preprocessed,export_cocoa_tokens_hep4m,fit_cocoa_truthpart_scaler}.py`, `scripts/pflow_*`,
  `scripts/copy_tokenizers_to_nersc.sh`, `tests/test_positions_out.py`.

## Preprocessing

```bash
pixi run python scripts/get_preprocessing.py --file_path mc-flavtag-ttbar-medium.h5 \
    --cst_mode log_standard --jet_mode log_standard --output_dir resources
```

This fits the feature scalers and writes `resources/cst_quantiles.joblib` (track features) and
`resources/jet_quantiles.joblib` (jet features), which the training configs read.

## Training a tokenizer

This is exactly the Table 1 JetSet configuration:

```bash
pixi run python scripts/train.py model=transformer_vqvae datamodule=atlas_mappable callbacks=encode \
    datamodule.data_path=mc-flavtag-ttbar-medium.h5 datamodule.num_jets=10000000 \
    model.encoder.model.d_model=256 model.encoder.model.n_heads=8 model.encoder.model.num_layers=6 \
    model.encoder.model.dim_feedforward=1024 model.decoder.model.d_model=256 model.decoder.model.n_heads=8 \
    model.decoder.model.num_layers=6 model.decoder.model.dim_feedforward=1024 \
    model.codebook_size=32768 model.codebook_dim=8 model.num_quantizers=4 +model.commitment_weight=1.0 \
    model.learning_rate=0.0003 model.optimizer.lr=0.0003 \
    trainer.max_epochs=200 trainer.gradient_clip_val=1.0 +trainer.num_sanity_val_steps=0 \
    trainer.check_val_every_n_epoch=1 trainer.val_check_interval=1.0 trainer.limit_val_batches=1.0 \
    callbacks.reconstruction_monitor.log_every_n_epochs=5 +callbacks.reconstruction_monitor.max_batches=100 \
    datamodule.transforms.preprocess.cst_fn.filename=$PWD/resources/cst_quantiles.joblib \
    datamodule.transforms.preprocess.jet_fn.filename=$PWD/resources/jet_quantiles.joblib \
    output_dir=$PWD/results project_name=vqvae_scan network_name=my_tokenizer
```

The configuration in detail:

- **Data:** 10M jets are loaded from the medium file and split 70/15/15, so the tokenizer trains on 7M jets.
- **Training:** batch size 1024, learning rate 3·10⁻⁴, seed 42, up to 200 epochs.
- **Early stopping and checkpoint:** training stops after 20 epochs without improvement in the validation loss
  (from `callbacks=encode`), and `checkpoints/best.ckpt` is the epoch with the lowest validation loss.

The three main hyperparameters are `model.codebook_size` (K), `model.codebook_dim` (d) and `model.num_quantizers` (Q).

## Example: the rate-distortion (Pareto) plot, Fig. 1

The scan runs through `workflow/Snakefile`, which submits one SLURM job per tokenizer with `sbatch`. Set the
account and partition with `--config slurm_account=... slurm_partition=...`.

1. **Train the paper's JetSet grid.** This is K ∈ {1024, 8192, 16384, 32768}, d ∈ {2, 4, 8, 16} and Q ∈ {3, 4},
   plus Q ∈ {2, 6, 8} at (32768, 8). Every point uses the command above.
   ```bash
   pixi run snakemake -s workflow/Snakefile all_paper_scan --cores 40
   ```
2. **Evaluate every run on the test set.** This writes `test_nofastpath/` in each run directory.
   ```bash
   pixi run snakemake -s workflow/Snakefile all_paper_scan_test --cores 40
   ```
3. **Make the plot.** This writes `rate_distortion_pareto_with_cells_binning.pdf` (Fig. 1), `design_rules.pdf`
   (Fig. 4) and the MAE version (Fig. E.10). Modalities with no runs are skipped.
   ```bash
   pixi run python scripts/plot_design_rules.py --e100 --outdir plots/design_rules
   ```

"nofp" means the test ran with PyTorch's fused-attention fast path off and fp32 matmuls. The tokenizer numbers
use it: Table 1, Figs. 1, 4, C.8, E.10 and Table C.4.

## Calorimeter cells

These use `workflow/vendored/*.smk` with your copy of `workflow/vendored/config.yaml`:

```bash
# Cell inputs from the COCOA ROOT files: the 2M training set, the 20M set and the test set
pixi run snakemake -s workflow/vendored/cells_preprocess.smk all_cells_20M_sparse --configfile my_paths.yaml --cores 8
# Cell scan: train and evaluate every (K, d, Q) point (includes Q = 24, 32 runs not shown in the paper)
pixi run snakemake -s workflow/vendored/cells_frontier_2M.smk all_frontier --configfile my_paths.yaml --cores 4
# Table 1 cells row: the (4096, 16, 16) run only
pixi run snakemake -s workflow/vendored/cells_frontier_2M.smk --configfile my_paths.yaml --cores 1 \
    <cells_results>/cells_scan_2M_a0.4_frontier/transformer/cb4096_cd16_nq16_a0.4_large/test_metrics.json
```

- **Table 1 cells row:** in that `test_metrics.json`, the resolution is `energy_resolution_iqr` and the
  utilization is `codebook_util_avg`.
- **Figures:** Fig. E.12 is rule `figE12` and Fig. D.9c is rule `figD9c` in `workflow/vendored/paper_figures.smk`.
- **Mean active pixels (Fig. E.12 caption):** the means of `evt_true_cnt` and `evt_pred_cnt` in
  `cell_diagnostics/diagnostics_arrays.npz`.

## Other Snakemake rules

Training and evaluation (`workflow/Snakefile`):

- `all_nofp_test`: evaluates the JetSet, track and topocluster scans (Table 1, Figs. 1, 4).
- `all_datascan_nofp`: dataset-size scans (Table C.4).
- `compare_roc_scan`: flavour-tagging classifiers (Fig. 6a, Table 2).
- `workflow/vendored/{track,topo}_token_regression.smk all`: jet-pT regression on tokens (Fig. 5 top).

Figures and tables (`workflow/vendored/paper_figures.smk`), for example:

```bash
pixi run snakemake -s workflow/vendored/paper_figures.smk table2 --configfile my_paths.yaml --cores 1
```

- `fig1_fig4_figE10`: Figs. 1, 4, E.10.
- `fig5_fig6b`: Fig. 5 and Fig. 6b. The particle-flow points in Fig. 5 (bottom) come from an external pipeline,
  fed with tokens exported by `scripts/build_hep4m_preprocessed.py`.
- `fig6a`: Fig. 6a.
- `table2`: Table 2.
- `figC8`: Fig. C.8.
- `tableC4`: Table C.4.
- `figD9ab_d`: Figs. D.9a, b, d.
- `figD9c`: Fig. D.9c.
- `figE12`: Fig. E.12.
- Fig. E.11: request the output files `selected_2d_correlation/cocoa_tracks_cb4096_cd8_nq3.pdf` and
  `selected_2d_correlation/cocoa_topos_cb4096_cd8_nq6.pdf` (the rule `figE11` needs a concrete target).

Not produced by this code: Figs. 2, 3 and B.7 and Table A.3 (illustrations), and Fig. 6c (its input is no longer available).

## Code layout

`src/heptokens` is the original code. It produced the JetSet, track and topocluster results.
`src/heptokens/vendored` is a copy of the later, refactored library code, used for the calorimeter cells and the
jet-pT regression.

## Citation

```bibtex
@article{precision_tokenization,
  title  = {Precision Tokenization for Multi-Modal Data in High Energy Physics},
  author = {Krupa, J. and Klein, S. and Kakati, N. and Murnane, D. and Hashemi, B. and Gross, E. and Heinrich, L. and Kagan, M.},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
}
```

## License

To be added.
