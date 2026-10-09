# heptokens
High-energy physics quantization and tokenization framework.

[![pytorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra-89b8cd?logo=hydra&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

This repository is built around Hydra configs, Lightning training loops, and Pixi-managed environments.

## Quick Start
1. Install [Pixi](https://pixi.sh/latest/installation/).
2. Clone the repo and `cd` to the top-level directory.
3. Create/use the environment:

```bash
pixi shell
```

or run commands without entering a shell:

```bash
pixi run python scripts/train.py ...
```

To see a simple example that demonstrates the low level API take a look at```scripts/simple_demo.py```

## Data

The initial training datasets for this project have been taken from here: https://opendata.cern.ch/record/93940

## Repository Layout
- `scripts/`: entry points for training, preprocessing, plotting, and profiling.
- `configs/`: Hydra config groups for models, callbacks, datamodules, and training defaults.
- `src/heptokens/`: library code (models, dataloaders, callbacks, utilities).
- `workflow/`: Snakemake workflows for larger experiment grids.
- `profiles/`: Snakemake executor profiles (`local`, `s3df`/Slurm).

## Data and Preprocessing
Preprocessing files are required before normal training runs.

The training config expects:
- an input HDF5 jet dataset via `datamodule.data_path`
- preprocessing transformers (`cst_fn`, `jet_fn`) for `preprocess_batch`

Generate transformers with:

```bash
pixi run python scripts/get_preprocessing.py \
  --file_path /path/to/dataset.h5 \
  --num_jets 1000000 \
  --cst_mode log_quantile \
  --jet_mode log_quantile \
  --output_dir /path/to/resources
```

Then pass transformer files into training:

```bash
datamodule.transforms.preprocess.cst_fn.filename=/path/to/cst_quantiles.joblib
datamodule.transforms.preprocess.jet_fn.filename=/path/to/jet_quantiles.joblib
```

Important path note:
- `configs/hydra/default.yaml` sets `hydra.job.chdir=true`, so relative paths can be resolved from the run directory (`<output_dir>/<project>/<network>`), not the repo root.
- Prefer absolute transformer paths in CLI overrides.

## First Run Sequence
1. Build preprocessing transformers with `scripts/get_preprocessing.py`.
2. Launch training with transformer paths passed via Hydra overrides.

## Core Training Pattern
All training is launched via `scripts/train.py` + Hydra overrides:

```bash
pixi run python scripts/train.py \
  model=vqvae \
  callbacks=encode \
  datamodule=atlas_mappable \
  project_name=my_project \
  network_name=my_run \
  output_dir=/path/to/results
```

Output structure is:

`<output_dir>/<project_name>/<network_name>/`

and includes checkpoints and `full_config.yaml` for reproducibility/resume.

## Common Runs
Train tokenizer (VQ-VAE):

```bash
pixi run python scripts/train.py \
  model=vqvae callbacks=encode \
  datamodule.num_jets=100000 \
  trainer.max_epochs=30
```

Train classifier on raw features:

```bash
pixi run python scripts/train.py \
  model=feature_classifier callbacks=classify \
  trainer.max_epochs=30
```

Train classifier on learned tokens:

```bash
pixi run python scripts/train.py \
  model=token_classifier callbacks=classify \
  model.tokenizer_ckpt=/path/to/tokenizer/checkpoints/last.ckpt \
  trainer.max_epochs=30
```

## W&B Logging
The default logger is `WandbLogger` in `configs/train.yaml`.

Useful overrides:

```bash
logger.entity=<your_username_or_team>
logger.project=<project_name>
logger.offline=true
```

## Resume Runs
Use `full_resume=true` only if the previous run saved `full_config.yaml` in the expected run directory.
This works best when the original run used an absolute `output_dir` and you reuse the same
`output_dir`, `project_name`, and `network_name`.
Also, ensure that `trainer.max_epochs` exceeds the current epoch.

```bash
pixi run python scripts/train.py \
  full_resume=true \
  ckpt_flag=last.ckpt \
  output_dir=/path/to/results \
  project_name=my_project \
  network_name=my_run
```

If `full_resume` cannot find the prior config, resume directly from a checkpoint path:

```bash
pixi run python scripts/train.py \
  full_resume=false \
  ckpt_path=/abs/path/to/checkpoints/last.ckpt \
  model=vqvae \
  callbacks=encode \
  output_dir=/path/to/results \
  project_name=my_project \
  network_name=my_run
```

## Snakemake Workflows
End-to-end experiment workflows are in:
- `workflow/Snakefile`
- `workflow/preprocessing.smk`

Run locally:

```bash
pixi run snakemake -s workflow/Snakefile --profile profiles/local
```

Run on Slurm:

```bash
pixi run snakemake -s workflow/Snakefile --profile profiles/s3df
```


## Hydra Reference
Main training entry point:
- `scripts/train.py`

Top-level config:
- `configs/train.yaml`

Config groups:
- `configs/model/*.yaml`
- `configs/callbacks/*.yaml`
- `configs/datamodule/*.yaml`
- `configs/hydra/default.yaml`

Hydra override pattern:
- choose config groups: `model=vqvae callbacks=encode datamodule=atlas_mappable`
- override scalar values: `trainer.max_epochs=30`
- use `+key=value` only when adding a new key that does not already exist

Common high-value overrides:

```bash
output_dir=/path/to/results project_name=my_project network_name=my_run
logger.entity=<username_or_team>
trainer.max_epochs=30 trainer.check_val_every_n_epoch=1
datamodule.transforms.preprocess.cst_fn.filename=/path/to/cst_quantiles.joblib
datamodule.transforms.preprocess.jet_fn.filename=/path/to/jet_quantiles.joblib
```

## Reconstruction Callback Settings
For `callbacks=encode`, reconstruction monitoring is configured in `configs/callbacks/encode.yaml`. For `callbacks=classify`, monitoring is configured in `configs/callbacks/classify.yaml`.


## Troubleshooting
### DataLoader workers exit unexpectedly
- start with `datamodule.num_workers=0` to confirm worker issue
- set `datamodule.pin_memory=false` if worker shutdown/restart appears unstable
- set `trainer.reload_dataloaders_every_n_epochs=0` to avoid costly per-epoch dataloader rebuild

### Training becomes slow
- set `trainer.reload_dataloaders_every_n_epochs=0`
- tune `datamodule.num_workers`, `datamodule.batch_size`, `datamodule.pin_memory`
- use `scripts/profile_dataloader.py` to profile dataloading throughput

### W&B run lands under wrong account
- set `logger.entity=<username_or_team>`

### Debugger uses wrong Python interpreter
- use `.vscode/launch.json`
- set `"python": "${workspaceFolder}/.pixi/envs/default/bin/python"`

## Reproducing the paper

This section covers "Precision Tokenization for Multi-Modal Data in High Energy Physics". The JetSet, COCOA track and COCOA
topocluster results were produced with the code in this repository. The calorimeter-cell tokenizer and the jet-pT
regression were produced with a separate code base, now included here unchanged apart from imports and paths:

| Origin | Location here |
|---|---|
| `heptokens` @ `73cb6ae` (`models/{vq_vae,coders,transformer,utils,vq_layers}.py`, `data/{base,collation}.py`, `callbacks/recon.py`) | `src/heptokens/vendored/{models,data,callbacks}/` |
| `heptokens-cocoa` @ `cc20cbc` (`src/heptokens_cocoa/{data/hep4m_cell,data/cell_image_helper,data/cocoa_base,callbacks/cell_recon}.py`) | `src/heptokens/vendored/cocoa/` |
| `heptokens-cocoa` @ `cc20cbc` scripts (cells, regression, Fig. 5/6b) | `scripts/vendored/` |
| `heptokens-cocoa` @ `cc20cbc` cell configs | `configs/train_cell.yaml`, `configs/*/hep4m_cell*.yaml` |
| `heptokens-cocoa` @ `cc20cbc` workflows | `workflow/vendored/*.smk` |

The vendored model code is a different version of `heptokens.models` and is used only by the cell and regression
code. Checkpoints written by that code pickle their encoder/decoder as `heptokens.models.*`; the vendored scripts load
them inside `heptokens.vendored.b_module_names()`, which resolves those names to the vendored modules.

### Environment

```bash
pixi install                       # one environment for everything (pixi.lock)
export WANDB_MODE=disabled         # or log in to W&B; training logs to W&B by default
```

### Data

- JetSet (ATLAS open data, flavour tagging): doi:[10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P).
  The JetSet scalers `resources/cst_quantiles.joblib` and `resources/jet_quantiles.joblib` are archived with the checkpoints.
- COCOA single-jet samples ([Charkin-Gorbulin et al.](https://doi.org/10.1088/2632-2153/acf186)): the directories
  `train_topup2_20M_cells256/`, `val_100K_cells256/` and `test_1M_cells256_isInfFalse/` of ROOT files. <COCOA_DATA_LOCATION>
- Checkpoints, scan metric files and the cell test data: <ZENODO_DOI>. Unpack them so that the repo-A runs sit under
  `results/` (`results/vqvae_scan/...`, `results/cocoa_scan/...`, `results/binning_baseline/...`), and point
  `workflow/vendored/config.yaml` at the rest.

### Paths

New workflows read every path from `workflow/vendored/config.yaml` (placeholders `<...>`; or pass
`--configfile my_paths.yaml`). Older files still hard-code the S3DF locations and must be edited to run elsewhere.
Path roots: `/sdf/data/atlas/u/jkrupa/` (repo, results, 2M cell data), `/sdf/scratch/users/j/jkrupa/data/atlas/`
(JetSet HDF5), `/fs/ddn/sdf/group/atlas/d/jkrupa/` (COCOA ROOT files, cell results), `/sdf/group/mli/samklein/`
(old defaults). Files: `workflow/Snakefile`, `workflow/preprocessing.smk`, `configs/train.yaml`,
`configs/datamodule/{atlas_iterable,atlas_iterable_multi,atlas_mappable,cocoa_topos,cocoa_tracks,cocoa_truthpart,token_npz}.yaml`,
`configs/model/token_classifier.yaml`, `scripts/plot_design_rules.py` (`CELLS["dir"]`, the cell frontier),
`scripts/codebook_vqvae_utilization.py`, `scripts/compare_roc.py`, `scripts/get_preprocessing.py`,
`scripts/build_hep4m_preprocessed.py`, `scripts/export_cocoa_tokens_hep4m.py`, `scripts/fit_cocoa_truthpart_scaler.py`,
`scripts/pflow_*.{py,sh}`, `scripts/copy_tokenizers_to_nersc.sh`, `scripts/simple_demo.py`, `tests/test_positions_out.py`.

### Training and evaluation

| What | Command |
|---|---|
| JetSet / COCOA track / topocluster scans and fast-path-off tests | `pixi run snakemake -s workflow/Snakefile all_nofp_test all_design_e100_nofp --profile profiles/s3df` |
| Dataset-size scans (Table C.4) | `pixi run snakemake -s workflow/Snakefile all_datascan_nofp --profile profiles/s3df` |
| Classifiers (Fig. 6, Table 2) | `pixi run snakemake -s workflow/Snakefile compare_roc_scan --profile profiles/s3df` |
| Cell preprocessing (2M and 20M sets) | `pixi run snakemake -s workflow/vendored/cells_preprocess.smk all_cells_20M_sparse --cores 100` |
| Cell frontier, 2M events (Fig. 1, Table 1, Fig. D.9c) | `pixi run snakemake -s workflow/vendored/cells_frontier_2M.smk all_frontier --cores 24` |
| Cells, 20M events (Table C.4) | `pixi run snakemake -s workflow/vendored/cells_20M.smk full_qscan_eval --cores 2` |
| Jet-pT regression (Fig. 5 top) | `pixi run snakemake -s workflow/vendored/{track,topo}_token_regression.smk all --cores 16` |

Set `slurm_gpu` / `slurm_cpu` in `workflow/vendored/config.yaml` to submit the vendored workflows to SLURM.

### Figures and tables

All commands below are `pixi run snakemake -s workflow/vendored/paper_figures.smk <target> --cores 1`, where `<F>` is
`figures_dir`. `all_figures` builds everything that has a script.

| Paper | Target / command | Notes |
|---|---|---|
| Fig. 1 | `<F>/design_rules/rate_distortion_pareto_with_cells_binning.pdf` | `scripts/plot_design_rules.py --e100`; binning baselines: rules `binning_baseline`, `binning_baseline_cells` |
| Fig. 2 | – | drawing, no code |
| Fig. 3 | – | illustrative; not reproduced by this code |
| Table 1 | metric files | JetSet/tracks/topos: `<run>/test_nofastpath/**/test_metrics.npz` (IQR/median of `pt_ratio`) and `results/codebook_utilization/{jetset_utilization,cocoa_utilization_nofp,cocoa_utilization_nofp_q6}.csv`; cells: `test_metrics.json` (`energy_resolution_iqr`, `codebook_util_avg`) |
| Fig. 4 | `<F>/design_rules/design_rules.pdf` | same command as Fig. 1 |
| Fig. 5 | `<F>/rate_distortion/rate_distortion_cocoa.pdf` | `scripts/vendored/plot_rate_distortion.py`; values are literals (regression: `<regression_results>/*_regression_results.csv`; particle flow: nanoHEP, see below) |
| Fig. 6a | `<F>/roc/token_clf_scan_vs_refs_btag_roc.pdf` | `scripts/compare_roc.py` (as rule `compare_roc_scan`) |
| Fig. 6b | `<F>/rate_distortion/rate_distortion_jetset.pdf` | literals in `scripts/vendored/plot_rate_distortion.py` |
| Fig. 6c | – | not reproduced: its input token export is no longer available |
| Table 2 | `<F>/tables/rejection_table.txt` | `scripts/rejection_table.py` |
| Table A.3 | – | no code |
| Fig. B.7 | – | illustrative; not reproduced by this code |
| Fig. C.8 | `<F>/mlp_vs_transformer/mlp_vs_transformer.pdf` | `scripts/compare_mlp_transformer.py` (reads `test/`, fast path on) |
| Table C.4 | metric files | `results/cocoa_scan/{tracks,topos}_datascan/*/test_nofastpath`, JetSet `*_nj1000000` / `*_nj10000000`, cells `cells_scan_2M_a0.4_frontier` and `cells_20M/full`; no table script |
| Fig. D.9a, b, d | `pixi run python scripts/cocoa_utilization_nofp.py compute && ... plot` | fixed outputs under `results/`; the published (a), (b) have Q = 2, 3, 4 panels (no `cocoa_utilization_nofp_q6.csv`) |
| Fig. D.9c | `<F>/cell_utilization/plots/util_vs_nq_transformer_large.pdf` | `scripts/vendored/make_cell_utilization.py` |
| Fig. E.10 | `<F>/design_rules/rate_distortion_pareto_mae_mean_with_binning.pdf` | same command as Fig. 1 |
| Fig. E.11 | `<F>/selected_2d_correlation/cocoa_{tracks_cb4096_cd8_nq3,topos_cb4096_cd8_nq6}.pdf` | `scripts/plot_2d_correlation.py` |
| Fig. E.12 | `<F>/cell_diagnostics/cell_diagnostics.pdf` | `scripts/vendored/make_cell_diagnostics.py` |

### Particle flow (Fig. 5, bottom)

The particle-flow model was trained and evaluated outside this repository, with nanoHEP (`FM-for-HEP/hep-any2any`)
at NERSC, following `docs/nanohep_pflow_handoff.md` on branch `feature/cocoa-dataloader`. Its inputs from this
repository are the tokenizers `results/cocoa_scan/{tracks,topos}_e100/cocoa_{tracks,topos}{,_posout}_cb{256,1024,4096}_cd8_nq4_lr0.001_e100`
and `results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001`, the scalers in `resources/`, and the token
stores built with `scripts/build_hep4m_preprocessed.py` (adapted at NERSC as described in the runbook). The plotted
values are literals in `scripts/vendored/plot_rate_distortion.py`.

## Resources
- [Pixi documentation](https://pixi.sh/)
- [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/)
- [Reproducible Machine Learning Workflows for Scientists, Matthew Feickert, 2025](https://carpentries-incubator.github.io/reproducible-ml-workflows/)
- [Good ML project structure](https://github.com/mattcleigh/JetSSL-Lite/tree/master)

## Contributors
Jeffrey Krupa, Samuel Klein, Michael Kagan
