# Precision Tokenization for Multi-Modal Data in High Energy Physics

This repository contains the code that reproduces the results of

> J. Krupa, S. Klein, N. Kakati, D. Murnane, B. Hashemi, E. Gross, L. Heinrich, M. Kagan,
> *Precision Tokenization for Multi-Modal Data in High Energy Physics*, `<arXiv/DOI placeholder>`.

**This is a frozen release, not a development repository.** The code has since been refactored into the
[heptokens library](https://github.com/treasure-AmSC/heptokens) (conda package: `<conda package placeholder>`). New
work, and any development, should use the library rather than this repository.

## Setup

### Environment

The environment is managed with [pixi](https://pixi.sh) and pinned in `pixi.lock`:

```bash
pixi install
export WANDB_MODE=disabled   # training logs to Weights & Biases by default
```

All commands below are run from the repository root with `pixi run ...`.

### Data

- **JetSet** (ATLAS tt̄ simulation for ML-based jet flavour tagging), CERN Open Data Portal,
  [doi:10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P). The three HDF5 files
  `mc-flavtag-ttbar-{small,medium,large}.h5` (about 200M jets in total).
- **COCOA** single-jet samples ([Charkin-Gorbulin et al., Mach. Learn. Sci. Tech. 4 (2023) 035042](https://doi.org/10.1088/2632-2153/acf186)):
  ROOT files in `train_topup2_20M_cells256/`, `val_100K_cells256/` and `test_1M_cells256_isInfFalse/`.
  `<COCOA data location placeholder>`. The calorimeter-cell inputs are derived from these files by
  `workflow/vendored/cells_preprocess.smk`.

### Checkpoints

Trained tokenizers, classifiers and regressors, the preprocessing scalers (`resources/*.joblib`) and the stored test
metrics are available at `<checkpoint archive placeholder>`. Unpack them so that the set-based runs sit under
`results/` (`results/vqvae_scan/`, `results/cocoa_scan/`, `results/binning_baseline/`, ...), and point the
`cells_results` and `regression_results` keys below at the cell and regression runs.

### Site-specific paths

The workflows in `workflow/vendored/` read every path from `workflow/vendored/config.yaml`. Copy it, fill in the
`<...>` placeholders, and pass it with `--configfile my_paths.yaml`:

| Key | Content |
|---|---|
| `cocoa_root` | COCOA ROOT directories (see Data) |
| `cells_h5_dir`, `cells_h5_2M_dir`, `cells_sparse_dir` | preprocessed calorimeter-cell data (20M, 2M, sparse) |
| `cells_results` | calorimeter-cell runs |
| `a_results` | set-based runs (normally `results`) |
| `regression_results` | jet-pT regression tokens, regressors and evaluations |
| `figures_dir` | output directory for figures and tables |
| `slurm_gpu`, `slurm_cpu` | job prefix; `local` runs jobs in the foreground, otherwise e.g. `sbatch -p <partition> ...` |

Older files hard-code the paths of the cluster the paper was produced on and must be edited to run elsewhere. Path
roots: `/sdf/data/atlas/u/jkrupa/`, `/sdf/scratch/users/j/jkrupa/`, `/sdf/scratch/users/s/samklein/`,
`/sdf/group/mli/samklein/`, `/fs/ddn/sdf/group/atlas/d/jkrupa/`, `/lscratch/`, `/global/cfs/`. Files:

- `workflow/Snakefile`, `workflow/preprocessing.smk`
- `configs/train.yaml`, `configs/model/token_classifier.yaml`,
  `configs/datamodule/{atlas_iterable,atlas_iterable_multi,atlas_mappable,cocoa_topos,cocoa_tracks,cocoa_truthpart,token_npz}.yaml`
- `scripts/plot_design_rules.py` (the cell-run directory, `CELLS["dir"]`), `scripts/compare_roc.py`,
  `scripts/codebook_vqvae_utilization.py`, `scripts/get_preprocessing.py`, `scripts/build_hep4m_preprocessed.py`,
  `scripts/export_cocoa_tokens_hep4m.py`, `scripts/fit_cocoa_truthpart_scaler.py`, `scripts/simple_demo.py`,
  `scripts/pflow_*.{py,sh}`, `scripts/copy_tokenizers_to_nersc.sh`
- `src/heptokens/data/token_npz.py`, `tests/test_positions_out.py`

## Reproducing the paper

Figures and tables are built by `workflow/vendored/paper_figures.smk`:

```bash
pixi run snakemake -s workflow/vendored/paper_figures.smk --configfile my_paths.yaml --cores 1 <target>
```

`all_figures` builds every target. In the table, `<F>` is `figures_dir` and `pf:` marks a target of
`paper_figures.smk`. Runtimes are approximate, on CPU; the long CPU times were measured on a shared, heavily loaded
node and are much shorter on a GPU. Figures built from stored metrics need only the checkpoint archive (no inference).

| Paper | What it shows | Command | Output | Runtime | GPU |
|---|---|---|---|---|---|
| Fig. 1 | Jet resolution vs information fraction, all modalities, with fixed-binning baselines | `pf:fig1_fig4_figE10` | `<F>/design_rules/rate_distortion_pareto_with_cells_binning.pdf` | < 1 min | no |
| Fig. 2 | Method overview | – | – | – | – |
| Fig. 3 | Calorimeter-cell patching | illustrative; not produced by this code | – | – | – |
| Table 1 | Selected tokenizer per modality: resolution and utilization | resolution: `pixi run snakemake -s workflow/Snakefile all_nofp_test` (set-based), `workflow/vendored/cells_frontier_2M.smk <cells_results>/cells_scan_2M_a0.4_frontier/transformer/cb4096_cd16_nq16_a0.4_large/test_metrics.json` (cells); utilization: `scripts/codebook_vqvae_utilization.py --run_dirs <JetSet run> --num_jets 50000 --output <csv>`, `scripts/cocoa_utilization_nofp.py compute` | `<run>/test_nofastpath/**/test_metrics.npz` (IQR/median of `pt_ratio`), cells `<run>/test_metrics.json`; utilization CSVs in `results/codebook_utilization/` | tracks 5 min, topoclusters 40 min, JetSet 3 h, cells 4 h, utilization 20 min–4 h | recommended |
| Fig. 4 | Design rules (a)–(d) | `pf:fig1_fig4_figE10` | `<F>/design_rules/design_rules.pdf` | < 1 min | no |
| Fig. 5 top | Regressed jet-pT resolution vs information fraction (tracks, topoclusters) | `pf:fig5_fig6b`; regressors: `pixi run snakemake -s workflow/vendored/{track,topo}_token_regression.smk all` | `<F>/rate_distortion/rate_distortion_cocoa.pdf` | plot < 1 min; regression scan: hours | regression training: yes |
| Fig. 5 bottom | Particle-flow jet-pT resolution | produced in an external pipeline (nanoHEP) from tokens exported by this repository, see [Particle flow](#particle-flow-fig-5-bottom); plotted by `pf:fig5_fig6b` | `<F>/rate_distortion/rate_distortion_cocoa.pdf` | plot < 1 min | external pipeline: yes |
| Fig. 6a | b-tagging ROC curves, tokenized vs continuous classifiers | `pf:fig6a` (from stored test predictions) | `<F>/roc/token_clf_scan_vs_refs_btag_roc.pdf` | 35 min | no |
| Fig. 6b | Light-jet rejection vs information fraction | `pf:fig5_fig6b` | `<F>/rate_distortion/rate_distortion_jetset.pdf` | < 1 min | no |
| Fig. 6c | t-SNE of the JetSet codebooks | not reproducible here: its input token export is no longer available | – | – | – |
| Table 2 | Background rejection at 50% and 70% b-jet efficiency | `pf:table2` (from stored test predictions) | `<F>/tables/rejection_table.txt` | 3 min | no |
| Table A.3 | Dataset summary | not produced by code | – | – | – |
| Fig. B.7 | Illustration | illustrative; not produced by this code | – | – | – |
| Fig. C.8 | MLP vs Transformer backbones, 1M and 10M training jets | `pf:figC8` | `<F>/mlp_vs_transformer/mlp_vs_transformer.pdf` | < 1 min | no |
| Table C.4 | Jet resolution vs training-set size | `pf:tableC4` | `<F>/tables/table_c4.csv` | < 1 min | no |
| Fig. D.9a, b, d | Codebook utilization heatmaps: COCOA tracks, topoclusters, JetSet | `pixi run python scripts/cocoa_utilization_nofp.py compute`, then `... plot` | `results/cocoa_scan/plots_nofp/{tracks,topos}/cocoa_utilization_heatmap.pdf`, `results/vqvae_scan/plots_nofp/jetset_utilization_heatmap.pdf` | compute: hours on CPU; plot < 1 min | compute: recommended |
| Fig. D.9c | Codebook utilization, calorimeter cells | `pf:figD9c` | `<F>/cell_utilization/plots/util_vs_nq_transformer_large.pdf` | < 1 min | no |
| Fig. E.10 | As Fig. 1, with normalized MAE | `pf:fig1_fig4_figE10` | `<F>/design_rules/rate_distortion_pareto_mae_mean_with_binning.pdf` | < 1 min | no |
| Fig. E.11 | Feature correlations and residuals, selected track and topocluster tokenizers | `pf:` the two output files | `<F>/selected_2d_correlation/cocoa_{tracks_cb4096_cd8_nq3,topos_cb4096_cd8_nq6}.pdf` | < 1 min | no |
| Fig. E.12 | Calorimeter-cell energy and occupancy reconstruction | `pf:figE12` | `<F>/cell_diagnostics/cell_diagnostics.pdf` | 45 min (50k events) | recommended |

The published Fig. D.9a, b show Q = 2, 3, 4. The plot step adds a Q = 6 panel if
`results/codebook_utilization/cocoa_utilization_nofp_q6.csv` (from `cocoa_utilization_nofp.py compute_q6`) exists.

Fixed-binning baselines for Figs. 1 and E.10 (`pf:` targets): `<a_results>/binning_baseline/{tracks,topos,jetset}/binning_uniform_K<K>/test_metrics/test_metrics.npz`
and `<a_results>/binning_baseline/cells/binning_uniform_K<K>/test_metrics.json`.

### Training

Training needs a GPU. The commands below retrain the stored checkpoints:

| Runs | Command |
|---|---|
| JetSet, COCOA track and topocluster scans; fast-path-off tests | `pixi run snakemake -s workflow/Snakefile all_nofp_test all_design_e100_nofp --profile profiles/s3df` |
| Dataset-size scans (Table C.4) | `pixi run snakemake -s workflow/Snakefile all_datascan_nofp --profile profiles/s3df` |
| Classifiers (Fig. 6, Table 2) | `pixi run snakemake -s workflow/Snakefile compare_roc_scan --profile profiles/s3df` |
| Calorimeter-cell preprocessing | `pixi run snakemake -s workflow/vendored/cells_preprocess.smk all_cells_20M_sparse --configfile my_paths.yaml` |
| Calorimeter-cell scan, 2M events | `pixi run snakemake -s workflow/vendored/cells_frontier_2M.smk all_frontier --configfile my_paths.yaml` |
| Calorimeter cells, 20M events (Table C.4) | `pixi run snakemake -s workflow/vendored/cells_20M.smk full_qscan_eval --configfile my_paths.yaml` |
| Jet-pT regression (Fig. 5 top) | `pixi run snakemake -s workflow/vendored/{track,topo}_token_regression.smk all --configfile my_paths.yaml` |

`profiles/s3df` is the SLURM profile of the cluster the paper was produced on; adapt it, or use `--cores N` to run
locally.

### Particle flow (Fig. 5, bottom)

The particle-flow model was trained and evaluated outside this repository, with nanoHEP. Its inputs are token stores
exported by this repository from the tokenizers `results/cocoa_scan/{tracks,topos}_e100/cocoa_{tracks,topos}{,_posout}_cb{256,1024,4096}_cd8_nq4_lr0.001_e100`
and the truth-particle tokenizer `results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001`:

```bash
pixi run python scripts/build_hep4m_preprocessed.py --data_dir <cocoa_root> \
    --split_dir train_topup2_20M_cells256 --split_name train --output_dir <token store> \
    --track_ckpt <track tokenizer>/checkpoints/best.ckpt \
    --topo_ckpt <topocluster tokenizer>/checkpoints/best.ckpt \
    --truthpart_ckpt results/cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt
```

Repeat for `val_100K_cells256` / `val` and `test_1M_cells256_isInfFalse` / `test`. This script writes positions-in
token stores. The positions-out stores, which append three position tokens per object from a fixed binning with
1024 bins, and the streaming export of the full training set were produced with an adapted version of the script in
the external pipeline. The plotted particle-flow values are literals in `scripts/vendored/plot_rate_distortion.py`.

## Repository layout

- `src/heptokens/`: the original code. It produced the set-based results (JetSet, COCOA tracks and topoclusters):
  tokenizers, classifiers, data modules and callbacks.
- `src/heptokens/vendored/`: a copy of the later, refactored library code. It is used only for the
  calorimeter-cell tokenizer and the jet-pT regression; the scripts in `scripts/vendored/`, the configs
  `configs/train_cell.yaml` and `configs/*/hep4m_cell*.yaml`, and the workflows in `workflow/vendored/` use it.
  Checkpoints written by that code store their model classes under `heptokens.models`; the vendored scripts load them
  inside `heptokens.vendored.b_module_names()`, which maps those names to the vendored modules.
- `scripts/`, `configs/`, `workflow/`: entry points, Hydra configs and Snakemake workflows for the set-based results.
  `scripts/train.py` trains every tokenizer (`-cn train_cell` for the calorimeter cells).

## Citation

```bibtex
@article{<citation key placeholder>,
  title   = {Precision Tokenization for Multi-Modal Data in High Energy Physics},
  author  = {Krupa, J. and Klein, S. and Kakati, N. and Murnane, D. and Hashemi, B. and Gross, E. and Heinrich, L. and Kagan, M.},
  journal = {<placeholder>},
  year    = {<placeholder>},
  eprint  = {<placeholder>},
}
```

## License

`<license placeholder>`
