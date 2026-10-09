# Precision Tokenization for Multi-Modal Data in High Energy Physics

Code for the paper *Precision Tokenization for Multi-Modal Data in High Energy Physics* (arXiv:XXXX.XXXXX).
This is a frozen release; for new work use the [heptokens library](https://github.com/treasure-AmSC/heptokens).

- **Data:** JetSet is public ([doi:10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P)). The COCOA samples are not public yet.
- **Trained models:** not hosted yet.

## Setup

```bash
pixi install
export WANDB_MODE=disabled   # training logs to Weights & Biases otherwise
```

## Preprocessing

Fit the JetSet feature scalers:

```bash
pixi run python scripts/get_preprocessing.py --file_path mc-flavtag-ttbar-medium.h5 \
    --cst_mode log_standard --jet_mode log_standard --output_dir resources
```

This writes `resources/cst_quantiles.joblib` (track features) and `resources/jet_quantiles.joblib` (jet features).
The training configs read these files.

Build the calorimeter-cell inputs from the COCOA ROOT files:

```bash
pixi run snakemake -s workflow/vendored/cells_preprocess.smk all_cells_20M_sparse --configfile my_paths.yaml --cores 8
```

This writes the cell patches as dense HDF5 files and a sparse copy, split into train, val and test.

## Training a tokenizer

```bash
pixi run python scripts/train.py model=transformer_vqvae datamodule=atlas_mappable callbacks=encode \
    datamodule.data_path=mc-flavtag-ttbar-medium.h5 datamodule.num_jets=10000000 \
    datamodule.transforms.preprocess.cst_fn.filename=$PWD/resources/cst_quantiles.joblib \
    datamodule.transforms.preprocess.jet_fn.filename=$PWD/resources/jet_quantiles.joblib \
    model.codebook_size=32768 model.codebook_dim=8 model.num_quantizers=4 \
    trainer.max_epochs=200 output_dir=$PWD/results project_name=my_scan network_name=my_tokenizer
```

The three main hyperparameters are `model.codebook_size` (K), `model.codebook_dim` (d) and `model.num_quantizers` (Q).
The checkpoint and the full config go to `results/my_scan/my_tokenizer/`.

## Example: the rate-distortion (Pareto) plot, Fig. 1

The scans are Snakemake workflows that submit one job per configuration. `workflow/Snakefile` submits
SLURM jobs through `sbatch`, set with `--config slurm_account=... slurm_partition=...`.
`workflow/vendored/*.smk` read their paths and job settings from a copy of `workflow/vendored/config.yaml`
(`my_paths.yaml` below).

1. **Choose the grid.** For COCOA tracks, set `cocoa_codebook_sizes`, `cocoa_codebook_dims` and
   `cocoa_num_quantizers` near the top of the COCOA section of `workflow/Snakefile`. Set the same grid in
   `nofp_tracks` further down, which defines the evaluation.
2. **Train the scan:**
   ```bash
   pixi run snakemake -s workflow/Snakefile all_cocoa_tracks --cores 20
   ```
3. **Evaluate every run on the test set:**
   ```bash
   pixi run snakemake -s workflow/Snakefile all_nofp_test --cores 20
   ```
   This writes `test_nofastpath/` in each run directory.
4. **Make the plot:**
   ```bash
   pixi run snakemake -s workflow/vendored/paper_figures.smk fig1_fig4_figE10 --configfile my_paths.yaml --cores 1
   ```
   This runs `scripts/plot_design_rules.py --e100`. It writes `design_rules/rate_distortion_pareto_with_cells_binning.pdf`
   (Fig. 1), `design_rules.pdf` (Fig. 4) and the MAE version (Fig. E.10).

## Other Snakemake rules

Training and evaluation:

- `workflow/Snakefile all_nofp_test`: evaluates the JetSet, track and topocluster scans (Table 1, Figs. 1, 4).
- `workflow/Snakefile all_datascan_nofp`: dataset-size scans (Table C.4).
- `workflow/Snakefile compare_roc_scan`: flavour-tagging classifiers (Fig. 6a, Table 2).
- `workflow/vendored/cells_frontier_2M.smk all_frontier`: calorimeter-cell scan (Table 1, Figs. 1, D.9c).
- `workflow/vendored/cells_20M.smk full_qscan_eval`: calorimeter cells at 20M events (Table C.4).
- `workflow/vendored/{track,topo}_token_regression.smk all`: jet-pT regression on tokens (Fig. 5 top).

Figures and tables (`workflow/vendored/paper_figures.smk`):

- `fig1_fig4_figE10`: Figs. 1, 4, E.10.
- `fig5_fig6b`: Fig. 5 and Fig. 6b. The particle-flow points in Fig. 5 (bottom) come from an external pipeline,
  fed with tokens exported by `scripts/build_hep4m_preprocessed.py`.
- `fig6a`: Fig. 6a.
- `table2`: Table 2.
- `figC8`: Fig. C.8.
- `tableC4`: Table C.4.
- `figD9ab_d`: Figs. D.9a, b, d.
- `figD9c`: Fig. D.9c.
- Fig. E.11: request the output files `selected_2d_correlation/cocoa_tracks_cb4096_cd8_nq3.pdf` and
  `selected_2d_correlation/cocoa_topos_cb4096_cd8_nq6.pdf` (the rule `figE11` needs a concrete target).
- `figE12`: Fig. E.12.
- `all_figures`: all of the above.

Not produced by this code: Figs. 2, 3 and B.7 and Table A.3 (illustrations), and Fig. 6c (its input is no longer available).

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
