# Precision Tokenization for Multi-Modal Data in High Energy Physics

Code for the paper *Precision Tokenization for Multi-Modal Data in High Energy Physics* (arXiv:XXXX.XXXXX).
This is a frozen release; for new work use the [heptokens library](https://github.com/treasure-AmSC/heptokens).

- **Data:** JetSet is public ([doi:10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P)). The COCOA samples are not public yet.
- **Trained models:** not hosted yet.

The examples below use JetSet.

## Setup

```bash
pixi install
export WANDB_MODE=disabled   # training logs to Weights & Biases otherwise
```

## Preprocessing

```bash
pixi run python scripts/get_preprocessing.py --file_path mc-flavtag-ttbar-medium.h5 \
    --cst_mode log_standard --jet_mode log_standard --output_dir resources
```

This fits the feature scalers and writes `resources/cst_quantiles.joblib` (track features) and
`resources/jet_quantiles.joblib` (jet features), which the training configs read.

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

The scan is a Snakemake workflow, `workflow/Snakefile`. Set `WORKSPACE` at its top to your clone. It submits
one SLURM job per tokenizer through `sbatch`; choose the account and partition with
`--config slurm_account=... slurm_partition=...`.

1. **Choose the grid.** `scan_configs` near the top of `workflow/Snakefile` lists the tokenizers to train. As
   committed it varies model size and training-set size at (K, d, Q) = (32768, 8, 4); loop over K, d and Q
   instead to scan them. The paper's grid is listed in `nofp_jetset` further down:
   K ∈ {1024, 8192, 16384, 32768}, d ∈ {2, 4, 8, 16}, Q ∈ {3, 4}. That list defines which runs are evaluated.
2. **Train the scan:**
   ```bash
   pixi run snakemake -s workflow/Snakefile all_vqvae_scan --cores 20
   ```
3. **Evaluate every run on the test set:**
   ```bash
   pixi run snakemake -s workflow/Snakefile all_nofp_test --cores 20
   ```
   This writes `test_nofastpath/` in each run directory. The target also covers the COCOA scans; remove
   `nofp_tracks` and `nofp_topos` from its inputs if you only have JetSet.
4. **Make the plot:**
   ```bash
   pixi run python scripts/plot_design_rules.py --e100 --outdir plots/design_rules
   ```
   This writes `rate_distortion_pareto_with_cells_binning.pdf` (Fig. 1), `design_rules.pdf` (Fig. 4) and the MAE
   version (Fig. E.10). Modalities with no runs are skipped. In the figure workflow this is rule `fig1_fig4_figE10`.

## Other Snakemake rules

Training and evaluation (`workflow/Snakefile`):

- `all_nofp_test`: evaluates the JetSet, track and topocluster scans (Table 1, Figs. 1, 4).
- `all_datascan_nofp`: dataset-size scans (Table C.4).
- `compare_roc_scan`: flavour-tagging classifiers (Fig. 6a, Table 2).
- `workflow/vendored/{track,topo}_token_regression.smk all`: jet-pT regression on tokens (Fig. 5 top).

Figures and tables (`workflow/vendored/paper_figures.smk`; it reads its paths from a copy of
`workflow/vendored/config.yaml` passed with `--configfile`):

- `fig1_fig4_figE10`: Figs. 1, 4, E.10.
- `fig5_fig6b`: Fig. 5 and Fig. 6b. The particle-flow points in Fig. 5 (bottom) come from an external pipeline,
  fed with tokens exported by `scripts/build_hep4m_preprocessed.py`.
- `fig6a`: Fig. 6a.
- `table2`: Table 2.
- `figC8`: Fig. C.8.
- `tableC4`: Table C.4.
- `figD9ab_d`: Figs. D.9a, b, d.
- Fig. E.11: request the output files `selected_2d_correlation/cocoa_tracks_cb4096_cd8_nq3.pdf` and
  `selected_2d_correlation/cocoa_topos_cb4096_cd8_nq6.pdf` (the rule `figE11` needs a concrete target).

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
