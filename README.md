# Precision Tokenization for Multi-Modal Data in High Energy Physics

Code for the paper *Precision Tokenization for Multi-Modal Data in High Energy Physics* ([arXiv:2610.13876](https://arxiv.org/abs/2610.13876)),
by Jeffrey Krupa, Samuel Klein, Nilotpal Kakati, Daniel Murnane, Baran Hashemi, Eilam Gross, Lukas Heinrich and Michael Kagan.
Code contributors: Jeffrey Krupa, Samuel Klein and Michael Kagan.
*Important: this is a frozen release. For new work, use the maintained [heptokens library](https://github.com/treasure-AmSC/heptokens).*

- **JetSet** is public: [doi:10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P).
- **COCOA:** we are still figuring out how to host the data.
- **Trained models:** not hosted yet.

## Setup

This repo uses [pixi](https://pixi.sh) for reproducible environments. Install it if you don't have it (`curl -fsSL https://pixi.sh/install.sh | bash`), then:

**Disk space:** the environment takes about 7 GB and the two JetSet files about 17 GB. On clusters with small home quotas, clone onto scratch.

```bash
pixi install                  # after installing pixi. 
export WANDB_MODE=disabled    # training logs to Weights & Biases otherwise
```

`pixi.lock` pins every package. If `pixi install` stops with a hash mismatch, a channel has re-uploaded that
package: run `pixi update <package>` (only that package is re-locked), then `pixi install` again.

```bash
export PATH=$PWD/.pixi/envs/default/bin:$PATH
```

A full training will require a GPU. Evaluation and plotting can run on CPU, but slowly (a JetSet test takes a few hours on CPU) and a GPU is preferred.

Download the JetSet files from the [record page](https://opendata.cern.ch/record/93940) or with the CERN Open Data client:

- `mc-flavtag-ttbar-medium.h5` is used to train the tokenizers.
- `mc-flavtag-ttbar-small.h5` is used to fit the scalers and to test.
- The classifiers (Fig. 6, Table 2) use all three files, including the 85 GB `mc-flavtag-ttbar-large.h5`.

Download the data with a simple `wget` (or use `cernopendata-client` if you have it):
```bash
wget -P 93940 https://opendata.cern.ch/record/93940/files/mc-flavtag-ttbar-medium.h5
wget -P 93940 https://opendata.cern.ch/record/93940/files/mc-flavtag-ttbar-small.h5
```
The files land in `./93940/` (as with `cernopendata-client`), so use `data_dir=$PWD/93940` below.

## Reproducing the JetSet scan

`workflow/Snakefile_simple` reproduces the JetSet points in Fig. 1 and Fig. 4a-c, and the Table 1 JetSet row.
Positions-out (Fig. 4d) is not covered by it. It needs only the two files above:

```bash
pixi run snakemake -s workflow/Snakefile_simple --cores 8 --config data_dir=/path/to/jetset out_dir=results/simple
```
Snakemake chains rules and running this workflow will actually run five rules:

1. **`preprocess`** fits the scalers once.
2. **`train`** trains one tokenizer per (K, d, Q).
3. **`test`** evaluates each tokenizer as for Table 1 and computes its codebook utilization.
4. **`export`** writes the token indices and codebooks of the test file.
5. **`summary`** collects everything into `summary.csv`, with K, d, Q, bits, information fraction, jet pT resolution (IQR/median), normalized MAE and utilization.

Details:

- **Hyperparameters:** taken from the stored config of the Table 1 JetSet run.
  - Large Transformer, learning rate 3·10⁻⁴, commitment weight 1, batch size 1024, seed 42.
  - 10M jets are loaded from the medium file and split 70/15/15, so each tokenizer trains on 7M jets.
  - Up to 200 epochs, with early stopping after 20 epochs without improvement in the validation loss.
- **Expected result:** with K = 32768, d = 8, Q = 4 the jet pT resolution comes out at about 0.8% (Table 1; the
  other modalities are listed there too).
- **Training speed:** on one A100 with 0.7M training jets, an epoch takes about 1.6, 2.2 and 4.7 minutes for K = 1024, 8192 and 32768. A full 200-epoch run therefore takes roughly 5–16 GPU-hours. 
- **Default grid:** the paper's, K ∈ {1024, 8192, 16384, 32768}, d ∈ {2, 4, 8, 16}, Q ∈ {3, 4}, plus additional scan Q ∈ {2, 6, 8} at (K, d) = (32768, 8).
- **Changing the grid:** the configuration is at the top of the file. The simplest way is a YAML file passed with
  `--configfile`. Set `extra_points: []` unless you also want the Q ∈ {2, 6, 8} runs at (32768, 8):
```yaml
# my_grid.yaml
grid: {K: [1024, 8192, 32768], d: [8], Q: [3, 4]}
extra_points: []
```

```bash
pixi run snakemake -s workflow/Snakefile_simple --cores 8 \
  --configfile my_grid.yaml --config data_dir=$PWD/93940 out_dir=results/simple
```

  The same works on the command line, `--config 'grid={"K": [...], "d": [...], "Q": [...]}' 'extra_points=[]'`.
  Snakemake passes these list entries as strings; the Snakefile converts them to integers. A commented tiny grid
  at the top of the file is a quick first pass.

- **Running on a cluster:** the rules contain no job submission, so the same file runs locally, with a SLURM executor plugin, or with your Snakemake profile (`--profile ...`).

Small complication: testing uses "nofp": PyTorch's fused-attention fast path (fp) is off and matmuls run in fp32. The JetSet, track and topocluster tokenizer numbers in the paper use it (Table 1, Figs. 1, 4, E.10, Table C.4, and the Transformer entries of Fig. C.8).
This doesn't affect MLPs.

## Breaking it down further into single steps

# Running preprocessing
```bash
# Scalers: resources/cst_quantiles.joblib (track features) and resources/jet_quantiles.joblib (jet features)
pixi run python scripts/get_preprocessing.py --file_path mc-flavtag-ttbar-small.h5 --num_jets 1000000 \
    --cst_mode log_standard --jet_mode log_standard --cst_log_features pt --jet_log_features pt --output_dir resources

# Running a single tokenizer
pixi run python scripts/train.py model=transformer_vqvae datamodule=atlas_mappable callbacks=encode \
    datamodule.data_path=$PWD/mc-flavtag-ttbar-medium.h5 \
    datamodule.transforms.preprocess.cst_fn.filename=$PWD/resources/cst_quantiles.joblib \
    datamodule.transforms.preprocess.jet_fn.filename=$PWD/resources/jet_quantiles.joblib \
    model.codebook_size=32768 model.codebook_dim=8 model.num_quantizers=4 \
    output_dir=$PWD/results project_name=my_scan network_name=my_tokenizer
```

The training command is illustrative: it uses the medium baseline model and settings. The exact Table 1 configuration is the `train` rule of `workflow/Snakefile_simple`. The three main hyperparameters are `model.codebook_size` (K), `model.codebook_dim` (d) and `model.num_quantizers` (Q).

## The full paper workflow

`workflow/Snakefile` and `workflow/vendored/*.smk` are the complete record of how every result in the paper was produced. 
They were written for the SLAC cluster which uses `slurm`. 
Jobs are submitted with `sbatch` inside the rules (set the account and partition with `--config slurm_account=... slurm_partition=...`).
In principle, snakemake should support other jobs, and these workflows have been tested at other submission systems like `pbs`. 
Paths are hard-coded (see below). `workflow/vendored/*.smk` read their paths from a copy of `workflow/vendored/config.yaml`, passed with `--configfile my_paths.yaml`.

### Calorimeter cells: special notes 

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

### Other rules

Training and evaluation (`workflow/Snakefile`):

- `all_nofp_test`: evaluates the JetSet, track and topocluster scans (Table 1, Figs. 1, 4).
- Positions-out runs (Fig. 4d): JetSet with `all_design_batch2` (it also trains the seed repeats), evaluated by
  `all_nofp_test`. COCOA tracks and topoclusters with `all_design_e100`, evaluated by `all_design_e100_nofp`.
- `all_datascan_nofp`: dataset-size scans (Table C.4).
- `compare_roc_scan`: flavour-tagging classifiers (Fig. 6a, Table 2).
- `workflow/vendored/{track,topo}_token_regression.smk all`: jet-pT regression on tokens (Fig. 5 top).

### Making the rest of the figures and tables (`workflow/vendored/paper_figures.smk`), for example:

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

### Site-specific paths

`workflow/Snakefile_simple` has none. The rest of the code was run on one cluster, and some paths are
hard-coded. Path roots:

- `/sdf/data/atlas/u/jkrupa/`: the repository (results, scalers) and the complete `mc-flavtag-ttbar-large.h5`
- `/sdf/scratch/users/j/jkrupa/data/atlas/`: the JetSet HDF5 files
- `/fs/ddn/sdf/group/atlas/d/jkrupa/`: COCOA ROOT files, calorimeter-cell data and runs, token exports
- `/sdf/scratch/users/s/samklein/`, `/sdf/group/mli/samklein/`: old defaults
- `/lscratch/`: node-local staging; `/global/cfs/`: NERSC

What to change:

- **Slurm account:** `workflow/Snakefile` and `profiles/s3df/config.yaml` have the placeholder `<your_account>`.
  Pass `--config slurm_account=... slurm_partition=...` or edit them. The `workflow/vendored/*.smk` workflows take
  their job prefix (`slurm_gpu`, `slurm_cpu`) from your copy of `workflow/vendored/config.yaml`.
- **PBS or HTCondor:** the rules submit with `sbatch --wait ... --wrap="..."`. Replace these prefixes with your
  scheduler's blocking submit command (for example `qsub -W block=true` on PBS). Alternatively set `slurm_gpu`
  and `slurm_cpu` to `local` to run the vendored workflows inside one batch job. `workflow/Snakefile_simple`
  needs no change: run it with the Snakemake executor plugin for your scheduler.
- **JetSet scans:** in `workflow/Snakefile`, `WORKSPACE` (top of the file) and the JetSet file paths
  (`/sdf/scratch/...`).
- **Training defaults:** `configs/train.yaml` (`output_dir`), `configs/datamodule/atlas_mappable.yaml` and
  `atlas_iterable.yaml` (`data_path`). Or pass `output_dir` and `datamodule.data_path` on the command line.
- **COCOA tracks and topoclusters:** `configs/datamodule/cocoa_{tracks,topos,truthpart}.yaml` (`data_dir`,
  scaler path) and `COCOA_DATA_DIR` in `workflow/Snakefile`.
- **Calorimeter cells, regression and figures:** copy `workflow/vendored/config.yaml` and fill in its `<...>`
  placeholders.
- **Figure scripts:** `scripts/plot_design_rules.py` (`CELLS["dir"]`, the cell runs),
  `scripts/codebook_vqvae_utilization.py` (`DATA_PATH`), `scripts/compare_roc.py`.
- **Classifiers:** `configs/datamodule/{atlas_iterable_multi,token_npz}.yaml`, `configs/model/token_classifier.yaml`.
- **Not needed for the paper commands:** `workflow/preprocessing.smk`, `scripts/{simple_demo,
  build_hep4m_preprocessed,export_cocoa_tokens_hep4m,fit_cocoa_truthpart_scaler}.py`, `scripts/pflow_*`,
  `scripts/copy_tokenizers_to_nersc.sh`, `tests/test_positions_out.py`. `scripts/get_preprocessing.py` only has
  a default `--file_path`.

## Code layout

`src/heptokens` is the original code. It produced the JetSet, track and topocluster results.
`src/heptokens/vendored` is a copy of the later, refactored library code, used for the calorimeter cells and the
jet-pT regression.

## Citation

```bibtex
@misc{krupa2026precisiontokenization,
  title         = {Precision Tokenization for Multi-Modal Data in High Energy Physics},
  author        = {Jeffrey Krupa and Samuel Klein and Nilotpal Kakati and Daniel Murnane and Baran Hashemi and Eilam Gross and Lukas Heinrich and Michael Kagan},
  year          = {2026},
  eprint        = {2610.13876},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph},
  url           = {https://arxiv.org/abs/2610.13876},
}
```

## License

MIT — see [LICENSE](LICENSE).
