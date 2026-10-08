# Paper release: review status (temporary, delete before release)

State as of 2026-10-08 evening, branch `paper-release`. Paper of record: `Tokenizing_HEP_datasets-33.pdf`. It is not
committed.

This is a temporary note, so it uses machine paths directly. Short names:

| Name | Location |
|---|---|
| `A` | this repo, `/sdf/data/atlas/u/jkrupa/heptokens` |
| `B` | `/sdf/data/atlas/u/jkrupa/heptokens-new` (`heptokens/` library + `heptokens-cocoa/`) |
| `LS` | `/lscratch/jkrupa/paper-release`. This is node-local on **sdfiana005** and holds all check outputs, logs and helper tools (`LS/tools`). |
| `CELLS` | `/fs/ddn/sdf/group/atlas/d/jkrupa/cocoa_results` (cell runs) |

## 1. Provenance

| What | Ref |
|---|---|
| A base before this work | `af8c104` |
| A stash@{0}, preserved as branch `stash-backup` | `bacbbab6` |
| B snapshot, `heptokens` (branch `cells-paper-snapshot`, parent `196775b`) | `73cb6ae` |
| B snapshot, `heptokens-cocoa` (branch `cells-paper-snapshot`, parent `4c8466c`) | `cc20cbc` |

- **B snapshot contents.** `73cb6ae` adds the uncommitted `models/utils.py` (`scheduler_interval`) and `tests/test_scheduler_interval.py`. `cc20cbc` adds every modified or untracked source, config and workflow file, plus `plots/codebook_utilization_cells/make_cell_utilization.py`. It excludes `SUCCESS.txt` and `plots/*.csv|*.npz`. Neither snapshot branch is pushed.
- **Files checked out from stash@{0}**, as approved (commit `a1117ac`): `scripts/plot_design_rules.py`, `scripts/compare_roc.py`, `scripts/plot_2d_correlation.py`, `workflow/Snakefile`, `scripts/export_tokens.py`, `configs/tokenize.yaml`. `profiles/s3df/config.yaml` from the stash was left out on purpose. The stash itself was not popped or dropped.

## 2. Committed on `paper-release` (not pushed, not tagged)

| Commit | Content |
|---|---|
| `a1117ac` | Six files restored from stash@{0} (published-figure code) |
| `5045f57` | `src/heptokens/vendored/` (B library, cell datamodule and callback), `scripts/vendored/` (13 scripts), cell configs (`configs/train_cell.yaml`, `configs/*/hep4m_cell*.yaml`) |
| `10ae49d` | Previously uncommitted A scripts behind paper results: `binning_baseline{,_cells}.py`, `cocoa_utilization_nofp.py`, `plot_heatmaps_nofp.py`, `rejection_table.py`, `configs/tokenize_cocoa.yaml` |
| `a26c654` | `workflow/vendored/`: config, cell, regression and paper-figure workflows |
| `0a9645e` | README section "Reproducing the paper" |
| (this commit) | `REVIEW_STATUS.md` |

**B → A file mapping.** Vendored files are byte copies except for the edits listed under "Edits" below.

| B | A |
|---|---|
| `heptokens/src/heptokens/models/{vq_vae,coders,transformer,utils,vq_layers}.py` | `src/heptokens/vendored/models/` |
| `heptokens/src/heptokens/data/{base,collation}.py`, `callbacks/recon.py` | `src/heptokens/vendored/{data,callbacks}/` |
| `heptokens-cocoa/src/heptokens_cocoa/data/{hep4m_cell,cell_image_helper,cocoa_base}.py`, `callbacks/cell_recon.py` | `src/heptokens/vendored/cocoa/{data,callbacks}/` |
| `heptokens-cocoa/scripts/{preprocess_hep4m_cells,sparsify_cell_h5,compute_cell_scan_resolution,eval_cell_wloss,make_cell_diagnostics,make_tokenization_schematic,plot_rate_distortion,export_track_tokens,export_topo_tokens,train_topo_regressor,eval_topo_regressor,make_topo_regression_plot}.py` | `scripts/vendored/` |
| `heptokens-cocoa/plots/codebook_utilization_cells/make_cell_utilization.py` | `scripts/vendored/make_cell_utilization.py` |
| `heptokens-cocoa/configs/{train_cell,callbacks/hep4m_cell_*,datamodule/hep4m_cell*,model/hep4m_cell_*}.yaml` | `configs/` (same relative paths) |
| `heptokens-cocoa/workflow/{cells_*,track_token_regression,topo_token_regression}.smk` | `workflow/vendored/` (paths moved to `config.yaml`) |
| (new) | `workflow/vendored/{common.smk,config.yaml,run_local.sh,paper_figures.smk}` |

**Edits**
- Imports changed from `heptokens.*` / `heptokens_cocoa.*` to `heptokens.vendored.*`.
- `_target_`s changed to `heptokens.vendored.*`.
- Absolute paths replaced by `<PLACEHOLDER>`s or `???`.
- `DEFAULT_CST_FN` is now repo-relative.
- `make_cell_utilization.py` gained `--scan_dir` and `--outdir`.
- `torch.load` and `load_from_checkpoint` are wrapped in `heptokens.vendored.b_module_names()`. Cell checkpoints pickle `heptokens.models.coders.CellDecoder`, which does not exist in A's `heptokens.models`.
- `plot_rate_distortion.py` is unchanged.
- `scripts/binning_baseline_cells.py` (an A file) now imports the vendored loader and requires `--data_dir`.

**New rules (not in B)**
- `preprocess_cells_2M`: regenerates the 2M cell set. The original came from an earlier project; B recorded relative energy differences ≤1.2e-7.
- `eval_cells_20M`: mirrors the manual sbatch eval jobs 39906461/2.

**Hygiene checks on the committed files**
- No absolute paths.
- No secrets.
- No import or `_target_` inside `vendored` points to A's own modules.
- No `__pycache__` files and no files >1 MB.
- Every committed script passes `--help` in A's env. The exception is `plot_heatmaps_nofp.py`, which has no CLI: it is a constants module imported by `cocoa_utilization_nofp.py`.

**Not committed (left untracked on purpose)**
- `scripts/{cells_pixel_mae,check_rejection_uncertainty,debug_ckpt,extract_ckpt_metric,feature_effective_dimension,make_encoder_training_plots,make_filtered_scan_plots,make_slides,pflow_slides_plots,plot_cocoa_datascan,prof_feat,prof_token,sanity_check_tokens,utilization_saturation}.py` and `scripts/{download_toptagging,run_cocoa_tests,submit_batch_tests,wandb_sweep_watcher}.sh`. None of these produce a paper figure or table; the possible exceptions are listed under open issues.
- `workflow/profiles/` (unknown origin).
- `HEP4M/`, `logs/`, `.claude/`, and the stray files `--from_predictions` and `--run_dirs`.
- `pixi.lock` is unchanged; see decision D1.

## 3. Verification checks 5a–f

All runs use `WANDB_MODE=disabled` and CPU (this node has no GPU), and write only to `LS`.

### 5a: set-based regression

CPU reruns of the stored `test_nofastpath` overrides with A's code (`LS/tools/run_5a.sh`, `LS/tools/util_5a.py`).

| Tokenizer (K, d, Q) | Quantity | Stored | Paper (Table 1) | Before port (A code, pre-commit) |
|---|---|---|---|---|
| Tracks (4096, 8, 3) | jet IQR/median | 0.006782 | 0.68% | **0.006782 (identical)** |
| Tracks | utilization | 0.705404 | 71% | **0.705404 (identical)** |
| Topos (4096, 8, 6) | jet IQR/median | 0.003872 | 0.39% | **0.003872 (identical)** |
| Topos | utilization | 0.985962 | 98% | running |
| JetSet (32768, 8, 4) | jet IQR/median | 0.008215 | 0.8% | running (`LS/logs/5a_before_jetset.log`) |
| JetSet | utilization | 0.768 | 77% | queued after the topos utilization |

- **After-port reruns: not run yet.** The commits do not touch A's shared library (`src/heptokens/*` outside `vendored/`). The set-based test path differs from before only through the stash files.

### 5b: cells, Table 1 row

- **Ported eval: running.** It runs `scripts/vendored/eval_cell_wloss.py` on `CELLS/cells_scan_2M_a0.4_frontier/transformer/cb4096_cd16_nq16_a0.4_large`, output in `LS/5b/A`.
- **Expected (stored, B, GPU):** `energy_resolution_iqr` 0.012415791494719063 (paper 1.24%), `codebook_util_avg` 0.8838958740234375 (paper 88%).
- **Diagnostics (active pixels 132.13 / 132.14; Fig. E.12): not run yet.** Plan: run ported and B-original on CPU with `--max_events 50000`, then compare `diagnostics_arrays.npz` with B's `plots/cell_diagnostics_transformer_cb4096_cd16_nq16_a0.4/`.
- **Checkpoint loading in A:** `build_model` loads with `b_module_names()`.
- **Config cross-check: done.** I composed `train_cell` in A and compared it with each stored run's saved config. All frontier and 20M runs match except for:
  - `_target_` renames
  - path placeholders
  - `decoder.energy_sum_level: patch`, present only in the saved configs. This was an uncommitted option, reverted in B mid-run (Oct 6 01:29). Its value `patch` with weight 0 reproduces the old behaviour, so I did not port it.
- **`scheduler_interval`:** it was in effect (`utils.py` modified Sep 29, before every run).
  - The 2M frontier configs do not set it, so they get the default `epoch` (a no-op).
  - The 20M runs set `step`, so it matters for the 20M rows of Table C.4.

### 5c: short training, A (ported) vs B (original), same seed and data

Both sides train for 4 epochs on 800 events of `singlejet_270.h5` (batch 64, 10 steps per epoch), so the run crosses 3 epoch boundaries. `LS/tools/steplog.py` logs loss and learning rate every step. The script is `LS/tools/run_5c.sh`; compare with `LS/tools/compare_5c.py`.

| Recipe | A | B | Comparison |
|---|---|---|---|
| frontier (warmup_large) | done (`LS/5c/frontier/A/steplog.json`) | running | **not done** |
| steplr (20M recipe, `scheduler_interval: step`) | queued | queued | **not done** |

**Settings to confirm against the configs: not yet confirmed.**

| Run set | max_epochs | Early stopping | Checkpoint selection |
|---|---|---|---|
| Frontier | 30 | patience 15 on `val/total_loss` | best by `val/total_loss` (min) |
| 20M | — | removed | — |

### 5d: dry runs

These ran on the working tree before committing; the committed files are the same.

- **All targets resolve:**
  - `workflow/Snakefile`: `all_nofp_test`, `all_design_e100_nofp`, `all_datascan_nofp`, `compare_roc_scan`
  - cell frontier, cells 20M and cell preprocessing: "Nothing to be done"
  - regression: only `make_plot` is pending (mtime)
  - every `paper_figures.smk` target and both binning targets
- **Exception:** `fig6c`. Its input `results/vqvae_scan/tokenized/transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj10000000/mc-flavtag-ttbar-small.npz` was deleted.
- **Note:** use `--rerun-triggers mtime`. With the default triggers, `train_design_topos` is scheduled for retraining because params 15/50 changed in the Snakefile.

### 5e: fresh clone + `pixi install` + dry run + one cell inference

**Not run.** It needs the commits, which now exist.

### 5f: regenerated outputs vs published

Outputs are in `LS/figures`. For text, I compared each output's `pdftotext` against the paper's `pdftotext`.

| Item | Result |
|---|---|
| Fig. 4 `design_rules` | identical to the on-disk render; text matches the paper |
| Fig. E.10 (MAE) | identical to the on-disk render |
| Fig. 1 | **differs from the on-disk render (0.05% of pixels).** The new render includes the cell point (4096, 16, 32), which was evaluated at 06:42, after the 05:56 render. See O5. |
| Fig. 5 `rate_distortion_cocoa` | identical to B's render; text matches. The top-panel x-label reads only "Information fraction", the same as B's render. |
| Fig. 6b `rate_distortion_jetset` | identical to B's render; text matches |
| Fig. 6a ROC | generated; labels and axes match the paper text. No published file exists for a pixel comparison. |
| Table 2 | 29 of 30 entries match the paper exactly. **One difference:** continuous τ at εb = 0.70 is 451 ± 10 here vs 444 ± 10 in the paper. Output: `LS/figures/tables/rejection_table.txt`. See O16. |
| Fig. C.8 | CSV identical; the numbers in the figure match the paper |
| Fig. D.9a, b, d | plot step run in a temporary copy of the repo (`LS/tmprepo_d9`, without the q6 CSV); every printed value matches the paper |
| Fig. D.9c | identical to the on-disk render; text matches |
| Fig. E.11 (both panels) | identical to the on-disk render |
| Fig. E.12 | **not run** (part of the 5b diagnostics) |
| Fig. 3 | **not run** (arguments and checkpoint unknown, see O9) |
| Fig. 6c | **cannot run** (input missing) |
| Table C.4 | values recomputed from the metric files match the paper (cells 2M: nq8 3.764, nq16 3.617; 20M: nq8 3.273, nq16 3.236). **No script.** |

### Regression (Fig. 5, top), checked from stored checkpoints without retraining

Config: track `cb4096_cd4_nq4`, seed 42.
- **Eval.** The ported `eval_topo_regressor.py` (A env, CPU) and B's original (B env, CPU) give byte-identical outputs: `iqr_median_overall` 0.011908. The stored value, computed on GPU, is 0.012045. The 1.1% gap comes from the device: the GPU `nn.TransformerEncoder` fast path uses tanh-GELU.
- **Token export** (10k test events, CPU):
  - Ported code in B's env: byte-identical to B's original.
  - A env vs B env: 96.4% of tracks bit-identical, 99.88% within 1e-5, and 0.008% differ by more than 1e-3 (numpy/sklearn versions).
  - B on CPU vs the stored GPU export: 44% bit-identical, 4.3% differ by more than 1e-3 (finest-level code flips).
- **Literals in `plot_rate_distortion.py`.**
  - Every regression mean matches B's `{track,topo}_regression_results.csv` (mean over 3 seeds).
  - Four SEMs differ by ≤2e-6: topo cb64, cb1024 and cb4096 at nq4, and track cb4096 at nq4.
  - The continuous baselines match.

### Still running on sdfiana005

All of these write only to `LS`.
- `util_5a.py` (topos, then JetSet)
- 5a JetSet test
- 5b eval
- 5c loop: B frontier, then A steplr, then B steplr

To stop them all: `pkill -u jkrupa -f '/lscratch/jkrupa/paper-release|run_5c|compare_roc.py|rejection_table.py'`.

## 4. Manifest

`pf` = `workflow/vendored/paper_figures.smk`, `SF` = `workflow/Snakefile`. Checkpoint paths are under `results/` (A) unless marked. The full list, 312 files and 12.9 GB grouped by artifact, is in `LS/out/checkpoints.csv`.

| Paper item | Script | Config | Rule | Checkpoints / inputs | Gaps |
|---|---|---|---|---|---|
| Fig. 1 | `scripts/plot_design_rules.py --e100` | – | `pf:fig1_fig4_figE10`, `binning_baseline`, `binning_baseline_cells` | `vqvae_scan/vqvae_scan`, `cocoa_scan/{tracks,tracks_posout,tracks_e50,tracks_e100,topos,topos_posout,topos_seed,topos_e100}` `test_nofastpath` metrics; cell frontier `CELLS/cells_scan_2M_a0.4_frontier` `test_metrics.json` (hard-coded `CELLS["dir"]`); `binning_baseline/` | extra cell point (O5) |
| Fig. 2 | – | – | – | – | drawing, no code |
| Fig. 3 | `scripts/vendored/make_tokenization_schematic.py` | – | `pf:fig3_schematic` | config key `schematic_ckpt` (unknown) | args, event and checkpoint not recorded (O9) |
| Table 1 | metric files | `configs/train.yaml` + experiment overrides; `configs/train_cell.yaml` | `SF:all_nofp_test`; `cells_frontier_2M.smk:eval_frontier` | JetSet `vqvae_scan/vqvae_scan/transformer_vqvae_enc_large_cb32768_cd8_nq4_lr0.0003_cw1.0_nj10000000`; tracks `cocoa_scan/tracks/cocoa_tracks_cb4096_cd8_nq3_lr0.001`; topos `cocoa_scan/topos/cocoa_topos_cb4096_cd8_nq6_lr0.001`; cells `CELLS/.../cb4096_cd16_nq16_a0.4_large`; utilization CSVs in `results/codebook_utilization/` | O1, O2 |
| Fig. 4 | `plot_design_rules.py --e100` | – | `pf:fig1_fig4_figE10` | as Fig. 1, incl. `_posout` e100 runs | caption vs axis (O4) |
| Fig. 5 top | `scripts/vendored/plot_rate_distortion.py` (literals) ← `train/eval_topo_regressor.py` | – | `pf:fig5_fig6b`; `{track,topo}_token_regression.smk:all` | tokenizers `cocoa_scan/tracks/*_cd4_*`, `cocoa_scan/topos/*_cd8_*`; regressors (B) `heptokens-cocoa/results/{track,topo}_token_regression/regressors/` (79 files) | literals, not read from the CSV |
| Fig. 5 bottom | `plot_rate_distortion.py` (literals) | – | `pf:fig5_fig6b` | pflow tokenizers `cocoa_scan/{tracks,topos}_e100/cocoa_*{,_posout}_cb{256,1024,4096}_cd8_nq4_lr0.001_e100`; truth `cocoa_scan/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001` | nanoHEP code and results not here (O8); `PARTICLE_FLOW_D` = 7.345, +30 bits for positions-out |
| Fig. 6a | `scripts/compare_roc.py --from_predictions` | – | `pf:fig6a` (= `SF:compare_roc_scan`) | `vqvae_scan/classifiers/{feature_clf_full_v3,token_clf_full_v2,token_clf_full_cb1024_nq3,token_clf_full_cb1024_nq4,token_clf_full_cb8192_nq4}` | – |
| Fig. 6b | `plot_rate_distortion.py` (literals) | – | `pf:fig5_fig6b` | classifier predictions as in Fig. 6a | literals |
| Fig. 6c | `scripts/codebook_tsne.py` (input from `scripts/export_tokens.py`) | – | `pf:fig6c` | token export of JetSet (32768, 8, 4) | **input npz deleted** (O7) |
| Table 2 | `scripts/rejection_table.py` | – | `pf:table2` | as Fig. 6a | O16 |
| Table A.3 | – | – | – | – | no code |
| Fig. B.7 | **none found** | – | – | – | O6 |
| Fig. C.8 | `scripts/compare_mlp_transformer.py` | – | `pf:figC8` | `vqvae_scan/vqvae_scan` MLP and transformer runs (MLP 10M runs: `last.ckpt` only) | reads `test/` with the fast path on (O3) |
| Table C.4 | **no script** | – | `SF:all_datascan_nofp`; `cells_20M.smk:eval_cells_20M` | `cocoa_scan/{tracks,topos}_datascan`, JetSet `*_nj1000000` / `*_nj10000000`, `CELLS/cells_scan_2M_a0.4_frontier`, `CELLS/cells_20M/full` | O10 |
| Fig. D.9a, b, d | `scripts/cocoa_utilization_nofp.py compute && plot` | – | `pf:figD9ab_d` | the Table 1 scans | writes to fixed `results/` paths; q6 CSV (O11) |
| Fig. D.9c | `scripts/vendored/make_cell_utilization.py` | – | `pf:figD9c` | cell frontier `test_metrics.json` | – |
| Fig. E.10 | `plot_design_rules.py --e100` | – | `pf:fig1_fig4_figE10` | as Fig. 1 | – |
| Fig. E.11 | `scripts/plot_2d_correlation.py` | – | `pf:figE11` | the Table 1 tracks/topos runs, `test_nofastpath` | – |
| Fig. E.12 | `scripts/vendored/make_cell_diagnostics.py` | – | `pf:figE12` | cells `cb4096_cd16_nq16_a0.4_large/checkpoints/best.ckpt` | debug title (O12) |

## 5. Open issues and decisions needed

**D1. Environment.** The README currently says one pixi environment covers everything, and `pixi.lock` is unchanged. Earlier plan: keep A's env unless 5c shows a difference, otherwise add a `cells` env. 5c is not finished. Re-exporting tokens in A's env gives float-level differences vs B's env (see Regression above).
**D2. Table 1, cells.** The frontier contains better points than the "best" 1.24% (4096, 16, 16): (4096, 16, 24) = 1.104% and (4096, 16, 32) = 1.085%. Should the table and text change, or should the choice of row be explained?
**D3. Table 1, topos utilization.** The stored value is 0.98596, which rounds to 99%; the paper says 98%.
**O3. Fig. C.8 vs Table C.4.** Fig. C.8 uses fast-path-on `test/` metrics (JetSet 10M: 0.00813), while Table C.4 uses `test_nofastpath` (0.82%). Which should be used for both?
**O4. Fig. 4 caption.** It says positions-out "excludes the bits spent on the binning", but the axis shows total bits (including `pos_bits` 20/30).
**O5. Fig. 1.** The current render includes cell point (4096, 16, 32), evaluated after the render on disk. Was the published Fig. 1 made before that point existed?
**O6. Fig. B.7.** No producing script was found. The PDF contains the stray text "Annotate better and put in tokenization appendix".
**O7. Fig. 6c.** The token export it reads was deleted. A re-export with the current `export_tokens.py` (fast path off) may differ slightly from the published one. Re-export, or accept the gap?
**O8. Particle flow.**
- The nanoHEP code and numeric outputs are not on S3DF; only the literals and the runbook (`docs/nanohep_pflow_handoff.md`, branch `feature/cocoa-dataloader`) exist.
- The runbook says 89M events and a ~110M-parameter or 13–15M-parameter model; the paper says 20M events and 15M parameters.
- Who records the NERSC exports and commits?
**O9. Fig. 3.** The published inputs (`schematic_v2_ds1`, July 14), checkpoint and event are not recorded. The script expects an MLP coder, so the transformer cell checkpoints do not fit it.
**O10. Table C.4.** It was computed ad hoc. Should I add a small script (new code), or document the metric paths only?
**O11. Fig. D.9a, b.**
- `results/codebook_utilization/cocoa_utilization_nofp_q6.csv` (Oct 8 06:53) exists. With it present, `plot` adds a Q = 6 panel, so the result would not match the published 3-panel figure.
- The heatmaps on disk in `results/` and `paper/figures_nofp/` (06:54) already have 4 panels.
- `figD9ab_d` writes into `results/` and `paper/`.
**O12. Fig. E.12.** The published title carries a debug string ("Transformer VQ-VAE (Q,K,d)=… MSE, α=0.4, 2M train"). The rule reproduces it as published.
**O13. Scalers.** `resources/cst_quantiles.joblib` and `resources/jet_quantiles.joblib` are gitignored. The plan is to ship them with the checkpoints on Zenodo; the README has placeholders `<ZENODO_DOI>` and `<COCOA_DATA_LOCATION>`.
**O14. Absolute paths in existing A files.** Per instruction these were not edited; the README lists the path roots and the files that contain them.
**O15. Regression device.** The stored results come from GPU (tanh-GELU fast path); CPU reproduces them to about 1% in IQR.
**O16. Table 2, continuous τ at εb = 0.70.** The regenerated value is 451 ± 10; the paper has 444 ± 10. The other 29 entries match. The paper value may come from an earlier version of the predictions or of the script. Should the paper be updated, or the source of 444 be found?

## 6. Remaining steps (not started)

1. Collect the running results: 5a JetSet and utilizations, 5b eval, 5c comparison.
2. 5b diagnostics, A vs B on CPU.
3. Decide D1.
4. 5a after-port reruns from the committed tree.
5. 5e fresh clone.
6. 5f for Fig. E.12, and Fig. 3 if O9 is resolved.
7. Final report.
8. Delete this file.
9. Tag. Not done, by instruction.
