# nanoHEP particle flow at NERSC with heptokens tokenizers: agent runbook

You are setting up and running **nanoHEP** (decoder-only GPT from `FM-for-HEP/hep-any2any`) for
particle flow, `(track, topo) token IDs -> truth-particle token IDs`, on ~89M COCOA single-jet
events at NERSC (Perlmutter). The point of the study is to compare **positions-in** and
**positions-out** track/topo tokenizers trained in the `heptokens` repo. Work through the
sections in order; every step ends with a check, so don't skip ahead when a check fails.

Fill in the paths in section 0 first. Everything else refers to them.

---

## 0. Paths (fill in)

```bash
export WORK=<scratch dir for this study, e.g. $SCRATCH/nanohep_pflow>
export RAW_TRAIN=<dir with the ~89M-event training ROOT files>
export RAW_VAL=<dir with validation ROOT files (disjoint from train)>
export RAW_TEST=<dir with test ROOT files (disjoint from train and val)>
export TOKENIZERS=/global/cfs/cdirs/m5386/data/COCOA/tokenizers   # copied from S3DF, layout in section 1
export HEPTOKENS=$WORK/heptokens          # clone, section 2
export ANY2ANY=$WORK/hep-any2any          # clone, section 2
export STORE=$WORK/stores                 # token stores, one subdir per tokenizer config
export NERSC_ACCOUNT=<allocation, e.g. mXXXX>
```

---

## 1. Inputs copied from S3DF

### 1a. Tokenizer runs

Each run directory needs `checkpoints/best.ckpt` (~20 MB) and `full_config.yaml`. The
checkpoints contain no absolute paths, so they load anywhere `heptokens` is installed.
`full_config.yaml` does contain S3DF paths (data dir, scaler file); use it for the
**features, position features, max_csts and pos_tokenizer**, not for paths.

```
$TOKENIZERS/
  MANIFEST.sha256                                                                   # verify: cd $TOKENIZERS && sha256sum -c MANIFEST.sha256
  tracks_e100/cocoa_tracks_cb{K}_cd8_nq4_lr0.001_e100/{checkpoints/best.ckpt,full_config.yaml,SUCCESS.txt}          # positions-in
  tracks_e100/cocoa_tracks_posout_cb{K}_cd8_nq4_lr0.001_e100/{checkpoints/best.ckpt,full_config.yaml,SUCCESS.txt}   # positions-out
  topos_e100/cocoa_topos_cb{K}_cd8_nq4_lr0.001_e100/...
  topos_e100/cocoa_topos_posout_cb{K}_cd8_nq4_lr0.001_e100/...
  truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/{checkpoints/best.ckpt,full_config.yaml,SUCCESS.txt}
```

K in {256, 1024, 4096}. These are copied from S3DF
(`/sdf/data/atlas/u/jkrupa/heptokens/results/cocoa_scan/<same relative paths>`) by
`scripts/copy_tokenizers_to_nersc.sh`, which only copies finished runs (`SUCCESS.txt`, 100 epochs).
Start with whatever is present and verified; check `SUCCESS.txt` before using a run.

### 1b. Scalers

In the heptokens repo (`resources/`, committed): tracks `cocoa_cst_log_standard.joblib`
(positions-out: `cocoa_cst_log_standard_posout.joblib`), topos `cocoa_topo_log_standard.joblib`
(positions-out: `cocoa_topo_log_standard_posout.joblib`), truthpart
`cocoa_truthpart_log_standard.joblib`. The scaler must match the tokenizer: `full_config.yaml`
-> `datamodule.transforms.preprocess.cst_fn.filename` names it (take the basename).

### 1c. Raw ROOT

COCOA single-jet ROOT files, tree `EventTree`, with branches `track_{pt,eta,phi,d0,z0}`,
`topo_{eta,phi,e,rho,sigma_eta,sigma_phi,ecal_e,hcal_e}`, `particle_{pt,eta,phi,e,pdgid,track_idx,...}`,
`eventNumber`. **Check (step 4.1):** every file opens, the branches exist, and no `eventNumber` is
shared between train and val/test (on S3DF one 24M sample overlapped the val/test segments).

---

## 2. Code and environments

```bash
cd $WORK
git clone -b feature/cocoa-dataloader git@github.com:Treasure-AmSC/heptokens.git
git clone git@github.com:FM-for-HEP/hep-any2any.git                                # main (checked at d5aac9b)
```

- **heptokens** (tokenizers, COCOA datasets, token export, truthpart decoding): `cd $HEPTOKENS && pixi install`.
  Run its Python as `pixi run --manifest-path $HEPTOKENS/pyproject.toml python ...`.
- **hep-any2any** (nanoHEP train/infer/report): `cd $ANY2ANY && pixi install` (CUDA env), and
  `cp .env.example .env`, then set in `.env`:
  `HEP4M_DATA=$WORK/any2any_data`, `HEP4M_WORK=$WORK/any2any_work`, `HEP4M_TOKENIZED=<store dir of the run>`,
  `HEP4M_LOGGER=wandb` (or `csv`). `python -m hep4m.paths` prints the resolved paths.
- The two repos have **different** `vector-quantize-pytorch` versions (heptokens 0.2.2, hep-any2any 1.22).
  Load heptokens tokenizers only in the heptokens env (export, decoding). Never load them in the hep-any2any env.
- Perlmutter GPU nodes: 4x A100 per node. Use `-C gpu -A $NERSC_ACCOUNT -q regular --gpus-per-node 4`;
  jobs are capped at 48 h, and nanoHEP resumes from `checkpoints/last.ckpt` (section 6).

**Check:** in the heptokens env, `python -c "from heptokens.models.vq_vae import LitVqVae; LitVqVae.load_from_checkpoint('$TOKENIZERS/truthpart/cocoa_truthpart_cb128_cd8_nq3_lr0.001/checkpoints/best.ckpt', map_location='cpu')"` succeeds.

---

## 3. What the model does (so you know what you're building)

- One sequence per jet: `<MOD_OUT:truthpart> <TASK_SEP> <MOD_IN:topo>`, then the track and topo
  elements interleaved (track, topo, track, topo, ..., then the remainder), then
  `<MOD_OUT:truthpart>`, the truth particles one by one, and `<EOS>`, right-padded to `block_size`.
- Element tokens: Q=4 content codes per track/topo; **positions-out** adds 3 position tokens
  (eta, cos phi, sin phi, 1024 bins each). Truth particles: 3 codes (truthpart tokenizer K=128,
  Q=3, positions-in), no position tokens.
- Loss: next-token cross-entropy on the truth-particle tokens and `<EOS>` only. At inference the
  model generates until `<EOS>`, so it also chooses the number of particles.
- Order matters: elements are interleaved and generated in on-disk order, which must be
  pT/energy-descending (section 4.2, item 4).

---

## 4. Build the token stores (heptokens env)

One store per **tokenizer config** = (positions-in | positions-out, K). The truthpart tokens are
identical across stores, so tokenize truthpart once and copy/symlink `truthpart_*` into each store.
Layout per split (what nanoHEP reads, `hep-any2any/hep4m/datasets/tokenized_memmap.py`):

```
$STORE/<config>/{train,val,test}/
    {track,topo,truthpart}_data.npy      headerless int16 [n_rows, n_codebooks + n_pos_codebooks]
    {track,topo,truthpart}_offsets.npy   headerless int64 [n_events + 1]
    {track,topo,truthpart}_meta.npz      n_events, n_codebooks, n_pos_codebooks
    {track,topo,truthpart}_is_empty.npy  int64 indices of events with no objects
    event_numbers.npy                    headerless int64 [n_events], identical order for all modalities
```

### 4.1 Validate the raw data

For train/val/test: count events per file, check the branches in 1c, and check that train and
val/test share no `eventNumber`. Record the event counts.

### 4.2 Adapt `scripts/build_hep4m_preprocessed.py`

The script already writes the layout above, but it was written for S3DF and positions-in only.
Change it so that:

1. **Paths:** no hard-coded `WS = /sdf/...` and no default `--data_dir`. Take `--data_dir`,
   `--track_ckpt/--topo_ckpt/--truthpart_ckpt`, and scaler paths (or resolve the scaler from the
   repo's `resources/` by the basename in `full_config.yaml`). Use `root_files_from_dir`, which globs
   `*.root`; pass the train/val/test dirs from section 0.
2. **Dataset from the tokenizer's `full_config.yaml`:** `datamodule.features`,
   `datamodule.position_features` (absent or null for positions-in) and `max_csts` (tracks 15,
   topos 50, truthpart 16). For reference:
   - positions-out tracks: features `[track_pt, track_d0, track_z0]`, positions `[track_eta, track_phi]`
   - positions-out topos: features `[topo_e, topo_rho, topo_sigma_eta, topo_sigma_phi, topo_ecal_e, topo_hcal_e]`, positions `[topo_eta, topo_phi]`
   - positions-in: the 5 track / 8 topo features including eta/phi; no positions.
3. **Positions-out: write position tokens.** With `position_features` the dataset returns
   `batch["positions"]` = (eta, cos phi, sin phi). Bin them with
   `heptokens.models.pos_tokenizer.PositionTokenizer(ranges=[(-3,3),(-1,1),(-1,1)], n_bins=1024)`
   (instantiate it from `callbacks.reconstruction_monitor.pos_tokenizer` in `full_config.yaml`),
   append the 3 indices after the Q content codes, and write `n_pos_codebooks=3` in `{mod}_meta.npz`.
   Positions-in: `n_pos_codebooks=0`. This matches nanoHEP's default position vocabulary
   (`pos_codebook_size=1024`, `num_q_pos=3`).
4. **Sort order:** sort each event's elements descending by **track_pt** (tracks), **topo_e**
   (topos), **particle pt** (truthpart), using the same sort for content and position columns.
   The current `--pt_idx 0` sorts by column 0 of the scaled features, which is `topo_eta` for
   positions-in topos (wrong) and `topo_e` for positions-out topos. Choose the column by
   feature name.
5. **Stream file by file.** The COCOA map-style datasets hold every event in RAM (89M topos are
   ~140 GB of float32). Process one ROOT file (or a few) at a time, append rows to the open
   `{mod}_data.npy`, collect per-event counts and event numbers, and write `offsets`, `meta`,
   `is_empty` and `event_numbers` at the end. Use a GPU for `encode_indices`, and a SLURM array
   over files for train if one job is too slow (then concatenate in file order).
6. **Alignment:** all three modalities must have the same events in the same order. Keep the
   existing `event_numbers` equality assert. Note the 16-particle / 15-track / 50-topo caps
   truncate <1% of events; report how many.

### 4.3 Checks on each store

- `meta.npz`: `n_codebooks` (4 track/topo, 3 truthpart), `n_pos_codebooks` (3 positions-out, else 0),
  identical `n_events` across the three modalities and equal to the raw count from 4.1.
- Code ranges: content `< K` (`< 128` for truthpart), positions `< 1024`, no negatives.
- Mean multiplicities per event (S3DF 20M sample: tracks 3.62, topos 12.95, truth particles 6.75).
- Spot-check 10 events: re-encode them in a notebook and compare to the stored rows; positions-out
  position tokens decode (`PositionTokenizer.decode`) to within half a bin of the raw eta/cos/sin.

---

## 5. Configure nanoHEP (hep-any2any)

Start from `configs/train/nanohep_pflow.yml` (paper recipe) and make one config per store.

```yaml
run_name: nanohep_pflow_<config>              # e.g. posin_K4096 / posout_K4096
tokenized_root: <$STORE/<config>>
all_modalities: [topo, track, truthpart]
vocab_args:                                    # give all four dicts; the defaults are the HEP4M tokenizers
    modalities: [topo, track, truthpart]
    codebook_sizes:     {topo: <K>, track: <K>, truthpart: 128}
    num_quantizers:     {topo: 4,   track: 4,   truthpart: 3}
    pos_codebook_sizes: {topo: 1024, track: 1024, truthpart: 1024}
    num_q_pos:          {topo: <3 posout | 0 posin>, track: <3 | 0>, truthpart: 0}
block_size: 512                                # data sequence length
gpt_config: {block_size: 640, n_layer: 12, n_head: 12, n_embd: 768, dropout: 0.0, bias: false, vocab_size: null}
pflow_eval: {enabled: false}                   # until section 7a is done
```

- **Sequence length:** 2 + 1 + sum over input elements of width + 1 + 3*N_truth + 1, width = 4
  (positions-in) or 7 (positions-out). Max with the caps: 313 (positions-in), 508 (positions-out).
  Longer sequences are **silently truncated**, so keep `block_size: 512` for every config
  and check on the store that no event exceeds it.
- **Model size:** 12 layers x 768 is about 85M in the transformer plus vocab x 768 shared
  embedding/output (vocab 32,777 positions-in / 38,921 positions-out at K=4096): about 110M / 115M.
  A smaller option: 6 layers x 256 (8 heads) is about 13-15M; 8 x 128 (4 heads) is about 6M.
  Pick one size and use it for every tokenizer config.
- **Recipe (paper):** 4 GPUs, bf16, batch 128 x accumulate 2 x 4 GPUs = 1024 sequences/step,
  360k steps (~4 passes over 89M), AdamW lr 1.2e-3 -> 1.2e-4 cosine, 2k warmup. Keep the recipe
  identical across tokenizer configs.

**Checks:**
1. `Vocab.build(**vocab_args)`; build 5 samples with `TokenizedMemmapDataset` on the val store;
   `vocab.decode_element_with_pos` must return the stored codes; print one decoded sequence and
   confirm the layout of section 3.
2. Over the whole val store: no sequence reaches `block_size`.
3. `configs/train/nanohep_pflow_tiny.yml` with your `vocab_args`/store/`block_size`: 50 steps on CPU or one GPU,
   loss finite and falling, checkpoint written and reloadable.

---

## 6. Train

```bash
cd $ANY2ANY
srun pixi run python -m hep4m.train_nano_hep -ct configs/train/<your config>.yml        # new run
srun pixi run python -m hep4m.train_nano_hep -edir <run dir>                            # resume from last.ckpt
```

Run under `sbatch -C gpu -A $NERSC_ACCOUNT -q regular -N 1 --gpus-per-node 4 -t 48:00:00`;
resubmit with `-edir` until `max_steps`. Order: positions-in K=4096 and positions-out K=4096
first, then K=1024, then K=256. Log to wandb; record the run dir, steps reached and wall time
per run.

---

## 7. Inference and metric

### 7a. Decoding truth-particle tokens

hep-any2any's `Detokenizer` (`hep4m/decoding/detokenizer.py`) builds HEP4M VQ-VAEs from a
modality dict and can't read heptokens checkpoints, and `hep4m/evaluations/nano_hep_inference_helper.py`
always calls it inline (truth side from the store, reco side from the generated tokens). Pick one:
- **Offline (recommended):** add an option to the nanoHEP inference helper that skips the
  `Detokenizer` and writes the **local** truthpart token IDs instead: per event
  `truthpart_truth_tokens` (flat `[N_truth*3]`), `truthpart_reco_tokens` (flat `[N_reco*3]`, from
  `vocab.decode_element_with_pos`), `truthpart_truth_card`, `truthpart_reco_card`. Then decode in the
  heptokens env with `scripts/pflow_decode_metrics.py` (token IDs -> pt, eta, phi, E, class via the
  truthpart `LitVqVae` + inverse scaler, writes a `pflow_report`-compatible ROOT). That script
  currently reads `truthpart_reco_token_logits` and argmaxes them, and assumes reco and truth have the
  same count. Add a path that reads `truthpart_reco_tokens` directly with its own cardinality.
- **In-process (optional, not needed for the offline route):** port the heptokens tokenizer adapter
  (on S3DF: `/sdf/data/atlas/u/jkrupa/heptokens/HEP4M/hep4m/models/heptokens_adapter.py`; copy it over
  if you go this way) into hep-any2any so a modality dict entry `vae_type: heptokens` works. Because of the
  vector-quantize-pytorch version clash, the heptokens decode has to run in a separate process/env.

### 7b. Generate

`python -m hep4m.eval_hep4m -i configs/infer/nanoHEP-pflow.yml` with `model_type: nano_hep`,
your checkpoint, `nano_hep.tokenized_root` = the store, `split: test`, argmax decoding,
`max_new_tokens` >= 3*16 + 1.

### 7c. Metric

`hep4m.performance.pflow_report.run_report(<decoded physics ROOT>, <outdir>, output_modality="truthpart")`
writes `metrics.json`. Headline: `resolution_iqr_over_median` = (P75 - P25) / median of
pT_reco/pT_truth, where each jet pT is the four-vector sum of its decoded particles; errors are
the bootstrap std. Also report median response, mean predicted multiplicity vs truth, and token
accuracy.

**Cardinality:** nanoHEP chooses how many particles to emit. The S3DF HEP4M reference below used
the true multiplicity. Report nanoHEP as-is, and if you can, also with generation forced to the
true number of particles (stop at N_truth, ignore early EOS), for a direct comparison.

**Floor:** decode the **stored** truthpart tokens of the test split (no model) and compute the
same metric. Every run includes this output-tokenizer floor; report it alongside.

---

## 8. Deliverables

A table: config (positions-in/out, K, Q=4, bits per element = 4*log2 K, +30 for positions-out;
and the continuous baseline of section 9) |
steps | val loss | test IQR/median +- err | median response | mean multiplicity (pred/true) |
token accuracy | run dir. Plus the floor from 7c and the store event counts from 4.

---

## 9. Continuous-input baseline

The reference for "what does tokenizing the inputs cost": the same nanoHEP, recipe, data and
tokenized truth-particle output, but tracks and topos enter as their raw (scaled) features. nanoHEP
only consumes token IDs, so this needs a small model change. Do it after one token config has
trained and evaluated end to end.

1. **Store:** for one store (any K; only its offsets/event order are used), add
   `{track,topo}_cont.npy` = headerless float32 `[n_rows, F]` with the **scaled** features in exactly
   the row order of `{mod}_data.npy`, plus `{mod}_cont_meta.npz` (`n_rows`, `n_feat`).
   Use the **positions-in** feature lists and scalers (tracks F=5, topos F=8, including eta/phi).
   `scripts/build_hep4m_preprocessed.py --cont_only` already does this next to an existing export
   (it asserts identical offsets and event numbers); apply the same path, streaming and sort-order
   changes as in 4.2.
2. **Sequence:** each input element becomes **one** slot holding a new special token
   `<ELEM:track>` / `<ELEM:topo>` (add them to `Vocab`); the interleaving, header and outputs are
   unchanged. The max length drops to 2 + 1 + 65 + 1 + 48 + 1 = 118.
   `TokenizedMemmapDataset` additionally returns `cont` `[block_size, F_max]` (zero-padded, F_max=8)
   aligned with the slots; the collate stacks it.
3. **Model:** per input modality an MLP `Linear(F, n_embd) -> GELU -> Linear(n_embd, n_embd)`.
   In `GPT.forward` / `forward_hidden`, after `x = wte(idx)`, replace `x` at `<ELEM:m>` positions
   with `MLP_m(cont[..., :F_m])` (then add `wpe` as usual). Keep the MLP in fp32 (disable autocast
   for it) so the inputs aren't rounded to bf16. Loss and loss mask are unchanged (inputs carry no loss).
4. **Generation:** `hep4m/models/nano_hep/decode.py::batched_ar_decode` prefills the prompt once
   (`model(cur, return_kv_list=True)`) and then steps with the KV cache, so `cont` is needed only for
   the prefill; generated positions are always ordinary tokens. Its no-cache fallback re-runs
   `model(cur)` on the whole sequence each step: there, pass `cont` (zero-padded to the current
   length) on every call, or force the cached path.
5. **Checks:** a slot's MLP output changes when its features change and nowhere else; a 50-step tiny
   run trains; parameter count = token run minus the track/topo rows of the embedding table plus
   the two MLPs (report both counts).
6. Train with the identical recipe and model size, evaluate exactly as in section 7 (same test
   split, same truth-particle decoding, both cardinality modes), and add it as a row in the table.

S3DF reference (HEP4M, true multiplicity): continuous 0.1285 vs best token input 0.1328, so the
expected gap is a few percent; a much larger gap points to a bug in the continuous path.

## Reference (S3DF, HEP4M 4M model, 20M training events, 99,840 val events, true multiplicity)

| input | IQR/median |
|---|---|
| positions-in d=8/8, Q=3, K=64 / 256 / 1024 / 4096 | 0.148 / 0.138 / 0.135 / 0.135 |
| track d=4, topo d=8, Q=4, K=64 / 256 / 1024 / 4096 | 0.144 / 0.137 / 0.138 / 0.133 |
| continuous inputs (MLP input head, same trunk) | 0.1285 |
