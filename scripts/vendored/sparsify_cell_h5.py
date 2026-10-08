#!/usr/bin/env python
"""Convert one preprocessed cell HDF5 (preprocess_hep4m_cells.py output) to a
sparse on-disk format that fits 20M events in RAM.

Output files in --output_dir, for input <stem>.h5:
  <stem>.values.npy        float32 [nnz]     non-zero pixel values (bit-exact copies)
  <stem>.index.npy         int16   [nnz]     flat index into the (n_patches * n_pixels) grid
  <stem>.offsets.npy       int64   [N + 1]   event i owns values[offsets[i]:offsets[i+1]]
  <stem>.positions.npy     float32 [N, n_patches, 3]   unchanged (memory-mapped at train time)
  <stem>.event_number.npy  int64   [N]
  <stem>.meta.json         {"n_events", "n_patches", "n_pixels"}  written last = completion marker

"Non-zero" is decided on the raw float32 bits, so -0.0 is kept as-is. Every
event is rebuilt from the sparse arrays and compared bit-for-bit against the
HDF5 before anything is written; any mismatch (or a NaN) aborts.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

BLOCK = 12_500  # events per read; a multiple of the HDF5 chunk height (3125)


def _save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(path)


def convert(h5_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = h5_path.stem
    with h5py.File(h5_path, "r") as f:
        patches = f["cell_patches"]
        n_events, n_patches, n_pixels = patches.shape
        if n_patches * n_pixels > np.iinfo(np.int16).max:
            raise ValueError(f"grid of {n_patches * n_pixels} does not fit int16 indices")

        values, index, counts = [], [], []
        for start in range(0, n_events, BLOCK):
            block = patches[start : start + BLOCK]
            if np.isnan(block).any():
                raise ValueError(f"NaN in {h5_path} events {start}..{start + len(block)}")
            flat = block.reshape(len(block), -1)
            ev, idx = np.nonzero(flat.view(np.uint32))
            vals = flat[ev, idx]
            cnt = np.bincount(ev, minlength=len(block))

            rebuilt = np.zeros_like(flat)
            rebuilt[ev, idx] = vals
            if not np.array_equal(rebuilt.view(np.uint32), flat.view(np.uint32)):
                raise AssertionError(f"sparse round-trip mismatch in {h5_path} block at {start}")

            values.append(vals.astype(np.float32, copy=False))
            index.append(idx.astype(np.int16))
            counts.append(cnt)

        positions = f["patch_positions"][:]
        event_number = f["event_number"][:]

    counts = np.concatenate(counts)
    offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    _save(out_dir / f"{stem}.values.npy", np.concatenate(values))
    _save(out_dir / f"{stem}.index.npy", np.concatenate(index))
    _save(out_dir / f"{stem}.offsets.npy", offsets)
    _save(out_dir / f"{stem}.positions.npy", positions.astype(np.float32, copy=False))
    _save(out_dir / f"{stem}.event_number.npy", event_number.astype(np.int64, copy=False))
    meta = {"n_events": int(n_events), "n_patches": int(n_patches), "n_pixels": int(n_pixels)}
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta))
    return {**meta, "nnz": int(offsets[-1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input cell HDF5 file")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    info = convert(Path(args.input), Path(args.output_dir))
    print(f"{args.input}: {info}")


if __name__ == "__main__":
    main()
