"""Cell preprocessing (ported from heptokens-cocoa workflow/hep4m_cell_preprocess_20M.smk @ cc20cbc).

Builds, under cells_h5_dir (dense HDF5, ds=1) and cells_sparse_dir (sparse .npy):
  train/singlejet_{270..469}   200 files x ~100k = ~20M events
  val/singlejet_10             val_100K_cells256 (never trained on)
  test/singlejet_{0..9}        test_1M_cells256_isInfFalse (never trained on)

The 2M-event training set (cells_h5_2M_dir, singlejet_270..289) is linked into train/, as in the
original workflow. In the original, those 20 files came from an earlier project and were linked, not
regenerated; preprocess_cells_2M (not in the original workflow) regenerates them with the same script
and settings. The original workflow records that this agrees with the stored files on singlejet_270
(2000 events): identical occupancy, positions and event numbers, energies within 1.2e-7 relative.

Usage:
  snakemake -s workflow/vendored/cells_preprocess.smk -n
  snakemake -s workflow/vendored/cells_preprocess.smk all_cells_20M_sparse --cores 100 --keep-going
"""


include: "common.smk"


ROOT_BASE = cfg_path("cocoa_root")
ROOT_DIRS = {
    "train": ROOT_BASE / "train_topup2_20M_cells256",
    "val": ROOT_BASE / "val_100K_cells256",
    "test": ROOT_BASE / "test_1M_cells256_isInfFalse",
}
OUT_ROOT = cfg_path("cells_h5_dir")
EXISTING_2M_DIR = cfg_path("cells_h5_2M_dir")
SPARSE_ROOT = cfg_path("cells_sparse_dir")

SPLIT_INDICES = {
    "train": list(range(270, 470)),
    "val": [10],
    "test": list(range(10)),
}
EXISTING_2M_INDICES = list(range(270, 290))

CPU_SLURM = f"{job_prefix('cpu')} -t 04:00:00 --mem=16G -D {REPO_DIR}"

TARGETS = [
    str(OUT_ROOT / f"{split}/singlejet_{idx}.h5")
    for split, idxs in SPLIT_INDICES.items()
    for idx in idxs
]
SPARSE_TARGETS = [
    str(SPARSE_ROOT / f"{split}/singlejet_{idx}.meta.json")
    for split, idxs in SPLIT_INDICES.items()
    for idx in idxs
]
NEW_TRAIN = "|".join(str(i) for i in SPLIT_INDICES["train"] if i not in EXISTING_2M_INDICES)


rule all_cells_20M:
    input:
        TARGETS,


rule all_cells_20M_sparse:
    input:
        SPARSE_TARGETS,


rule sparsify_cells:
    """HDF5 -> sparse .npy (scripts/sparsify_cell_h5.py); bit-exact round trip checked per event."""
    input:
        h5=str(OUT_ROOT / "{split}/singlejet_{idx}.h5"),
    output:
        meta=str(SPARSE_ROOT / "{split}/singlejet_{idx}.meta.json"),
    wildcard_constraints:
        split="train|val|test",
        idx=r"\d+",
    params:
        python_bin=PYTHON_BIN,
        slurm=f"{job_prefix('cpu')} -t 00:30:00 --mem=8G -D {REPO_DIR}",
        out_dir=lambda wc: str(SPARSE_ROOT / wc.split),
        log_dir=str(SPARSE_ROOT / "logs"),
    shell:
        """
        mkdir -p {params.log_dir}
        {params.slurm} -J cellsparse_{wildcards.split}_{wildcards.idx} \
            -o {params.log_dir}/{wildcards.split}_{wildcards.idx}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/sparsify_cell_h5.py \
                --input {input.h5} --output_dir {params.out_dir}"
        """


rule link_existing_2M:
    input:
        str(EXISTING_2M_DIR / "singlejet_{idx}.h5"),
    output:
        str(OUT_ROOT / "train/singlejet_{idx}.h5"),
    wildcard_constraints:
        idx="|".join(str(i) for i in EXISTING_2M_INDICES),
    localrule: True
    shell:
        "ln -s {input} {output}"


rule preprocess_cells_2M:
    input:
        root=lambda wc: str(ROOT_DIRS["train"] / f"singlejet_{wc.idx}_100kseg_bw100.0.root"),
    output:
        h5=str(EXISTING_2M_DIR / "singlejet_{idx}.h5"),
    wildcard_constraints:
        idx="|".join(str(i) for i in EXISTING_2M_INDICES),
    params:
        python_bin=PYTHON_BIN,
        slurm=CPU_SLURM,
        log_dir=str(EXISTING_2M_DIR / "logs"),
    shell:
        """
        mkdir -p {params.log_dir}
        {params.slurm} -J cellpre_2M_{wildcards.idx} \
            -o {params.log_dir}/train_{wildcards.idx}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/preprocess_hep4m_cells.py \
                --input {input.root} \
                --output {output.h5} \
                --downsample 1 \
                --visualize 0"
        """


rule preprocess_cells_ds1:
    input:
        root=lambda wc: str(ROOT_DIRS[wc.split] / f"singlejet_{wc.idx}_100kseg_bw100.0.root"),
    output:
        h5=str(OUT_ROOT / "{split}/singlejet_{idx}.h5"),
    wildcard_constraints:
        # train excludes 270..289 (linked by link_existing_2M instead).
        split="train|val|test",
        idx=f"{NEW_TRAIN}|" + "|".join(str(i) for i in SPLIT_INDICES["test"] + SPLIT_INDICES["val"]),
    params:
        python_bin=PYTHON_BIN,
        slurm=CPU_SLURM,
        log_dir=str(OUT_ROOT / "logs"),
    shell:
        """
        mkdir -p {params.log_dir}
        {params.slurm} -J cellpre_{wildcards.split}_{wildcards.idx} \
            -o {params.log_dir}/{wildcards.split}_{wildcards.idx}_%j.log \
            --wrap="{params.python_bin} {SCRIPTS}/preprocess_hep4m_cells.py \
                --input {input.root} \
                --output {output.h5} \
                --downsample 1 \
                --visualize 0"
        """
