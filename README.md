# scAGCR

`scAGCR` (single-cell Adaptive Graph-Contrastive Refinement) is a graph-based deep clustering method for single-cell RNA-seq data. The method combines:

- learnable graph construction
- transformer-based graph refinement
- graph-aware contrastive learning
- staged clustering refinement
- ZINB-based count-aware regularization

This `repo/` folder is a cleaned package-style extraction centered on the scAGCR method and the core figure-generation scripts used in the manuscript.

## Structure

```text
repo/
├── README.md
├── requirements.txt
├── scagcr/
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   ├── model.py
│   └── config.py
├── scripts/
│   ├── run_scagcr_repro.sh
│   └── run_scagcr_custom.sh
├── figures/
└── examples/
```

## Installation

```bash
pip install -r repo/requirements.txt
```

Note: `torch-geometric` must match your installed PyTorch/CUDA stack.

## Input data

The method expects a Scanpy-readable `.h5ad` file.

### Label columns
The loader will search for labels in `adata.obs` using these names, in order:

- `cell_type`
- `cell_type_label`
- `CellType`
- `celltype`
- `label`
- `labels`
- `Group`

### Built-in preprocessing
At runtime the loader will:

- subsample to 20,000 cells if the dataset is larger
- normalize and log-transform if expression values are not already in a log-scale range
- select top 2,000 highly variable genes when `n_vars > 2000`
- use a fixed 80/20 train-test split with `random_state=1`

## Quick start

### Reproduce a default run on a supported dataset

```bash
bash repo/scripts/run_scagcr_repro.sh muraro_pancreas 1
```

Supported dataset names in the bundled script:

- `baron`
- `zheng68k`
- `multiome`
- `muraro_pancreas`

### Run on a custom dataset

```bash
DATA_PATH=/path/to/your_data.h5ad \
N_CLUSTERS=10 \
EPOCHS=150 \
TAU=0.8 \
LAMBDA_CL=0.7 \
DROPOUT=0.3 \
bash repo/scripts/run_scagcr_custom.sh my_dataset 1
```

## Outputs

The scripts write:

- a training log under `results/.../logs/`
- a checkpoint under `results/.../checkpoints/`

The main training script prints final clustering metrics as a Python dict containing:

- `CA`
- `NMI`
- `ARI`

## Plotting code

The main manuscript plotting scripts can be organized under `repo/figures/`. The most useful ones to carry over are:

- benchmark comparison plots
- ablation plots
- parameter-sensitivity plots
- success / failure case-study plots
- qualitative clustering panel plots

These scripts are especially useful when they are written with frozen source data so they can reproduce figures without depending on the full benchmark workspace.

## Core dependencies

Minimal runtime dependencies for scAGCR itself:

- `torch`
- `numpy`
- `scipy`
- `scanpy`
- `anndata`
- `scikit-learn`
- `torch-geometric`

Additional plotting dependencies included in `repo/requirements.txt`:

- `pandas`
- `matplotlib`
- `scienceplots`
- `umap-learn`

## Notes

This package-style extraction is intended to make scAGCR easier to understand and reuse without the rest of the original multi-method benchmark repository.
