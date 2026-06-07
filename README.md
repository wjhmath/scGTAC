# scGTAC

**scGTAC: Graph Transformer-based Adaptive Contrastive Learning for Deep Single-Cell Clustering**

## Overview

scGTAC is an end-to-end deep clustering framework for scRNA-seq data. It jointly learns an adaptive cell–cell graph, latent cell representations and cluster assignments through a unified two-phase training process.

- **Adaptive graph construction** — a multi-head attention constructor with a learnable sparsity threshold replaces the fixed KNN graph, enabling graph–representation co-evolution.
- **Topology-aware contrastive learning** — graph neighbours serve as affinity-weighted soft positives, injecting topological information into the contrastive objective.
- **Count-aware modelling** — a ZINB decoder preserves overdispersion and zero inflation of raw counts.
- **Two-phase training** — reconstruction, ZINB and contrastive losses stabilise the latent space before the clustering loss is activated.

## Installation

    git clone https://github.com/wjhmath/scGTAC.git
    cd scGTAC
    conda create -n scgtac python=3.10 -y
    conda activate scgtac
    pip install -r requirements.txt

## Quick Start

Run on a single dataset:

    bash scripts/run_scagcr_repro.sh muraro_pancreas 1

Run with custom data:

    DATA_PATH=/path/to/data.h5ad N_CLUSTERS=10 bash scripts/run_scagcr_custom.sh my_dataset 1

Input: an `.h5ad` file with raw UMI counts in `adata.X` and cell-type labels in `adata.obs['cell_type']` (labels are used only for evaluation).

## Reproducing Paper Results

    ./run.sh train       # Train scGTAC on all 15 benchmark datasets
    ./run.sh baselines   # Run baseline methods
    ./run.sh ablation    # Run ablation experiments
    ./run.sh figs        # Generate paper figures

## Benchmark Datasets

All 15 benchmark datasets are publicly available:

| Dataset | Source | Accession |
|---------|--------|-----------|
| Muraro | GEO | GSE85241 |
| Baron | GEO | GSE84133 |
| Airway | GEO | GSE103354 |
| Puram | GEO | GSE103322 |
| Tonsil | Broad SCP | SCP2169 |
| Mammary | GEO | GSE150580 |
| UUO kidney | GEO | GSE119531 |
| BMMC-B1 / BMMC-test / Multiome | GEO | GSE194122 |
| Kidney ccRCC | GEO | GSE159115 |
| Goolam | ArrayExpress | E-MTAB-3321 |
| Crohn | Broad SCP | SCP359 |
| 68k PBMC | 10x Genomics | — |
| Intestine | GEO | GSE123516 |

## Project Structure

    scGTAC/
    ├── scagcr/          # Core model (config, model, main, utils)
    ├── baselines/       # Baseline implementations
    ├── pipeline/        # Experiment scripts and figure generation
    ├── scripts/         # Run scripts
    ├── requirements.txt
    └── run.sh

## Citation

If you find scGTAC useful in your research, please cite:

    @article{wang2026scgtac,
      title={scGTAC: Graph Transformer-based Adaptive Contrastive Learning
             for Deep Single-Cell Clustering},
      author={Wang, Jiahe and Wu, Yan and Li, Yang and Zhang, Yapu},
      journal={Journal of the Operations Research Society of China},
      year={2026}
    }

## License

MIT License