# scGTAC

**Graph Transformer-based Adaptive Contrastive Learning for Deep Single-Cell Clustering**

scGTAC is an end-to-end deep clustering framework for single-cell RNA sequencing (scRNA-seq) data. Unlike existing methods that construct a fixed cell–cell graph before training, scGTAC jointly learns an adaptive graph topology, latent cell representations and cluster assignments, enabling the graph and representations to co-evolve throughout training.

## Key Features

- **Adaptive graph construction**: A multi-head attention constructor with a learnable sparsity threshold infers cell–cell connectivity directly from expression data. The resulting affinity is blended with Graph Transformer encoder attention through a trainable coefficient, creating a feedback loop between graph topology and cell embeddings.

- **Topology-aware contrastive learning**: Beyond standard instance-level contrastive pairs, scGTAC treats neighbours in the adaptively refined graph as affinity-weighted soft positives. Combined with stochastic edge dropping and network dropout, this encourages representations that are simultaneously noise-robust and discriminative across cell types.

- **Count-aware modelling**: A ZINB (zero-inflated negative binomial) decoder models the raw count matrix to preserve the overdispersion and zero inflation characteristic of scRNA-seq data, while a masked reconstruction branch anchors the latent space to the expression manifold.

- **Two-phase training**: In the pre-training phase, reconstruction, ZINB and contrastive losses jointly stabilise the latent space. Cluster centroids are then initialised by K-means, and a DEC-style clustering loss is activated in the joint-training phase. This avoids the premature cluster collapse documented in DEC-based methods.

## Installation

**Requirements**: Python 3.10+, PyTorch 2.0+, PyTorch Geometric

    git clone https://github.com/wjhmath/scGTAC.git
    cd scGTAC

    # Option 1: Create a new conda environment
    conda create -n scgtac python=3.10 -y
    conda activate scgtac
    pip install -r requirements.txt

    # Option 2: Use the existing environment
    conda activate /path/to/scGTAC/scagcr_env

## Quick Start

**Run on a built-in dataset:**

    bash scripts/run_scagcr_repro.sh muraro_pancreas 1

Supported built-in datasets: muraro_pancreas, baron, multiome, Goolam.

**Run on a custom dataset:**

    DATA_PATH=/path/to/your_data.h5ad \
    N_CLUSTERS=10 \
    EPOCHS=200 \
    bash scripts/run_scagcr_custom.sh my_dataset 1

**Run directly with Python:**

    python scgtac/main.py \
      --data_path data/muraro_pancreas/muraro_pancreas.h5ad \
      --n_clusters 10 \
      --pretrain_epochs 20 --epochs 200 \
      --seed 1

**Input format**: An .h5ad file (AnnData) with raw UMI counts in adata.X and cell-type labels in adata.obs['cell_type']. Labels are used only for external evaluation (ARI, NMI, ACC) and never during training.

## Reproducing Paper Results

    # Step 1: Train scGTAC on all 15 benchmark datasets (3 seeds each)
    ./run.sh train

    # Step 2: Run all baseline methods
    ./run.sh baselines

    # Step 3: Run ablation, parameter sweep and robustness experiments
    ./run.sh ablation

    # Step 4: Aggregate results and generate paper figures
    ./run.sh agg
    ./run.sh figs

All results are saved to results/ and figures to paper_figures_final/.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| gcn_dim | 256 | Latent dimension of TransformerConv encoder |
| mlp_dim | 128 | Output dimension of contrastive projection head |
| graph_head | 5 | Number of attention heads in graph constructor |
| phi | 0.45 | Initial learnable sparsity threshold |
| tau | 0.8 | Contrastive temperature |
| lambda_cl | 0.7 | Contrastive loss weight |
| lambda_cluster | 0.3 | Clustering loss weight |
| lambda_zinb | 0.2 | ZINB loss weight |
| alpha | 0.55 | Asymmetric contrastive weight |
| beta | 0.4 | Topology soft-positive weight |
| prob_edge | 0.5 | Edge dropping probability |
| dropout | 0.3 | Encoder dropout rate |
| lr | 0.001 | Adam learning rate |
| pretrain_epochs | 20 | Pre-training epochs (without clustering loss) |
| epochs | 200 | Total training epochs |

## Benchmark Datasets

All 15 benchmark datasets are publicly available:

| Dataset | Cells | Types | Source | Accession |
|---------|-------|-------|--------|-----------|
| Muraro | 2,126 | 10 | GEO | GSE85241 |
| Baron | 8,569 | 14 | GEO | GSE84133 |
| Airway | 7,193 | 7 | GEO | GSE103354 |
| Puram | 3,363 | 8 | GEO | GSE103322 |
| Tonsil | 5,778 | 13 | Broad SCP | SCP2169 |
| Mammary | 13,684 | 13 | GEO | GSE150580 |
| UUO kidney | 6,147 | 17 | GEO | GSE119531 |
| BMMC-B1 | 16,311 | 43 | GEO | GSE194122 |
| Multiome | ~16,000 | 19 | GEO | GSE194122 |
| Kidney ccRCC | 20,748 | 13 | GEO | GSE159115 |
| BMMC-test | 16,750 | 40 | GEO | GSE194122 |
| Goolam | 124 | 8 | ArrayExpress | E-MTAB-3321 |
| Crohn | 39,563 | 27 | Broad SCP | SCP359 |
| 68k PBMC | 68,579 | 11 | 10x Genomics | — |
| Intestine | 18,000 | 5 | GEO | GSE123516 |

Data repositories:
- GEO: https://www.ncbi.nlm.nih.gov/geo/
- Broad Single Cell Portal: https://singlecell.broadinstitute.org/
- ArrayExpress: https://www.ebi.ac.uk/biostudies/arrayexpress/
- 10x Genomics: https://www.10xgenomics.com/datasets/

## Project Structure

    scGTAC/
    ├── scgtac/                  # Core algorithm
    │   ├── config.py            # Default hyperparameters
    │   ├── model.py             # Model architecture and losses
    │   ├── main.py              # Training loop
    │   └── utils.py             # Data loading, preprocessing, evaluation
    ├── baselines/               # Baseline method implementations
    │   ├── run_seurat.py        # Seurat (Leiden clustering)
    │   ├── run_scdeepcluster.py # scDeepCluster
    │   ├── run_scvi.py          # scVI
    │   ├── run_scgnn.py         # scGNN
    │   ├── run_dec.py           # DEC
    │   ├── run_scdsc.py         # scDSC
    │   ├── run_sc3.py           # SC3
    │   ├── run_leiden.py        # Leiden
    │   ├── run_simlr.py         # SIMLR
    │   ├── run_sctag.py         # scTAG
    │   ├── run_scgpt.py         # scGPT
    │   └── aggregate.py         # Aggregate baseline results
    ├── pipeline/                # Experiment pipeline
    │   ├── train/               # Training scripts
    │   ├── baselines/           # Baseline batch scripts
    │   ├── ablation/            # Ablation experiments
    │   ├── param_sweep/         # Hyperparameter sensitivity
    │   ├── scalability/         # Timing and down-sampling
    │   ├── aggregate/           # Result aggregation
    │   └── figures/             # Paper figure generation (fig2–fig8)
    │       ├── scgtac_palette.py # Shared colour palette / Nature style
    │       └── emb/             # Per-method embedding extraction
    ├── scripts/                 # Run entry points
    │   ├── run_scagcr_repro.sh  # Reproducible run on built-in datasets
    │   └── run_scagcr_custom.sh # Run on custom datasets
    ├── requirements.txt
    ├── run.sh                   # One-command experiment runner
    └── README.md

## Evaluation Metrics

scGTAC is evaluated using three standard clustering metrics:

- **ARI** (Adjusted Rand Index): Measures pairwise agreement between predicted and true partitions, corrected for chance. Range: [-1, 1].
- **NMI** (Normalised Mutual Information): Measures shared information between partitions, normalised by entropy. Range: [0, 1].
- **ACC** (Clustering Accuracy): Best accuracy under the optimal one-to-one label mapping found by the Hungarian algorithm. Range: [0, 1].

All metrics are computed using the same evaluation code in scgtac/utils.py for both scGTAC and all baselines to ensure a fair comparison.

## License

MIT License