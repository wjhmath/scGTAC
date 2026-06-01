#!/bin/bash
# setup_baseline_envs.sh —— 按框架家族建独立 conda 环境(避免依赖打架)。
# 在登录节点(有网)上跑一次即可。GPU 节点不一定有网, 别在 sbatch 里装包。
#
# 4 个新算法的环境归属:
#   SC3            -> R + Bioconductor   (env: bl_r)
#   DESC           -> TensorFlow/Keras   (env: bl_tf)
#   scDeepCluster  -> TensorFlow(原版)或 PyTorch(重实现) (env: bl_tf 或 bl_torch)
#   scTAG          -> PyTorch + DGL      (env: bl_torch)
#   Leiden/Seurat  -> 直接用你现有的 scagcr_env, 不用建
set -e

CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"

# 国内超算下载慢可先配镜像(清华源), 不需要就注释掉:
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "===== [1/3] bl_torch (scTAG / PyTorch 版 scDeepCluster) ====="
conda create -y -n bl_torch python=3.9
conda activate bl_torch
# 按你的 CUDA 版本选 torch(你 scAGCR 用 cu121, 这里给 cu118 通用版, 不行就换)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install scanpy anndata scikit-learn scipy pandas h5py munkres
conda deactivate

echo "===== [2/3] bl_tf (DESC / 原版 scDeepCluster) ====="
conda create -y -n bl_tf python=3.8
conda activate bl_tf
pip install "tensorflow==2.10.*" "keras==2.10.*"
pip install desc scanpy anndata scikit-learn scipy pandas h5py
conda deactivate

echo "===== [3/3] bl_r (SC3) ====="
conda create -y -n bl_r -c conda-forge -c bioconda \
    r-base=4.2 bioconductor-sc3 bioconductor-singlecellexperiment \
    r-matrix anndata2ri r-reticulate
conda deactivate

echo "全部环境就绪。Leiden 用现有 scagcr_env 即可。"
