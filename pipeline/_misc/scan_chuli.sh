#!/bin/bash
#SBATCH --job-name=scan_chuli
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/scan_chuli_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/scan_chuli_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=a100g1
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100

PROJ="/home/liyang/BioJiaheWang/scAGCR"
ENV="${PROJ}/scagcr_env"
CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"

source "${CONDA_SH}"
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
conda activate "${ENV}" 2>/dev/null || source activate "${ENV}" 2>/dev/null

cd "${PROJ}"
mkdir -p log results

python scan_all_chuli.py
