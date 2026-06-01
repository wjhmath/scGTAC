#!/bin/bash
# ===== scAGCR 全流程单一入口 =====
#   用法: ./run.sh <阶段>   阶段: train | baselines | ablation | agg | figs
#   train/baselines/ablation 是 SLURM 异步作业 → 投出去等 squeue 跑完, 再 agg → figs
set -euo pipefail
cd /home/liyang/BioJiaheWang/scAGCR
CONDA_SH=/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
case "${1:-help}" in
  train)      bash pipeline/train/run_all.sh ;;
  baselines)  sbatch pipeline/baselines/run_all.sh; sbatch pipeline/baselines/fill.sh ;;
  ablation)   bash pipeline/ablation/scan_all.sh ;;
  agg)        source "$CONDA_SH"; conda activate "$PWD/scagcr_env"
              python pipeline/aggregate/scagcr_final.py
              python pipeline/aggregate/chuli.py
              python pipeline/aggregate/ablation.py ;;
  figs)       source "$CONDA_SH"; conda activate "$PWD/scagcr_env"
              python pipeline/figures/fig1.py
              python pipeline/figures/fig3.py
              python pipeline/figures/fig4.py ;;
  *)          echo "用法: ./run.sh {train|baselines|ablation|agg|figs}";;
esac
