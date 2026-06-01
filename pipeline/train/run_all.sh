#!/bin/bash
# 用法: ./pipeline/train/run_all.sh [seed] [epochs_override]
#   把全部 17 个数据集各投一个 GPU 作业 (并行排队). run_one 内部自动跳过已完成的.
set -uo pipefail
PROJ="/home/liyang/BioJiaheWang/scAGCR"
CHULI="/home/liyang/BioJiaheWang/RARECELL/data/chuli"
SEED=${1:-1}; EP_OVERRIDE=${2:-}
DATASETS=(
  "baron|data/baron/baron.h5ad|14|200"
  "muraro_pancreas|data/muraro_pancreas/muraro_pancreas.h5ad|10|200"
  "multiome|data/multiome/multiome.h5ad|19|200"
  "zheng68k|data/zheng68k/pbmc68k.h5ad|11|50"
  "GSE103322|${CHULI}/GSE103322.h5ad||200"
  "Tonsil|${CHULI}/Tonsil.h5ad||200"
  "GSE150580_Mammary|${CHULI}/GSE150580_Mammary.h5ad||200"
  "GSE119531|${CHULI}/GSE119531.h5ad||200"
  "GSE194122_PBMC_Bench_1|${CHULI}/GSE194122_PBMC_Bench_1.h5ad||200"
  "GSE159115_ccRCC|${CHULI}/GSE159115_ccRCC.h5ad||200"
  "10X_PBMC|${CHULI}/10X_PBMC.h5ad||200"
  "GSE103354|${CHULI}/GSE103354.h5ad||200"
  "68kPBMC|${CHULI}/68kPBMC.h5ad||200"
  "Crohn|${CHULI}/Crohn.h5ad||200"
  "GSE123516_labeled|${CHULI}/GSE123516_labeled.h5ad||200"
  "GSE194122_PBMC_Test|${CHULI}/GSE194122_PBMC_Test.h5ad||200"
  "Goolam|${CHULI}/Goolam.h5ad||200"
)
n=0
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r NAME DATA NCLUST EP <<< "${entry}"
  [ -n "${EP_OVERRIDE}" ] && EP="${EP_OVERRIDE}"
  sbatch "${PROJ}/pipeline/train/run_one.sh" "${NAME}" "${DATA}" "${NCLUST:-auto}" "${EP}" "${SEED}"
  n=$((n+1))
done
echo "已提交 ${n} 个作业 (seed=${SEED}). squeue -u liyang 查看."
