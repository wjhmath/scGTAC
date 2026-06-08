#!/bin/bash
# 用法: ./pipeline/ablation/scan_all.sh [maxcells=40000] [epochs=200]
set -uo pipefail
PROJ="/home/liyang/BioJiaheWang/scGTAC"; CHULI="/home/liyang/BioJiaheWang/RARECELL/data/chuli"
MAXCELLS=${1:-40000}; EP=${2:-200}
DATASETS=(
  "baron|data/baron/baron.h5ad|14" "muraro_pancreas|data/muraro_pancreas/muraro_pancreas.h5ad|10"
  "multiome|data/multiome/multiome.h5ad|19" "zheng68k|data/zheng68k/pbmc68k.h5ad|11"
  "GSE103322|${CHULI}/GSE103322.h5ad|" "Tonsil|${CHULI}/Tonsil.h5ad|"
  "GSE150580_Mammary|${CHULI}/GSE150580_Mammary.h5ad|" "GSE119531|${CHULI}/GSE119531.h5ad|"
  "GSE194122_PBMC_Bench_1|${CHULI}/GSE194122_PBMC_Bench_1.h5ad|" "GSE159115_ccRCC|${CHULI}/GSE159115_ccRCC.h5ad|"
  "10X_PBMC|${CHULI}/10X_PBMC.h5ad|" "GSE103354|${CHULI}/GSE103354.h5ad|"
  "68kPBMC|${CHULI}/68kPBMC.h5ad|" "Crohn|${CHULI}/Crohn.h5ad|"
  "GSE123516_labeled|${CHULI}/GSE123516_labeled.h5ad|" "GSE194122_PBMC_Test|${CHULI}/GSE194122_PBMC_Test.h5ad|"
  "Goolam|${CHULI}/Goolam.h5ad|"
)
n=0
for e in "${DATASETS[@]}"; do
  IFS='|' read -r NAME DATA NCLUST <<< "$e"
  [ ! -f "$DATA" ] && { echo "跳过(无文件) $NAME"; continue; }
  NC=$(python3 -c "import scanpy as sc;print(sc.read_h5ad('$DATA',backed='r').n_obs)" 2>/dev/null)
  if [ -n "$NC" ] && [ "$NC" -gt "$MAXCELLS" ]; then echo "跳过(过大 ${NC} cells) $NAME"; continue; fi
  sbatch "${PROJ}/pipeline/ablation/scan_one.sh" "$NAME" "$DATA" "${NCLUST:-auto}" "$EP"
  echo "已投 $NAME (${NC} cells)"; n=$((n+1))
done
echo "共提交 ${n} 个数据集作业. squeue -u liyang 查看."
