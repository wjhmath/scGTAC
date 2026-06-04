#!/bin/bash
#SBATCH --job-name=scalability
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/scalability_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/scalability_%j.err
#SBATCH --nodes=1 -n 1 --cpus-per-task=4 --mem=64G --time=48:00:00
#SBATCH --qos=a100g2 --gres=gpu:a100:1 -p a100
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scAGCR; ENV=$PROJ/scagcr_env
source /home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
source activate "$ENV" 2>/dev/null || conda activate "$ENV"; cd "$PROJ"
CHULI=/home/liyang/BioJiaheWang/RARECELL/data/chuli
EP=50; CK=/tmp/scale_${SLURM_JOB_ID:-$$}.pt
CSV=$PROJ/results/scalability/timing.csv
[ -f "$CSV" ] || echo "dataset,n_cells,n_clusters,seconds" > "$CSV"
DATASETS=(
  "Goolam|${CHULI}/Goolam.h5ad|"
  "muraro_pancreas|data/muraro_pancreas/muraro_pancreas.h5ad|10"
  "GSE103322|${CHULI}/GSE103322.h5ad|"
  "GSE123516_labeled|${CHULI}/GSE123516_labeled.h5ad|"
  "GSE119531|${CHULI}/GSE119531.h5ad|"
  "Crohn|${CHULI}/Crohn.h5ad|"
  "baron|data/baron/baron.h5ad|14"
  "GSE103354|${CHULI}/GSE103354.h5ad|"
  "GSE150580_Mammary|${CHULI}/GSE150580_Mammary.h5ad|"
  "GSE159115_ccRCC|${CHULI}/GSE159115_ccRCC.h5ad|"
  "Tonsil|${CHULI}/Tonsil.h5ad|"
  "GSE194122_PBMC_Bench_1|${CHULI}/GSE194122_PBMC_Bench_1.h5ad|"
  "10X_PBMC|${CHULI}/10X_PBMC.h5ad|"
  "GSE194122_PBMC_Test|${CHULI}/GSE194122_PBMC_Test.h5ad|"
  "multiome|data/multiome/multiome.h5ad|19"
  "68kPBMC|${CHULI}/68kPBMC.h5ad|"
  "zheng68k|data/zheng68k/pbmc68k.h5ad|11"
)
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r NAME DATA NCLUST <<< "$entry"
  [ ! -f "$DATA" ] && { echo "[skip] $NAME: 无文件"; continue; }
  grep -q "^$NAME," "$CSV" && { echo "[skip] $NAME: 已计时"; continue; }
  NCELLS=$(python3 -c "import scanpy as sc;print(sc.read_h5ad('$DATA',backed='r').n_obs)")
  if [ -z "$NCLUST" ]; then
    NCLUST=$(python3 -c "import scanpy as sc;a=sc.read_h5ad('$DATA',backed='r')
c=[x for x in ['cell_type','cell_type_label','CellType','celltype','label','labels','Group'] if x in a.obs.columns]
print(a.obs[c[0]].astype(str).nunique() if c else 0)")
  fi
  echo ">>> $NAME ($NCELLS cells, k=$NCLUST)"
  T0=$(date +%s)
  python scagcr/main.py --data_path "$DATA" --n_clusters "$NCLUST" --epochs "$EP" \
    --seed 1 --save_model_path "$CK" > /dev/null 2>&1
  T1=$(date +%s); SECS=$((T1-T0))
  echo "$NAME,$NCELLS,$NCLUST,$SECS" >> "$CSV"
  echo "    → ${SECS}s"
done
rm -f "$CK"; echo "=== 完成 ==="; cat "$CSV"
