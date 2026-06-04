#!/bin/bash
#SBATCH --job-name=abl_fill
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/abl_fill_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/abl_fill_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=a100g2
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scAGCR; ENV=$PROJ/scagcr_env
source /home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
source activate "$ENV" 2>/dev/null || conda activate "$ENV"; cd "$PROJ"
NAME="$1"; DATA="$2"; NCLUST="${3:-auto}"; EP="${4:-200}"; CK=/tmp/fill_${SLURM_JOB_ID:-$$}.pt
if [ "$NCLUST" = auto ]; then NCLUST=$(python3 -c "
import scanpy as sc;a=sc.read_h5ad('$DATA',backed='r')
c=[x for x in ['cell_type','cell_type_label','CellType','celltype','label','labels','Group'] if x in a.obs.columns]
print(a.obs[c[0]].astype(str).nunique() if c else 0)"); fi
echo "数据集=$NAME k=$NCLUST ep=$EP"
declare -A F=( [full]="" [wo_cl]="--lambda_cl 0" [wo_aug]="--prob_feature 0 --prob_edge 0" [wo_graph]="--no_graph" )
for V in full wo_cl wo_aug wo_graph; do for S in 1 42 84; do
  O=$PROJ/results/ablation/$V/$NAME; mkdir -p "$O"; L=$O/run_seed$S.log
  if [ -f "$L" ] && grep -q "ARI" "$L"; then echo "[skip] $V s$S"; continue; fi
  if python scagcr/main.py --data_path "$DATA" --n_clusters "$NCLUST" --epochs "$EP" \
       --seed "$S" --save_model_path "$CK" ${F[$V]} > "$L" 2>&1; then
    echo "[ok] $V s$S $(tail -1 "$L")"; else echo "[fail] $V s$S"; tail -3 "$L"; fi
done; done; rm -f "$CK"
