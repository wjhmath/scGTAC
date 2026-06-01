#!/bin/bash
#SBATCH --job-name=abl_scan
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/abl_scan_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/abl_scan_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=a100g2
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100
# 用法: sbatch scan_one.sh <name> <data> <k|auto> <epochs>
set -uo pipefail
PROJ="/home/liyang/BioJiaheWang/scAGCR"; ENV="${PROJ}/scagcr_env"
CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"; export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
source activate "${ENV}" 2>/dev/null || conda activate "${ENV}" 2>/dev/null || true
cd "${PROJ}"
NAME="$1"; DATA="$2"; NCLUST="$3"; EP="${4:-200}"
CK="/tmp/abl_scratch_${SLURM_JOB_ID:-$$}.pt"   # checkpoint 丢弃, 出图只用 log

if [ -z "${NCLUST}" ] || [ "${NCLUST}" = "auto" ]; then
  NCLUST=$(python3 -c "
import scanpy as sc
a=sc.read_h5ad('${DATA}', backed='r')
cands=['cell_type','cell_type_label','CellType','celltype','label','labels','Group']
col=next((c for c in cands if c in a.obs.columns), None)
print(a.obs[col].astype(str).nunique() if col else 0)" 2>/dev/null)
fi
{ [ -z "${NCLUST}" ] || [ "${NCLUST}" = "0" ]; } && { echo "[SKIP] 无簇数 ${NAME}"; exit 0; }
echo "=== ${NAME} k=${NCLUST} ep=${EP} ==="

declare -A FLAGS=(
  [full]="" [wo_cl]="--lambda_cl 0" [wo_zinb]="--lambda_zinb 0"
  [wo_cluster]="--lambda_cluster 0" [wo_aug]="--prob_feature 0 --prob_edge 0" [wo_graph]="--no_graph"
)
run(){ local V=$1 S=$2 OUT="${PROJ}/results/ablation/$V/${NAME}"; mkdir -p "$OUT"
  local LOG="$OUT/run_seed$S.log"
  if [ -f "$LOG" ] && grep -q ARI "$LOG"; then echo "[skip] $V s$S"; return; fi
  if python scagcr/main.py --data_path "${DATA}" --n_clusters "${NCLUST}" --epochs "${EP}" \
       --seed "$S" --save_model_path "$CK" ${FLAGS[$V]} > "$LOG" 2>&1; then
    echo "[ok] $V s$S $(tail -n1 "$LOG")"; else echo "[fail] $V s$S"; tail -n3 "$LOG"; fi; }

for V in full wo_cl wo_zinb wo_cluster wo_aug wo_graph; do run "$V" 1; done   # 热图: seed1 全变体
for S in 42 84; do for V in full wo_cl; do run "$V" "$S"; done; done           # 主图: full/no_cl 补2seed
rm -f "$CK"
