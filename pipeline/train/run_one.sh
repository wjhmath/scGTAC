#!/bin/bash
#SBATCH --job-name=scAGCR_one
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/scAGCR_one_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/scAGCR_one_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=a100g2
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100
# 用法: sbatch pipeline/train/run_one.sh <name> <data> <k|auto> <epochs> <seed>
set -uo pipefail
PROJ="/home/liyang/BioJiaheWang/scAGCR"; ENV="${PROJ}/scagcr_env"
CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"
RESULT_BASE="${PROJ}/results/scagcr_final"
source "${CONDA_SH}"; export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
source activate "${ENV}" 2>/dev/null || conda activate "${ENV}" 2>/dev/null || true
cd "${PROJ}"; mkdir -p "${RESULT_BASE}" "${PROJ}/log"

NAME="${1:?需要数据集名}"; DATA="${2:?需要数据路径}"
NCLUST="${3:-}"; EP="${4:-200}"; SEED="${5:-1}"
OUTDIR="${RESULT_BASE}/${NAME}"; mkdir -p "${OUTDIR}"
RUN_LOG="${OUTDIR}/run_seed${SEED}.log"; CKPT="${OUTDIR}/${NAME}_seed${SEED}.pt"
echo ">>> [${NAME}] seed=${SEED} ep=${EP}  $(date)"

[ ! -f "${DATA}" ] && { echo "[SKIP] 数据不存在: ${DATA}"; exit 0; }
if [ -f "${RUN_LOG}" ] && grep -q "ARI" "${RUN_LOG}"; then
  echo "[SKIP] 已完成: $(tail -n1 "${RUN_LOG}")"; exit 0; fi

if [ -z "${NCLUST}" ] || [ "${NCLUST}" = "auto" ]; then
  NCLUST=$(python3 -c "
import scanpy as sc
a=sc.read_h5ad('${DATA}', backed='r')
cands=['cell_type','cell_type_label','CellType','celltype','label','labels','Group']
col=next((c for c in cands if c in a.obs.columns), None)
print(a.obs[col].astype(str).nunique() if col else 0)
" 2>/dev/null)
fi
{ [ -z "${NCLUST}" ] || [ "${NCLUST}" = "0" ]; } && { echo "[SKIP] 检测不到簇数"; exit 0; }
echo "    n_clusters=${NCLUST}"

T0=$(date +%s)
if python scagcr/main.py --data_path "${DATA}" --n_clusters "${NCLUST}" \
     --epochs "${EP}" --seed "${SEED}" --save_model_path "${CKPT}" > "${RUN_LOG}" 2>&1; then
  echo "[OK] ${NAME} ($(( $(date +%s)-T0 ))s)  $(tail -n1 "${RUN_LOG}")"
else echo "[FAIL] ${NAME} (详见 ${RUN_LOG})"; tail -n5 "${RUN_LOG}"; exit 1; fi
