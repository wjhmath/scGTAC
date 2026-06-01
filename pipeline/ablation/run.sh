#!/bin/bash
#SBATCH --job-name=abl_i3
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/abl_i3_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/abl_i3_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=a100g2
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100
set -uo pipefail
PROJ="/home/liyang/BioJiaheWang/scAGCR"; ENV="${PROJ}/scagcr_env"
CONDA_SH="/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh"
RESULT_BASE="${PROJ}/results/ablation_iter3"
source "${CONDA_SH}"; export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
source activate "${ENV}" 2>/dev/null || conda activate "${ENV}" 2>/dev/null || true
cd "${PROJ}"; mkdir -p "${RESULT_BASE}" "${PROJ}/log"

EPOCHS=${1:-15}                 # ← 严格复刻=15; 想让 no_cluster 有意义改 25
SEEDS=(1 42 84)
# 完全照 iter3: 6 变体, 拆 feature/edge, 无 graph
VARIANTS=(
  "baseline|"
  "no_cl|--lambda_cl 0"
  "no_cluster|--lambda_cluster 0"
  "no_zinb|--lambda_zinb 0"
  "no_feature_aug|--prob_feature 0"
  "no_edge_aug|--prob_edge 0"
)
DATASETS=( "baron|data/baron/baron.h5ad|14" "multiome|data/multiome/multiome.h5ad|19" )

echo "════ ABLATION iter3复刻 EPOCHS=${EPOCHS} $(date) ════"
ok=0; fail=0; skip=0
for SEED in "${SEEDS[@]}"; do
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r NAME DATA NCLUST <<< "${entry}"
    for ventry in "${VARIANTS[@]}"; do
      IFS='|' read -r VAR FLAGS <<< "${ventry}"
      OUTDIR="${RESULT_BASE}/${VAR}/${NAME}"; mkdir -p "${OUTDIR}"
      LOG="${OUTDIR}/run_seed${SEED}.log"; CKPT="${OUTDIR}/${VAR}_${NAME}_seed${SEED}.pt"
      echo ""; echo ">>> [${NAME}|${VAR}] seed=${SEED} ep=${EPOCHS} $(date +%H:%M:%S)"
      if [ -f "${LOG}" ] && grep -q ARI "${LOG}"; then echo "[SKIP] 已完成"; skip=$((skip+1)); continue; fi
      if python scagcr/main.py --data_path "${DATA}" --n_clusters "${NCLUST}" \
           --epochs "${EPOCHS}" --seed "${SEED}" --save_model_path "${CKPT}" ${FLAGS} > "${LOG}" 2>&1; then
        echo "[OK] ${NAME}/${VAR} $(tail -n1 "${LOG}")"; ok=$((ok+1))
      else echo "[FAIL] ${NAME}/${VAR}"; tail -n5 "${LOG}"; fail=$((fail+1)); fi
    done
  done
done
echo ""; echo "成功=${ok} 失败=${fail} 跳过=${skip}"
