#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --output=/home/liyang/BioJiaheWang/scGTAC/log/param_sweep_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scGTAC/log/param_sweep_%j.err
#SBATCH --nodes=1 -n 1 --cpus-per-task=4 --mem=64G --time=24:00:00
#SBATCH --qos=a100g1 --gres=gpu:a100:1 -p a100
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scGTAC; ENV=$PROJ/scagcr_env
source /home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
source activate "$ENV" 2>/dev/null || conda activate "$ENV"; cd "$PROJ"
DATA=data/baron/baron.h5ad; K=14; EP=200; SEEDS=(1 42 84)
RESULT=$PROJ/results/param_sweep; CK=/tmp/sweep_${SLURM_JOB_ID:-$$}.pt
CFG=scgtac/config.py; BAK=${CFG}.sweep_bak; cp "$CFG" "$BAK"
trap 'cp "$BAK" "$CFG"; rm -f "$CK"' EXIT
restore(){ cp "$BAK" "$CFG"; }
setparam(){ sed -i "s/'$1': *[0-9.]*/'$1': $2/" "$CFG"; }
run(){
  local TAG="${1}_${2}"
  local ODIR="$RESULT/$TAG/baron"
  mkdir -p "$ODIR"
  for S in "${SEEDS[@]}"; do
    local LOG="$ODIR/run_seed${S}.log"
    [ -f "$LOG" ] && grep -q ARI "$LOG" && { echo "[skip] $TAG s$S"; continue; }
    python scgtac/main.py --data_path "$DATA" --n_clusters "$K" --epochs "$EP" \
      --seed "$S" --save_model_path "$CK" > "$LOG" 2>&1 \
      && echo "[ok] $TAG s$S $(tail -1 "$LOG")" || echo "[fail] $TAG s$S"
  done
}
echo "=== 参数扫描开始 $(date) ==="
for V in 0.2 0.4 0.6 0.8 1.0; do restore; setparam tau "$V"; run tau "$V"; done
echo "--- tau 完成 ---"
for V in 0.1 0.3 0.5 0.7 0.9; do restore; setparam lambda_cl "$V"; run lambda_cl "$V"; done
echo "--- lambda_cl 完成 ---"
for V in 0.0 0.1 0.2 0.3 0.5; do restore; setparam dropout "$V"; run dropout "$V"; done
echo "=== 全部完成 $(date) ==="
