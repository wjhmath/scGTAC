#!/bin/bash
#SBATCH --job-name=bl_blanks
#SBATCH --output=/home/liyang/BioJiaheWang/scGTAC/log/bl_blanks_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scGTAC/log/bl_blanks_%j.err
#SBATCH -N 1 -n 1 --cpus-per-task=4 --mem=64G --time=12:00:00
#SBATCH --qos=a100g1 --gres=gpu:a100:1 -p a100
PROJ=/home/liyang/BioJiaheWang/scGTAC
CHULI=/home/liyang/BioJiaheWang/RARECELL/data/chuli
source /home/liyang/miniconda3/bin/activate $PROJ/envs/bl_torch
cd $PROJ/baselines
run(){ M=$1; DS=$2; OUT=../results/baselines_chuli/$M; mkdir -p "$OUT"
  [ -f "$OUT/${DS}_seed1_metrics.json" ] && { echo "[skip] $M/$DS exists"; return; }
  echo ">>> $M | $DS | $(date +%T) (无超时)"
  python run_${M}.py --data "$CHULI/${DS}.h5ad" --out "$OUT" --dataset "$DS" --seed 1 \
    && echo "[done] $M/$DS" || echo "[FAIL] $M/$DS"; }
run scdeepcluster Crohn
run scdsc GSE159115_ccRCC
run scdsc Crohn
run scdsc 68kPBMC      # 最大,可能仍 OOM/失败 → 那就如实标 —
echo "BLANKS FILL DONE $(date)"
