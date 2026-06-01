#!/bin/bash
#SBATCH --job-name=bl_fill
#SBATCH --output=/home/liyang/BioJiaheWang/scAGCR/log/bl_fill_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scAGCR/log/bl_fill_%j.err
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=a100g1
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100

METHOD=$1                       # scdeepcluster 或 scdsc
PROJ=/home/liyang/BioJiaheWang/scAGCR
source /home/liyang/miniconda3/bin/activate $PROJ/envs/bl_torch
cd $PROJ/baselines
OUT=../results/baselines_chuli/$METHOD
mkdir -p "$OUT"
SKIP="GSE164378 GSE194122_BMMC"

# 清理超过 3 小时的残留旧锁(死进程留下的)
find "$OUT" -name "*.lock" -mmin +180 -delete 2>/dev/null

for f in $(ls -S -r /home/liyang/BioJiaheWang/RARECELL/data/chuli/*.h5ad); do
  DS=$(basename "$f" .h5ad)
  echo "$SKIP" | grep -qw "$DS" && continue
  [ -f "$OUT/${DS}_seed1_metrics.json" ] && continue
  if ( set -o noclobber; echo $$ > "$OUT/${DS}.lock" ) 2>/dev/null; then
    echo ">>> $METHOD | $DS | $(date +%T)"
    python run_${METHOD}.py --data "$f" --out "$OUT" --dataset "$DS" --seed 1
    rm -f "$OUT/${DS}.lock"
  fi
done
