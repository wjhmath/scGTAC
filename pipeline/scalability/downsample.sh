#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --output=/home/liyang/BioJiaheWang/scGTAC/log/downsample_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scGTAC/log/downsample_%j.err
#SBATCH --nodes=1 -n 1 --cpus-per-task=4 --mem=64G --time=12:00:00
#SBATCH --qos=a100g1 --gres=gpu:a100:1 -p a100
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scGTAC
source /home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
source activate "$PROJ/scagcr_env"; cd "$PROJ"

CSV=results/robustness/downsample.csv
mkdir -p results/robustness
echo "dataset,ratio,seed,ARI,NMI,CA" > "$CSV"

CHULI=/home/liyang/BioJiaheWang/RARECELL/data/chuli

# 3 数据集 × 5 比例 × 3 seeds = 45 runs
for DS_INFO in \
  "baron|data/baron/baron.h5ad|14" \
  "GSE103322|${CHULI}/GSE103322.h5ad|8" \
  "GSE119531|${CHULI}/GSE119531.h5ad|17"; do

  IFS='|' read -r DS H5 K <<< "$DS_INFO"
  echo "=== $DS (k=$K) ==="

  for RATIO in 0.1 0.3 0.5 0.7 0.9; do
    # 生成 downsample 数据
    SUB_H5="/tmp/ds_${DS}_${RATIO}.h5ad"
    python3 -c "
import scanpy as sc, numpy as np
adata=sc.read_h5ad('$H5')
np.random.seed(42)
n=int(adata.n_obs*$RATIO)
idx=np.random.choice(adata.n_obs,n,replace=False)
adata[idx].write('$SUB_H5')
print(f'  {\"$DS\"} ratio=$RATIO: {n}/{adata.n_obs} cells')
"
    for S in 1 42 84; do
      CK="/tmp/ds_tmp_${SLURM_JOB_ID}.pt"
      OUT=$(python scgtac/main.py --data_path "$SUB_H5" --n_clusters "$K" \
        --epochs 200 --seed "$S" --save_model_path "$CK" 2>&1 | tail -1)
      ARI=$(echo "$OUT" | grep -oP "'ARI': [0-9.]+" | grep -oP "[0-9.]+$")
      NMI=$(echo "$OUT" | grep -oP "'NMI': [0-9.]+" | grep -oP "[0-9.]+$")
      CA=$(echo "$OUT" | grep -oP "'CA': [^,}]+" | grep -oP "[0-9.]+$")
      echo "$DS,$RATIO,$S,$ARI,$NMI,$CA" >> "$CSV"
      echo "    [ok] ratio=$RATIO s=$S ARI=$ARI"
      rm -f "$CK"
    done
    rm -f "$SUB_H5"
  done

  # ratio=1.0 从已有结果提取
  for S in 1 42 84; do
    LOG="results/scagcr_final/${DS}/run_seed${S}.log"
    [ -f "$LOG" ] || continue
    OUT=$(tail -1 "$LOG")
    ARI=$(echo "$OUT" | grep -oP "'ARI': [0-9.]+" | grep -oP "[0-9.]+$")
    NMI=$(echo "$OUT" | grep -oP "'NMI': [0-9.]+" | grep -oP "[0-9.]+$")
    CA=$(echo "$OUT" | grep -oP "'CA': [^,}]+" | grep -oP "[0-9.]+$")
    echo "$DS,1.0,$S,$ARI,$NMI,$CA" >> "$CSV"
  done
done

echo "=== done $(date) ==="
cat "$CSV"
