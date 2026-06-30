#!/bin/bash
#SBATCH --job-name=synth_time
#SBATCH --output=/home/liyang/BioJiaheWang/scGTAC/log/synth_time_%j.out
#SBATCH -N 1 -n 1 --cpus-per-task=4 --mem=64G --time=12:00:00
#SBATCH --qos=a100g1 --gres=gpu:a100:1 -p a100
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scGTAC; ENV=$PROJ/scagcr_env
export PATH=$ENV/bin:$PATH; source $ENV/bin/activate 2>/dev/null; cd "$PROJ"
CSV=results/scalability/synthetic_timing.csv; CK=/tmp/synth_$$.pt
echo "n_cells,seconds" > "$CSV"
for H5 in $(ls -v data/synthetic/synth_*.h5ad); do
  NC=$($ENV/bin/python -c "import scanpy as sc;print(sc.read_h5ad('$H5',backed='r').n_obs)")
  K=$($ENV/bin/python -c "import scanpy as sc;a=sc.read_h5ad('$H5',backed='r');c=[x for x in ['cell_type','label','labels','Group','celltype'] if x in a.obs.columns];print(a.obs[c[0]].nunique() if c else 8)")
  echo ">>> $H5 NC=$NC K=$K"; T0=$(date +%s)
  $ENV/bin/python scgtac/main.py --data_path "$H5" --n_clusters "$K" --epochs 50 --seed 1 --save_model_path "$CK" >/dev/null 2>&1
  T1=$(date +%s); echo "$NC,$((T1-T0))" >> "$CSV"; echo "   $NC -> $((T1-T0))s"
done
rm -f "$CK"; echo "SYNTH DONE"; cat "$CSV"
