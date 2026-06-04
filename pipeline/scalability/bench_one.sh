#!/bin/bash
#SBATCH --nodes=1 -n 1 --cpus-per-task=4 --mem=64G --time=4:00:00
#SBATCH --qos=a100g1 --gres=gpu:a100:1 -p a100
set -uo pipefail
DS=$1; K=$2; H5=$3
PROJ=/home/liyang/BioJiaheWang/scAGCR
source /home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
CSV=$PROJ/results/scalability/bench_${DS}.csv
NC=$(source activate "$PROJ/scagcr_env" 2>/dev/null && python3 -c "import scanpy as sc;print(sc.read_h5ad('$H5',backed='r').n_obs)")
echo "method,dataset,n_cells,seconds" > "$CSV"
CK="/tmp/bench_${SLURM_JOB_ID}_${DS}.pt"; BLOUT="/tmp/bl_${SLURM_JOB_ID}_${DS}"; mkdir -p "$BLOUT"
timeit(){
  echo ">>> $1 x $DS ($NC cells)"
  SECONDS=0
  if timeout 1800 bash -c "$2" >/dev/null 2>&1; then
    echo "$1,$DS,$NC,$SECONDS" >> "$CSV"; echo "    [ok] ${SECONDS}s"
  else
    echo "$1,$DS,$NC,FAIL" >> "$CSV"; echo "    [fail/timeout]"
  fi
}
# scAGCR
source activate "$PROJ/scagcr_env" 2>/dev/null
timeit "scAGCR" "cd $PROJ && python scagcr/main.py --data_path $H5 --n_clusters $K --epochs 200 --seed 1 --save_model_path $CK && rm -f $CK"
# Baselines
PYT="$PROJ/envs/bl_torch/bin/python"; PYR="$PROJ/envs/bl_r/bin/python"
for spec in "DEC|run_dec.py|$PYT" "scDeepCluster|run_scdeepcluster.py|$PYT" "scVI|run_scvi.py|$PYT" \
            "scDSC|run_scdsc.py|$PYT" "scGNN|run_scgnn.py|$PYT" "Seurat|run_seurat.py|$PYR"; do
  IFS='|' read -r NM RN PY <<< "$spec"
  [ -f "$PY" ] || { echo "skip $NM"; continue; }
  timeit "$NM" "cd $PROJ/baselines && $PY $RN --data $H5 --out $BLOUT --dataset $DS --seed 1"
done
rm -rf "$BLOUT" "$CK"
echo "=== $DS done $(date) ==="; cat "$CSV"
