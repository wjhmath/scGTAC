#!/bin/bash
#SBATCH --job-name=time_bench
#SBATCH --output=/home/liyang/BioJiaheWang/scGTAC/log/time_bench_%j.out
#SBATCH --error=/home/liyang/BioJiaheWang/scGTAC/log/time_bench_%j.err
#SBATCH --nodes=1 -n 1 --cpus-per-task=4 --mem=64G --time=24:00:00
#SBATCH --qos=a100g1 --gres=gpu:a100:1 -p a100
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scGTAC
CHULI=/home/liyang/BioJiaheWang/RARECELL/data/chuli
CONDA_SH=/home/liyang/BioJiaheWang/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"

CSV=$PROJ/results/scalability/benchmark.csv
mkdir -p "$(dirname "$CSV")"
echo "method,dataset,n_cells,seconds" > "$CSV"

# 5 representative datasets: small → large
declare -A DS_K=(
  [Goolam]=8
  [muraro_pancreas]=10
  [baron]=14
  [GSE159115_ccRCC]=13
  [zheng68k]=11
)
DS_ORDER=(Goolam muraro_pancreas baron GSE159115_ccRCC zheng68k)

get_h5ad(){
  local ds=$1
  for d in "$CHULI/${ds}.h5ad" "$PROJ/data/${ds}/${ds}.h5ad"; do
    [ -f "$d" ] && echo "$d" && return; done
  echo ""
}

get_ncells(){
  local h5=$1
  source activate "$PROJ/scagcr_env" 2>/dev/null
  python3 -c "import scanpy as sc; print(sc.read_h5ad('$h5',backed='r').n_obs)"
}

timeit(){
  local method=$1 ds=$2 ncells=$3 cmd=$4
  echo ">>> $method x $ds ($ncells cells)"
  SECONDS=0
  if timeout 1800 bash -c "$cmd" >/dev/null 2>&1; then
    echo "$method,$ds,$ncells,$SECONDS" >> "$CSV"
    echo "    [ok] ${SECONDS}s"
  else
    echo "$method,$ds,$ncells,FAIL" >> "$CSV"
    echo "    [fail/timeout]"
  fi
}

echo "=== Timing benchmark $(date) ==="

for ds in "${DS_ORDER[@]}"; do
  H5=$(get_h5ad "$ds"); K=${DS_K[$ds]}
  [ -z "$H5" ] && { echo "SKIP $ds (no h5ad)"; continue; }
  NC=$(get_ncells "$H5")
  CK="/tmp/bench_${SLURM_JOB_ID:-$$}.pt"
  BLOUT="/tmp/bench_bl_${SLURM_JOB_ID:-$$}"
  mkdir -p "$BLOUT"

  # --- scAGCR (200ep) ---
  source activate "$PROJ/scagcr_env" 2>/dev/null
  timeit "scAGCR" "$ds" "$NC" \
    "cd $PROJ && source activate $PROJ/scagcr_env 2>/dev/null && python scgtac/main.py --data_path $H5 --n_clusters $K --epochs 200 --seed 1 --save_model_path $CK && rm -f $CK"

  # --- Baselines ---
  PY_TORCH="$PROJ/envs/bl_torch/bin/python"
  PY_R="$PROJ/envs/bl_r/bin/python"

  for spec in \
    "DEC|run_dec.py|$PY_TORCH" \
    "scDeepCluster|run_scdeepcluster.py|$PY_TORCH" \
    "scVI|run_scvi.py|$PY_TORCH" \
    "scDSC|run_scdsc.py|$PY_TORCH" \
    "scGNN|run_scgnn.py|$PY_TORCH" \
    "Seurat|run_seurat.py|$PY_R"; do
    IFS='|' read -r MNAME RUNNER PY <<< "$spec"
    [ -x "$PY" ] || [ -f "$PY" ] || { echo "    skip $MNAME (no env)"; continue; }
    timeit "$MNAME" "$ds" "$NC" \
      "cd $PROJ/baselines && $PY $RUNNER --data $H5 --out $BLOUT --dataset $ds --seed 1"
  done
  rm -rf "$BLOUT"
done

echo "=== Benchmark 完成 $(date) ==="
echo "结果: $CSV"
cat "$CSV"
