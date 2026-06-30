#!/bin/bash
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scGTAC; ENV=$PROJ/scagcr_env
export PATH=$ENV/bin:$PATH; source $ENV/bin/activate 2>/dev/null; cd "$PROJ"
DATA=/home/liyang/BioJiaheWang/RARECELL/data/chuli/GSE103322.h5ad
NCL=$($ENV/bin/python -c "import scanpy as sc;a=sc.read_h5ad('$DATA',backed='r');c=[x for x in ['cell_type','cell_type_label','CellType','celltype','label','labels','Group'] if x in a.obs.columns];print(a.obs[c[0]].astype(str).nunique() if c else 0)")
echo "GSE103322 k=$NCL"
declare -A FLAG=( [wo_cl]="--lambda_cl 0" [wo_edge]="--prob_edge 0" )
for V in wo_cl wo_edge; do
  have=$(grep -l ARI results/ablation/$V/GSE103322/run_seed*.log 2>/dev/null | wc -l)
  for S in 7 21 35 50 63; do
    [ "$have" -ge 3 ] && break
    L=results/ablation/$V/GSE103322/run_seed$S.log
    [ -f "$L" ] && grep -q ARI "$L" && continue
    CK=/tmp/alt_${V}_$S.pt
    echo ">>> $V s$S"
    if $ENV/bin/python scgtac/main.py --data_path "$DATA" --n_clusters "$NCL" --epochs 200 --seed "$S" --save_model_path "$CK" ${FLAG[$V]} > "$L" 2>&1 && grep -q ARI "$L"; then
      echo "[ok] $V s$S $(tail -1 $L)"; have=$((have+1))
    else echo "[diverge] $V s$S"; rm -f "$L"; fi
    rm -f "$CK"
  done
  echo ">>> $V GSE103322 最终有效seed数=$have"
done
echo "ALTSEED DONE $(date)"
