#!/bin/bash
set -uo pipefail
PROJ=/home/liyang/BioJiaheWang/scGTAC; ENV=$PROJ/scagcr_env
export PATH=$ENV/bin:$PATH; source $ENV/bin/activate 2>/dev/null; cd "$PROJ"
$ENV/bin/python -c "import torch" || { echo "ENV BROKEN"; exit 1; }
CHULI=/home/liyang/BioJiaheWang/RARECELL/data/chuli
declare -A DS=(
 [muraro_pancreas]="data/muraro_pancreas/muraro_pancreas.h5ad|10"
 [baron]="data/baron/baron.h5ad|14"
 [GSE103322]="$CHULI/GSE103322.h5ad|auto"
 [GSE119531]="$CHULI/GSE119531.h5ad|auto"
 [GSE159115_ccRCC]="$CHULI/GSE159115_ccRCC.h5ad|auto" )
declare -A F=( [full]="" [wo_cl]="--lambda_cl 0" [wo_graph]="--no_graph" [wo_edge]="--prob_edge 0" [wo_recon]="--lambda_zinb 0" )
for NAME in "${!DS[@]}"; do
  IFS='|' read -r DATA NCL <<< "${DS[$NAME]}"
  if [ "$NCL" = auto ]; then NCL=$($ENV/bin/python -c "import scanpy as sc;a=sc.read_h5ad('$DATA',backed='r');c=[x for x in ['cell_type','cell_type_label','CellType','celltype','label','labels','Group'] if x in a.obs.columns];print(a.obs[c[0]].astype(str).nunique() if c else 0)"); fi
  echo "== $NAME k=$NCL =="
  for V in full wo_cl wo_graph wo_edge wo_recon; do for S in 1 42 84; do
    O=$PROJ/results/ablation/$V/$NAME; mkdir -p "$O"; L=$O/run_seed$S.log
    [ -f "$L" ] && grep -q ARI "$L" && { echo "[skip] $NAME $V s$S"; continue; }
    CK=/tmp/ablg_${NAME}_${V}_$S.pt
    $ENV/bin/python scgtac/main.py --data_path "$DATA" --n_clusters "$NCL" --epochs 200 --seed "$S" --save_model_path "$CK" ${F[$V]} > "$L" 2>&1 \
      && echo "[ok] $NAME $V s$S $(tail -1 $L)" || { echo "[fail] $NAME $V s$S"; tail -3 "$L"; }
    rm -f "$CK"
  done; done
done
echo "ABLATION DONE $(date)"
