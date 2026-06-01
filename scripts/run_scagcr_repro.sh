#!/bin/bash
set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <baron|zheng68k|multiome|muraro_pancreas> [seed]"
  exit 1
fi

SEED="${2:-1}"
RESULTS_DIR="${RESULTS_DIR:-results/scagcr_repro}"
LOG_DIR="${RESULTS_DIR}/logs"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

GRAPH_HEAD=5
PHI=0.45
GCN_DIM=277
MLP_DIM=118
PROB_FEATURE=0.1
PROB_EDGE=0.5
TAU=0.8
ALPHA=0.55
BETA=0.4
LAMBDA_CL=0.7
LAMBDA_CLUSTER=0.3
CLUSTER_ALPHA=1.0
LAMBDA_ZINB=0.2
DROPOUT=0.3
LR=0.001
PRETRAIN_EPOCHS=20
EPOCHS=200

case "$DATASET" in
  baron)
    DATA_PATH="data/baron/baron.h5ad"
    N_CLUSTERS=14
    ;;
  zheng68k)
    DATA_PATH="data/zheng68k/pbmc68k.h5ad"
    N_CLUSTERS=11
    EPOCHS=50
    ;;
  multiome)
    DATA_PATH="data/multiome/multiome.h5ad"
    N_CLUSTERS=19
    ;;
  muraro_pancreas)
    DATA_PATH="data/muraro_pancreas/muraro_pancreas.h5ad"
    N_CLUSTERS=10
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    exit 1
    ;;
esac

LOG_PATH="$LOG_DIR/${DATASET}_seed${SEED}.log"
CKPT_PATH="$CKPT_DIR/${DATASET}_seed${SEED}.pt"

python scagcr/main.py \
  --data_path "$DATA_PATH" \
  --n_clusters "$N_CLUSTERS" \
  --graph_head "$GRAPH_HEAD" \
  --phi "$PHI" \
  --gcn_dim "$GCN_DIM" \
  --mlp_dim "$MLP_DIM" \
  --prob_feature "$PROB_FEATURE" \
  --prob_edge "$PROB_EDGE" \
  --tau "$TAU" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --lambda_cl "$LAMBDA_CL" \
  --lambda_cluster "$LAMBDA_CLUSTER" \
  --cluster_alpha "$CLUSTER_ALPHA" \
  --lambda_zinb "$LAMBDA_ZINB" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --pretrain_epochs "$PRETRAIN_EPOCHS" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --save_model_path "$CKPT_PATH" \
  > "$LOG_PATH" 2>&1

echo "log: $LOG_PATH"
echo "checkpoint: $CKPT_PATH"
