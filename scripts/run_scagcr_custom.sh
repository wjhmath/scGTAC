#!/bin/bash
set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <dataset> [seed]"
  exit 1
fi

SEED="${2:-1}"
RESULTS_DIR="${RESULTS_DIR:-results/scagcr_custom}"
LOG_DIR="${RESULTS_DIR}/logs"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

GRAPH_HEAD="${GRAPH_HEAD:-5}"
PHI="${PHI:-0.45}"
GCN_DIM="${GCN_DIM:-277}"
MLP_DIM="${MLP_DIM:-118}"
PROB_FEATURE="${PROB_FEATURE:-0.1}"
PROB_EDGE="${PROB_EDGE:-0.5}"
TAU="${TAU:-0.8}"
ALPHA="${ALPHA:-0.55}"
BETA="${BETA:-0.4}"
LAMBDA_CL="${LAMBDA_CL:-0.7}"
LAMBDA_CLUSTER="${LAMBDA_CLUSTER:-0.3}"
CLUSTER_ALPHA="${CLUSTER_ALPHA:-1.0}"
LAMBDA_ZINB="${LAMBDA_ZINB:-0.2}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-0.001}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
EPOCHS="${EPOCHS:-200}"
DATA_PATH="${DATA_PATH:-}"
N_CLUSTERS="${N_CLUSTERS:-}"

if [[ -z "$DATA_PATH" || -z "$N_CLUSTERS" ]]; then
  echo "Please set DATA_PATH and N_CLUSTERS in the environment."
  exit 1
fi

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
