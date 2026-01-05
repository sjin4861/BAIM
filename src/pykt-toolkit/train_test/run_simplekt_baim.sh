#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
VENV_DIR="../../../.venv"
VENV_ACT="$VENV_DIR/bin/activate"
if [[ -f "$VENV_ACT" ]]; then
  source "$VENV_ACT"
else
  echo "Virtualenv not found at $VENV_DIR" >&2
  exit 1
fi
export PYTHONPATH=..
export CUDA_VISIBLE_DEVICES=0

DATASET="nips_task34"
MODEL="simplekt_baim"
EMB_TYPE="qid"

SEED=42
DROPOUT=0.1
LR=1e-4
BATCH_SIZE=128
EMB_SIZE=256
PRETRAIN_DIM=768
EPOCHS=200
SUBSET=1.0

USE_WANDB=1
ADD_UUID=1
WANDB_PROJECT="${DATASET}_${MODEL}_all_layers_mean"

EMB_PATH="../../../embedding/${DATASET}/polya_tensor_all_layers_mean_pca768.pt"
SAVE_DIR="./saved_model"

echo "===================================================="
echo " [SimpleKT-BAIM] All Layers Mean Pooling Embedding"
echo " Dataset=${DATASET} | GPU=${CUDA_VISIBLE_DEVICES}"
echo " Emb File=${EMB_PATH}"
echo "===================================================="
echo ""

for FOLD in 0 1 2 3 4; do
  echo "=============================================="
  echo " Running Fold ${FOLD} for ${MODEL}"
  echo "=============================================="

  python simplekt_baim_train.py \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL}" \
    --emb_type "${EMB_TYPE}" \
    --emb_path "${EMB_PATH}" \
    --seed "${SEED}" \
    --dropout "${DROPOUT}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --num_epochs "${EPOCHS}" \
    --fold "${FOLD}" \
    --train_subset_rate "${SUBSET}" \
    --save_dir "${SAVE_DIR}" \
    --flag_load_emb \
    --use_wandb "${USE_WANDB}" \
    --wandb_project_name "${WANDB_PROJECT}" \
    --add_uuid "${ADD_UUID}" \
    --d_model "${EMB_SIZE}" \
    --pretrain_dim "${PRETRAIN_DIM}"

  echo ""
done

echo "===================================================="
echo " Done: All folds completed for ${MODEL}"
echo "===================================================="
