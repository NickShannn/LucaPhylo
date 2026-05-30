#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example:
# NUM_GPUS=2 MODEL_NAME_OR_PATH=/path/to/base_model TRAIN_FILE=/path/to/stage1.txt \
# OUTPUT_DIR=/path/to/stage1_output bash scripts/run_stage_1.sh

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-facebook/esm2_t33_650M_UR50D}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/stage1_data.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/stage1_mlm}"

python -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" \
  --master_port "$MASTER_PORT" \
  "$ROOT_DIR/train_stage_1.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 1024 \
  --masking_probability 0.15 \
  --validation_split 0.2 \
  --num_train_epochs 30 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --fp16 \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --save_steps 2000 \
  --logging_steps 1000 \
  --metric_for_best_model eval_loss \
  --load_best_model_at_end \
  --overwrite_output_dir \
  "$@"
