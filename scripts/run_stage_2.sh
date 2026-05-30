#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example:
# NUM_GPUS=2 MODEL_NAME_OR_PATH=/path/to/stage1_checkpoint TRAIN_FILE=/path/to/stage2.csv \
# OUTPUT_DIR=/path/to/stage2_output bash scripts/run_stage_2.sh

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ROOT_DIR/outputs/stage1_mlm}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/stage2_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/stage2_contrastive}"

python -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" \
  --master_port "$MASTER_PORT" \
  "$ROOT_DIR/train_stage_2_3.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --training_stage 2 \
  --col_anchor seq \
  --col_label label \
  --pooler_type avg \
  --base_temp 0.2 \
  --num_train_epochs 150 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --max_seq_length 1024 \
  --fp16 \
  --do_train \
  --save_strategy steps \
  --logging_strategy steps \
  --save_steps 2000 \
  --logging_steps 100 \
  --overwrite_output_dir \
  "$@"
