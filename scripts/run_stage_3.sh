#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example:
# NUM_GPUS=2 MODEL_NAME_OR_PATH=/path/to/stage2_checkpoint TRAIN_FILE=/path/to/stage3.csv \
# OUTPUT_DIR=/path/to/stage3_output bash scripts/run_stage_3.sh

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ROOT_DIR/outputs/stage2_contrastive}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/stage3_data.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/stage3_multiscale}"

python -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" \
  --master_port "$MASTER_PORT" \
  "$ROOT_DIR/train_stage_2_3.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --training_stage 3 \
  --col_anchor polyprotein_seq \
  --col_positive domain_seq \
  --pooler_type avg \
  --base_temp 0.07 \
  --ortho_lambda 0.3 \
  --separation_weight 2.0 \
  --num_train_epochs 50 \
  --per_device_train_batch_size 1 \
  --learning_rate 5e-5 \
  --max_seq_length 1024 \
  --fp16 \
  --do_train \
  --save_strategy steps \
  --evaluation_strategy no \
  --save_steps 2000 \
  --logging_steps 100 \
  --overwrite_output_dir \
  "$@"
