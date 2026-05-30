#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example:
# NUM_GPUS=2 MODEL_NAME_OR_PATH=/path/to/stage3_checkpoint TREE_PATH=/path/to/backbone.nwk \
# FASTA_PATH=/path/to/backbone.fasta OUTPUT_DIR=/path/to/stage4_output \
# bash scripts/run_stage_4_train.sh
#
# If you want to pin visible GPUs, use standard shell syntax before the command:
# CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/run_stage_4_train.sh

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29503}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ROOT_DIR/outputs/stage3_multiscale}"
TREE_PATH="${TREE_PATH:-$ROOT_DIR/data/stage4_backbone.nwk}"
FASTA_PATH="${FASTA_PATH:-$ROOT_DIR/data/stage4_backbone.fasta}"
SPLIT_FILE="${SPLIT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/stage4_lora}"

TRAIN_ARGS=(
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --tree_path "$TREE_PATH"
  --fasta_path "$FASTA_PATH"
  --output_dir "$OUTPUT_DIR"
  --pooler_type avg
  --use_lora true
  --max_seq_length 1024
  --num_train_epochs 10
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 8
  --gradient_accumulation_steps 1
  --learning_rate 5e-5
  --fp16
  --do_train
  --do_eval
  --evaluation_strategy epoch
  --save_strategy epoch
  --logging_steps 20
  --overwrite_output_dir
)

if [[ -n "$SPLIT_FILE" ]]; then
  TRAIN_ARGS+=(--split_file "$SPLIT_FILE")
fi


python -m torch.distributed.run \
  --nproc_per_node "$NUM_GPUS" \
  --master_port "$MASTER_PORT" \
  "$ROOT_DIR/train_stage_4.py" \
  "${TRAIN_ARGS[@]}" \
  "$@"
