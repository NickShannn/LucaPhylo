#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example:
# MODEL_NAME_OR_PATH=/path/to/stage3_checkpoint STAGE4_ADAPTER_PATH=/path/to/stage4_adapter \
# BACKBONE_TREE=/path/to/backbone.nwk BACKBONE_SEQS=/path/to/backbone.fasta \
# QUERY_SEQS=/path/to/query.fasta OUTPUT_DIR=/path/to/infer_output \
# bash scripts/run_stage_4_infer.sh
#
# To pin one GPU:
# CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash scripts/run_stage_4_infer.sh

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$ROOT_DIR/outputs/stage3_multiscale}"
STAGE4_ADAPTER_PATH="${STAGE4_ADAPTER_PATH:-$ROOT_DIR/outputs/stage4_lora}"
BACKBONE_TREE="${BACKBONE_TREE:-$ROOT_DIR/data/stage4_backbone.nwk}"
BACKBONE_SEQS="${BACKBONE_SEQS:-$ROOT_DIR/data/stage4_backbone.fasta}"
QUERY_SEQS="${QUERY_SEQS:-$ROOT_DIR/data/stage4_query.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/stage4_infer}"
DEVICE="${DEVICE:-cuda:0}"

python "$ROOT_DIR/infer_stage_4.py" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --stage4_adapter_path "$STAGE4_ADAPTER_PATH" \
  --backbone_tree "$BACKBONE_TREE" \
  --backbone_seqs "$BACKBONE_SEQS" \
  --query_seqs "$QUERY_SEQS" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 1024 \
  --batch_size 8 \
  --distance_batch_size 32 \
  --device "$DEVICE" \
  "$@"
