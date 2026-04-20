#!/usr/bin/env bash
# Launch a plain vLLM OpenAI server (no LoRA) for ablation backbones.
# Usage: ./launch_vllm_plain.sh <GPU_ID> <PORT> <MODEL_PATH> <SERVED_NAME> [MAX_LEN]
set -euo pipefail

GPU_ID="${1:?GPU_ID required}"
PORT="${2:?PORT required}"
MODEL="${3:?MODEL path required}"
SERVED="${4:?served-model-name required}"
MAX_LEN="${5:-16384}"
GPU_MEM="${GPU_MEM:-0.85}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-/home/aiscuser/GenPT/logs/vllm_ablation}"
mkdir -p "$LOG_DIR"

SAFE_NAME="$(echo "$SERVED" | tr '/' '_')"
LOG="$LOG_DIR/${SAFE_NAME}_gpu${GPU_ID}_p${PORT}.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup /opt/conda/envs/ptca/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --no-enable-log-requests \
  $EXTRA_ARGS \
  > "$LOG" 2>&1 &

echo "vLLM plain server launching on GPU $GPU_ID port $PORT; pid=$!; log=$LOG; model=$MODEL; served=$SERVED"
