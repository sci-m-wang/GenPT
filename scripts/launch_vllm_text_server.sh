#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for Qwen3-8B (text-only) on one GPU.
# Usage: ./launch_vllm_text_server.sh <GPU_ID> <PORT>
set -euo pipefail

GPU_ID="${1:-0}"
PORT="${2:-9000}"
LOG_DIR="${LOG_DIR:-/home/aiscuser/GenPT/logs/vllm}"
MODEL="${MODEL:-/home/aiscuser/models/Qwen/Qwen3-8B}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_MEM="${GPU_MEM:-0.85}"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/text_gpu${GPU_ID}_p${PORT}.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup /opt/conda/envs/ptca/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "Qwen/Qwen3-8B" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --no-enable-log-requests \
  > "$LOG" 2>&1 &

echo "vLLM text server launching on GPU $GPU_ID port $PORT; pid=$!; log=$LOG"
