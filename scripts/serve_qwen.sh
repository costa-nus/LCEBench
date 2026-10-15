#!/usr/bin/env bash
# Serve a Qwen model with vLLM's OpenAI-compatible server.
#
#   MODEL=Qwen/Qwen3-8B PORT=8000 bash scripts/serve_qwen.sh
#   MODEL=/path/to/Qwen3-8B SERVED_NAME=Qwen/Qwen3-8B bash scripts/serve_qwen.sh   # local weights
#
# All three methods request the model as "Qwen/<name>" where <name> is the --student_lm /
# --teacher_lm value (e.g. Qwen3-8B -> Qwen/Qwen3-8B), so SERVED_NAME must have that form.
# For teacher != student, start a second server on another port/GPU and pass --teacher_api_base.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"                         # HF id or local snapshot path
SERVED_NAME="${SERVED_NAME:-$MODEL}"
case "$SERVED_NAME" in
    Qwen/*) ;;
    *) echo "SERVED_NAME must look like Qwen/<model> (got '$SERVED_NAME'); set it when MODEL is a local path" >&2; exit 1 ;;
esac
PORT="${PORT:-8000}"
TP="${TP:-1}"                                           # tensor parallel size
DP="${DP:-1}"                                           # data parallel replicas
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"                 # keep equal to MAX_MODEL_LEN used by ACE / TF-GRPO
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enable-prefix-caching
