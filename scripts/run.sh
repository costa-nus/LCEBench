#!/usr/bin/env bash
# Train one method on the four datasets x five seeds against running vLLM server(s).
#
#   bash scripts/run.sh {gepa|ace|tfgrpo} [extra arguments for <method>.run]
#
# Environment (defaults in brackets):
#   TASKS             tasks to run            [hotpotqa ifbench math500 middlematch]
#   SEEDS             seeds                   [0 1 2 3 4]
#   STUDENT, TEACHER  model names             [Qwen3-8B, = STUDENT]
#   STUDENT_API_BASE, TEACHER_API_BASE        [$VLLM_BASE_URL or http://localhost:8000/v1]
#   PARALLEL          seeds run concurrently  [5]
# Seeds run as separate processes; logs go to logs/<method>/<task>_seed<N>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

METHOD="${1:?usage: bash scripts/run.sh {gepa|ace|tfgrpo} [extra args]}"
shift
case "$METHOD" in
    gepa) MODULE=gepa_dspy ;;
    ace|tfgrpo) MODULE=$METHOD ;;
    *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac

TASKS="${TASKS:-hotpotqa ifbench math500 middlematch}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STUDENT="${STUDENT:-Qwen3-8B}"
TEACHER="${TEACHER:-$STUDENT}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
STUDENT_API_BASE="${STUDENT_API_BASE:-$VLLM_BASE_URL}"
TEACHER_API_BASE="${TEACHER_API_BASE:-$VLLM_BASE_URL}"
PARALLEL="${PARALLEL:-5}"

# Prompt budgeting (ACE, TF-GRPO): Qwen tokenizer and the vLLM --max-model-len.
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/${STUDENT#Qwen/}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
# ACE bullet de-duplication embeddings (run on CPU next to the vLLM server).
export ACE_EMBEDDING_MODEL="${ACE_EMBEDDING_MODEL:-sentence-transformers/all-mpnet-base-v2}"
export ACE_EMBEDDING_DEVICE="${ACE_EMBEDDING_DEVICE:-cpu}"

mkdir -p "logs/$METHOD"
for task in $TASKS; do
    running=0
    for seed in $SEEDS; do
        python -u -m "$MODULE.run" --task "$task" --seed "$seed" \
            --student_lm "$STUDENT" --teacher_lm "$TEACHER" \
            --student_api_base "$STUDENT_API_BASE" --teacher_api_base "$TEACHER_API_BASE" "$@" \
            > "logs/$METHOD/${task}_seed${seed}.log" 2>&1 &
        running=$((running + 1))
        if [ "$running" -ge "$PARALLEL" ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
    echo "[$METHOD] $task done (logs/$METHOD/${task}_seed*.log)"
done
