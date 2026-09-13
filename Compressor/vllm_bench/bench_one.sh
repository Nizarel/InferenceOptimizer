#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat >&2 <<'EOF'
Usage: bench_one.sh MODEL_KEY [OUTPUT_DIR]
MODEL_KEY: bf16, gptq, awq, or fp8
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
MODEL_KEY="$1"
OUTPUT_DIR="${2:-$RESULT_ROOT/$MODEL_KEY}"
mkdir -p "$OUTPUT_DIR"
TOKENIZER_PATH="$(model_path "$MODEL_KEY")"
TOKENIZER_ARG="$TOKENIZER_PATH"
TOKENIZER_MOUNT=()
if [[ "$TOKENIZER_PATH" == /* ]]; then
  TOKENIZER_ARG=/model
  TOKENIZER_MOUNT=(-v "$TOKENIZER_PATH:/model:ro")
fi
CLIENT_CACHE_MOUNT=()
if [[ -d "$HF_CACHE_DIR" ]]; then
  CLIENT_CACHE_MOUNT=(-v "$HF_CACHE_DIR:/root/.cache/huggingface/hub:ro")
fi

require_tools
model_path "$MODEL_KEY" >/dev/null
if [[ "${SERVER_ALREADY_RUNNING:-0}" == "1" ]]; then
  log "using existing server at http://$HOST:$PORT"
else
  serve_model "$MODEL_KEY"
  CURRENT_MODEL="$MODEL_KEY"
  trap cleanup_current EXIT
  wait_health
  capture_server_facts "$MODEL_KEY"
fi

# Discard one warm-up before collecting the seeded measurements.
for warmup in $(seq 1 "$WARMUPS"); do
  docker run --rm --network host -v "$OUTPUT_DIR:/results" "${CLIENT_CACHE_MOUNT[@]}" "${TOKENIZER_MOUNT[@]}" --entrypoint "" "$VLLM_IMAGE" \
    vllm bench serve --backend vllm --base-url "http://$HOST:$PORT" \
    --endpoint /v1/completions --model "$(served_name "$MODEL_KEY")" \
    --tokenizer "$TOKENIZER_ARG" \
    --dataset-name random --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" --num-prompts 20 \
    --request-rate inf --max-concurrency 1 --ignore-eos --seed "$warmup" \
    > "$OUTPUT_DIR/warmup_${warmup}.log" 2>&1
  reset_caches
done

for concurrency in $CONCURRENCY_LEVELS; do
  for seed in $SEEDS; do
    output="$OUTPUT_DIR/bench_c${concurrency}_s${seed}"
    log "benchmarking $MODEL_KEY concurrency=$concurrency seed=$seed"
    reset_caches
    docker run --rm --network host -v "$OUTPUT_DIR:/results" "${CLIENT_CACHE_MOUNT[@]}" "${TOKENIZER_MOUNT[@]}" --entrypoint "" "$VLLM_IMAGE" \
      vllm bench serve --backend vllm --base-url "http://$HOST:$PORT" \
      --endpoint /v1/completions --model "$(served_name "$MODEL_KEY")" \
      --tokenizer "$TOKENIZER_ARG" \
      --dataset-name random --random-input-len "$RANDOM_INPUT_LEN" \
      --random-output-len "$RANDOM_OUTPUT_LEN" --num-prompts "$NUM_PROMPTS" \
      --request-rate inf --max-concurrency "$concurrency" --ignore-eos \
      --seed "$seed" --save-result --save-detailed \
      --result-dir /results --result-filename "$(basename "$output").json" \
      > "${output}.log" 2>&1
  done
done
