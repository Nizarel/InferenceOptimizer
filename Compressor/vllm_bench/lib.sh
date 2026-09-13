#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

require_tools() {
  command -v docker >/dev/null || die "docker is required"
  command -v curl >/dev/null || die "curl is required"
  docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || die "missing image: $VLLM_IMAGE"
}

model_path() {
  local key="$1"
  [[ -n "${MODEL_PATHS[$key]+x}" ]] || die "unknown model key: $key"
  printf '%s' "${MODEL_PATHS[$key]}"
}

served_name() {
  local key="$1"
  [[ -n "${SERVED_NAMES[$key]+x}" ]] || die "unknown model key: $key"
  printf '%s' "${SERVED_NAMES[$key]}"
}

container_name() { printf 'vllm-bench-%s' "${1:-server}"; }

serve_model() {
  local key="$1"
  local image="${2:-$VLLM_IMAGE}"
  local name
  name="$(container_name "$key")"
  stop_server "$name"
  mkdir -p "$RESULT_ROOT/$key"
  local model="$(model_path "$key")"
  local served="$(served_name "$key")"
  local model_arg=()
  if [[ "$model" == /* ]]; then
    model_arg=(/model)
  else
    model_arg=("$model")
  fi
  local mounts=()
  if [[ "$model" == /* ]]; then
    mounts+=("$model:/model:ro")
  fi
  if [[ -d "$HF_CACHE_DIR" ]]; then
    mounts+=("$HF_CACHE_DIR:/root/.cache/huggingface/hub:ro")
  fi
  local mount_args=()
  local mount
  for mount in "${mounts[@]}"; do mount_args+=( -v "$mount" ); done
  log "starting $key ($model) in $name"
  docker run -d --name "$name" --gpus all --ipc host \
    --ulimit memlock=-1 --ulimit stack=67108864 --entrypoint "" \
    -p "$PORT:8000" "${mount_args[@]}" \
    "$image" vllm serve "${model_arg[@]}" \
    --served-model-name "$served" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    > "$RESULT_ROOT/$key/container_id.txt"
  docker logs -f "$name" > "$RESULT_ROOT/$key/server.log" 2>&1 &
  echo "$!" > "$RESULT_ROOT/$key/server_log_pid.txt"
}

wait_health() {
  local timeout_seconds="${1:-900}"
  local deadline=$((SECONDS + timeout_seconds))
  until curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "$(container_name "$CURRENT_MODEL")"; then
      docker logs "$(container_name "$CURRENT_MODEL")" | tail -80 >&2 || true
      die "server container exited before becoming healthy"
    fi
    (( SECONDS >= deadline )) && { docker logs "$(container_name "$CURRENT_MODEL")" | tail -80 >&2 || true; die "server health timeout"; }
    sleep 5
  done
  log "server is healthy"
}

capture_server_facts() {
  local key="$1"
  local log_file="$RESULT_ROOT/$key/server.log"
  grep -Ei 'compressed-tensors|marlin|machete|awq|fp8|quantization|cuda graph|GPU KV cache size|Maximum concurrency|decompress' "$log_file" \
    > "$RESULT_ROOT/$key/server_facts.txt" || true
  if grep -Eiq 'decompressing model|decompress model' "$log_file"; then
    die "native-kernel gate failed: server log reports decompression"
  fi
  if [[ "$key" != "bf16" ]]; then
    grep -Eiq 'compressed-tensors|marlin|machete|awq|fp8|quantization' "$log_file" \
      || die "native-kernel gate inconclusive: no quantization/kernel evidence in server log"
  fi
}

reset_caches() {
  curl -fsS -X POST "http://$HOST:$PORT/reset_prefix_cache" >/dev/null 2>&1 || true
}

stop_server() {
  local name="${1:-$(container_name server)}"
  docker rm -f "$name" >/dev/null 2>&1 || true
}

cleanup_current() {
  [[ -n "${CURRENT_MODEL:-}" ]] && stop_server "$(container_name "$CURRENT_MODEL")"
}
