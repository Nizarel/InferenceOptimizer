#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

WINNER_KEY="${1:-gptq}"
NODE2_HOST="${NODE2_HOST:-}"
NODE2_ROOT="${NODE2_ROOT:-$ROOT_DIR}"
NODE2_BENCH_HOST="${NODE2_BENCH_HOST:-$NODE2_HOST}"
ROUTER_PORT="${ROUTER_PORT:-8080}"
REMOTE_CONTAINER="vllm-bench-$WINNER_KEY"
ROUTER_CONTAINER="vllm-bench-router"
HAPROXY_IMAGE="${HAPROXY_IMAGE:-haproxy:3.2-alpine}"
ROUTER_MODE=""

[[ -n "$NODE2_HOST" ]] || die "set NODE2_HOST to the second DGX hostname or QSFP IP"
model_path "$WINNER_KEY" >/dev/null
command -v ssh >/dev/null || die "ssh is required for Phase B"

cleanup_phase_b() {
  cleanup_current
  ssh "$NODE2_HOST" "docker rm -f '$REMOTE_CONTAINER' >/dev/null 2>&1 || true" || true
  if [[ "$ROUTER_MODE" == "native" && -n "${HAPROXY_PID:-}" ]]; then
    kill "$HAPROXY_PID" >/dev/null 2>&1 || true
  elif [[ "$ROUTER_MODE" == "docker" ]]; then
    docker rm -f "$ROUTER_CONTAINER" >/dev/null 2>&1 || true
  fi
}

RESULT_ROOT="$RESULT_ROOT/phase_b"
export RESULT_ROOT
mkdir -p "$RESULT_ROOT"
printf '%s\n' "phase_b_replicas" > "$RESULT_ROOT/phase.txt"

remote_model="${MODEL_PATHS[$WINNER_KEY]}"
[[ "$remote_model" == /* ]] || die "Phase B currently requires a local model path on both nodes"

ssh "$NODE2_HOST" "test -d '$remote_model' && docker image inspect '$VLLM_IMAGE' >/dev/null" \
  || die "node2 is missing $remote_model or $VLLM_IMAGE"

serve_model "$WINNER_KEY"
CURRENT_MODEL="$WINNER_KEY"
trap cleanup_phase_b EXIT
wait_health
capture_server_facts "$WINNER_KEY"

ssh "$NODE2_HOST" "docker rm -f '$REMOTE_CONTAINER' >/dev/null 2>&1 || true; docker run -d --name '$REMOTE_CONTAINER' --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --entrypoint '' -p 8000:8000 -v '$remote_model:/model:ro' -v '$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro' '$VLLM_IMAGE' vllm serve /model --served-model-name '$(served_name "$WINNER_KEY")' --dtype bfloat16 --max-model-len '$MAX_MODEL_LEN' --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' >/tmp/$REMOTE_CONTAINER.id"

remote_deadline=$((SECONDS + 900))
until curl -fsS "http://$NODE2_BENCH_HOST:8000/health" >/dev/null 2>&1; do
  if ! ssh "$NODE2_HOST" "docker ps --format '{{.Names}}' | grep -Fxq '$REMOTE_CONTAINER'"; then
    ssh "$NODE2_HOST" "docker logs '$REMOTE_CONTAINER' | tail -80" >&2 || true
    die "node2 server exited before becoming healthy"
  fi
  if (( SECONDS >= remote_deadline )); then
    ssh "$NODE2_HOST" "docker logs '$REMOTE_CONTAINER' | tail -80" >&2 || true
    die "node2 server health timeout"
  fi
  sleep 5
done
ssh "$NODE2_HOST" "docker logs '$REMOTE_CONTAINER'" > "$RESULT_ROOT/$WINNER_KEY/server-node2.log" 2>&1
grep -Ei 'compressed-tensors|marlin|machete|awq|fp8|quantization|cuda graph|GPU KV cache size|Maximum concurrency|decompress' \
  "$RESULT_ROOT/$WINNER_KEY/server-node2.log" > "$RESULT_ROOT/$WINNER_KEY/server-node2-facts.txt" || true
if grep -Eiq 'decompressing model|decompress model' "$RESULT_ROOT/$WINNER_KEY/server-node2.log"; then
  die "node2 native-kernel gate failed: server log reports decompression"
fi
grep -Eiq 'compressed-tensors|marlin|machete|awq|fp8|quantization' "$RESULT_ROOT/$WINNER_KEY/server-node2.log" \
  || die "node2 native-kernel gate inconclusive: no quantization/kernel evidence in server log"

sed "s|\${NODE2_BENCH_HOST:-node2}|$NODE2_BENCH_HOST|" "$SCRIPT_DIR/haproxy.cfg" > "$RESULT_ROOT/haproxy.cfg"
if command -v haproxy >/dev/null; then
  ROUTER_MODE="native"
  haproxy -f "$RESULT_ROOT/haproxy.cfg" -D -p "$RESULT_ROOT/haproxy.pid"
  HAPROXY_PID="$(cat "$RESULT_ROOT/haproxy.pid")"
else
  ROUTER_MODE="docker"
  docker image inspect "$HAPROXY_IMAGE" >/dev/null 2>&1 || docker pull "$HAPROXY_IMAGE" >/dev/null
  docker rm -f "$ROUTER_CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$ROUTER_CONTAINER" --network host \
    -v "$RESULT_ROOT/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro" \
    "$HAPROXY_IMAGE" >/dev/null
fi

SERVER_ALREADY_RUNNING=1 HOST=127.0.0.1 PORT="$ROUTER_PORT" "$SCRIPT_DIR/bench_one.sh" "$WINNER_KEY" "$RESULT_ROOT/$WINNER_KEY" \
  || die "replica sweep failed"

log "Phase B complete: $RESULT_ROOT"
