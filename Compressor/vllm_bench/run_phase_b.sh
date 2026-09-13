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

[[ -n "$NODE2_HOST" ]] || die "set NODE2_HOST to the second DGX hostname or QSFP IP"
model_path "$WINNER_KEY" >/dev/null
command -v ssh >/dev/null || die "ssh is required for Phase B"
command -v haproxy >/dev/null || die "haproxy is required for the router"

mkdir -p "$RESULT_ROOT/phase_b"
printf '%s\n' "phase_b_replicas" > "$RESULT_ROOT/phase_b/phase.txt"

remote_model="${MODEL_PATHS[$WINNER_KEY]}"
[[ "$remote_model" == /* ]] || die "Phase B currently requires a local model path on both nodes"

ssh "$NODE2_HOST" "test -d '$remote_model' && docker image inspect '$VLLM_IMAGE' >/dev/null" \
  || die "node2 is missing $remote_model or $VLLM_IMAGE"

serve_model "$WINNER_KEY"
CURRENT_MODEL="$WINNER_KEY"
trap 'cleanup_current; ssh "$NODE2_HOST" "docker rm -f "$REMOTE_CONTAINER" >/dev/null 2>&1 || true"; [[ -n "${HAPROXY_PID:-}" ]] && kill "$HAPROXY_PID" >/dev/null 2>&1 || true' EXIT
wait_health
capture_server_facts "$WINNER_KEY"

ssh "$NODE2_HOST" "docker rm -f '$REMOTE_CONTAINER' >/dev/null 2>&1 || true; docker run -d --name '$REMOTE_CONTAINER' --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --entrypoint '' -p 8000:8000 -v '$remote_model:/model:ro' -v '$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro' '$VLLM_IMAGE' vllm serve /model --served-model-name '$(served_name "$WINNER_KEY")' --dtype bfloat16 --max-model-len '$MAX_MODEL_LEN' --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' >/tmp/$REMOTE_CONTAINER.id"

until curl -fsS "http://$NODE2_BENCH_HOST:8000/health" >/dev/null 2>&1; do sleep 5; done
sed "s|\${NODE2_BENCH_HOST:-node2}|$NODE2_BENCH_HOST|" "$SCRIPT_DIR/haproxy.cfg" > "$RESULT_ROOT/phase_b/haproxy.cfg"
haproxy -f "$RESULT_ROOT/phase_b/haproxy.cfg" -D -p "$RESULT_ROOT/phase_b/haproxy.pid"
HAPROXY_PID="$(cat "$RESULT_ROOT/phase_b/haproxy.pid")"

SERVER_ALREADY_RUNNING=1 HOST=127.0.0.1 PORT="$ROUTER_PORT" "$SCRIPT_DIR/bench_one.sh" "$WINNER_KEY" "$RESULT_ROOT/phase_b/$WINNER_KEY" \
  || die "replica sweep failed"

log "Phase B complete: $RESULT_ROOT/phase_b"
