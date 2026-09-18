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

# --- Phase C: cross-node distributed serving --------------------------------

node2_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE2_HOST" "$@"; }

# Upstream run_cluster.sh installs an EXIT trap that kills the container when its
# shell exits, so Phase C issues the equivalent docker run itself, detached.
# All interpolated values are shell-safe tokens (IPs, interface names, abs paths).
phase_c_run_prefix() {
  local ip="$1" name="$2" outdir="$3" model="$4" dev=""
  [[ -e /dev/infiniband ]] && dev="--device /dev/infiniband "
  printf 'docker run -d --name %s --network host --gpus all --ipc host --shm-size=16g -v /dev/shm:/dev/shm --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 %s-v %s:/model:ro -v %s:/out -e VLLM_HOST_IP=%s -e MASTER_ADDR=%s -e NCCL_SOCKET_IFNAME=%s -e GLOO_SOCKET_IFNAME=%s -e TP_SOCKET_IFNAME=%s -e NCCL_IB_HCA=%s -e NCCL_IB_DISABLE=%s -e NCCL_MAX_NCHANNELS=%s -e NCCL_BUFFSIZE=%s -e NCCL_CUMEM_ENABLE=%s -e NCCL_CUMEM_HOST_ENABLE=%s -e VLLM_USE_PRECOMPILED_NCCL=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET -e RAY_memory_monitor_refresh_ms=0 --entrypoint %s %s' \
    "$name" "$dev" "$model" "$outdir" "$ip" "$HEAD_NODE_IP" \
    "$MN_IF_NAME" "$MN_IF_NAME" "$MN_IF_NAME" "$RDMA_HCA" "$NCCL_IB_DISABLE" \
    "$NCCL_MAX_NCHANNELS" "$NCCL_BUFFSIZE" "$NCCL_CUMEM_ENABLE" "$NCCL_CUMEM_HOST_ENABLE" "''" "$VLLM_IMAGE"
}

phase_c_serve_args() {
  local served="$1" tp="$2" pp="$3" backend="$4"
  printf 'vllm serve /model --served-model-name %s --dtype bfloat16 --max-model-len %s --gpu-memory-utilization %s --tensor-parallel-size %s --pipeline-parallel-size %s --distributed-executor-backend %s --host 0.0.0.0 --port %s' \
    "$served" "$MAX_MODEL_LEN" "$GPU_MEMORY_UTILIZATION" "$tp" "$pp" "$backend" "$PORT"
}

# Launch commands go through bind-mounted scripts to avoid nested ssh/docker quoting.
phase_c_write_launch() {
  local outdir="$1" role="$2" args="$3"
  printf '#!/usr/bin/env bash\nexec %s > /out/server-%s.log 2>&1\n' "$args" "$role" > "$outdir/launch-$role.sh"
}

phase_c_teardown() {
  docker rm -f "$PHASE_C_HEAD_CONTAINER" >/dev/null 2>&1 || true
  node2_ssh "docker rm -f $PHASE_C_WORKER_CONTAINER >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
}

# Containers idle on `sleep infinity` so logs survive a vLLM crash.
phase_c_containers_up() {
  local outdir="$1" model="$2"
  phase_c_teardown
  mkdir -p "$outdir"
  node2_ssh "mkdir -p '$PHASE_C_REMOTE_OUT' && rm -f '$PHASE_C_REMOTE_OUT'/*" || die "cannot prepare remote out dir"
  eval "$(phase_c_run_prefix "$HEAD_NODE_IP" "$PHASE_C_HEAD_CONTAINER" "$outdir" "$model") sleep infinity" \
    > "$outdir/head_container_id.txt" || die "head container failed to start"
  node2_ssh "$(phase_c_run_prefix "$NODE2_HOST" "$PHASE_C_WORKER_CONTAINER" "$PHASE_C_REMOTE_OUT" "$model") sleep infinity" \
    > "$outdir/worker_container_id.txt" || die "worker container failed to start"
}

ray_up() {
  local outdir="$1"
  log "installing $RAY_SPEC in head container"
  docker exec "$PHASE_C_HEAD_CONTAINER" pip install -q --root-user-action=ignore "$RAY_SPEC" \
    > "$outdir/ray-install-head.log" 2>&1 || { tail -20 "$outdir/ray-install-head.log" >&2; return 1; }
  local ray_version
  ray_version="$(docker exec "$PHASE_C_HEAD_CONTAINER" python3 -c 'import ray;print(ray.__version__)')" || return 1
  log "ray $ray_version on head; pinning worker to the same version"
  node2_ssh "docker exec $PHASE_C_WORKER_CONTAINER pip install -q --root-user-action=ignore 'ray[cgraph]==$ray_version'" \
    > "$outdir/ray-install-worker.log" 2>&1 || { tail -20 "$outdir/ray-install-worker.log" >&2; return 1; }
  docker exec "$PHASE_C_HEAD_CONTAINER" ray start --head --node-ip-address "$HEAD_NODE_IP" \
    --port "$RAY_PORT" --num-gpus 1 --disable-usage-stats > "$outdir/ray-head.log" 2>&1 || return 1
  node2_ssh "docker exec $PHASE_C_WORKER_CONTAINER ray start --address $HEAD_NODE_IP:$RAY_PORT --node-ip-address $NODE2_HOST --num-gpus 1 --disable-usage-stats" \
    > "$outdir/ray-worker.log" 2>&1 || return 1
}

ray_verify() {
  local outdir="$1"
  local deadline=$((SECONDS + 180)) shape=""
  while (( SECONDS < deadline )); do
    shape="$(docker exec "$PHASE_C_HEAD_CONTAINER" python3 -c \
      "import ray;ray.init(address='auto',logging_level='ERROR');print(len([n for n in ray.nodes() if n['Alive']]),int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null)" || shape=""
    [[ "$shape" == "2 2" ]] && break
    sleep 5
  done
  docker exec "$PHASE_C_HEAD_CONTAINER" ray status > "$outdir/ray-status.txt" 2>&1 || true
  [[ "$shape" == "2 2" ]] || { log "ray cluster shape is '$shape', expected '2 2'"; return 1; }
  log "ray cluster: 2 alive nodes, 2 GPUs"
}

phase_c_wait_health() {
  local outdir="$1" timeout="${2:-$PHASE_C_HEALTH_TIMEOUT}"
  local deadline=$((SECONDS + timeout))
  until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! docker exec "$PHASE_C_HEAD_CONTAINER" pgrep -f 'vllm serve' >/dev/null 2>&1; then
      log "vllm serve is no longer running on the head node"
      return 1
    fi
    (( SECONDS >= deadline )) && { log "health timeout after ${timeout}s"; return 1; }
    sleep 10
  done
  log "distributed server is healthy"
}

phase_c_collect_logs() {
  local outdir="$1"
  node2_ssh "cat '$PHASE_C_REMOTE_OUT'/server-worker.log 2>/dev/null || true" > "$outdir/server-worker.log" 2>/dev/null || true
  docker logs "$PHASE_C_HEAD_CONTAINER" > "$outdir/head-container.log" 2>&1 || true
}

# Classifies the NCCL transport actually negotiated over the RoCE fabric.
phase_c_capture_facts() {
  local outdir="$1"
  local head_log="$outdir/server-head.log"
  [[ -f "$head_log" ]] || return 0
  grep -Ei 'compressed-tensors|marlin|machete|awq|fp8|quantization|cuda graph|GPU KV cache size|Maximum concurrency|decompress|NET/IB|GDRDMA|NET/Socket|Using network' \
    "$head_log" > "$outdir/server_facts.txt" || true
  local verdict="unknown"
  if grep -q 'GDRDMA' "$head_log"; then verdict="NET/IB/GDRDMA"
  elif grep -q 'NET/IB' "$head_log"; then verdict="NET/IB"
  elif grep -q 'NET/Socket' "$head_log"; then verdict="NET/Socket"
  fi
  printf 'nccl_transport=%s\n' "$verdict" > "$outdir/nccl-facts.txt"
  grep -E 'NET/IB|NET/Socket|GDRDMA|NCCL INFO Using|nccl version' "$head_log" >> "$outdir/nccl-facts.txt" 2>/dev/null || true
  log "NCCL transport: $verdict"
  if grep -Eiq 'decompressing model|decompress model' "$head_log"; then
    die "native-kernel gate failed: head log reports decompression"
  fi
  grep -Eiq 'compressed-tensors|marlin|machete|awq|fp8|quantization' "$head_log" \
    || die "native-kernel gate inconclusive: no quantization evidence in head log"
}
