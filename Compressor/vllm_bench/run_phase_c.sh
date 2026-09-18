#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

WINNER_KEY="gptq"
MODE="all"
for arg in "$@"; do
  case "$arg" in
    --smoke-only) MODE="smoke" ;;
    --full-only)  MODE="full" ;;
    -*) die "unknown flag: $arg" ;;
    *) WINNER_KEY="$arg" ;;
  esac
done

[[ -n "$NODE2_HOST" ]]    || die "set NODE2_HOST to node2's QSFP IP"
[[ -n "$HEAD_NODE_IP" ]]  || die "set HEAD_NODE_IP to node1's QSFP IP"
[[ -n "$MN_IF_NAME" ]]    || die "set MN_IF_NAME to the QSFP interface"
command -v ssh >/dev/null || die "ssh is required for Phase C"
require_tools
model_path "$WINNER_KEY" >/dev/null

WINNER_MODEL="${MODEL_PATHS[$WINNER_KEY]}"
[[ "$WINNER_MODEL" == /* ]] || die "Phase C requires a local checkpoint on both nodes"
SERVED="$(served_name "$WINNER_KEY")"

node2_ssh "test -d '$WINNER_MODEL' && docker image inspect '$VLLM_IMAGE' >/dev/null" \
  || die "node2 is missing $WINNER_MODEL or $VLLM_IMAGE"

if [[ "${PHASE_C_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  (( busy == 0 )) || die "node1 GPU has $busy compute process(es); free it or set PHASE_C_ALLOW_BUSY_GPU=1"
  busy2="$(node2_ssh 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true')"
  (( busy2 == 0 )) || die "node2 GPU has $busy2 compute process(es); free it or set PHASE_C_ALLOW_BUSY_GPU=1"
fi

RESULT_ROOT="$RESULT_ROOT/phase_c"
export RESULT_ROOT
mkdir -p "$RESULT_ROOT"
printf '%s\n' "phase_c_distributed" > "$RESULT_ROOT/phase.txt"

trap phase_c_teardown EXIT

capture_fabric() {
  {
    printf 'captured_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'vllm_image=%s\n' "$VLLM_IMAGE"
    printf 'head_ip=%s worker_ip=%s iface=%s hca=%s\n' "$HEAD_NODE_IP" "$NODE2_HOST" "$MN_IF_NAME" "$RDMA_HCA"
    printf 'node1_speed=%s node1_mtu=%s\n' \
      "$(ethtool "$MN_IF_NAME" 2>/dev/null | awk -F': ' '/Speed/{print $2}')" \
      "$(cat "/sys/class/net/$MN_IF_NAME/mtu" 2>/dev/null)"
    printf 'node2_speed=%s node2_mtu=%s\n' \
      "$(node2_ssh "ethtool $MN_IF_NAME 2>/dev/null | awk -F': ' '/Speed/{print \$2}'")" \
      "$(node2_ssh "cat /sys/class/net/$MN_IF_NAME/mtu 2>/dev/null")"
    printf '\n[node1 %s]\n' "$RDMA_HCA"; ibv_devinfo -d "$RDMA_HCA" 2>/dev/null || true
    printf '\n[node2 %s]\n' "$RDMA_HCA"; node2_ssh "ibv_devinfo -d $RDMA_HCA 2>/dev/null" || true
  } > "$RESULT_ROOT/fabric.txt"
  log "fabric captured: $(sed -n '4p;5p' "$RESULT_ROOT/fabric.txt" | paste -sd' ' -)"
}

smoke_verdict() {
  local benchdir="$1"
  if grep -RqE 'Failed requests:[[:space:]]*[1-9]' "$benchdir"/bench_*.log 2>/dev/null; then
    echo "FAIL failed-requests"; return 1
  fi
  python3 - "$benchdir" "$PHASE_C_MIN_C1_TPS" <<'PY'
import glob, json, os, sys
benchdir, floor = sys.argv[1], float(sys.argv[2])
files = sorted(glob.glob(os.path.join(benchdir, "bench_*.json")))
if not files:
    print("FAIL no-result-json")
    raise SystemExit(1)
ok, parts = True, []
for path in files:
    with open(path) as handle:
        row = json.load(handle)
    concurrency = int(row.get("max_concurrency") or 0)
    throughput = float(row.get("output_throughput") or 0.0)
    parts.append(f"c={concurrency}:{throughput:.2f}")
    if concurrency == 1 and throughput < floor:
        ok = False
        parts.append(f"(c=1 below floor {floor})")
print(("PASS " if ok else "FAIL ") + " ".join(parts))
raise SystemExit(0 if ok else 1)
PY
}

bring_up() {
  local config="$1" outdir="$2" tp="$3" pp="$4"
  local backend="${config%%-*}"
  phase_c_containers_up "$outdir" "$WINNER_MODEL"
  if [[ "$backend" == ray ]]; then
    ray_up "$outdir"     || return 1
    ray_verify "$outdir" || return 1
    phase_c_write_launch "$outdir" head "$(phase_c_serve_args "$SERVED" "$tp" "$pp" ray)"
    docker exec -d "$PHASE_C_HEAD_CONTAINER" bash /out/launch-head.sh
  else
    local base
    base="$(phase_c_serve_args "$SERVED" "$tp" "$pp" mp)"
    phase_c_write_launch "$outdir" head \
      "$base --nnodes 2 --node-rank 0 --master-addr $HEAD_NODE_IP --master-port $MASTER_PORT"
    phase_c_write_launch "$outdir" worker \
      "$base --nnodes 2 --node-rank 1 --master-addr $HEAD_NODE_IP --master-port $MASTER_PORT --headless"
    scp -q "$outdir/launch-worker.sh" "$NODE2_HOST:$PHASE_C_REMOTE_OUT/launch-worker.sh" || return 1
    docker exec -d "$PHASE_C_HEAD_CONTAINER" bash /out/launch-head.sh
    node2_ssh "docker exec -d $PHASE_C_WORKER_CONTAINER bash /out/launch-worker.sh" || return 1
  fi
  phase_c_wait_health "$outdir" || return 1
}

run_config() {
  local config="$1" stage="$2"
  local topo="${config##*-}" tp pp outdir benchdir
  case "$topo" in
    tp2) tp=2; pp=1 ;;
    pp2) tp=1; pp=2 ;;
    *) die "unknown topology in config: $config" ;;
  esac
  outdir="$RESULT_ROOT/$config"
  benchdir="$outdir/$stage"
  mkdir -p "$benchdir"
  log "=== $config ($stage): tp=$tp pp=$pp backend=${config%%-*} ==="

  if ! bring_up "$config" "$outdir" "$tp" "$pp"; then
    phase_c_collect_logs "$outdir"
    printf '%s %s bring-up-failed\n' "$config" "$stage" >> "$RESULT_ROOT/gate.txt"
    phase_c_teardown
    return 1
  fi
  phase_c_collect_logs "$outdir"
  phase_c_capture_facts "$outdir"

  local status=0
  if [[ "$stage" == smoke ]]; then
    SERVER_ALREADY_RUNNING=1 HOST=127.0.0.1 PORT="$PORT" \
      CONCURRENCY_LEVELS="$PHASE_C_SMOKE_CONCURRENCY" SEEDS="$PHASE_C_SMOKE_SEEDS" \
      NUM_PROMPTS="$PHASE_C_SMOKE_PROMPTS" WARMUPS=1 \
      "$SCRIPT_DIR/bench_one.sh" "$WINNER_KEY" "$benchdir" || status=1
  else
    SERVER_ALREADY_RUNNING=1 HOST=127.0.0.1 PORT="$PORT" \
      "$SCRIPT_DIR/bench_one.sh" "$WINNER_KEY" "$benchdir" || status=1
  fi
  phase_c_collect_logs "$outdir"
  phase_c_teardown

  if (( status != 0 )); then
    printf '%s %s bench-failed\n' "$config" "$stage" >> "$RESULT_ROOT/gate.txt"
    return 1
  fi
  if [[ "$stage" == smoke ]]; then
    local verdict
    verdict="$(smoke_verdict "$benchdir")" || { printf '%s smoke %s\n' "$config" "$verdict" >> "$RESULT_ROOT/gate.txt"; log "$config smoke: $verdict"; return 1; }
    printf '%s smoke %s\n' "$config" "$verdict" >> "$RESULT_ROOT/gate.txt"
    log "$config smoke: $verdict"
  fi
  return 0
}

capture_fabric
: > "$RESULT_ROOT/gate.txt"

QUALIFIED=()
if [[ "$MODE" == full ]]; then
  read -r -a QUALIFIED <<< "$PHASE_C_CONFIGS"
else
  for config in $PHASE_C_CONFIGS; do
    if run_config "$config" smoke; then
      QUALIFIED+=("$config")
    else
      log "$config did not clear the smoke gate; excluded from the full sweep"
    fi
  done
fi

log "configs qualified for the full sweep: ${QUALIFIED[*]:-none}"

if [[ "$MODE" != smoke ]]; then
  for config in "${QUALIFIED[@]:-}"; do
    [[ -n "$config" ]] || continue
    run_config "$config" sweep || log "$config full sweep failed"
  done
fi

python3 "$SCRIPT_DIR/collect.py" "$RESULT_ROOT" || true
log "Phase C complete: $RESULT_ROOT"
