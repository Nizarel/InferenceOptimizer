#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

WINNER_KEY="${1:-gptq}"
NODE2_HOST="${NODE2_HOST:-}"
MN_IF_NAME="${MN_IF_NAME:-}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
VLLM_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
RUN_CLUSTER_URL="${RUN_CLUSTER_URL:-https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh}"

[[ -n "$NODE2_HOST" ]] || die "set NODE2_HOST to the second DGX hostname or QSFP IP"
[[ -n "$MN_IF_NAME" ]] || die "set MN_IF_NAME to the QSFP interface"
[[ -n "$HEAD_NODE_IP" ]] || die "set HEAD_NODE_IP to node1's QSFP IP"
model_path "$WINNER_KEY" >/dev/null
[[ "${MODEL_PATHS[$WINNER_KEY]}" == /* ]] || die "Phase C requires a local winner checkpoint on both nodes"
command -v ssh >/dev/null || die "ssh is required for Phase C"

mkdir -p "$RESULT_ROOT/phase_c"
printf '%s\n' "phase_c_tp2" > "$RESULT_ROOT/phase_c/phase.txt"

for host in localhost "$NODE2_HOST"; do
  if [[ "$host" == localhost ]]; then
    docker pull "$VLLM_IMAGE" >/dev/null
    curl -fsSL "$RUN_CLUSTER_URL" -o "$RESULT_ROOT/phase_c/run_cluster.sh"
    sed -i "s|^RAY_START_CMD=\"ray start|RAY_START_CMD=\"pip install -q --root-user-action=ignore 'ray[default]>=2.9' \\&\\& ray start|" "$RESULT_ROOT/phase_c/run_cluster.sh"
    chmod +x "$RESULT_ROOT/phase_c/run_cluster.sh"
  else
    ssh "$host" "docker pull '$VLLM_IMAGE' >/dev/null"
    scp "$RESULT_ROOT/phase_c/run_cluster.sh" "$host:$RESULT_ROOT/phase_c/run_cluster.sh"
  fi
done

log "Phase C is staged. Start the head and worker in tmux using the pinned script, then launch vLLM with tensor-parallel-size=2."
cat > "$RESULT_ROOT/phase_c/launch.env" <<EOF
MN_IF_NAME=$MN_IF_NAME
HEAD_NODE_IP=$HEAD_NODE_IP
VLLM_IMAGE=$VLLM_IMAGE
WINNER_MODEL=${MODEL_PATHS[$WINNER_KEY]}
SERVED_NAME=$(served_name "$WINNER_KEY")
EOF
cat > "$RESULT_ROOT/phase_c/launch_command.txt" <<EOF
bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --head ~/.cache/huggingface
vllm serve /model --served-model-name $(served_name "$WINNER_KEY") --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --tensor-parallel-size 2 --distributed-executor-backend ray
EOF
