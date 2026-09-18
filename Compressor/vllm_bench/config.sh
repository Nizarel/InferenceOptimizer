#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/Compressor/vllm_bench"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/Compressor/benchmark_results/vllm_$(date -u +%Y%m%dT%H%M%SZ)}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/vllm:26.05-py3}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT_DIR/Compressor/.hf-cache}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-512}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
REPEATS="${REPEATS:-3}"
WARMUPS="${WARMUPS:-1}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16 32 64}"
SEEDS="${SEEDS:-101 202 303}"

MODEL_KEYS=(bf16 gptq awq fp8)
declare -A MODEL_PATHS=(
  [bf16]="Qwen/Qwen3-8B"
  [gptq]="$ROOT_DIR/models/Qwen3-8B-W4A16"
  [awq]="$ROOT_DIR/models/Qwen3-8B-AWQ-W4A16"
  [fp8]="$ROOT_DIR/models/Qwen3-8B-FP8-Dynamic"
)
declare -A SERVED_NAMES=(
  [bf16]="qwen3-8b-bf16"
  [gptq]="qwen3-8b-gptq"
  [awq]="qwen3-8b-awq"
  [fp8]="qwen3-8b-fp8"
)

# --- Phase C: cross-node distributed serving --------------------------------
NODE2_HOST="${NODE2_HOST:-}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
MN_IF_NAME="${MN_IF_NAME:-}"
RDMA_HCA="${RDMA_HCA:-rocep1s0f1}"
# Without nvidia_peermem, NCCL pins host bounce buffers; 64 default channels exhaust
# RDMA memory registration (ibv_reg_mr_iova2: Cannot allocate memory).
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
# NCCL >=2.28 allocates even the host-side proxy bounce buffers through the CUDA VMM
# (cuMem) allocator and registers them with the NIC via dmabuf. This box has no
# nvidia_peermem and its libmlx5 lacks mlx5dv_reg_dmabuf_mr, so that registration
# always fails. Forcing the legacy allocators restores plain host-memory registration.
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
# 0 = RoCE/IB verbs transport, 1 = TCP sockets over the same QSFP interface.
# This platform has no nvidia_peermem and its libmlx5 lacks mlx5dv_reg_dmabuf_mr, so
# NCCL cannot register its proxy buffers with the NIC and IB init fails; see
# basic/Docs/multi-node-phase-c.md for the full diagnosis.
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
PHASE_C_CONFIGS="${PHASE_C_CONFIGS:-ray-tp2 ray-pp2 mp-tp2 mp-pp2}"
PHASE_C_HEAD_CONTAINER="${PHASE_C_HEAD_CONTAINER:-phase-c-head}"
PHASE_C_WORKER_CONTAINER="${PHASE_C_WORKER_CONTAINER:-phase-c-worker}"
PHASE_C_REMOTE_OUT="${PHASE_C_REMOTE_OUT:-/tmp/phase-c-out}"
RAY_PORT="${RAY_PORT:-6379}"
MASTER_PORT="${MASTER_PORT:-29501}"
RAY_SPEC="${RAY_SPEC:-ray[cgraph]}"
# Smoke gate: cheap screen before committing ~70 min to a full sweep per config.
PHASE_C_SMOKE_CONCURRENCY="${PHASE_C_SMOKE_CONCURRENCY:-1 8}"
PHASE_C_SMOKE_SEEDS="${PHASE_C_SMOKE_SEEDS:-101}"
PHASE_C_SMOKE_PROMPTS="${PHASE_C_SMOKE_PROMPTS:-40}"
PHASE_C_MIN_C1_TPS="${PHASE_C_MIN_C1_TPS:-20}"
PHASE_C_HEALTH_TIMEOUT="${PHASE_C_HEALTH_TIMEOUT:-1800}"

export ROOT_DIR BENCH_DIR RESULT_ROOT VLLM_IMAGE NGC_IMAGE HF_CACHE_DIR PORT HOST
export MAX_MODEL_LEN GPU_MEMORY_UTILIZATION NUM_PROMPTS RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN
export REPEATS WARMUPS CONCURRENCY_LEVELS SEEDS
export NODE2_HOST HEAD_NODE_IP MN_IF_NAME RDMA_HCA PHASE_C_CONFIGS
export PHASE_C_HEAD_CONTAINER PHASE_C_WORKER_CONTAINER PHASE_C_REMOTE_OUT
export RAY_PORT MASTER_PORT RAY_SPEC PHASE_C_HEALTH_TIMEOUT
export NCCL_MAX_NCHANNELS NCCL_BUFFSIZE NCCL_CUMEM_ENABLE NCCL_CUMEM_HOST_ENABLE NCCL_IB_DISABLE
export PHASE_C_SMOKE_CONCURRENCY PHASE_C_SMOKE_SEEDS PHASE_C_SMOKE_PROMPTS PHASE_C_MIN_C1_TPS
