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

export ROOT_DIR BENCH_DIR RESULT_ROOT VLLM_IMAGE NGC_IMAGE HF_CACHE_DIR PORT HOST
export MAX_MODEL_LEN GPU_MEMORY_UTILIZATION NUM_PROMPTS RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN
export REPEATS WARMUPS CONCURRENCY_LEVELS SEEDS
