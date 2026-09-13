# vLLM Serving Benchmark Report

Run: `vllm_20260913T000908Z`

## Executive Summary

GPTQ W4A16 is the best overall serving configuration in this run. At maximum
tested concurrency (64), it reached a median output throughput of 1,041.95
tokens/s, ahead of AWQ at 996.59 tokens/s, FP8 at 805.61 tokens/s, and BF16 at
532.36 tokens/s.

FP8 had the lowest high-concurrency P99 TTFT (416.81 ms), so it remains a
reasonable latency-oriented option. GPTQ is the recommended default when
aggregate throughput and memory efficiency matter.

## Models

| Key | Model | Runtime checkpoint |
|---|---|---|
| `bf16` | Qwen3-8B BF16 | `Qwen/Qwen3-8B` from the local Hugging Face cache |
| `gptq` | GPTQ W4A16 | `models/Qwen3-8B-W4A16/` |
| `awq` | AWQ W4A16 | `models/Qwen3-8B-AWQ-W4A16/` |
| `fp8` | FP8 dynamic | `models/Qwen3-8B-FP8-Dynamic/` |

## vLLM Results

Values are medians across three seeds at each concurrency level. Each model
has 21 measured runs: concurrency 1, 2, 4, 8, 16, 32, and 64, with seeds
101, 202, and 303. All 84 runs completed with zero failed requests.

### Output Throughput

| Model | C=1 output tok/s | C=8 output tok/s | C=64 output tok/s | C=64 total tok/s |
|---|---:|---:|---:|---:|
| BF16 | 13.56 | 109.95 | 532.36 | 2,661.80 |
| **GPTQ W4A16** | **40.14** | **309.09** | **1,041.95** | **5,209.74** |
| AWQ W4A16 | 39.98 | 292.92 | 996.59 | 4,982.93 |
| FP8 dynamic | 23.79 | 188.86 | 805.61 | 4,028.03 |

GPTQ's C=64 output throughput is approximately 1.96x BF16, 1.05x AWQ, and
1.29x FP8.

### Latency

| Model | C=1 P99 TTFT | C=8 P99 TTFT | C=64 P99 TTFT | C=64 median TPOT |
|---|---:|---:|---:|---:|
| BF16 | 148.78 ms | 242.85 ms | 494.86 ms | 98.16 ms |
| GPTQ W4A16 | 153.55 ms | 121.84 ms | 508.86 ms | 53.27 ms |
| AWQ W4A16 | 164.02 ms | 126.13 ms | 511.39 ms | 55.73 ms |
| **FP8 dynamic** | **104.92 ms** | **139.08 ms** | **416.81 ms** | **67.26 ms** |

At low concurrency, FP8 has the best TTFT. At concurrency 8 and above, GPTQ
has the best output throughput and lower TPOT than AWQ and FP8. At concurrency
64, GPTQ and AWQ have similar TTFT; GPTQ has the higher throughput.

## Earlier Transformers Baseline

These measurements came from `Compressor/Compress2.ipynb` and are not serving
benchmarks. They use Transformers `.generate()` rather than a vLLM server.

| Model | Checkpoint size | Transformers output tok/s | Perplexity |
|---|---:|---:|---:|
| BF16 | 15.27 GiB | 9.36 | 14.83 |
| GPTQ W4A16 | 5.67 GiB | 7.77 | 15.37 |
| AWQ W4A16 | 5.69 GiB | 7.58 | 15.57 |
| FP8 dynamic | 8.80 GiB | 6.42 | 15.00 |

The apparent Transformers slowdown for GPTQ is explained by compressed-tensors
decompression during HF loading. The vLLM server log confirms the intended
runtime path for GPTQ:

- `quantization=compressed-tensors`
- `Using MarlinLinearKernel for CompressedTensorsWNA16`
- GPU KV cache: 507,744 tokens
- Maximum concurrency at 4,096 tokens/request: 123.96x

This is why the serving result, rather than the earlier local `.generate()`
result, should guide deployment decisions.

## Protocol

- vLLM image: `vllm/vllm-openai:latest`, vLLM `0.28.0`
- Docker with `--ipc host`, CUDA graphs enabled, no `--enforce-eager`
- `--max-model-len 4096`
- `--gpu-memory-utilization 0.65`
- Random workload: 512 input tokens, 128 output tokens, 200 prompts
- `--request-rate inf`, concurrency 1/2/4/8/16/32/64
- One warmup per model, then three seeds per concurrency level
- Cache reset requested between measured runs
- Results collected from `vllm bench serve`

Raw artifacts are organized under the model directories in this run. The
machine-readable aggregate is `Compressor/benchmark_results/vllm_20260913T000908Z/serving_results.csv`.

## Interpretation and Caveats

1. The GPU had an unrelated Python process using approximately 16.9 GiB, so
   the benchmark used `GPU_MEMORY_UTILIZATION=0.65` instead of 0.85. This is a
   controlled comparison, but not a maximum-capacity result.
2. vLLM 0.28.0 reported prefix caching enabled by default. Cache resets were
   requested between runs, but a production-style cold-cache validation should
   be run explicitly with prefix caching disabled or with a fresh server per
   trial.
3. The benchmark used one DGX Spark. The planned two-DGX comparison has not
   run because the second node's hostname or QSFP IP is not configured.
4. The earlier PPL values are quality indicators, not a substitute for task
   accuracy evaluation or production validation.

## Recommendation

Use GPTQ W4A16 as the default Qwen3-8B vLLM deployment candidate. Use FP8 when
the workload prioritizes lower TTFT over maximum aggregate throughput. Before
production selection, repeat GPTQ and FP8 on an otherwise idle GPU with a
cold-cache protocol, then compare GPTQ on one Spark against two independent
GPTQ replicas across both Sparks.
