# vLLM Serving Benchmark Report

Run: `vllm_20260913T000908Z`

## Executive Summary

GPTQ W4A16 is the best overall serving configuration in this run. At maximum
tested concurrency (64), it reached a median output throughput of 1,041.95
tokens/s, ahead of AWQ at 996.59 tokens/s, FP8 at 805.61 tokens/s, and BF16 at
532.36 tokens/s.

With two independent GPTQ replicas behind HAProxy, C=64 output throughput rose
to 1,446.19 tokens/s, a 1.39x improvement over the single-DGX GPTQ result.

Splitting one model across both DGX nodes was also tested (Phase C). It never
beat two independent replicas. Pipeline parallelism reached 1,216.04 tokens/s at
C=64 (1.17x single node); tensor parallelism reached only 595.71 tokens/s
(0.57x), losing to a single node at every concurrency level. **Replication beats
model splitting for any model that fits on one GPU.**

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

## Two-DGX Replica Results

Phase B ran one GPTQ replica per DGX and balanced requests with HAProxy over the
direct `192.168.100.0/24` fabric. Values are medians across the same three seeds.
All 21 runs completed with zero failed requests.

| Concurrency | Single output tok/s | Two replicas output tok/s | Scaling | Two-replica median TTFT | Two-replica P99 TTFT | Two-replica median TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.14 | 40.68 | 1.01x | 212.47 ms | 218.49 ms | 23.26 ms |
| 2 | 86.54 | 83.69 | 0.97x | 213.16 ms | 219.19 ms | 21.98 ms |
| 4 | 166.89 | 175.28 | 1.05x | 216.86 ms | 222.77 ms | 20.97 ms |
| 8 | 309.09 | 335.23 | 1.08x | 217.53 ms | 533.44 ms | 22.07 ms |
| 16 | 531.36 | 611.38 | 1.15x | 218.88 ms | 542.50 ms | 24.12 ms |
| 32 | 803.63 | 1,008.14 | 1.25x | 234.17 ms | 503.39 ms | 27.25 ms |
| 64 | 1,041.95 | 1,446.19 | 1.39x | 536.48 ms | 708.21 ms | 34.79 ms |

Scaling improves as concurrency rises, but it is not linear at the tested load.
At C=64, decoding latency improved from 53.27 ms to 34.79 ms TPOT while median
TTFT increased from 304.75 ms to 536.48 ms. The replica topology therefore
favors aggregate throughput over first-token latency.

## Cross-Node Distributed Results

Phase C split a single GPTQ model across both DGX nodes over the QSFP fabric,
testing tensor parallelism (TP=2) and pipeline parallelism (PP=2) under both the
Ray and Ray-free `mp` executor backends. Four configurations x 21 runs = 84 runs,
all completing with zero failed requests. Medians across the same three seeds.

Full setup, bring-up and troubleshooting detail is in
[../multi-node-phase-c.md](../multi-node-phase-c.md).

### Output throughput, all topologies

| Concurrency | 1 node | 2 replicas | PP=2 (ray) | PP=2 (mp) | TP=2 (ray) | TP=2 (mp) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.14 | 40.68 | 40.20 | 39.48 | 30.17 | 27.44 |
| 2 | 86.54 | 83.69 | 65.38 | 65.00 | 57.02 | 46.75 |
| 4 | 166.89 | 175.28 | 135.93 | 142.13 | 119.42 | 105.80 |
| 8 | 309.09 | 335.23 | 275.54 | 275.69 | 178.91 | 156.90 |
| 16 | 531.36 | 611.38 | 497.08 | 493.36 | 310.30 | 298.69 |
| 32 | 803.63 | 1,008.14 | 822.34 | 820.20 | 489.00 | 452.16 |
| 64 | 1,041.95 | **1,446.19** | 1,216.04 | 1,180.02 | 595.71 | 598.93 |

### Scaling relative to a single node

| Concurrency | 2 replicas | PP=2 (ray) | PP=2 (mp) | TP=2 (ray) | TP=2 (mp) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.01x | 1.00x | 0.98x | 0.75x | 0.68x |
| 2 | 0.97x | 0.76x | 0.75x | 0.66x | 0.54x |
| 4 | 1.05x | 0.81x | 0.85x | 0.72x | 0.63x |
| 8 | 1.08x | 0.89x | 0.89x | 0.58x | 0.51x |
| 16 | 1.15x | 0.94x | 0.93x | 0.58x | 0.56x |
| 32 | 1.25x | 1.02x | 1.02x | 0.61x | 0.56x |
| 64 | **1.39x** | 1.17x | 1.13x | 0.57x | 0.58x |

Pipeline parallelism only overtakes a single node at C>=32. Tensor parallelism
never does, and its penalty is roughly constant at 0.55-0.60x from C=8 upward.

### Latency

| Concurrency | 1 node TTFT | PP=2 TTFT | TP=2 TTFT | 1 node TPOT | PP=2 TPOT | TP=2 TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 144.8 ms | 146.0 ms | 257.3 ms | 23.87 ms | 23.92 ms | 29.96 ms |
| 2 | 74.6 ms | **57.8 ms** | 130.4 ms | 22.64 ms | 30.44 ms | 32.80 ms |
| 4 | 77.9 ms | **58.4 ms** | 152.8 ms | 23.45 ms | 29.25 ms | 32.72 ms |
| 8 | 81.7 ms | **62.9 ms** | 206.5 ms | 25.21 ms | 28.62 ms | 43.19 ms |
| 16 | 104.5 ms | **69.0 ms** | 239.4 ms | 28.54 ms | 30.53 ms | 48.79 ms |
| 32 | 162.9 ms | **136.6 ms** | 359.7 ms | 36.15 ms | 34.70 ms | 58.50 ms |
| 64 | 304.8 ms | **223.3 ms** | 462.6 ms | 53.27 ms | 43.27 ms | 94.96 ms |

PP=2 columns are the Ray backend; TP=2 columns are the Ray backend.

Pipeline parallelism has the **best median TTFT of any topology tested**, beating
a single node at every concurrency from 2 upward and beating the two-replica
setup by a wide margin (223.3 ms vs 536.48 ms at C=64). Each node holds half the
layers, so per-node prefill work halves. Its cost is TPOT at low concurrency
(30.44 ms vs 22.64 ms at C=2), where the pipeline cannot be kept full.

### Why tensor parallelism loses

TP=2 issues an all-reduce at every transformer layer, so each generated token
crosses the interconnect dozens of times. PP=2 transfers activations once per
stage boundary. The measured fabric ceiling is 12.95 GiB/s all-reduce bus
bandwidth (about 111 Gb/s of the 200 Gb/s link), which is orders of magnitude
below the on-package bandwidth TP assumes. TP=2 across nodes is therefore only
justified when a model does not fit in one GPU's memory.

### Executor backend: Ray vs multiprocessing

Under pipeline parallelism the backends are indistinguishable (1,216.04 vs
1,180.02 at C=64, within seed spread). Under tensor parallelism Ray is
consistently ahead at low-to-mid concurrency (178.91 vs 156.90 at C=8), but both
lose to a single node, so the difference has no practical consequence here.
Ray costs about 30 s of extra bring-up per run for `pip install` and cluster
formation. **Prefer the `mp` backend** unless Ray is already required.

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
Two-DGX artifacts and their aggregate CSV are under
`Compressor/benchmark_results/vllm_20260913T000908Z/phase_b/`.
Cross-node distributed artifacts are under
`Compressor/benchmark_results/vllm_20260913T000908Z/phase_c/<config>/sweep/`.

Phase C additions to the protocol:

- Both nodes on the direct `192.168.100.0/24` QSFP fabric (200 GbE, MTU 1500)
- NCCL pinned to `enp1s0f1np1`; transport `NET/Socket` (see caveat 5)
- Same image, model checkpoint, workload, seeds and concurrency levels as Phase A

## Interpretation and Caveats

1. The GPU had an unrelated Python process using approximately 16.9 GiB, so
   the benchmark used `GPU_MEMORY_UTILIZATION=0.65` instead of 0.85. This is a
   controlled comparison, but not a maximum-capacity result.
2. vLLM 0.28.0 reported prefix caching enabled by default. Cache resets were
   requested between runs, but a production-style cold-cache validation should
   be run explicitly with prefix caching disabled or with a fresh server per
   trial.
3. The two-DGX test used independent replicas behind HAProxy, not tensor
   parallelism. It measures horizontal serving scale and includes router and
   network overhead; it does not show single-request TP=2 performance.
4. The earlier PPL values are quality indicators, not a substitute for task
   accuracy evaluation or production validation.
5. Phase C ran NCCL over TCP sockets, not RoCE. RDMA initialization fails on
   these nodes because `nvidia_peermem` is not loaded and the image's `libmlx5`
   lacks `mlx5dv_reg_dmabuf_mr`, so NCCL cannot register its proxy buffers with
   the NIC. Traffic still crossed the 200 GbE QSFP link. A working RDMA path
   would raise the fabric ceiling and should improve the TP=2 numbers in
   particular; the TP-vs-PP ordering is unlikely to change, since the gap is
   roughly 2x and structural.
6. Phase C conclusions are specific to an 8B model that fits comfortably in one
   GB10's memory. For a model too large for a single GPU, splitting is not a
   choice but a requirement, and PP=2 is then the better of the two options
   tested.

## Recommendation

Use GPTQ W4A16 as the default Qwen3-8B vLLM deployment candidate. Use FP8 when
the workload prioritizes lower TTFT over maximum aggregate throughput. Before
production selection, repeat GPTQ and FP8 on otherwise idle GPUs with a
cold-cache protocol. Use two GPTQ replicas when aggregate throughput is the
priority; retain a single replica when first-token latency is more important.

For the two-node topology, choose by objective:

| Objective | Topology | Evidence |
|---|---|---|
| Maximum aggregate throughput | two independent replicas | 1,446.19 tok/s at C=64, 1.39x |
| Lowest time to first token | PP=2 across both nodes | 223.3 ms at C=64 vs 304.8 / 536.5 |
| Model too large for one GPU | PP=2, `mp` backend | 1.17x at C=64; TP=2 only reaches 0.57x |
| Anything else | single node | splitting an 8B model costs more than it returns |

Do not use cross-node TP=2 for models of this size. It was the worst topology at
every concurrency level tested.
