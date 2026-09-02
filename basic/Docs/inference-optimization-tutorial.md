# LLM Inference Optimization: A Hands-On Tutorial

This tutorial teaches inference optimization as an experimental discipline:
measure a baseline, change one variable, measure again, and keep only changes
that improve the target workload.

It focuses on vLLM and NVIDIA GPUs because that matches this repository. The
measurement method also applies to other inference runtimes.

## What You Will Learn

By the end, you will be able to:

- distinguish latency, throughput, and capacity;
- explain prefill, decode, model weights, and the KV cache;
- create a repeatable serving benchmark;
- tune context length, concurrency, batching, and memory utilization;
- evaluate prefix caching and quantization without misleading yourself;
- choose between replication, tensor parallelism, and pipeline parallelism;
- profile a slow workload and form an evidence-based optimization hypothesis.

Complete the lessons in order. After each lesson, save the requested result.
When following this tutorial interactively, paste that result into the chat
before moving to the next lesson.

## Prerequisites

The serving labs assume:

- Linux on the inference host;
- an NVIDIA GPU with a working driver;
- Docker with NVIDIA Container Toolkit support;
- enough storage for a model;
- a Hugging Face token if the selected model is gated.

DGX Spark users can use the repository's
[single-node instructions](instructions.md) and
[multi-node guide](MultiDGX.md). Start with one node even if several are
available.

Use a small instruct model that comfortably fits on one GPU. Keep the same
model, runtime image, prompts, and output lengths until a lesson explicitly
changes one of them. Pin the container image rather than using `latest` for
recorded experiments.

## Lesson 1: Define the Optimization Target

Inference optimization is not simply "make it faster." Improvements in one
dimension can hurt another.

Learn these metrics first:

| Metric | Meaning | Typical priority |
|---|---|---|
| TTFT | Time from request arrival to the first output token | Interactive chat |
| ITL | Time between consecutive output tokens | Streaming smoothness |
| TPOT | Average time per generated output token | Generation speed |
| End-to-end latency | Time until the entire response completes | Batch jobs and APIs |
| Request throughput | Completed requests per second | Serving capacity |
| Token throughput | Input or output tokens processed per second | GPU efficiency |
| Goodput | Throughput that satisfies latency objectives | Production capacity |
| Peak memory | Maximum accelerator memory consumed | Model and concurrency capacity |

An LLM request has two important phases:

1. **Prefill** processes prompt tokens, usually as large parallel matrix
   operations. Long prompts tend to increase TTFT.
2. **Decode** generates tokens one at a time. It repeatedly reads model weights
   and the request's KV cache. Long outputs tend to expose ITL and TPOT.

Write a workload contract before tuning:

```text
Use case:
Model:
Hardware:
Prompt length: p50=___, p95=___ tokens
Output length: p50=___, p95=___ tokens
Concurrency or arrival rate:
Latency objective: p95 TTFT < ___ ms; p95 ITL < ___ ms
Primary goal: latency / throughput / memory / cost
Quality constraint:
```

**Checkpoint:** Save the completed workload contract. Do not optimize an
undefined workload.

## Lesson 2: Inventory the System

Run:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

Record:

```text
GPU model and count:
Memory per GPU:
Driver version:
Docker version:
Host OS:
Interconnect (PCIe, NVLink, or network):
```

The exact CUDA test-image tag is not important to the later benchmark, but it
must be compatible with the installed driver. If the container test fails,
repair GPU container access before continuing.

**Checkpoint:** The host and container must both see every intended GPU.

## Lesson 3: Run Offline Inference

Offline inference removes HTTP and queueing from the first experiment. Use the
repository example:

```bash
python basic/offline_inference/basic.py
```

Then inspect the configurable example:

```bash
python basic/offline_inference/generate.py --help
python basic/offline_inference/generate.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --max-tokens 64 \
  --temperature 0
```

The first invocation includes model loading and runtime warm-up, so it is not a
steady-state latency measurement. Notice that sending multiple prompts lets the
runtime schedule work together; this is the beginning of batching.

**Experiment:** Run one prompt and then the four-prompt example. Observe GPU
utilization with `nvidia-smi dmon` in another terminal.

**Checkpoint:** Explain why model-load time must not be compared with
steady-state request time.

## Lesson 4: Establish an Online Baseline

Follow [instructions.md](instructions.md) to start a single vLLM server. For a
small learning model, use a realistic context limit rather than reserving the
model's maximum advertised context:

```bash
export MODEL_HANDLE=meta-llama/Llama-3.2-1B-Instruct
export MAX_MODEL_LEN=4096
```

Confirm health and inspect the exposed metrics:

```bash
curl -f http://localhost:8000/health
curl -s http://localhost:8000/metrics | grep '^vllm:'
```

Run the included client from an environment with the OpenAI Python package:

```bash
python basic/online_serving/openai_chat_completion_client.py --stream
```

Record the model identifier, exact image tag or digest, launch command, and
startup memory use. This is configuration **B0**, the baseline.

**Checkpoint:** A streamed request succeeds and `/metrics` returns vLLM
metrics.

## Lesson 5: Build a Reproducible Benchmark

Do not benchmark by watching one request. Use `vllm bench serve`. Check the
installed version's exact options first:

```bash
vllm bench serve --help
```

A representative synthetic test is:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model "$MODEL_HANDLE" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 2
```

If the CLI is only inside the server container, execute the benchmark from a
second environment containing the same vLLM version. Do not include model
startup in the measurement.

Benchmark rules:

1. warm up the server;
2. use fixed input and output token distributions;
3. run at least three measured trials;
4. change one variable at a time;
5. report percentiles, not only averages;
6. preserve raw output and failed-request counts;
7. monitor GPU utilization, memory, power, and KV-cache use;
8. validate response quality after precision-changing optimizations.

Use this table for every experiment:

| ID | Change | Input/output | Rate/concurrency | p50/p95 TTFT | p50/p95 ITL | Output tok/s | Req/s | Peak memory | Errors |
|---|---|---|---|---|---|---|---|---|---|
| B0 | Baseline | 512/128 | 2 req/s | | | | | | |

Run a second workload with long prompts and short outputs, such as 2048/32, and
a third with short prompts and long outputs, such as 128/512. This separates
prefill-sensitive behavior from decode-sensitive behavior.

**Checkpoint:** Produce three repeatable baseline rows whose run-to-run
variation is understood.

## Lesson 6: Find the Saturation Point

The fastest single request does not reveal server capacity. Sweep offered load:

```text
0.5, 1, 2, 4, 8, 16, ... requests/second
```

Stop when errors appear, throughput stops increasing, or the latency objective
is violated. Plot or tabulate offered load against:

- achieved request and token throughput;
- p50 and p95 TTFT;
- p50 and p95 ITL;
- running, waiting, and preempted requests;
- KV-cache utilization;
- GPU utilization and memory.

At low load, latency should be close to service time. Near saturation, queueing
causes TTFT to rise sharply. The highest point that still meets the service
level objective is the useful capacity, or goodput.

**Checkpoint:** Identify the "knee" of the latency-throughput curve.

## Lesson 7: Understand and Tune Memory

Memory is approximately divided among:

```text
model weights + KV cache + activations/workspace + runtime overhead
```

A rough decoder-only KV-cache size per request is:

```text
2 * layers * KV heads * head dimension * bytes per element * cached tokens
```

The factor `2` represents keys and values. Architecture details, block
allocation, alignment, and cache precision affect actual usage, so confirm with
runtime metrics.

Run controlled comparisons:

1. baseline context limit;
2. half the context limit;
3. a modest increase in `--gpu-memory-utilization`;
4. longer prompts at the same offered load.

Never raise memory utilization until the server barely fits. Leave operating
headroom and test the longest expected prompt at maximum useful concurrency.
CPU offload expands capacity but transfers weights over the CPU-GPU link during
forward passes, so treat it as a fit mechanism rather than a free speedup.

**Checkpoint:** Explain how context length changes KV-cache capacity and
concurrency.

## Lesson 8: Test Prefix Caching

Automatic prefix caching helps when requests reuse an identical long prefix,
such as a shared system prompt or document. It does not make the decode phase
faster.

Create two otherwise identical workloads:

- **cold/diverse:** every prompt has a different prefix;
- **warm/shared:** requests share a long prefix and differ near the end.

Start a new baseline server, measure both workloads, then restart with prefix
caching enabled:

```bash
vllm serve "$MODEL_HANDLE" \
  --max-model-len 4096 \
  --enable-prefix-caching
```

Compare TTFT and prefix-cache hit metrics. Restart between cold and warm tests
when you need an empty cache.

**Checkpoint:** Show that caching improves reused prefill work, not unrelated
prompts or token decode.

## Lesson 9: Evaluate Quantization

Quantization reduces the precision of weights, activations, or the KV cache.
Potential benefits are lower memory use, more KV-cache capacity, and faster
supported kernels. A smaller model file does not guarantee faster inference:
the hardware and runtime need efficient kernels for that format.

Compare a full-precision baseline with one hardware-supported quantized model:

1. keep the model family and workload equivalent;
2. verify that the runtime uses the intended quantization path;
3. repeat the load sweep;
4. record startup time, peak memory, TTFT, ITL, and throughput;
5. evaluate quality on a fixed task set.

The repository's [offline inference guide](../offline_inference/README.md)
contains a GGUF loading example. For production GPU experiments, select a
format from vLLM's hardware compatibility table rather than assuming every
format accelerates every GPU.

**Checkpoint:** State the memory, performance, and quality deltas separately.

## Lesson 10: Profile Before Advanced Tuning

Use server metrics first. If the cause remains unclear, use:

1. **Nsight Systems** for the end-to-end CPU/GPU timeline;
2. **Nsight Compute** only for selected expensive kernels;
3. a profiler-supported vLLM run with a short, representative workload.

Questions to answer:

- Is the GPU idle because requests are sparse or the CPU cannot feed it?
- Is time spent in prefill, decode, collectives, or queueing?
- Are kernels separated by large launch gaps?
- Is distributed communication using the intended high-speed interface?
- Is the workload memory-bandwidth-bound or compute-bound?

Do not begin with low-level kernel tuning when the bottleneck is queueing,
networking, oversized context reservation, or an unsuitable deployment shape.

**Checkpoint:** Write one profile-backed hypothesis and one experiment that
could disprove it.

## Lesson 11: Scale Up and Scale Out

Use additional GPUs only after the single-GPU baseline is understood:

- **Data parallel replicas** increase aggregate throughput when one model fits
  on one GPU or node.
- **Tensor parallelism** shards each layer and can make a large model fit, but
  adds frequent collective communication.
- **Pipeline parallelism** partitions layers and can reduce cross-node
  communication frequency, but introduces pipeline scheduling tradeoffs.

For two DGX Spark systems, follow [MultiDGX.md](MultiDGX.md). Compare:

1. one node, one replica;
2. two nodes, two independent replicas;
3. two nodes, one distributed replica only when required for fit.

Do not call a two-node result a speedup unless it improves the metric in the
workload contract. Distributed execution may increase capacity while worsening
single-request latency.

**Checkpoint:** Justify the chosen parallelism using fit, latency, throughput,
and communication evidence.

## Lesson 12: Capstone Optimization Report

Choose the best configuration using the original workload contract. Your final
report should contain:

```text
Goal and service-level objective:
Hardware and software versions:
Model and precision:
Prompt/output distribution:
Baseline configuration and results:
Winning configuration and results:
Percentage improvement:
Quality comparison:
Rejected optimizations and why:
Known limits:
Next experiment:
```

The winning configuration must survive a sustained test at the target load
without out-of-memory failures, unacceptable errors, or latency regression.

## Common Benchmarking Mistakes

- Timing the first request and calling compilation or warm-up "inference."
- Comparing different models, prompt lengths, or output lengths.
- Reporting average latency without p95 or p99.
- Maximizing tokens per second while violating interactive latency.
- Ignoring failed or preempted requests.
- Changing several flags at once.
- Assuming quantization always improves speed.
- Allocating the advertised maximum context when the workload does not need it.
- Distributing a model that already fits comfortably on one device.
- Trusting synthetic benchmarks without one realistic workload.

## Primary References

- [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)
- [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Hugging Face LLM optimization](https://huggingface.co/docs/transformers/llm_tutorial_optimization)
- [NVIDIA Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)

Prefer stable, version-matched documentation for the runtime image used in an
experiment. CLI `--help` output from that exact version is the final authority
for available flags.
