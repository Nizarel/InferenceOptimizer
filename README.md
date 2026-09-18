# InferenceOptimizer

Learn LLM inference optimization through the hands-on
[step-by-step tutorial](basic/Docs/inference-optimization-tutorial.md).

See the completed [vLLM serving benchmark report](basic/Docs/reports/vllm-serving-benchmark.md)
for the BF16, GPTQ, AWQ, and FP8 comparison, raw result location, and the
two-DGX results covering both independent replicas and cross-node model splitting.

For the two-node distributed serving procedure itself — Ray and Ray-free bring-up,
tensor vs pipeline parallelism, NCCL transport verification, and troubleshooting —
see the [multi-node Phase C runbook](basic/Docs/multi-node-phase-c.md).
