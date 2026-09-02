---
title: "Serve LLMs with vLLM — Instructions"
canonical: "https://build.nvidia.com/spark/vllm/instructions.md"
---

> [!NOTE] These instructions target **Linux** (containerized vLLM). WSL and Windows Native are not applicable to the containerized vLLM workflow at this time.

# Step 1. Set up Docker permissions

To manage containers without `sudo`, add your user to the `docker` group. Open a terminal and test Docker access:

```shell
docker ps
```

If you see a permission-denied error, add your user to the docker group (skip if it already works):

```shell
sudo usermod -aG docker $USER
newgrp docker
```

# Step 2. Set up environment variables

Find your model's HuggingFace handle and launch settings on [vLLM Recipes](https://recipes.vllm.ai/browse) for your hardware platform. Set these so the vLLM container can download and serve your model:

```shell
# HuggingFace token (required for gated / private models)
# Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"

# Model to serve (HuggingFace handle from vLLM Recipes for your hardware platform)
export MODEL_HANDLE="<HF_HANDLE>"

# Tag for the vLLM image (recommended in the vLLM Recipes), then pull
export VLLM_IMAGE=vllm/vllm-openai:latest 
docker pull "$VLLM_IMAGE"

# Maximum context length (prompt + output). Size to your workload and VRAM.
export MAX_MODEL_LEN=131072
```

# Step 3. Start the vLLM server

## Hardware platform launch notes

Container flags differ slightly by hardware platform. `--gpus all` is correct on all supported hardware platforms unless noted below. Apply the note for your hardware platform to any recipe below:

| Hardware platform | Launch notes |
| :---- | :---- |
| **DGX Spark** | Unified memory (UMA). If you hit memory pressure even within capacity, flush the buffer cache (see Troubleshooting). For multi-node serving, use the **Multi-node serving** tab (multi-node capable hardware only). |
| **DGX Station** | Add `--ipc host`. `--gpus all` uses the GB300; to pin the GB300 when both GPUs are present, use `--gpus '"device=N"'` where `N` is the GB300 device id from `nvidia-smi`. |

## Base configuration (most models)

Recommended starting point for any model that fits in memory on a single node. 

```shell
docker run -d \
--name vllm-server \
--gpus all \
--ipc host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--entrypoint "" \
-p 8000:8000 \
-e HF_TOKEN="$HF_TOKEN" \
-v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
"$VLLM_IMAGE" \
vllm serve "$MODEL_HANDLE" \
--max-model-len $MAX_MODEL_LEN \
--gpu-memory-utilization 0.8
```

Settings used:

- `--max-model-len` — maximum context length (prompt + output) per request. Larger values reserve more GPU memory for the KV cache; size it to your workload.  
- `--gpu-memory-utilization 0.8` — fraction of GPU memory vLLM may use for weights and KV cache. `0.8` leaves headroom; raise toward `0.95` on a dedicated GPU to fit more KV cache.

## Agent-ready models

For agentic workloads (tool calling, reasoning, long multi-turn sessions), see the **Agent-ready Models** tab for hardware-platform recommendations and launch guidance.

## Watch startup

Check the server logs for startup progress:

```shell
docker logs -f vllm-server
```

Expected output includes:

- Model download progress (first run only)  
- Model loading into GPU memory  
- `Application startup complete.`

Or wait for the health endpoint to come up (model loading can take several minutes):

```shell
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done' \
|| { echo "Server failed to start within 900s"; docker logs vllm-server | tail -50; exit 1; }
```

# Step 4. Test the API

Send a test request to verify the server:

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_HANDLE"'",
"messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
"max_tokens": 2048
}'
```

The response should contain a `choices` array with the model's answer in `message.content`.

> Recipes that enable a reasoning parser (or models that think by default) spend part of the completion budget on a thinking pass before the answer. Use a large enough `max_tokens` (this example uses `2048`) so generation can finish with `finish_reason: stop` and a non-null `content`. If you lower the budget too far, you may see `finish_reason: length` with thinking text only (often under `reasoning` or `reasoning_content`) and `content: null`.

# Step 5. Stop the container

Stop and remove the container when you are done testing (non-destructive — your model cache is preserved):

```shell
docker stop vllm-server
docker rm vllm-server
```

Optionally, remove the image and cached model:

```shell
docker rmi "<docker image name>"
rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"
```

# Next steps

- **Production deployment:** configure vLLM for your specific model and workload  
- **Performance tuning:** adjust batch sizes, `--max-model-len`, and memory settings  
- **Monitoring:** set up logging and metrics collection  
- **Agent-ready models:** tool-calling and reasoning workloads — see the **Agent-ready Models** tab  
- **Scale out:** serve larger models across multiple nodes on multi-node capable hardware — see the **Multi-node serving** tab