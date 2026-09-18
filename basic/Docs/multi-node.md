---
title: "Serve LLMs with vLLM — Multi-node serving"
canonical: "https://build.nvidia.com/spark/vllm/multi-node.md"
---

# Multi-node serving

> **This repository's Phase C runbook is [multi-node-phase-c.md](multi-node-phase-c.md).** It documents the
> automated two-node procedure actually used here, including the Ray-free `mp` backend, pipeline
> parallelism, and why RDMA/RoCE could not be used on these nodes (NCCL falls back to TCP over the
> same QSFP link). It deliberately keeps `vllm/vllm-openai:latest` rather than
> the `nvcr.io/nvidia/vllm:26.05-py3` image pinned below, to stay comparable with Phases A and B.
>
> Note that the tensor-parallel topology this page describes measured **worse than a single node** at
> every concurrency level tested, for a model that fits on one GPU. See section 13 of the runbook.

Serve models larger than a single node can hold by pooling GPUs across multiple **multi-node capable hardware** systems with a Ray cluster and tensor parallelism. Two topologies are covered:

- **Two nodes (direct QSFP cable)** — connect two nodes back-to-back.  
- **Four or more nodes through a QSFP switch** — scale out over a switch fabric.

>   
> This tab applies to **multi-node capable hardware** only. Other supported hardware platforms serve models on a single node (see the Instructions tab).

## Prerequisites

### Docker permissions

If `docker ps` fails with a permission error, complete [Step 1 in the Instructions tab](http://instructions.md) on every node in the cluster before continuing.

---

## A. Two nodes (direct QSFP cable)

### Step 1. Configure network connectivity

Follow the [Connect two nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-two-sparks) playbook to establish connectivity on multi-node capable hardware: physical QSFP cable, network interface configuration, passwordless SSH, and connectivity verification.

> **Heads up:** the connectivity script from that playbook writes its SSH key to `~/.ssh/` and fails if the directory does not exist. Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both nodes first if you have never used SSH on them.

### Step 2. Download the cluster deployment script

On **both nodes**, download and patch the Ray cluster script:

```shell
wget https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh

sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' run_cluster.sh

chmod +x run_cluster.sh
```

### Step 3. Pull the NGC vLLM image

Pull the image **on both nodes**:

```shell
docker pull nvcr.io/nvidia/vllm:26.05-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
```

### Step 4. Start the Ray head node (Node 1)

Run inside tmux/screen so an SSH drop doesn't tear down the cluster (`run_cluster.sh` has an EXIT trap that stops the container).

Set `MN_IF_NAME` to the QSFP interface name from your connectivity playbook (validated example on multi-node capable hardware: `enp1s0f1np1`). Substitute if your interface differs.

```shell
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
-e VLLM_HOST_IP=$VLLM_HOST_IP \
-e UCX_NET_DEVICES=$MN_IF_NAME \
-e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
-e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
-e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
-e TP_SOCKET_IFNAME=$MN_IF_NAME \
-e RAY_memory_monitor_refresh_ms=0 \
-e MASTER_ADDR=$VLLM_HOST_IP
```

Leave this terminal open — closing it stops the head node and tears down the cluster.

### Step 5. Start the Ray worker node (Node 2)

Open a second terminal, SSH to Node 2, and join the cluster. Replace `<NODE_1_IP_ADDRESS>` with Node 1's QSFP IP (run `echo $VLLM_HOST_IP` on Node 1). Run inside tmux/screen on Node 2 as well. Use the same `MN_IF_NAME` guidance as Step 4.

```shell
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
-e VLLM_HOST_IP=$VLLM_HOST_IP \
-e UCX_NET_DEVICES=$MN_IF_NAME \
-e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
-e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
-e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
-e TP_SOCKET_IFNAME=$MN_IF_NAME \
-e RAY_memory_monitor_refresh_ms=0 \
-e MASTER_ADDR=$HEAD_NODE_IP
```

### Step 6. Verify cluster status

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"
docker exec $VLLM_CONTAINER ray status
```

Expected output shows 2 nodes with available GPU resources.

### Step 7. Download Llama 3.3 70B

Llama 3.3 70B is gated — accept its license at [https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and create an HF token with read permission. Authenticate inside the container so the cache lands at `/root/.cache/huggingface`:

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
hf auth login
hf download meta-llama/Llama-3.3-70B-Instruct'
```

### Step 8. Launch inference server (tensor parallel across both nodes)

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
vllm serve meta-llama/Llama-3.3-70B-Instruct \
--tensor-parallel-size 2 --max-model-len 2048 \
--distributed-executor-backend ray'
```

### Step 9. Test inference

Run on Node 1; from an external client, replace `localhost` with Node 1's reachable IP.

```shell
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "meta-llama/Llama-3.3-70B-Instruct",
"prompt": "Write a haiku about a GPU",
"max_tokens": 32,
"temperature": 0.7
}'
```

### Step 10. (Optional) Llama 3.1 405B — two-node topology only

> [!WARNING] The 405B model has insufficient memory headroom for production use — testing only.

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
hf download hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4'

docker exec -it $VLLM_CONTAINER /bin/bash -c '
vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
--tensor-parallel-size 2 --max-model-len 64 --gpu-memory-utilization 0.9 \
--max-num-seqs 1 --max-num-batched-tokens 64 \
--distributed-executor-backend ray'
```

The server is ready when you see `Application startup complete.`

---

## B. Four or more nodes through a QSFP switch

Same Ray + tensor-parallel workflow as Section A, scaled to more nodes over a QSFP switch. Set `--tensor-parallel-size` equal to your node count.

> **Topology note:** the four-or-more-node path uses a different validated container image and `run_cluster.sh` source than the two-node path above. Follow the steps in this section exactly — do not mix image tags or script versions between topologies.

### Step 1. Configure network connectivity

Follow the [Connect multiple nodes through a switch](https://build.nvidia.com/playbooks/connect-sparks-via-switch) playbook for multi-node capable hardware: QSFP cabling between nodes and switch, interface configuration, passwordless SSH, connectivity verification, and the NCCL bandwidth test.

### Step 2. Download the cluster deployment script (all nodes)

On **every node**, download the Ray cluster script:

```shell
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/ray_serving/run_cluster.sh
chmod +x run_cluster.sh
```

### Step 3. Pull the NGC vLLM image (all nodes)

```shell
docker pull nvcr.io/nvidia/vllm:26.02-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
```

### Step 4. Start the Ray head node (Node 1)

Run inside tmux/screen so an SSH drop doesn't tear down the cluster.

Set `MN_IF_NAME` to the QSFP interface name from your connectivity playbook (validated example on multi-node capable hardware: `enp1s0f1np1`). Substitute if your interface differs.

```shell
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
-e VLLM_HOST_IP=$VLLM_HOST_IP \
-e UCX_NET_DEVICES=$MN_IF_NAME \
-e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
-e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
-e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
-e TP_SOCKET_IFNAME=$MN_IF_NAME \
-e RAY_memory_monitor_refresh_ms=0 \
-e MASTER_ADDR=$VLLM_HOST_IP
```

Leave this terminal open — closing it stops the head node and tears down the cluster.

### Step 5. Start the Ray worker nodes (all other nodes)

Repeat the block below on **each worker node** (Nodes 2 through N). SSH to each node in turn, run inside tmux/screen, and replace `<NODE_1_IP_ADDRESS>` with Node 1's QSFP interface IP from the switch playbook. Use the same `MN_IF_NAME` guidance as Step 4.

```shell
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
-e VLLM_HOST_IP=$VLLM_HOST_IP \
-e UCX_NET_DEVICES=$MN_IF_NAME \
-e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
-e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
-e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
-e TP_SOCKET_IFNAME=$MN_IF_NAME \
-e RAY_memory_monitor_refresh_ms=0 \
-e MASTER_ADDR=$HEAD_NODE_IP
```

### Step 6. Verify cluster status

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status
```

Expected output shows all nodes with available GPU resources.

### Step 7. Download MiniMax M2.5

With four or more nodes you can run this model with tensor parallelism. Authenticate and download inside the head-node container (the cache is shared across the cluster):

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
hf auth login
hf download MiniMaxAI/MiniMax-M2.5'
```

### Step 8. Launch inference server (tensor parallel = node count)

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
vllm serve MiniMaxAI/MiniMax-M2.5 \
--tensor-parallel-size 4 --max-model-len 129000 --max-num-seqs 4 --trust-remote-code \
--distributed-executor-backend ray'
```

Set `--tensor-parallel-size` to match your node count (example above uses 4).

### Step 9. Test inference

Run on Node 1; from an external client, replace `localhost` with Node 1's reachable IP.

```shell
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "MiniMaxAI/MiniMax-M2.5",
"prompt": "Write a haiku about a GPU",
"max_tokens": 32,
"temperature": 0.7
}'
```

---

## Validate and monitor (both topologies)

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

curl http://localhost:8000/health

nvidia-smi
```

On hardware platforms with unified memory, `nvidia-smi --query-gpu` memory fields report `N/A` — use plain `nvidia-smi` instead.

The **Ray dashboard** runs on port 8265 of the head node under host networking, so it is only directly reachable from Node 1. Tunnel it from a workstation:

```shell
ssh -L 8265:localhost:8265 nvidia@<NODE_1_IP>
# then open http://localhost:8265
```

## Next steps

Consider for production:

- Health checks and automatic restarts  
- Log rotation for long-running services  
- Persistent model caching across restarts  
- Alternative quantization (FP8, NVFP4, INT4) to fit more models on the cluster