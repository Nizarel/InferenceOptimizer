#!/usr/bin/env python3
"""Measure cross-node NCCL all-reduce bandwidth so Phase C results can be read against the fabric."""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

SIZES_MIB = (1, 4, 16, 64, 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(0)

    for size_mib in SIZES_MIB:
        buffer = torch.ones(size_mib * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        for _ in range(args.warmup):
            dist.all_reduce(buffer)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(buffer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Ring all-reduce moves 2*(W-1)/W of the buffer per rank.
        moved_gib = 2 * (args.world_size - 1) / args.world_size * size_mib / 1024
        if args.rank == 0:
            print(
                f"size={size_mib:4d}MiB  latency={elapsed / args.iters * 1e3:8.3f} ms  "
                f"busbw={moved_gib * args.iters / elapsed:7.2f} GiB/s",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
