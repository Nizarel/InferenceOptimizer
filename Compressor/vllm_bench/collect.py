#!/usr/bin/env python3
"""Collect vllm bench JSON results into a stable CSV without extra dependencies."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def flatten(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            output.update(flatten(item, child))
        return output
    return {prefix: value}


BENCH_NAME = re.compile(r"bench_c(?P<concurrency>\d+)_s(?P<seed>\d+)")
PHASE_DIRS = {"phase_b": "phase_b", "phase_c": "phase_c"}
STAGES = {"smoke", "sweep"}


def describe(relative: Path, absolute: Path) -> dict:
    """Derive phase/config/stage/concurrency/seed from the artifact's path."""
    parts = relative.parts
    info = {"phase": "phase_a", "config": "", "stage": "", "concurrency": "", "seed": ""}
    # Phase comes from the absolute path: collect may be rooted inside the phase directory,
    # in which case the phase name is not part of the relative path at all.
    for part in absolute.parts:
        if part in PHASE_DIRS:
            info["phase"] = PHASE_DIRS[part]
    directories = [part for part in parts[:-1] if part not in PHASE_DIRS]
    if directories:
        if directories[-1] in STAGES:
            info["stage"] = directories[-1]
            directories = directories[:-1]
        if directories:
            info["config"] = directories[-1]
    match = BENCH_NAME.search(relative.name)
    if match:
        info["concurrency"] = match.group("concurrency")
        info["seed"] = match.group("seed")
    return info


def collect(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not (path.suffix == ".json" or path.name.startswith("bench_")):
            continue
        if path.name in {"manifest.json", "hardware.json", "final_report.json"}:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        row = flatten(data)
        relative = path.relative_to(root)
        row["artifact"] = str(relative)
        row.update(describe(relative, path.resolve()))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = collect(args.result_dir)
    output = args.result_dir / "serving_results.csv"
    keys = sorted({key for row in rows for key in row})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
