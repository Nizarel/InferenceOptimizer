#!/usr/bin/env python3
"""Collect vllm bench JSON results into a stable CSV without extra dependencies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            output.update(flatten(item, child))
        return output
    return {prefix: value}


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
        row["artifact"] = str(path.relative_to(root))
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
