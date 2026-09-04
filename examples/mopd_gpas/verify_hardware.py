#!/usr/bin/env python3
"""Verify the revised protocol's distinct 96GB-train/48GB-inference GPU pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

MIN_TRAIN_MEMORY_MIB = 90_000
MIN_INFERENCE_MEMORY_MIB = 45_000
MAX_INFERENCE_MEMORY_MIB = 55_000
QUERY_ATTEMPTS = 3
QUERY_TIMEOUT_SECONDS = 2.0
QUERY_RETRY_DELAY_SECONDS = 0.1


def query() -> dict[int, dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    last_error = "unknown error"
    for attempt in range(QUERY_ATTEMPTS):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode:
                last_error = f"exit_code={result.returncode}, stdout={stdout!r}, stderr={stderr!r}"
            else:
                try:
                    output = {}
                    for line in stdout.splitlines():
                        index, name, memory, uuid, driver = (part.strip() for part in line.split(",", 4))
                        output[int(index)] = {
                            "name": name,
                            "memory_total_mib": int(memory),
                            "uuid": uuid,
                            "driver_version": driver,
                        }
                    if not output:
                        raise ValueError("empty GPU inventory")
                except ValueError as exc:
                    last_error = f"invalid stdout={stdout!r}: {exc}"
                else:
                    return output

        if attempt + 1 < QUERY_ATTEMPTS:
            time.sleep(QUERY_RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"failed to query the frozen-protocol GPU inventory after {QUERY_ATTEMPTS} attempts: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-gpu", type=int, required=True)
    parser.add_argument("--inference-gpu", type=int, required=True)
    args = parser.parse_args()
    requested = [args.training_gpu, args.inference_gpu]
    if args.training_gpu == args.inference_gpu:
        raise ValueError("training and inference GPUs must be distinct")
    inventory = query()
    for index in requested:
        if index not in inventory:
            raise ValueError(f"physical GPU {index} is absent from nvidia-smi")
    train = inventory[args.training_gpu]
    inference = inventory[args.inference_gpu]
    if int(train["memory_total_mib"]) < MIN_TRAIN_MEMORY_MIB:
        raise ValueError(f"training GPU {args.training_gpu} has only {train['memory_total_mib']} MiB")
    inference_memory_mib = int(inference["memory_total_mib"])
    allow_larger_inference_gpu = os.environ.get("MOPD_ALLOW_LARGER_INFERENCE_GPU") == "1"
    if inference_memory_mib < MIN_INFERENCE_MEMORY_MIB or (
        inference_memory_mib > MAX_INFERENCE_MEMORY_MIB and not allow_larger_inference_gpu
    ):
        raise ValueError(
            f"inference GPU {args.inference_gpu} has {inference['memory_total_mib']} MiB; "
            "the frozen slot must be a 48GB-class GPU "
            "(set MOPD_ALLOW_LARGER_INFERENCE_GPU=1 to record and allow a larger substitute)"
        )
    print(
        json.dumps(
            {
                "training_gpu": {"index": args.training_gpu, **train},
                "inference_gpu": {"index": args.inference_gpu, **inference},
                "larger_inference_gpu_substitution": inference_memory_mib > MAX_INFERENCE_MEMORY_MIB,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
