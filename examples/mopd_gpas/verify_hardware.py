#!/usr/bin/env python3
"""Verify the GPU topology selected for a MOPD protocol run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

MIN_TRAIN_MEMORY_MIB = 90_000
MIN_48GB_MEMORY_MIB = 45_000
MAX_48GB_MEMORY_MIB = 55_000
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


def validate(
    profile: str,
    training_gpus: list[int],
    inference_gpu: int,
    inventory: dict[int, dict[str, object]],
    service_gpus: list[int] | None = None,
) -> dict[str, object]:
    """Validate and describe one of the two explicit protocol topologies."""

    if profile == "frozen-96gb-tp1":
        expected_training_gpus = 1
        tensor_parallel_size = 1
    elif profile == "dual-48gb-tp2":
        expected_training_gpus = 2
        tensor_parallel_size = 2
    else:
        raise ValueError(f"unknown MOPD hardware profile {profile!r}")
    if len(training_gpus) != expected_training_gpus:
        raise ValueError(
            f"{profile} requires {expected_training_gpus} training GPU(s), got {len(training_gpus)}"
        )
    service_gpus = service_gpus or []
    run_gpus = [*training_gpus, inference_gpu]
    if len(run_gpus) != len(set(run_gpus)):
        raise ValueError("training and inference GPUs must all be distinct")
    if set(training_gpus) & set(service_gpus):
        raise ValueError("teacher service GPUs must not overlap training GPUs")
    requested = [*run_gpus, *service_gpus]
    for index in requested:
        if index not in inventory:
            raise ValueError(f"physical GPU {index} is absent from nvidia-smi")

    training = []
    for index in training_gpus:
        device = inventory[index]
        memory_mib = int(device["memory_total_mib"])
        if profile == "frozen-96gb-tp1" and memory_mib < MIN_TRAIN_MEMORY_MIB:
            raise ValueError(f"training GPU {index} has only {memory_mib} MiB")
        if profile == "dual-48gb-tp2" and not MIN_48GB_MEMORY_MIB <= memory_mib <= MAX_48GB_MEMORY_MIB:
            raise ValueError(
                f"training GPU {index} has {memory_mib} MiB; dual-48gb-tp2 requires 48GB-class GPUs"
            )
        training.append({"index": index, **device})

    inference = inventory[inference_gpu]
    inference_memory_mib = int(inference["memory_total_mib"])
    allow_larger_inference_gpu = os.environ.get("MOPD_ALLOW_LARGER_INFERENCE_GPU") == "1"
    if inference_memory_mib < MIN_INFERENCE_MEMORY_MIB or (
        inference_memory_mib > MAX_INFERENCE_MEMORY_MIB and not allow_larger_inference_gpu
    ):
        raise ValueError(
            f"inference GPU {inference_gpu} has {inference_memory_mib} MiB; "
            "the frozen slot must be a 48GB-class GPU "
            "(set MOPD_ALLOW_LARGER_INFERENCE_GPU=1 to record and allow a larger substitute)"
        )
    services = []
    for index in dict.fromkeys(service_gpus):
        device = inventory[index]
        memory_mib = int(device["memory_total_mib"])
        if memory_mib < MIN_48GB_MEMORY_MIB:
            raise ValueError(f"teacher service GPU {index} has only {memory_mib} MiB")
        services.append({"index": index, **device})
    return {
        "profile": profile,
        "tensor_model_parallel_size": tensor_parallel_size,
        "training_gpus": training,
        "inference_gpu": {"index": inference_gpu, **inference},
        "teacher_service_gpus": services,
        "larger_inference_gpu_substitution": inference_memory_mib > MAX_INFERENCE_MEMORY_MIB,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("frozen-96gb-tp1", "dual-48gb-tp2"),
        default=os.environ.get("MOPD_HARDWARE_PROFILE", "frozen-96gb-tp1"),
    )
    parser.add_argument("--training-gpu", type=int, action="append", required=True)
    parser.add_argument("--inference-gpu", type=int, required=True)
    parser.add_argument("--service-gpu", type=int, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.profile, args.training_gpu, args.inference_gpu, query(), args.service_gpu),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
