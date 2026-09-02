#!/usr/bin/env python3
"""Verify the GPU revision frozen by the four-task MOPD protocol."""

from __future__ import annotations

import argparse
import json
import subprocess

EXPECTED_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_MEMORY_MIB = 97_887


def query() -> dict[int, dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = {}
    for line in result.stdout.splitlines():
        index, name, memory, uuid, driver = (part.strip() for part in line.split(",", 4))
        output[int(index)] = {
            "name": name,
            "memory_total_mib": int(memory),
            "uuid": uuid,
            "driver_version": driver,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU indices")
    args = parser.parse_args()
    requested = [int(value) for value in args.gpu_ids.split(",")]
    if len(requested) != len(set(requested)) or not requested:
        raise ValueError("GPU IDs must be nonempty and unique")
    inventory = query()
    selected = {}
    for index in requested:
        if index not in inventory:
            raise ValueError(f"physical GPU {index} is absent from nvidia-smi")
        record = inventory[index]
        if record["name"] != EXPECTED_NAME or record["memory_total_mib"] != EXPECTED_MEMORY_MIB:
            raise ValueError(
                f"GPU {index} is {record['name']} ({record['memory_total_mib']} MiB); "
                f"the frozen protocol requires {EXPECTED_NAME} ({EXPECTED_MEMORY_MIB} MiB)"
            )
        selected[str(index)] = record
    print(json.dumps({"gpu_ids": requested, "gpus": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
