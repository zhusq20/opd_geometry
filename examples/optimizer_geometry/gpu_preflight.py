#!/usr/bin/env python3
"""Fail-closed GPU health and capacity check for experiment launchers."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO


QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.total",
)


def _positive_or_zero_int(value: str, *, field: str) -> int:
    cleaned = value.strip()
    if not cleaned.isdigit():
        raise ValueError(f"{field} is unavailable or non-numeric: {value!r}")
    return int(cleaned)


def parse_devices(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one GPU device is required")
    devices = [_positive_or_zero_int(value, field="device index") for value in values]
    if len(set(devices)) != len(devices):
        raise ValueError(f"GPU device list contains duplicates: {raw!r}")
    return devices


def query_gpus() -> dict[int, dict[str, object]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: dict[int, dict[str, object]] = {}
    for columns in csv.reader(StringIO(completed.stdout), skipinitialspace=True):
        if not columns:
            continue
        if len(columns) != len(QUERY_FIELDS):
            raise RuntimeError(
                f"nvidia-smi returned {len(columns)} columns; expected {len(QUERY_FIELDS)}: {columns!r}"
            )
        raw = dict(zip(QUERY_FIELDS, columns, strict=True))
        index = _positive_or_zero_int(str(raw["index"]), field="index")
        rows[index] = {
            "index": index,
            "uuid": str(raw["uuid"]).strip(),
            "name": str(raw["name"]).strip(),
            "memory_total_mib": _positive_or_zero_int(str(raw["memory.total"]), field="memory.total"),
            "memory_used_mib": _positive_or_zero_int(str(raw["memory.used"]), field="memory.used"),
            "memory_free_mib": _positive_or_zero_int(str(raw["memory.free"]), field="memory.free"),
            "volatile_uncorrected_ecc": _positive_or_zero_int(
                str(raw["ecc.errors.uncorrected.volatile.total"]),
                field="ecc.errors.uncorrected.volatile.total",
            ),
            "aggregate_uncorrected_ecc": _positive_or_zero_int(
                str(raw["ecc.errors.uncorrected.aggregate.total"]),
                field="ecc.errors.uncorrected.aggregate.total",
            ),
        }
    return rows


def validate_gpus(
    requested: list[int],
    inventory: dict[int, dict[str, object]],
    *,
    min_free_mib: int,
    min_free_fraction: float,
) -> tuple[list[dict[str, object]], list[str]]:
    selected: list[dict[str, object]] = []
    errors: list[str] = []
    for index in requested:
        gpu = inventory.get(index)
        if gpu is None:
            errors.append(f"GPU {index} is not present")
            continue
        selected.append(gpu)
        free_mib = int(gpu["memory_free_mib"])
        total_mib = int(gpu["memory_total_mib"])
        if free_mib < min_free_mib:
            errors.append(f"GPU {index} has {free_mib} MiB free; requires at least {min_free_mib} MiB")
        free_fraction = free_mib / total_mib if total_mib else 0.0
        if free_fraction < min_free_fraction:
            errors.append(
                f"GPU {index} is only {free_fraction:.1%} free; "
                f"requires at least {min_free_fraction:.1%}"
            )
        volatile = int(gpu["volatile_uncorrected_ecc"])
        aggregate = int(gpu["aggregate_uncorrected_ecc"])
        if volatile or aggregate:
            errors.append(
                f"GPU {index} has uncorrected ECC errors (volatile={volatile}, aggregate={aggregate})"
            )
    return selected, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="Comma-separated physical nvidia-smi GPU indices")
    parser.add_argument("--role", default="worker", help="Human-readable allocation role")
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument(
        "--min-free-fraction",
        type=float,
        default=0.0,
        help="Also require this fraction of total device memory to be free (0--1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_free_mib < 0:
        raise SystemExit("--min-free-mib must be non-negative")
    if not 0 <= args.min_free_fraction <= 1:
        raise SystemExit("--min-free-fraction must be between 0 and 1")
    try:
        requested = parse_devices(args.devices)
        inventory = query_gpus()
        selected, errors = validate_gpus(
            requested,
            inventory,
            min_free_mib=args.min_free_mib,
            min_free_fraction=args.min_free_fraction,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        selected = []
        errors = [f"GPU health query failed closed: {exc}"]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "role": args.role,
        "requested_devices": args.devices,
        "min_free_mib": args.min_free_mib,
        "min_free_fraction": args.min_free_fraction,
        "devices": selected,
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
