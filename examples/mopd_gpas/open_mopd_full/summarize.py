#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAINS = {
    "math": ("aime24", "aime25"),
    "code": ("livecodebench_v5", "livecodebench_v6"),
    "if": ("ifeval", "ifbench_test"),
}


def summarize(root: Path) -> dict:
    datasets = {}
    for names in DOMAINS.values():
        for name in names:
            path = root / name / "scores.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing score file: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            datasets[name] = float(payload["pct"])
    domains = {
        domain: sum(datasets[name] for name in names) / len(names)
        for domain, names in DOMAINS.items()
    }
    return {
        "schema_version": 1,
        "datasets": datasets,
        "domains": domains,
        "total": sum(domains.values()) / len(domains),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the six-dataset Open-MOPD paper protocol.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
