#!/usr/bin/env python3
"""Compute exact realized update geometry for torch-dist model checkpoints.

The analysis streams one tensor chunk from each checkpoint at a time, so
training checkpoints may retain their optimizer state without forcing the
optimizer tensors into memory.  Only tensor entries present in the reference
model checkpoint are compared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import re
from collections import defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    iteration: int | None


@dataclass
class Contribution:
    parameter_count: int
    reference_square_sum: float
    update_square_sums: dict[str, float]
    update_dots: dict[tuple[str, str], float]
    changed_counts: dict[str, int]


@dataclass
class ScopeAccumulator:
    parameter_count: int = 0
    reference_square_sum: float = 0.0
    update_square_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    update_dots: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    changed_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, contribution: Contribution) -> None:
        self.parameter_count += contribution.parameter_count
        self.reference_square_sum += contribution.reference_square_sum
        for label, value in contribution.update_square_sums.items():
            self.update_square_sums[label] += value
        for pair, value in contribution.update_dots.items():
            self.update_dots[pair] += value
        for label, value in contribution.changed_counts.items():
            self.changed_counts[label] += value


class TorchDistTensorReader:
    """Read tensor chunks from a PyTorch distributed checkpoint."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        metadata_path = self.root / ".metadata"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing torch-dist metadata: {metadata_path}")
        with metadata_path.open("rb") as stream:
            # Checkpoints are trusted local experiment outputs in PyTorch's
            # native distributed-checkpoint pickle format.
            self.metadata = pickle.load(stream)  # noqa: S301
        self.tensor_names = tuple(
            name
            for name, metadata in self.metadata.state_dict_metadata.items()
            if hasattr(metadata, "size") and not name.startswith("optimizer.")
        )
        self.entries: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        for index, storage in self.metadata.storage_data.items():
            if index.offset is not None and index.fqn in self.tensor_names:
                self.entries[index.fqn].append((index, storage))
        for entries in self.entries.values():
            entries.sort(key=lambda item: tuple(int(value) for value in item[0].offset))
        self.stack = ExitStack()
        self.streams: dict[str, Any] = {}

    def __enter__(self) -> TorchDistTensorReader:
        return self

    def __exit__(self, *_: Any) -> None:
        self.stack.close()

    def shape(self, name: str) -> tuple[int, ...]:
        metadata = self.metadata.state_dict_metadata[name]
        return tuple(int(value) for value in metadata.size)

    def load_chunks(
        self,
        name: str,
        *,
        leading_index: int | None = None,
    ) -> dict[tuple[int, ...], torch.Tensor]:
        chunks: dict[tuple[int, ...], torch.Tensor] = {}
        for index, storage in self.entries.get(name, []):
            if leading_index is not None and (
                not index.offset or int(index.offset[0]) != leading_index
            ):
                continue
            descriptors = getattr(storage, "transform_descriptors", None)
            if descriptors:
                raise ValueError(f"Unsupported transformed checkpoint chunk for {name}: {descriptors}")
            relative_path = str(storage.relative_path)
            stream = self.streams.get(relative_path)
            if stream is None:
                stream = self.stack.enter_context((self.root / relative_path).open("rb"))
                self.streams[relative_path] = stream
            stream.seek(int(storage.offset))
            payload = stream.read(int(storage.length))
            if len(payload) != int(storage.length):
                raise OSError(
                    f"Short read for {name} from {relative_path}: "
                    f"expected {storage.length}, received {len(payload)} bytes."
                )
            tensor = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected a tensor payload for {name}, received {type(tensor)!r}.")
            offset = tuple(int(value) for value in index.offset)
            if offset in chunks:
                raise ValueError(f"Duplicate tensor chunk for {name} at offset {offset} in {self.root}.")
            chunks[offset] = tensor
        if not chunks:
            raise KeyError(f"Checkpoint {self.root} has no tensor chunks for {name}.")
        return chunks


def parse_checkpoint(value: str) -> CheckpointSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoints must use LABEL=PATH syntax.")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Checkpoints must use a non-empty LABEL=PATH pair.")
    path = Path(raw_path)
    match = re.fullmatch(r"iter_(\d+)", path.name)
    return CheckpointSpec(label, path, int(match.group(1)) if match else None)


def fp64_square_sum(value: torch.Tensor, *, block_elements: int = 4 * 1024 * 1024) -> float:
    flat = value.reshape(-1)
    total = 0.0
    for start in range(0, flat.numel(), block_elements):
        block = flat[start : start + block_elements].to(torch.float64)
        total += float(torch.dot(block, block))
    return total


def fp64_gram(
    vectors: Mapping[str, torch.Tensor],
    *,
    block_elements: int = 4 * 1024 * 1024,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    labels = list(vectors)
    if not labels:
        return {}, {}
    flattened = {label: vectors[label].reshape(-1) for label in labels}
    numel = flattened[labels[0]].numel()
    mismatched = {label: value.numel() for label, value in flattened.items() if value.numel() != numel}
    if mismatched:
        raise ValueError(f"Gram-matrix vector sizes differ from {numel}: {mismatched}.")
    squares = {label: 0.0 for label in labels}
    dots = {(left, right): 0.0 for left_index, left in enumerate(labels) for right in labels[left_index + 1 :]}
    for start in range(0, numel, block_elements):
        blocks = {label: value[start : start + block_elements].to(torch.float64) for label, value in flattened.items()}
        for label, block in blocks.items():
            squares[label] += float(torch.dot(block, block))
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                dots[(left, right)] += float(torch.dot(blocks[left], blocks[right]))
    return squares, dots


def contribution(reference: torch.Tensor, updates: dict[str, torch.Tensor]) -> Contribution:
    labels = tuple(updates)
    update_square_sums, update_dots = fp64_gram(updates)
    return Contribution(
        parameter_count=reference.numel(),
        reference_square_sum=fp64_square_sum(reference),
        update_square_sums=update_square_sums,
        update_dots=update_dots,
        changed_counts={label: int(torch.count_nonzero(updates[label])) for label in labels},
    )


def scope_row(scope: str, accumulator: ScopeAccumulator, labels: list[str]) -> dict[str, Any]:
    reference_norm = math.sqrt(accumulator.reference_square_sum)
    row: dict[str, Any] = {
        "scope": scope,
        "parameter_count": accumulator.parameter_count,
        "reference_norm": reference_norm,
    }
    for label in labels:
        update_norm = math.sqrt(accumulator.update_square_sums[label])
        row[f"update_norm/{label}"] = update_norm
        row[f"relative_update_norm/{label}"] = update_norm / reference_norm if reference_norm else None
        row[f"changed_count/{label}"] = accumulator.changed_counts[label]
        row[f"changed_fraction/{label}"] = (
            accumulator.changed_counts[label] / accumulator.parameter_count if accumulator.parameter_count else None
        )
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            dot = accumulator.update_dots[(left, right)]
            denominator = math.sqrt(accumulator.update_square_sums[left] * accumulator.update_square_sums[right])
            row[f"dot/{left}__{right}"] = dot
            row[f"cosine/{left}__{right}"] = dot / denominator if denominator else None
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_plots(output_dir: Path, rows: list[dict[str, Any]], labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global_row = next(row for row in rows if row["scope"] == "global")
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.bar(labels, [global_row[f"relative_update_norm/{label}"] for label in labels])
    axis.set_ylabel(r"$\|\Delta\theta\|_2 / \|\theta_0\|_2$")
    axis.set_title("Realized specialist update norms")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "global_relative_update_norms.png", dpi=200)
    plt.close(figure)

    cosine_matrix = np.eye(len(labels), dtype=np.float64)
    for left_index, left in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right = labels[right_index]
            value = global_row[f"cosine/{left}__{right}"]
            cosine_matrix[left_index, right_index] = value
            cosine_matrix[right_index, left_index] = value
    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    image = axis.imshow(cosine_matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_yticks(np.arange(len(labels)), labels)
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            axis.text(
                column_index,
                row_index,
                f"{cosine_matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis, label="cosine")
    axis.set_title("Specialist checkpoint-update cosine")
    figure.tight_layout()
    figure.savefig(output_dir / "global_update_cosines.png", dpi=200)
    plt.close(figure)

    layer_rows = sorted(
        (row for row in rows if str(row["scope"]).startswith("layer/")),
        key=lambda row: int(str(row["scope"]).split("/", 1)[1]),
    )
    if not layer_rows:
        return
    heatmap = np.asarray(
        [[row[f"relative_update_norm/{label}"] for label in labels] for row in layer_rows],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(6.8, 8.0))
    image = axis.imshow(np.log10(np.maximum(heatmap, np.finfo(np.float64).tiny)), aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_yticks(
        np.arange(len(layer_rows)),
        [str(row["scope"]).split("/", 1)[1] for row in layer_rows],
    )
    axis.set_xlabel("specialist")
    axis.set_ylabel("transformer layer")
    axis.set_title(r"Layer-wise $\log_{10}(\|\Delta\theta\|_2 / \|\theta_0\|_2)$")
    figure.colorbar(image, ax=axis, label="log10 relative update norm")
    figure.tight_layout()
    figure.savefig(output_dir / "layer_relative_update_norms.png", dpi=200)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    reference_path = args.reference.resolve()
    checkpoints = [CheckpointSpec(spec.label, spec.path.resolve(), spec.iteration) for spec in args.checkpoint]
    labels = [checkpoint.label for checkpoint in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Checkpoint labels must be unique: {labels}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to overwrite outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    accumulators: dict[str, ScopeAccumulator] = defaultdict(ScopeAccumulator)
    with TorchDistTensorReader(reference_path) as reference_reader, ExitStack() as stack:
        trained_readers = {
            checkpoint.label: stack.enter_context(TorchDistTensorReader(checkpoint.path)) for checkpoint in checkpoints
        }
        reference_names = set(reference_reader.tensor_names)
        for label, reader in trained_readers.items():
            missing = sorted(reference_names.difference(reader.tensor_names))
            if missing:
                raise ValueError(f"Checkpoint {label} is missing reference tensors: {missing}")
            changed_shapes = {
                name: (reference_reader.shape(name), reader.shape(name))
                for name in reference_names
                if reference_reader.shape(name) != reader.shape(name)
            }
            if changed_shapes:
                raise ValueError(f"Checkpoint {label} has changed tensor shapes: {changed_shapes}")

        for tensor_index, name in enumerate(reference_reader.tensor_names, start=1):
            print(f"[{tensor_index}/{len(reference_reader.tensor_names)}] {name}", flush=True)
            reference_chunks = reference_reader.load_chunks(name)
            trained_chunks = {label: trained_readers[label].load_chunks(name) for label in labels}
            expected_offsets = set(reference_chunks)
            for label in labels:
                actual_offsets = set(trained_chunks[label])
                if actual_offsets != expected_offsets:
                    raise ValueError(
                        f"Tensor layout changed for {name} in {label}: "
                        f"reference={sorted(expected_offsets)}, trained={sorted(actual_offsets)}"
                    )

            for offset, reference_chunk in reference_chunks.items():
                reference = reference_chunk.to(torch.float32)
                updates: dict[str, torch.Tensor] = {}
                for label in labels:
                    trained = trained_chunks[label][offset]
                    if trained.shape != reference_chunk.shape:
                        raise ValueError(
                            f"Chunk shape changed for {name} at {offset} in {label}: "
                            f"{tuple(reference_chunk.shape)} != {tuple(trained.shape)}"
                        )
                    updates[label] = trained.to(torch.float32).sub_(reference)

                whole = contribution(reference, updates)
                accumulators["global"].add(whole)
                accumulators[f"parameter/{name}"].add(whole)
                if name.startswith("decoder.layers."):
                    if not offset or reference.ndim == 0:
                        raise ValueError(f"Layer tensor {name} has no leading layer coordinate.")
                    for local_layer in range(reference.shape[0]):
                        layer = offset[0] + local_layer
                        layer_updates = {label: value[local_layer] for label, value in updates.items()}
                        accumulators[f"layer/{layer}"].add(contribution(reference[local_layer], layer_updates))
                del reference, updates
            del reference_chunks, trained_chunks

    ordered_scopes = ["global"]
    ordered_scopes.extend(
        sorted(
            (scope for scope in accumulators if scope.startswith("layer/")),
            key=lambda value: int(value.split("/", 1)[1]),
        )
    )
    ordered_scopes.extend(sorted(scope for scope in accumulators if scope.startswith("parameter/")))
    rows = [scope_row(scope, accumulators[scope], labels) for scope in ordered_scopes]
    write_csv(output_dir / "update_geometry.csv", rows)

    global_row = rows[0]
    cosine_rows = []
    for left in labels:
        row = {"checkpoint": left}
        for right in labels:
            if left == right:
                row[right] = 1.0
            else:
                pair = (left, right) if labels.index(left) < labels.index(right) else (right, left)
                row[right] = global_row[f"cosine/{pair[0]}__{pair[1]}"]
        cosine_rows.append(row)
    write_csv(output_dir / "global_cosine_matrix.csv", cosine_rows)

    summary = {
        "schema_version": 1,
        "method": (
            "Exact dot products over realized model tensors in the torch-dist checkpoints; "
            "optimizer and RNG entries are excluded."
        ),
        "reference": {
            "path": str(reference_path),
            "metadata_sha256": sha256(reference_path / ".metadata"),
            "common_sha256": sha256(reference_path / "common.pt"),
        },
        "checkpoints": [
            {
                "label": checkpoint.label,
                "path": str(checkpoint.path),
                "iteration": checkpoint.iteration,
                "metadata_sha256": sha256(checkpoint.path / ".metadata"),
                "common_sha256": sha256(checkpoint.path / "common.pt"),
            }
            for checkpoint in checkpoints
        ],
        "global": global_row,
        "layerwise": [row for row in rows if str(row["scope"]).startswith("layer/")],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.no_plots:
        save_plots(output_dir, rows, labels)
    print(f"Wrote checkpoint-update analysis to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        action="append",
        required=True,
        help="Trained checkpoint as LABEL=PATH; repeat for each specialist.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite known outputs in a non-empty directory.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
