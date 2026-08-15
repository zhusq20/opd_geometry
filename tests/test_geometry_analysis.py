"""Unit tests for cross-task geometry analysis helpers."""

import json
import runpy
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0
MODULE = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "optimizer_geometry" / "analyze_geometry.py"))


@pytest.mark.unit
def test_cosine_handles_parallel_orthogonal_and_zero_vectors():
    cosine = MODULE["cosine"]
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])) == pytest.approx(0.0)
    assert cosine(torch.zeros(2), torch.ones(2)) is None


@pytest.mark.unit
def test_homogeneous_task_rejects_mixed_batches():
    homogeneous_task = MODULE["homogeneous_task"]
    assert homogeneous_task({"math": 8}) == "math"
    assert homogeneous_task({"math": 4, "code": 4}) is None


def _write_geometry_run(path: Path, gradients: dict[str, torch.Tensor]) -> None:
    (path / "vectors").mkdir(parents=True)
    lines = []
    for observation_id, (task, gradient) in enumerate(gradients.items()):
        vector_name = f"vectors/{observation_id}.pt"
        torch.save(
            {"groups": {"global": {"gradient": gradient, "update": -gradient}}},
            path / vector_name,
        )
        lines.append(
            json.dumps(
                {
                    "optimizer": "adamw",
                    "advantage_estimator": "grpo",
                    "use_opd": False,
                    "role": "actor",
                    "rollout_id": observation_id,
                    "step_id": 0,
                    "observation_id": observation_id,
                    "source_counts": {task: 1},
                    "groups": {"global": {}},
                    "vector_file": vector_name,
                    "projection_dim": 2,
                    "projection_seed": 1,
                }
            )
        )
    (path / "metrics.jsonl").write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_analysis_never_mixes_projection_coordinates_across_runs(tmp_path):
    first = tmp_path / "seed1"
    second = tmp_path / "seed2"
    _write_geometry_run(first, {"math": torch.tensor([1.0, 0.0]), "code": torch.tensor([0.0, 1.0])})
    _write_geometry_run(second, {"math": torch.tensor([1.0, 0.0]), "code": torch.tensor([1.0, 0.0])})

    rows, matrices = MODULE["analyze"]([first, second])

    first_rows = [row for row in rows if row["observation_id"] == 0]
    assert len(first_rows) == 2
    assert all(row["cos_gradient_previous_gradient_sketch"] is None for row in first_rows)
    assert matrices[str(first.resolve())]["global"]["cos_math__code_sketch"] == pytest.approx(0.0)
    assert matrices[str(second.resolve())]["global"]["cos_math__code_sketch"] == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
