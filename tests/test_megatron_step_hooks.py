"""Static contract tests for train-step observation boundaries."""

import ast
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).parents[1]


def _call_line(function: ast.FunctionDef, name: str) -> int:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]
    assert len(calls) == 1, f"expected one {name} call, found {len(calls)}"
    return calls[0].lineno


def _attribute_call_line(function: ast.FunctionDef, owner: str, name: str) -> int:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    ]
    assert len(calls) == 1, f"expected one {owner}.{name} call, found {len(calls)}"
    return calls[0].lineno


@pytest.mark.unit
def test_geometry_hooks_bracket_optimizer_step():
    module = ast.parse((ROOT / "slime" / "backends" / "megatron_utils" / "model.py").read_text())
    train_step = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "train_one_step"
    )
    optimizer_steps = [
        node
        for node in ast.walk(train_step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "optimizer"
    ]
    assert len(optimizer_steps) == 1

    assert _call_line(train_step, "after_backward") < optimizer_steps[0].lineno
    assert optimizer_steps[0].lineno < _call_line(train_step, "after_optimizer_step")
    assert _call_line(train_step, "after_optimizer_step") < _attribute_call_line(
        train_step, "opt_param_scheduler", "step"
    )


@pytest.mark.unit
def test_generic_after_step_hooks_are_registered():
    source = (ROOT / "slime" / "utils" / "arguments.py").read_text()
    assert "--custom-megatron-after-backward-hook-path" in source
    assert "--custom-megatron-after-train-step-hook-path" in source


@pytest.mark.unit
def test_geometry_training_logs_peak_cuda_memory_without_charging_normal_training():
    source_path = ROOT / "slime" / "backends" / "megatron_utils" / "model.py"
    source = source_path.read_text()
    module = ast.parse(source)
    train = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "train")

    assert _call_line(train, "train_one_step") < _call_line(train, "_distributed_cuda_memory_mib")
    assert "torch.cuda.reset_peak_memory_stats" in source
    assert "gpu_peak_allocated_mib" in source
    assert "gpu_peak_reserved_mib" in source
    assert "if args.geometry_output_dir and torch.cuda.is_available()" in source
    assert "_distributed_cuda_memory_mib() if args.geometry_output_dir else {}" in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
