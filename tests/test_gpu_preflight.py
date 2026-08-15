"""CPU tests for fail-closed experiment GPU allocation checks."""

from __future__ import annotations

import pytest

from examples.optimizer_geometry.gpu_preflight import parse_devices, validate_gpus


NUM_GPUS = 0


def _gpu(index: int, *, total: int = 97_887, free: int = 97_251, volatile: int = 0, aggregate: int = 0):
    return {
        "index": index,
        "uuid": f"gpu-{index}",
        "name": "test",
        "memory_total_mib": total,
        "memory_used_mib": total - free,
        "memory_free_mib": free,
        "volatile_uncorrected_ecc": volatile,
        "aggregate_uncorrected_ecc": aggregate,
    }


@pytest.mark.unit
def test_gpu_preflight_accepts_idle_healthy_devices():
    selected, errors = validate_gpus(
        [1, 4],
        {1: _gpu(1), 4: _gpu(4)},
        min_free_mib=75_000,
        min_free_fraction=0.9,
    )

    assert [gpu["index"] for gpu in selected] == [1, 4]
    assert errors == []


@pytest.mark.unit
def test_gpu_preflight_rejects_shared_or_ecc_devices():
    _, errors = validate_gpus(
        [0, 2],
        {0: _gpu(0, volatile=5, aggregate=8), 2: _gpu(2, free=72_085)},
        min_free_mib=70_000,
        min_free_fraction=0.9,
    )

    assert any("uncorrected ECC" in error for error in errors)
    assert any("only 73.6% free" in error for error in errors)


@pytest.mark.unit
def test_gpu_preflight_rejects_duplicate_or_non_numeric_indices():
    with pytest.raises(ValueError, match="duplicates"):
        parse_devices("1,1")
    with pytest.raises(ValueError, match="non-numeric"):
        parse_devices("GPU-uuid")
