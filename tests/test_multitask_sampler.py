"""Tests for deterministic multi-task sampling and curriculum state."""

from types import SimpleNamespace

import pytest

from slime.utils.types import Sample
from slime_plugins.m2rl.data_source import MultiTaskRolloutDataSource, TaskSampler, _Source

NUM_GPUS = 0
SOURCES = [
    {"name": "math", "weight": 1, "phase_samples": 2},
    {"name": "science", "weight": 3, "phase_samples": 3},
]


@pytest.mark.unit
def test_exact_single_epoch_refuses_to_wrap_source():
    data_source = object.__new__(MultiTaskRolloutDataSource)
    data_source.args = SimpleNamespace(rollout_shuffle=False)
    data_source.sources = [_Source(config={"name": "math"}, dataset=[Sample(prompt="only")], offset=1)]
    data_source.strict_single_epoch = True

    with pytest.raises(RuntimeError, match="refusing to wrap and repeat prompts"):
        data_source._next_prompt(0)


@pytest.mark.unit
def test_round_robin_batch_sampling_is_task_homogeneous():
    sampler = TaskSampler(SOURCES, [10, 20], {"strategy": "round_robin", "unit": "batch"})

    assert sampler.select(4) == [0, 0, 0, 0]
    assert sampler.select(3) == [1, 1, 1]


@pytest.mark.unit
def test_sequential_curriculum_respects_phase_sizes():
    sampler = TaskSampler(SOURCES, [10, 20], {"strategy": "sequential", "repeat": False})

    assert sampler.select(5) == [0, 0, 1, 1, 1]
    with pytest.raises(StopIteration, match="exhausted"):
        sampler.select(1)


@pytest.mark.unit
def test_sequential_batch_schedule_never_mixes_boundary_batch():
    sampler = TaskSampler(
        SOURCES,
        [10, 20],
        {"strategy": "sequential", "unit": "batch", "repeat": False},
    )

    assert sampler.select(3) == [0, 0, 0]
    assert sampler.select(3) == [1, 1, 1]
    with pytest.raises(StopIteration, match="exhausted"):
        sampler.select(3)


@pytest.mark.unit
def test_weighted_sampler_resume_is_exact():
    sampling = {"strategy": "weighted", "seed": 123}
    sampler = TaskSampler(SOURCES, [10, 20], sampling)
    sampler.select(17)
    state = sampler.state_dict()
    expected = sampler.select(50)

    resumed = TaskSampler(SOURCES, [10, 20], sampling)
    resumed.load_state_dict(state)
    assert resumed.select(50) == expected


@pytest.mark.unit
def test_stratified_sampler_has_exact_batch_composition_and_resumes():
    sources = [{"name": "opd", "weight": 0.75}, {"name": "sft", "weight": 0.25}]
    sampling = {"strategy": "stratified", "unit": "prompt", "seed": 7}
    sampler = TaskSampler(sources, [10, 10], sampling)

    first = sampler.select(8)
    assert first.count(0) == 6
    assert first.count(1) == 2

    state = sampler.state_dict()
    expected = sampler.select(8)
    resumed = TaskSampler(sources, [10, 10], sampling)
    resumed.load_state_dict(state)
    assert resumed.select(8) == expected


@pytest.mark.unit
def test_stratified_sampler_rejects_batch_homogeneous_unit():
    sampler = TaskSampler(
        [{"name": "opd", "weight": 1}, {"name": "sft", "weight": 1}],
        [10, 10],
        {"strategy": "stratified", "unit": "batch"},
    )
    with pytest.raises(ValueError, match="unit: prompt"):
        sampler.select(4)


@pytest.mark.unit
def test_stratified_sampler_keeps_each_positive_component_in_small_batch():
    sampler = TaskSampler(
        [{"name": "opd", "weight": 0.99}, {"name": "sft", "weight": 0.01}],
        [10, 10],
        {"strategy": "stratified", "unit": "prompt", "seed": 1},
    )
    selected = sampler.select(4)
    assert selected.count(0) == 3
    assert selected.count(1) == 1
