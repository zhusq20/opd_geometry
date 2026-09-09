"""Tests for deterministic multi-task sampling and curriculum state."""

import json
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
def test_nonrepeating_mixed_source_refuses_to_wrap_source():
    data_source = object.__new__(MultiTaskRolloutDataSource)
    data_source.args = SimpleNamespace(rollout_shuffle=False)
    data_source.sources = [_Source(config={"name": "code"}, dataset=[Sample(prompt="only")], offset=1)]
    data_source.strict_single_epoch = False
    data_source.repeat_sources = False

    with pytest.raises(RuntimeError, match="Non-repeating multi-task source exhausted"):
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


@pytest.mark.unit
def test_four_task_rollout_batch_is_balanced_while_grpo_groups_share_one_prompt():
    names = ("code", "math", "qa", "if")
    data_source = object.__new__(MultiTaskRolloutDataSource)
    data_source.args = SimpleNamespace(n_samples_per_prompt=16, rollout_shuffle=False)
    data_source.sources = [
        _Source(
            config={"name": name},
            dataset=[Sample(prompt=f"{name}-{index}") for index in range(8)],
        )
        for name in names
    ]
    data_source.sampler = TaskSampler(
        [{"name": name, "weight": 1.0} for name in names],
        [8, 8, 8, 8],
        {"strategy": "stratified", "unit": "prompt", "seed": 42, "repeat": False},
    )
    data_source.repeat_sources = False
    data_source.strict_single_epoch = False
    data_source.sample_group_index = 0
    data_source.sample_index = 0

    groups = data_source.get_samples(16)

    assert len(groups) == 16
    assert {name: sum(group[0].source == name for group in groups) for name in names} == {name: 4 for name in names}
    assert all(len(group) == 16 for group in groups)
    assert all(len({sample.prompt for sample in group}) == 1 for group in groups)
    assert all(len({sample.group_index for sample in group}) == 1 for group in groups)


def _data_source_args(manifest):
    return SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=str(manifest),
        hf_checkpoint="unused",
        rollout_max_prompt_len=128,
        input_key="prompt",
        multimodal_keys=None,
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        rollout_shuffle=True,
        m2rl_task_sampling_seed=None,
        include_epoch_tail=False,
        num_epoch=None,
        num_rollout=1,
        n_samples_per_prompt=2,
    )


@pytest.mark.unit
def test_source_shuffle_seed_can_match_independent_single_task_streams(tmp_path, monkeypatch):
    manifest = tmp_path / "mixed.yaml"
    manifest.write_text(
        "sampling:\n  strategy: stratified\n  unit: prompt\n"
        "sources:\n"
        "- {name: code, path: code.jsonl, weight: 1, shuffle_seed: 42, required_samples: 8}\n"
        "- {name: math, path: math.jsonl, weight: 1, shuffle_seed: 42, required_samples: 8}\n"
    )
    seeds = []

    class FakeDataset:
        def __init__(self, path, *, seed, **kwargs):
            del path, kwargs
            self.seed = seed
            self.samples = list(range(8))
            seeds.append(seed)

        def __len__(self):
            return len(self.samples)

        def shuffle(self, epoch):
            del epoch

    monkeypatch.setattr("slime_plugins.m2rl.data_source.load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr("slime_plugins.m2rl.data_source.load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr("slime_plugins.m2rl.data_source.Dataset", FakeDataset)

    data_source = MultiTaskRolloutDataSource(_data_source_args(manifest))

    assert seeds == [42, 42]
    assert [source.config["shuffle_seed"] for source in data_source.sources] == [42, 42]


@pytest.mark.unit
def test_source_required_samples_checks_post_filter_length(tmp_path, monkeypatch):
    manifest = tmp_path / "mixed.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "code",
                        "path": "code.jsonl",
                        "required_samples": 9,
                    }
                ]
            }
        )
    )

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            self.samples = list(range(8))

        def __len__(self):
            return len(self.samples)

    monkeypatch.setattr("slime_plugins.m2rl.data_source.load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr("slime_plugins.m2rl.data_source.load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr("slime_plugins.m2rl.data_source.Dataset", FakeDataset)

    with pytest.raises(ValueError, match="requires 9 usable prompts, but only 8 remain"):
        MultiTaskRolloutDataSource(_data_source_args(manifest))


@pytest.mark.unit
def test_student_prompt_removal_and_prerendered_source_override_match(tmp_path, monkeypatch):
    suffix = "<think>\n\n</think>\n\n"
    student_prompt = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n"
    (tmp_path / "raw.jsonl").write_text(json.dumps({"prompt": "question", "label": "answer"}) + "\n")
    (tmp_path / "rendered.jsonl").write_text(json.dumps({"prompt": student_prompt, "label": "answer"}) + "\n")
    manifest = tmp_path / "mixed.json"
    manifest.write_text(
        json.dumps(
            {
                "sampling": {"strategy": "round_robin", "repeat": False},
                "sources": [
                    {"name": "raw", "path": "raw.jsonl"},
                    {
                        "name": "rendered",
                        "path": "rendered.jsonl",
                        "apply_chat_template": False,
                        "chat_template_suffix_to_remove": None,
                    },
                ],
            }
        )
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs["enable_thinking"] is False
            return student_prompt + suffix

    monkeypatch.setattr("slime_plugins.m2rl.data_source.load_tokenizer", lambda *_a, **_kw: Tokenizer())
    monkeypatch.setattr("slime_plugins.m2rl.data_source.load_processor", lambda *_a, **_kw: None)
    args = _data_source_args(manifest)
    args.rollout_max_prompt_len = None
    args.rollout_shuffle = False
    args.apply_chat_template = True
    args.apply_chat_template_kwargs = {"enable_thinking": False}
    args.chat_template_suffix_to_remove = suffix
    source = MultiTaskRolloutDataSource(args)
    assert [s.dataset.samples[0].prompt for s in source.sources] == [student_prompt, student_prompt]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
