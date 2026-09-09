"""CPU unit tests for ``slime.utils.eval_config.build_eval_dataset_configs``.

The documented contract (examples/eval_multi_task/README.md) is that
``eval.defaults`` "defines inference parameters shared by every dataset entry.
Override them inside an individual dataset block if needed." — i.e. resolution
is dataset entry > defaults > args, for every ``EvalDatasetConfig`` field.

Historically only the fields listed in the two spec tables flowed through
``defaults``; everything else (``rm_type``, ``repetition_penalty``,
``app_service``, ...) was silently dropped, and a typo'd key in ``defaults``
was silently accepted while the same typo in a dataset entry raised. These
tests pin the full contract, including the ``stop`` / ``stop_token_ids`` /
``min_new_tokens`` fields that lost their resolution in the #1005 refactor.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from slime.utils.eval_config import build_eval_dataset_configs

NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        rollout_temperature=0.8,
        rollout_top_p=1.0,
        rollout_stop=["</train_stop>"],
        rollout_stop_token_ids=[7],
        eval_min_new_tokens=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_non_spec_defaults_reach_every_dataset():
    datasets = build_eval_dataset_configs(
        _args(),
        [{"name": "aime", "path": "/d/aime.jsonl"}, {"name": "gpqa", "path": "/d/gpqa.jsonl"}],
        defaults={"rm_type": "deepscaler", "repetition_penalty": 1.05, "eval_task_timeout": 120},
    )

    for dataset in datasets:
        assert dataset.rm_type == "deepscaler"
        assert dataset.repetition_penalty == 1.05
        assert dataset.eval_task_timeout == 120


@pytest.mark.unit
def test_dataset_entry_overrides_default():
    datasets = build_eval_dataset_configs(
        _args(),
        [
            {"name": "aime", "path": "/d/aime.jsonl", "rm_type": "math", "temperature": 0.2},
            {"name": "gpqa", "path": "/d/gpqa.jsonl"},
        ],
        defaults={"rm_type": "deepscaler", "temperature": 0.7},
    )

    assert datasets[0].rm_type == "math"
    assert datasets[0].temperature == 0.2
    assert datasets[1].rm_type == "deepscaler"
    assert datasets[1].temperature == 0.7


@pytest.mark.unit
def test_stop_fields_resolve_dataset_then_default_then_args():
    datasets = build_eval_dataset_configs(
        _args(eval_min_new_tokens=4),
        [
            {"name": "a", "path": "/d/a.jsonl", "stop": ["</answer>"], "min_new_tokens": 8},
            {"name": "b", "path": "/d/b.jsonl"},
            {"name": "c", "path": "/d/c.jsonl"},
        ],
        defaults={"stop_token_ids": [11, 12]},
    )

    # dataset entry wins
    assert datasets[0].stop == ["</answer>"]
    assert datasets[0].min_new_tokens == 8
    # eval.defaults fills in
    assert datasets[1].stop_token_ids == [11, 12]
    # args are the last fallback
    assert datasets[1].stop == ["</train_stop>"]
    assert datasets[2].stop_token_ids == [11, 12]
    assert datasets[2].min_new_tokens == 4


@pytest.mark.unit
def test_unknown_default_key_raises():
    with pytest.raises(ValueError, match="temperture"):
        build_eval_dataset_configs(
            _args(),
            [{"name": "aime", "path": "/d/aime.jsonl"}],
            defaults={"temperture": 0.7},
        )


@pytest.mark.unit
def test_spec_fields_still_fall_back_to_args():
    datasets = build_eval_dataset_configs(
        _args(rollout_temperature=0.9),
        [{"name": "aime", "path": "/d/aime.jsonl"}],
        defaults={},
    )

    assert datasets[0].temperature == 0.9


@pytest.mark.unit
def test_prompt_suffix_resolves_overrides_and_separates_eval_cache():
    suffix = "<think>\n\n</think>\n\n"
    datasets = build_eval_dataset_configs(
        _args(chat_template_suffix_to_remove="from args"),
        [
            {"name": "same", "path": "/d/math.jsonl"},
            {"name": "same", "path": "/d/math.jsonl", "chat_template_suffix_to_remove": None},
            {"name": "same", "path": "/d/math.jsonl", "chat_template_suffix_to_remove": "custom"},
        ],
        defaults={"chat_template_suffix_to_remove": suffix},
    )
    assert [cfg.chat_template_suffix_to_remove for cfg in datasets] == [suffix, None, "custom"]
    assert len({cfg.cache_key for cfg in datasets}) == 3
    fallback = build_eval_dataset_configs(
        _args(chat_template_suffix_to_remove=suffix), [{"name": "math", "path": "/d/math.jsonl"}], {}
    )
    assert fallback[0].chat_template_suffix_to_remove == suffix


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + (
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [None, "<think>\n\n</think>\n\n"])
def test_dataset_removes_only_final_control_prefix(tmp_path, suffix):
    from slime.utils.data import Dataset

    embedded = "<think>\n\n</think>\n\n"
    messages = [{"role": "user", "content": f"Explain the literal text {embedded} and solve 1+1."}]
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"prompt": messages}) + "\n")
    dataset = Dataset(
        str(path),
        _ChatTokenizer(),
        None,
        None,
        prompt_key="prompt",
        apply_chat_template=True,
        apply_chat_template_kwargs={"enable_thinking": False},
        chat_template_suffix_to_remove=suffix,
    )
    expected = "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
    assert dataset.samples[0].prompt == expected + ("" if suffix else embedded)
    assert embedded in dataset.samples[0].prompt  # Literal user content is never stripped.


@pytest.mark.unit
def test_dataset_rejects_wrong_suffix_and_prerendered_suffix_removal(tmp_path):
    from slime.utils.data import Dataset

    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"prompt": "question"}) + "\n")
    with pytest.raises(ValueError, match="does not end"):
        Dataset(
            str(path),
            _ChatTokenizer(),
            None,
            None,
            prompt_key="prompt",
            apply_chat_template=True,
            apply_chat_template_kwargs={"enable_thinking": False},
            chat_template_suffix_to_remove="wrong",
        )
    with pytest.raises(ValueError, match="requires apply_chat_template"):
        Dataset(str(path), None, None, None, prompt_key="prompt", chat_template_suffix_to_remove="suffix")


@pytest.mark.unit
def test_eval_generation_honors_student_removal_and_teacher_null_override(tmp_path, monkeypatch):
    from slime.rollout import sglang_rollout

    suffix = "<think>\n\n</think>\n\n"
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"prompt": "question", "label": "answer"}) + "\n")
    args = _args(
        hf_checkpoint="same-model",
        group_rm=False,
        multimodal_keys=None,
        apply_chat_template=True,
        apply_chat_template_kwargs={"enable_thinking": False},
        chat_template_suffix_to_remove=suffix,
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        eval_max_prompt_len=None,
        rollout_skip_special_tokens=False,
        eval_reward_key=None,
        reward_key=None,
        n_samples_per_prompt=1,
    )
    datasets = build_eval_dataset_configs(
        args,
        [
            {"name": "math", "path": str(path)},
            {"name": "math", "path": str(path), "chat_template_suffix_to_remove": None},
        ],
        {},
    )
    monkeypatch.setattr(sglang_rollout, "EVAL_PROMPT_DATASET", {})
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *_a, **_kw: _ChatTokenizer())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *_a, **_kw: None)

    async def generate(_args, sample, **_kwargs):
        sample.response, sample.reward = "answer", 1.0
        return sample

    monkeypatch.setattr(sglang_rollout, "_generate_eval_sample", generate)
    results = [asyncio.run(sglang_rollout.eval_rollout_single_dataset(args, 0, cfg)) for cfg in datasets]
    student = results[0]["math"]["samples"][0].prompt
    teacher = results[1]["math"]["samples"][0].prompt
    assert student.endswith("<|im_start|>assistant\n")
    assert teacher == student + suffix
    assert len(sglang_rollout.EVAL_PROMPT_DATASET) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
