from types import SimpleNamespace

import pytest

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.types import Sample
from slime_plugins.mopd import eval as mopd_eval


def _payload(log_probs):
    return {"meta_info": {"input_token_logprobs": [[0.0, 0], *[[value, 0] for value in log_probs]]}}


def _sample(student, teacher, mask=None):
    return Sample(
        response="x",
        tokens=[1] * (len(student) + 1),
        response_length=len(student),
        rollout_log_probs=student,
        loss_mask=mask or [1] * len(student),
        reward=_payload(teacher),
        metadata={},
    )


def test_eval_reports_raw_relative_and_equal_task_mean(monkeypatch):
    samples = {
        "math": _sample([-1.0, -2.0], [-2.0, -99.0], [1, 0]),
        "code": _sample([-1.0], [-1.5]),
        "if": _sample([-2.0], [-3.0]),
        "science": _sample([-4.0], [-6.0]),
    }
    output = RolloutFnEvalOutput(
        data={task: {"samples": [sample], "rewards": [0.0]} for task, sample in samples.items()}
    )
    monkeypatch.setattr(mopd_eval, "default_generate_rollout", lambda *_args, **_kwargs: output)

    async def score(_args, task, task_samples, *, failure_penalty):
        assert failure_penalty == 10.0
        assert task_samples == [samples[task]]
        return {}

    async def activate(_args, task):
        assert task == "math"
        return {}

    monkeypatch.setattr(mopd_eval, "score_teacher_samples", score)
    monkeypatch.setattr(mopd_eval, "activate_teacher", activate)
    sources = [
        SimpleNamespace(config={"name": task, "relative_loss_scale": scale})
        for task, scale in zip(("math", "code", "if", "science"), (1.0, 2.0, 3.0, 4.0), strict=True)
    ]
    data_source = SimpleNamespace(sources=sources, controller=SimpleNamespace(resident_teacher=0))
    result = mopd_eval.generate_teacher_loss_eval(
        SimpleNamespace(mopd_failure_penalty=10.0), 7, data_source, evaluation=True
    )
    assert result.metrics["eval/teacher_loss/math"] == pytest.approx(1.0)
    assert result.metrics["eval/relative_teacher_loss/code"] == pytest.approx(1.0)
    assert result.metrics["eval/relative_teacher_loss/if"] == pytest.approx(3.0)
    assert result.metrics["eval/relative_teacher_loss/science"] == pytest.approx(8.0)
    assert result.metrics["eval/mean_relative_teacher_loss"] == pytest.approx(3.25)
    assert samples["math"].metadata["relative_teacher_loss"] == pytest.approx(1.0)


def test_eval_uses_numeric_penalty_for_empty_or_teacher_failure(monkeypatch):
    samples = {task: _sample([-1.0], [-1.0]) for task in ("math", "code", "if", "science")}
    samples["if"].metadata["mopd_failure_penalty"] = 10.0
    output = RolloutFnEvalOutput(
        data={task: {"samples": [sample], "rewards": [0.0]} for task, sample in samples.items()}
    )
    monkeypatch.setattr(mopd_eval, "default_generate_rollout", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(mopd_eval, "score_teacher_samples", lambda *_a, **_k: _async_value({}))
    monkeypatch.setattr(mopd_eval, "activate_teacher", lambda *_a, **_k: _async_value({}))
    sources = [SimpleNamespace(config={"name": task, "relative_loss_scale": 1.0}) for task in samples]
    result = mopd_eval.generate_teacher_loss_eval(
        SimpleNamespace(mopd_failure_penalty=10.0),
        0,
        SimpleNamespace(sources=sources, controller=SimpleNamespace(resident_teacher=0)),
        evaluation=True,
    )
    assert result.metrics["eval/teacher_loss/if"] == 10.0


async def _async_value(value):
    return value
