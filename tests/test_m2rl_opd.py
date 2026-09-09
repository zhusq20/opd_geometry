"""Unit tests for task-aware OPD routing and token alignment."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.m2rl import opd
from slime_plugins.m2rl.opd import _teacher_log_probs, teacher_route

NUM_GPUS = 0


@pytest.fixture
def suffix_tokenizers(monkeypatch):
    suffix = "<think>\n\n</think>\n\n"
    suffix_ids = [31, 32, 33, 34]

    def tokenizer():
        return SimpleNamespace(
            get_vocab=Mock(return_value={str(index): index for index in range(128)}),
            encode=Mock(side_effect=lambda text, **_kwargs: suffix_ids if text == suffix else pytest.fail(text)),
            decode=Mock(
                side_effect=lambda ids, **_kwargs: suffix if list(ids) == suffix_ids else pytest.fail(str(ids))
            ),
        )

    teacher, student = tokenizer(), tokenizer()
    load = Mock(side_effect=lambda path, **_kwargs: {"teacher": teacher, "student": student}[path])
    monkeypatch.setattr(opd, "load_tokenizer", load)
    monkeypatch.setattr(opd, "_opd_model_vocab_size", lambda _path: 160)
    opd._opd_tokenizer.cache_clear()
    opd._teacher_suffix_tokens.cache_clear()
    yield SimpleNamespace(suffix=suffix, ids=suffix_ids, teacher=teacher, student=student, load=load)
    opd._opd_tokenizer.cache_clear()
    opd._teacher_suffix_tokens.cache_clear()


@pytest.mark.unit
def test_teacher_routing_uses_task_then_default(tmp_path):
    path = tmp_path / "teachers.json"
    path.write_text(json.dumps({"teachers": {"math": "http://math/generate"}, "default": "http://base/generate"}))
    args = SimpleNamespace(opd_teacher_router_config=str(path), rm_url=None)

    assert teacher_route(args, Sample(metadata={"task_name": "math"}))["url"] == "http://math/generate"
    assert teacher_route(args, Sample(metadata={"task_name": "code"}))["url"] == "http://base/generate"


@pytest.mark.unit
def test_teacher_log_probs_are_aligned_to_response_tail():
    response = {"meta_info": {"input_token_logprobs": [[0.0, 0], [-1.0, 1], [-2.0, 2], [-3.0, 3]]}}
    assert _teacher_log_probs(response, 2).tolist() == [-2.0, -3.0]


@pytest.mark.unit
def test_teacher_request_scores_exact_tokens_without_generating_or_requesting_vocab_targets():
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {"meta_info": {"input_token_logprobs": [[0.0, 10]]}}

    class FakeSession:
        payload = None

        def post(self, _url, *, json, timeout):
            del timeout
            self.payload = json
            return FakeResponse()

    session = FakeSession()
    sample = Sample(tokens=[10, 11, 12])
    asyncio.run(
        opd._teacher_request(
            session,
            asyncio.Semaphore(1),
            {"url": "http://teacher/generate", "request_timeout": 1},
            sample,
        )
    )

    assert session.payload == {
        "input_ids": [10, 11, 12],
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }


@pytest.mark.unit
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("generated_id", [20, 159])
def test_teacher_suffix_keeps_response_ids_and_aligns_first_token_through_eos(suffix_tokenizers, dense, generated_id):
    from slime_plugins.mopd.loss import teacher_topk

    fixture = suffix_tokenizers
    # Thinking tokens inside the question or generated response are preserved.
    prompt = [10, *fixture.ids, 11]
    response_ids = [generated_id, *fixture.ids, 99]
    sample = Sample(tokens=prompt + response_ids, response_length=len(response_ids))
    original_tokens = sample.tokens

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            start = session.payload["logprob_start_len"]
            ids = session.payload["input_ids"]
            return {
                "meta_info": {
                    "input_token_logprobs": [[-float(index), ids[index]] for index in range(start, len(ids))],
                    "input_top_logprobs": [
                        [[-(index + token + 1) / 1000, token] for token in range(64)]
                        for index in range(start, len(ids))
                    ],
                }
            }

    class FakeSession:
        def post(self, _url, *, json, timeout):
            self.payload = json
            return FakeResponse()

    session = FakeSession()
    route = {
        "url": "http://teacher/generate",
        "model_path": "teacher",
        "prompt_suffix": fixture.suffix,
        "_student_model_path": "student",
    }
    if dense:
        route["top_logprobs_num"] = 64
    result = asyncio.run(opd._teacher_request(session, asyncio.Semaphore(1), route, sample))
    # Repeated batches reuse tokenizer loading and suffix encoding.
    asyncio.run(opd._teacher_request(session, asyncio.Semaphore(1), route, sample))

    teacher_prompt_length = len(prompt) + len(fixture.ids)
    assert session.payload["input_ids"] == prompt + fixture.ids + response_ids
    assert session.payload["sampling_params"]["max_new_tokens"] == 0
    assert sample.tokens is original_tokens
    assert sample.tokens == prompt + response_ids
    assert sample.response_length == len(response_ids)
    assert session.payload["logprob_start_len"] == (teacher_prompt_length - 1 if dense else 0)
    expected = [-float(index) for index in range(teacher_prompt_length, teacher_prompt_length + len(response_ids))]
    assert _teacher_log_probs(result, len(response_ids)).tolist() == expected
    if dense:
        ids, log_probs = teacher_topk(result, len(response_ids))
        assert ids.shape == log_probs.shape == (len(response_ids), 64)
        assert log_probs[:, 0].tolist() == pytest.approx(
            [-(index + 1) / 1000 for index in range(teacher_prompt_length, teacher_prompt_length + len(response_ids))]
        )
    assert fixture.load.call_count == 2
    fixture.teacher.encode.assert_called_once_with(fixture.suffix, add_special_tokens=False)
    fixture.student.encode.assert_called_once_with(fixture.suffix, add_special_tokens=False)
    fixture.teacher.decode.assert_called_once_with(
        tuple(fixture.ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


@pytest.mark.unit
@pytest.mark.parametrize("batch", [False, True])
def test_teacher_reward_validates_student_vocabulary_for_suffix_routes(monkeypatch, tmp_path, batch):
    path = tmp_path / "teachers.json"
    path.write_text(
        json.dumps(
            {
                "teachers": {
                    "math": {"url": "http://teacher/generate", "model_path": "teacher", "prompt_suffix": "suffix"}
                }
            }
        )
    )
    request = AsyncMock(return_value={"meta_info": {}})
    monkeypatch.setattr(opd, "_teacher_request", request)
    args = SimpleNamespace(opd_teacher_router_config=str(path), hf_checkpoint="student")
    sample = Sample(tokens=[10, 11], response_length=1, metadata={"task_name": "math"})

    asyncio.run(opd.teacher_reward(args, [sample, sample] if batch else sample))

    assert request.await_count == (2 if batch else 1)
    assert all(call.args[2]["_student_model_path"] == "student" for call in request.await_args_list)
    assert "_student_model_path" not in opd.load_teacher_router(str(path))["teachers"]["math"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "route_changes,tokens,response_length,message",
    [
        ({"prompt_suffix": None}, [10, 11], 1, "must be a string"),
        ({"model_path": None}, [10, 11], 1, "model_path tokenizer"),
        ({}, [10, 11], -1, "valid response_length"),
        ({}, [10, 11], 3, "valid response_length"),
        ({}, [10, 11], 2, "non-empty student prompt"),
        ({}, [10, 31, 32, 33, 34, 11], 1, "already ends"),
        ({}, [10, 999], 1, "outside the teacher model vocabulary"),
    ],
)
def test_teacher_suffix_rejects_invalid_boundaries_and_token_inputs(
    suffix_tokenizers, route_changes, tokens, response_length, message
):
    route = {"url": "http://teacher/generate", "model_path": "teacher", "prompt_suffix": suffix_tokenizers.suffix}
    route.update(route_changes)
    session = SimpleNamespace(post=Mock(side_effect=AssertionError("Invalid input reached the teacher")))
    with pytest.raises(ValueError, match=message):
        asyncio.run(
            opd._teacher_request(
                session, asyncio.Semaphore(1), route, Sample(tokens=tokens, response_length=response_length)
            )
        )
    session.post.assert_not_called()


def test_teacher_scoring_accepts_unnamed_model_vocabulary_rows(suffix_tokenizers):
    suffix_ids, accepted_ids = opd._teacher_suffix_tokens("teacher", suffix_tokenizers.suffix, "student")
    assert suffix_ids == tuple(suffix_tokenizers.ids)
    assert 159 not in suffix_tokenizers.teacher.get_vocab().values()
    assert 159 in accepted_ids
    assert -1 not in accepted_ids
    assert 160 not in accepted_ids


def test_teacher_scoring_rejects_different_model_vocabulary_dimensions(suffix_tokenizers, monkeypatch):
    monkeypatch.setattr(opd, "_opd_model_vocab_size", lambda path: 160 if path == "teacher" else 161)
    with pytest.raises(ValueError, match="same model vocabulary size"):
        opd._teacher_suffix_tokens("teacher", suffix_tokenizers.suffix, "student")


@pytest.mark.unit
@pytest.mark.parametrize("mismatch", ["vocabulary", "suffix_ids", "suffix_text"])
def test_teacher_suffix_rejects_incompatible_tokenizer_semantics(suffix_tokenizers, mismatch):
    fixture = suffix_tokenizers
    if mismatch == "vocabulary":
        fixture.student.get_vocab.return_value = {"0": 1, "1": 0}
        message = "identical token-to-ID"
    elif mismatch == "suffix_ids":
        fixture.student.encode.side_effect = lambda *_args, **_kwargs: [30, 31, 32]
        message = "same token IDs"
    else:
        fixture.teacher.decode.side_effect = lambda *_args, **_kwargs: fixture.suffix.rstrip()
        message = "encode and decode exactly"
    with pytest.raises(ValueError, match=message):
        opd._teacher_suffix_tokens("teacher", fixture.suffix, "student")


@pytest.mark.unit
@pytest.mark.parametrize(
    "error,phase,failures,attempts,succeeds",
    [
        (aiohttp.ServerDisconnectedError(), "enter", 1, 2, True),
        (aiohttp.ServerDisconnectedError(), "enter", 3, 3, False),
        (aiohttp.ClientPayloadError("incomplete response"), "json", 1, 2, True),
        (asyncio.TimeoutError(), "json", 1, 2, True),
        (
            aiohttp.ClientResponseError(SimpleNamespace(real_url="http://teacher/generate"), (), status=400),
            "status",
            1,
            1,
            False,
        ),
        (ValueError("invalid teacher JSON"), "json", 1, 1, False),
        (asyncio.CancelledError(), "json", 1, 1, False),
    ],
)
def test_teacher_request_retries_only_transient_failures(monkeypatch, error, phase, failures, attempts, succeeds):
    calls = []
    result = {"meta_info": {"input_token_logprobs": [[0.0, 10]]}}
    sleep = AsyncMock()
    monkeypatch.setattr(opd.asyncio, "sleep", sleep)

    class FakeResponse:
        def fail(self, stage):
            if phase == stage and len(calls) <= failures:
                raise error

        async def __aenter__(self):
            self.fail("enter")
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            self.fail("status")

        async def json(self):
            self.fail("json")
            return result

    class FakeSession:
        def post(self, url, *, json, timeout):
            calls.append((url, json, timeout.total))
            return FakeResponse()

    request = opd._teacher_request(
        FakeSession(),
        asyncio.Semaphore(1),
        {"url": "http://teacher/generate", "request_timeout": 1, "top_logprobs_num": 64},
        Sample(index=17, tokens=[10, 11, 12, 13, 14], response_length=2),
    )
    if succeeds:
        assert asyncio.run(request) == result
    else:
        with pytest.raises(type(error)) as caught:
            asyncio.run(request)
        assert caught.value is error

    assert len(calls) == attempts
    assert calls == [
        (
            "http://teacher/generate",
            {
                "input_ids": [10, 11, 12, 13, 14],
                "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
                "return_logprob": True,
                "logprob_start_len": 2,
                "top_logprobs_num": 64,
            },
            1,
        )
    ] * attempts
    assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1.0][: attempts - 1]


@pytest.mark.unit
def test_teacher_log_probs_reject_full_vocab_payloads():
    response = {
        "meta_info": {
            "input_token_logprobs": [
                [[0.0, -1.0], 0],
                [[-2.0, -3.0], 1],
                [[-4.0, -5.0], 2],
            ]
        }
    }

    with pytest.raises(ValueError, match="one scalar teacher log-probability"):
        _teacher_log_probs(response, 2)


@pytest.mark.unit
def test_opd_advantage_is_sampled_token_log_ratio_and_rejects_vocab_axis():
    from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages

    args = SimpleNamespace(opd_type="sglang", opd_kl_coef=1.0)
    advantages = [torch.zeros(2)]
    student = [torch.tensor([-1.0, -1.5])]
    teacher = [torch.tensor([-0.2, -2.0])]
    rollout_data = {"teacher_log_probs": teacher}

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student)

    # A_t = log p_T(a_t|h_t) - log p_S(a_t|h_t), for the sampled a_t only.
    torch.testing.assert_close(advantages[0], torch.tensor([0.8, -0.5]))
    torch.testing.assert_close(rollout_data["sampled_reverse_kl_logratio"][0], torch.tensor([-0.8, 0.5]))

    with pytest.raises(ValueError, match=r"Full-vocabulary \[tokens, vocab\]"):
        apply_opd_kl_to_advantages(
            args,
            {"teacher_log_probs": [torch.zeros(2, 8)]},
            [torch.zeros(2)],
            [torch.zeros(2)],
        )


@pytest.mark.unit
def test_combined_reward_accepts_batched_custom_rm_contract(monkeypatch):
    async def fake_gather(_args, sample, _kwargs):
        return {"id": sample.index}, float(sample.index)

    monkeypatch.setattr(opd, "_gather_teacher_and_task", fake_gather)
    result = asyncio.run(opd.combined_reward(SimpleNamespace(), [Sample(index=1), Sample(index=2)]))

    assert result == [
        {"teacher": {"id": 1}, "task_reward": 1.0},
        {"teacher": {"id": 2}, "task_reward": 2.0},
    ]


@pytest.mark.unit
def test_combined_reward_postprocess_preserves_native_task_reward(monkeypatch):
    monkeypatch.setattr(
        opd,
        "_teacher_log_probs",
        lambda _response, response_length: torch.zeros(response_length),
    )
    sample = Sample(
        response_length=2,
        reward={"teacher": {"scores": []}, "task_reward": 0.75},
        metadata={"task_name": "science"},
    )
    args = SimpleNamespace(
        reward_key=None,
        opd_task_reward_weight=0.2,
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        grpo_std_normalization=False,
    )

    weighted, normalized = opd.post_process_combined_rewards(args, [sample])

    assert weighted == pytest.approx([0.15])
    assert normalized == pytest.approx([0.15])
    assert sample.metadata["task_reward_observed"] == pytest.approx(0.75)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
