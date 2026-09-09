import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.mopd import loss as mopd_loss, topk
from slime_plugins.mopd.paper_diagnostics import local_distillation_loss

NUM_GPUS = 0


@pytest.mark.parametrize("k", [16, 64])
def test_normalized_topk_gradient_matches_mass_scaled_corrected_kl(k):
    torch.manual_seed(231)
    logits = torch.randn(3, 2 * k, requires_grad=True)
    teacher = torch.randn(3, 2 * k).log_softmax(-1).requires_grad_()
    ids = logits.detach().topk(k, dim=-1).indices
    log_p = logits.log_softmax(-1).gather(-1, ids)
    log_q = teacher.gather(-1, ids)
    advantage = mopd_loss.student_topk_advantage(log_p, log_q)
    assert not advantage.requires_grad
    position_weights = torch.tensor([0.1, 0.3, 0.6])
    objective = (-(advantage * log_p).sum(-1) * position_weights).sum()
    gradient, teacher_gradient = torch.autograd.grad(objective, (logits, teacher), retain_graph=True, allow_unused=True)
    mass = log_p.detach().exp().sum(-1)
    reference = (mopd_loss.corrected_reverse_kl(log_p, log_q) / mass * position_weights).sum()
    expected = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(gradient, expected)
    assert teacher_gradient is None
    # The full softmax denominator must still update tokens outside the support.
    outside = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, ids, False)
    assert gradient[outside].abs().sum() > 0


def test_student_selection_differs_from_teacher_and_full_support_matches_full_kl():
    logits = torch.tensor([[3., 2., 1., 0.]], requires_grad=True)
    teacher = torch.tensor([[0.01, 0.09, 0.3, 0.6]]).log()
    # Student chooses token 0, while the teacher would choose token 3.
    objective = local_distillation_loss(logits, teacher, loss="student_topk", topk=1, weights=torch.ones(1))
    log_p = logits.log_softmax(-1)
    expected = (log_p[:, 0] - teacher[:, 0]).detach() * log_p[:, 0]
    torch.testing.assert_close(objective, expected.sum())
    full_support = local_distillation_loss(logits, teacher, loss="student_topk", weights=torch.ones(1))
    actual = torch.autograd.grad(full_support, logits)[0]
    full_kl = (logits.softmax(-1) * (logits.log_softmax(-1) - teacher)).sum()
    torch.testing.assert_close(actual, torch.autograd.grad(full_kl, logits)[0])


def test_equal_teacher_has_zero_gradient_and_tiny_retained_mass_is_stable():
    logits = torch.linspace(-1, 1, 32).unsqueeze(0).requires_grad_()
    log_p = logits.log_softmax(-1)[:, :16]
    advantage = mopd_loss.student_topk_advantage(log_p, log_p.detach())
    (-(advantage * log_p).sum()).backward()
    assert torch.count_nonzero(logits.grad) == 0
    # A saved rollout support can have tiny current mass after an actor refresh.
    tiny = torch.arange(16, dtype=torch.float32).unsqueeze(0) - 1000
    assert tiny.exp().sum() == 0
    advantage = mopd_loss.student_topk_advantage(tiny, tiny + 2)
    assert torch.isfinite(advantage).all()
    torch.testing.assert_close(advantage.sum(-1), torch.tensor([2.]))


@pytest.mark.parametrize("k", [16, 64])
def test_online_student_topk_refreshes_advantage_keeps_masks_and_reduction(monkeypatch, k):
    from slime.backends.megatron_utils import loss as megatron_loss

    torch.manual_seed(23)
    vocab_size = k + 8
    logits = torch.randn(3, vocab_size, requires_grad=True)
    # Freeze rollout-selected IDs; current student ranking is deliberately different.
    ids = torch.arange(k).repeat(3, 1)
    teacher = torch.randn(3, vocab_size).log_softmax(-1).gather(-1, ids)
    metadata = {"student_topk_k": k, "student_topk_ids": ids, "student_topk_log_probs": torch.full((3, k), -999.),
                "teacher_on_student_topk_log_probs": teacher}
    monkeypatch.setattr(megatron_loss, "get_responses", lambda *a, **kw: iter([(logits, None)]))
    monkeypatch.setattr(mopd_loss, "selected_log_probs", lambda values, chosen, **kw: values.log_softmax(-1).gather(-1, chosen))
    batch = {"unconcat_tokens": [], "total_lengths": [], "response_lengths": [3],
             "metadata": [metadata], "loss_masks": [[1, 0, 1]]}
    weights = torch.tensor([0.25, 0., 0.75])
    objective, metrics = mopd_loss.paper_loss(
        SimpleNamespace(mopd_loss="student_topk", mopd_topk=k, vocab_size=vocab_size, mopd_pg_advantage_clip=0.001),
        batch, logits, lambda positions: (positions * weights).sum(),
    )
    objective.backward()
    p = logits.detach().softmax(-1)
    lp = logits.detach().log_softmax(-1).gather(-1, ids)
    normalized_p = p.gather(-1, ids) / p.gather(-1, ids).sum(-1, keepdim=True)
    # Independent score-function gradient, with no PPO ratio or advantage clip.
    coefficients = normalized_p * (lp - teacher)
    expected = -p * coefficients.sum(-1, keepdim=True)
    expected.scatter_add_(-1, ids, coefficients)
    torch.testing.assert_close(logits.grad, expected * weights[:, None])
    assert logits.grad[1].abs().sum() == 0
    expected_ratio = coefficients.sum(-1)
    assert metadata["mopd_teacher_loss"] == pytest.approx(float(expected_ratio[[0, 2]].mean()))
    torch.testing.assert_close(metrics[f"student_top{k}_normalized_logratio"], (expected_ratio * weights).sum())
    torch.testing.assert_close(metrics[f"student_top{k}_retained_mass"], (p.gather(-1, ids).sum(-1) * weights).sum())


@pytest.mark.parametrize("loss,enabled,k,requested", [("student_topk", True, 16, 16), ("student_topk", True, 64, 64),
    ("topk_intersection", True, 16, 16), ("topk_intersection", True, 64, 64),
    ("student_topk", False, 16, None), ("teacher_topk", True, 16, None), ("sampled_reverse_kl", True, 16, None)])
def test_generation_collects_only_student_topk_in_response_order(monkeypatch, loss, enabled, k, requested):
    from slime.rollout import sglang_rollout

    payloads = []
    rows = [[[-1 - token / 100, token + k * position] for token in range(k)] for position in range(2)]
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: SimpleNamespace(tokenizer=None, processor=None))
    monkeypatch.setattr(sglang_rollout, "trace_span", lambda *a, **kw: nullcontext(SimpleNamespace(update=lambda value: None)))

    async def post(url, payload, headers=None):
        payloads.append(payload)
        return {"text": "response", "meta_info": {"finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-2., 17], [-3., 99]], "output_top_logprobs": rows}}

    monkeypatch.setattr(sglang_rollout, "post", post)
    sample = Sample(tokens=[2, 3])
    args = SimpleNamespace(ci_test=False, sglang_router_ip="localhost", sglang_router_port=1,
                           use_rollout_routing_replay=False, mopd_enabled=enabled, mopd_loss=loss, mopd_topk=k)
    asyncio.run(sglang_rollout.generate(args, sample, {"max_new_tokens": 2}))
    assert payloads[0].get("top_logprobs_num") == requested
    assert sample.tokens == [2, 3, 17, 99]
    if requested:
        assert sample.train_metadata["student_topk_k"] == k
        assert sample.train_metadata["student_topk_ids"] == [list(range(k)), list(range(k, 2 * k))]
        assert sample.train_metadata["student_topk_log_probs"][0] == pytest.approx([row[0] for row in rows[0]])
    else:
        assert not sample.train_metadata


@pytest.mark.parametrize("suffix", [[], [100, 101]])
@pytest.mark.parametrize("k", [16, 64])
def test_teacher_scores_chunked_student_ids_with_exact_prefixes_and_eos(monkeypatch, suffix, k):
    from slime_plugins.m2rl import opd

    chunk_size = topk.teacher_score_chunk_size(k)
    length = chunk_size + 3
    response = list(range(10, 10 + length - 1)) + [99]
    ids = [list(range(position, position + k)) for position in range(length)]
    sample = Sample(tokens=[2, 3, *response], response_length=length,
                    train_metadata={"student_topk_k": k, "student_topk_ids": ids})
    original = sample.tokens.copy()
    payloads = []
    monkeypatch.setattr(opd, "_teacher_suffix_tokens", lambda *args: (tuple(suffix), range(200)))

    class Response:
        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            payload = self.payload
            positions = range(payload["logprob_start_len"], len(payload["input_ids"]))
            return {"meta_info": {
                "input_token_logprobs": [[-float(position + 1), payload["input_ids"][position]] for position in positions],
                "input_token_ids_logprobs": [
                    [[-float(position + 1) - token / 1000, token, None] for token in reversed(payload["token_ids_logprob"])]
                    for position in positions],
            }}

    class Session:
        def post(self, url, *, json, timeout):
            payloads.append(json)
            return Response(json)

    route = {"url": "http://teacher/generate", "student_topk": True, "student_topk_k": k, "model_path": "teacher",
             "prompt_suffix": "suffix" if suffix else ""}
    sample.reward = asyncio.run(opd._teacher_request(Session(), asyncio.Semaphore(1), route, sample))
    assert len(payloads) == 2
    prompt_length = 2 + len(suffix)
    for start, payload in zip(range(0, length, chunk_size), payloads, strict=True):
        end = min(start + chunk_size, length)
        assert payload["input_ids"] == [2, 3, *suffix, *response[:end]]
        assert payload["logprob_start_len"] == prompt_length + start - 1
        assert payload["token_ids_logprob"] == sorted({token for row in ids[start:end] for token in row})
        assert len(payload["token_ids_logprob"]) <= topk.TEACHER_SCORE_MAX_IDS
        assert payload["sampling_params"]["max_new_tokens"] == 0
        assert "top_logprobs_num" not in payload
    assert sample.tokens == original
    assert sample.reward["meta_info"]["input_token_logprobs"][-1][1] == 99
    mopd_loss.post_process_rewards(SimpleNamespace(mopd_loss="student_topk", mopd_topk=k), [sample])
    torch.testing.assert_close(sample.train_metadata["student_topk_ids"], torch.tensor(ids))
    expected = torch.tensor([[-float(prompt_length + position + 1) - token / 1000 for token in row]
                             for position, row in enumerate(ids)])
    torch.testing.assert_close(sample.train_metadata["teacher_on_student_topk_log_probs"], expected)


def test_top64_configuration_rejects_top16_rollout_support():
    sample = Sample(tokens=[1, 2], response_length=1,
                    train_metadata={"student_topk_k": 16, "student_topk_ids": [list(range(16))]})
    with pytest.raises(ValueError, match="Student Top64 IDs"):
        topk.student_support(sample, k=64)


@pytest.mark.parametrize("defect", ["absent", "wrong_k", "duplicates", "nonfinite", "wrong_order"])
def test_student_topk_payload_rejects_missing_or_misaligned_targets(defect):
    ids = list(range(16))
    rows = [[[-1., token] for token in ids]]
    sample = Sample(tokens=[1, 2], response_length=1, train_metadata={"student_topk_ids": [ids]},
                    reward={"meta_info": {"input_token_ids_logprobs": rows}})
    if defect == "absent":
        sample.reward = {"meta_info": {"input_top_logprobs": rows}}
    elif defect == "wrong_k":
        rows[0].pop()
    elif defect == "duplicates":
        rows[0][-1][1] = 0
    elif defect == "nonfinite":
        rows[0][0][0] = float("nan")
    else:
        rows[0].reverse()
    with pytest.raises(ValueError):
        mopd_loss.post_process_rewards(SimpleNamespace(mopd_loss="student_topk"), [sample])


def test_teacher_selected_scores_reject_response_shift_and_missing_id():
    rows = [[[-1., token] for token in range(16)]]
    result = {"meta_info": {"input_token_ids_logprobs": rows, "input_token_logprobs": [[-1., 99]]}}
    with pytest.raises(ValueError, match="response token IDs"):
        topk.select_teacher_rows(result, [list(range(16))], [98])
    with pytest.raises(ValueError, match="missing student IDs"):
        topk.select_teacher_rows(result, [list(range(1, 17))], [99])


@pytest.mark.parametrize("boundary_position", [0, 1, 2, 3])
def test_teacher_transport_restores_last_valid_vocab_id_only(monkeypatch, boundary_position):
    from slime_plugins.m2rl import opd

    monkeypatch.setattr(opd, "_opd_model_vocab_size", lambda path: 128)
    tokens = [2, 3, 4, 99]
    tokens[boundary_position] = 127
    top_rows = [None, *[[[-1. - i / 100, i] for i in range(16)] for _ in range(3)]]
    sampled = [[None if i == 0 else -float(i), 0 if token == 127 else token, None]
               for i, token in enumerate(tokens)]
    original = {"meta_info": {"input_token_logprobs": sampled, "input_top_logprobs": top_rows}}
    payload = {"input_ids": [1, *tokens], "logprob_start_len": 1}

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            return original

    class Session:
        def post(self, *args, **kwargs):
            assert kwargs["json"] is payload
            return Response()

    result = asyncio.run(opd._post_teacher_request(
        Session(), asyncio.Semaphore(1), {"url": "http://teacher/generate", "model_path": "teacher"},
        Sample(), payload, None,
    ))
    assert [row[1] for row in result["meta_info"]["input_token_logprobs"]] == tokens
    assert [row[0] for row in result["meta_info"]["input_token_logprobs"]] == [row[0] for row in sampled]
    assert result["meta_info"]["input_top_logprobs"] is top_rows
    assert sampled[boundary_position][1] == 0
    assert payload["input_ids"] == [1, *tokens]
    sample = Sample(tokens=payload["input_ids"], response_length=3, reward=result,
                    train_metadata={"student_topk_ids": [list(range(16))] * 3})
    topk.teacher_student_topk_intersection(sample)


@pytest.mark.parametrize("defect", ["shifted", "missing_row", "wrong_id", "not_boundary", "mixed", "fixed_server"])
def test_vocab_boundary_compatibility_keeps_alignment_checks(monkeypatch, defect):
    from slime_plugins.m2rl import opd

    monkeypatch.setattr(opd, "_opd_model_vocab_size", lambda path: 128)
    tokens = [3, 127, 5, 99]
    rows = [[None, 3], [-1., 0], [-2., 5], [-3., 99]]
    if defect == "shifted":
        rows = rows[1:] + [rows[0]]
    elif defect == "missing_row":
        rows.pop(0)
    elif defect == "wrong_id":
        rows[1][1] = 126
    elif defect == "not_boundary":
        tokens[1] = 126
    elif defect == "mixed":
        rows[2][1] = 6
    else:
        rows[1][1] = 127
    original = {"meta_info": {"input_token_logprobs": rows}}
    result = opd._restore_teacher_vocab_boundary_id(
        original, {"input_ids": tokens, "logprob_start_len": 0}, "teacher"
    )
    assert result is original
    if defect != "fixed_server":
        sample = Sample(tokens=tokens, response_length=3, reward=result,
                        train_metadata={"student_topk_ids": [list(range(16))] * 3})
        with pytest.raises(ValueError, match="response token IDs"):
            topk.teacher_student_topk_intersection(sample)


def test_student_support_append_requires_existing_position_alignment():
    row = [[-1., token] for token in range(16)]
    sample = Sample(response_length=2)
    with pytest.raises(ValueError, match="previously generated"):
        topk.append_student_topk(sample, {"output_top_logprobs": [row]}, 1)
    sample.train_metadata = {"student_topk_ids": [list(range(16))], "student_topk_log_probs": [[-1.] * 16]}
    topk.append_student_topk(sample, {"output_top_logprobs": [row]}, 1)
    assert topk.student_support(sample).shape == (2, 16)


@pytest.mark.parametrize("loss", ["student_topk", "topk_intersection"])
def test_student_scoring_failure_aborts_without_a_sampled_scalar_penalty(monkeypatch, loss):
    from slime_plugins.mopd import teacher_slot

    monkeypatch.setattr(teacher_slot, "_slot_config", lambda args: ({}, {}))
    monkeypatch.setattr(teacher_slot, "teacher_memory_mib", lambda *args: None)

    async def failed(*args, **kwargs):
        return [ValueError("missing student IDs")]

    monkeypatch.setattr(teacher_slot, "teacher_reward", failed)
    sample = Sample(tokens=[1, 2], response_length=1)
    with pytest.raises(RuntimeError, match=f"{loss} teacher scoring failed"):
        asyncio.run(teacher_slot.score_teacher_samples(SimpleNamespace(mopd_loss=loss), "math", [sample],
                                                       failure_penalty=10.))
    assert sample.reward is None


@pytest.mark.parametrize("k", [16, 64])
def test_intersection_aligns_native_teacher_rows_and_allows_empty_support(k):
    student_ids = torch.arange(0, 2 * k, 2).repeat(3, 1)
    teacher_ids = [list(reversed(range(k))), list(range(2 * k, 3 * k)), student_ids[2].flip(0).tolist()]
    rows = [[[-1. - token / 1000, token, None] for token in row] for row in teacher_ids]
    sample = Sample(tokens=[2, 3, 4, 5, 99], response_length=3,
                    train_metadata={"student_topk_k": k, "student_topk_ids": student_ids},
                    reward={"meta_info": {"input_top_logprobs": [None, *rows],
                            "input_token_logprobs": [[None, 3], [-2., 4], [-3., 5], [-4., 99]]}})
    mopd_loss.post_process_rewards(SimpleNamespace(mopd_loss="topk_intersection", mopd_topk=k), [sample])
    data = sample.train_metadata
    shared = data["student_teacher_topk_mask"]
    assert shared.dtype == torch.bool
    assert shared.sum(-1).tolist() == [k // 2, 0, k]
    expected = torch.tensor([[-1. - int(token) / 1000 if int(token) in teacher_ids[i] else 0.
                              for token in row] for i, row in enumerate(student_ids)])
    torch.testing.assert_close(data["teacher_on_student_topk_log_probs"], expected)
    torch.testing.assert_close(data["student_topk_ids"], student_ids)
    sample.reward["meta_info"]["input_token_logprobs"][-1][1] = 98
    with pytest.raises(ValueError, match="response token IDs"):
        topk.teacher_student_topk_intersection(sample, k=k)


@pytest.mark.parametrize("k", [16, 64])
def test_intersection_gradient_keeps_original_student_normalization_and_reduction(monkeypatch, k):
    from slime.backends.megatron_utils import loss as megatron_loss

    torch.manual_seed(34)
    logits = torch.randn(4, 3 * k, requires_grad=True)
    ids = torch.arange(k).repeat(4, 1)
    teacher = torch.randn_like(logits).log_softmax(-1).gather(-1, ids)
    shared = torch.zeros_like(ids, dtype=torch.bool)
    shared[0, ::2] = True
    shared[2] = True
    shared[3, ::3] = True
    teacher = teacher.masked_fill(~shared, 0.)
    metadata = {"student_topk_k": k, "student_topk_ids": ids, "teacher_on_student_topk_log_probs": teacher,
                "student_teacher_topk_mask": shared}
    monkeypatch.setattr(megatron_loss, "get_responses", lambda *a, **kw: iter([(logits, None)]))
    monkeypatch.setattr(mopd_loss, "selected_log_probs", lambda values, chosen, **kw: values.log_softmax(-1).gather(-1, chosen))
    batch = {"unconcat_tokens": [], "total_lengths": [], "response_lengths": [4],
             "metadata": [metadata], "loss_masks": [[1, 1, 1, 0]]}
    weights = torch.tensor([0.2, 0.3, 0.5, 0.])
    objective, metrics = mopd_loss.paper_loss(
        SimpleNamespace(mopd_loss="topk_intersection", mopd_topk=k, vocab_size=3 * k),
        batch, logits, lambda positions: (positions * weights).sum(),
    )
    objective.backward()
    p = logits.detach().softmax(-1)
    selected_p = p.gather(-1, ids)
    coefficients = selected_p / selected_p.sum(-1, keepdim=True) * (selected_p.log() - teacher) * shared
    expected = -p * coefficients.sum(-1, keepdim=True)
    expected.scatter_add_(-1, ids, coefficients)
    torch.testing.assert_close(logits.grad, expected * weights[:, None])
    assert torch.count_nonzero(logits.grad[[1, 3]]) == 0
    prefix = f"student_teacher_top{k}_intersection"
    assert metrics[f"{prefix}_empty_fraction"] == pytest.approx(0.3)
    torch.testing.assert_close(metrics[f"{prefix}_retained_weight"],
                               ((selected_p * shared).sum(-1) / selected_p.sum(-1) * weights).sum())
    torch.testing.assert_close(metrics[f"teacher_on_student_top{k}_intersection_mass"],
                               ((teacher.exp() * shared).sum(-1) * weights).sum())
    assert metadata["mopd_teacher_loss"] == pytest.approx(float(coefficients.sum(-1)[:3].mean()))
    metadata["student_teacher_topk_mask"] = shared.float()
    with pytest.raises(ValueError, match="mask must be boolean"):
        mopd_loss.paper_loss(SimpleNamespace(mopd_loss="topk_intersection", mopd_topk=k, vocab_size=3 * k),
                             batch, logits, lambda x: x.sum())


def test_local_intersection_drops_missing_teacher_ids_without_renormalization():
    logits = torch.tensor([[4., 3., 2., 1.]], requires_grad=True)
    teacher = torch.tensor([[0.03, 0.8, 0.07, 0.1]]).log()
    objective = local_distillation_loss(logits, teacher, loss="topk_intersection", topk=2, weights=torch.ones(1))
    log_p = logits.log_softmax(-1)
    # Student {0, 1} and teacher {1, 3} share only token 1, which keeps its original weight.
    expected = (log_p[:, :2].softmax(-1)[:, 1] * (log_p[:, 1] - teacher[:, 1])).detach() * log_p[:, 1]
    torch.testing.assert_close(objective, expected.sum())
    torch.testing.assert_close(torch.autograd.grad(objective, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), logits)[0])


@pytest.mark.parametrize("suffix", [[], [100, 101]])
def test_native_top64_scores_entire_response_in_one_request_with_eos(monkeypatch, suffix):
    from slime_plugins.m2rl import opd

    payloads = []
    response = list(range(10, 110)) + [99]
    sample = Sample(tokens=[2, 3, *response], response_length=len(response))
    monkeypatch.setattr(opd, "_teacher_suffix_tokens", lambda *args: (tuple(suffix), range(200)))

    async def post(session, semaphore, route, item, payload, timeout):
        payloads.append(payload)
        return {"meta_info": {}}

    monkeypatch.setattr(opd, "_post_teacher_request", post)
    route = {"url": "http://teacher/generate", "top_logprobs_num": 64, "model_path": "teacher",
             "prompt_suffix": "suffix" if suffix else ""}
    asyncio.run(opd._teacher_request(None, asyncio.Semaphore(1), route, sample))
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["input_ids"] == [2, 3, *suffix, *response]
    assert payload["logprob_start_len"] == 1 + len(suffix)
    assert payload["top_logprobs_num"] == 64
    assert "token_ids_logprob" not in payload
    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert sample.tokens == [2, 3, *response]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
