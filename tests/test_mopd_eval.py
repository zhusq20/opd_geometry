import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.types import Sample
from slime_plugins.mopd import eval as mopd_eval
from slime_plugins.mopd.sampler import TASKS

NUM_GPUS = 0


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


def _data_source(weights=(0.1, 0.2, 0.3, 0.4), initial=(0.5, 0.5, 0.5, 0.5)):
    return SimpleNamespace(
        sources=[
            SimpleNamespace(config={"name": task, "target_weight": weight, "initial_teacher_loss": loss})
            for task, weight, loss in zip(TASKS, weights, initial, strict=True)
        ]
    )


def test_eval_reports_per_task_normalized_and_fixed_weighted_loss(monkeypatch):
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

    monkeypatch.setattr(mopd_eval, "score_teacher_samples", score)
    result = mopd_eval.generate_teacher_loss_eval(
        SimpleNamespace(mopd_failure_penalty=10.0), 0, _data_source(), evaluation=True
    )
    assert result.metrics["eval/teacher_loss/math"] == pytest.approx(1.0)
    assert result.metrics["eval/teacher_loss/code"] == pytest.approx(0.5)
    assert result.metrics["eval/teacher_loss/if"] == pytest.approx(1.0)
    assert result.metrics["eval/teacher_loss/science"] == pytest.approx(2.0)
    assert result.metrics["eval/normalized_teacher_loss/science"] == pytest.approx(4.0)
    assert result.metrics["eval/weighted_teacher_loss"] == pytest.approx(1.3)
    assert samples["science"].metadata["weighted_teacher_loss"] == pytest.approx(0.8)


def test_eval_uses_numeric_penalty_for_empty_or_teacher_failure(monkeypatch):
    samples = {task: _sample([-1.0], [-1.0]) for task in TASKS}
    samples["if"].metadata["mopd_failure_penalty"] = 10.0
    output = RolloutFnEvalOutput(
        data={task: {"samples": [sample], "rewards": [0.0]} for task, sample in samples.items()}
    )
    monkeypatch.setattr(mopd_eval, "default_generate_rollout", lambda *_args, **_kwargs: output)

    async def score(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(mopd_eval, "score_teacher_samples", score)
    result = mopd_eval.generate_teacher_loss_eval(
        SimpleNamespace(mopd_failure_penalty=10.0), 0, _data_source(), evaluation=True
    )
    assert result.metrics["eval/teacher_loss/if"] == 10.0
    assert result.metrics["eval/weighted_teacher_loss"] == pytest.approx(3.0)


def test_corrected_topk_loss_keeps_full_normalization_and_minus_p_gradient():
    from slime_plugins.mopd.loss import corrected_reverse_kl

    logits = torch.tensor([[1.0, -0.5, 0.7, 2.0]], requires_grad=True)
    teacher = torch.tensor([[0.4, 0.1, 0.2, 0.3]])
    ids = torch.tensor([[0, 2]])
    log_probs = logits.log_softmax(dim=-1)
    loss = corrected_reverse_kl(log_probs.gather(-1, ids), teacher.log().gather(-1, ids)).sum()
    p = log_probs.exp()
    supported_log_ratio = (log_probs - teacher.log()) * torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    expected_gradient = p * (supported_log_ratio - (p * supported_log_ratio).sum(dim=-1, keepdim=True))
    loss.backward()
    torch.testing.assert_close(logits.grad, expected_gradient)
    assert logits.grad[0, 3].abs() > 0  # Unselected vocabulary still participates in normalization.
    assert loss > 0


def test_teacher_topk_payload_retains_the_last_response_positions():
    from slime_plugins.mopd.loss import teacher_topk

    rows = [[[float(-i - 1), i] for i in range(64)] for _ in range(2)]
    ids, probs = teacher_topk({"meta_info": {"input_top_logprobs": [None, *rows]}}, 2)
    assert ids.shape == probs.shape == (2, 64)
    assert ids[0].tolist() == list(range(64))
    with pytest.raises(ValueError, match="exactly 64"):
        teacher_topk({"meta_info": {"input_top_logprobs": [rows[0][:16]]}}, 1)


def test_reference_bank_reuses_responses_and_teacher_scores(tmp_path, monkeypatch):
    from slime.rollout import sglang_rollout
    from slime_plugins.mopd import reference_bank

    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"version":5}')
    args = SimpleNamespace(experiment_data_index=str(protocol), mopd_reference_bank=str(tmp_path / "bank.pt"))
    calls = []

    def generate(*a, **kw):
        calls.append("generate")
        return RolloutFnEvalOutput(
            data={
                task: {"samples": [Sample(tokens=[0, 1], response_length=1, loss_mask=[1]) for _ in range(64)]}
                for task in TASKS
            }
        )

    async def score(args, output):
        calls.append("teacher")
        for info in output.data.values():
            for sample in info["samples"]:
                sample.reward = {"meta_info": {"input_top_logprobs": [None, [[-5.0, i] for i in range(64)]]}}

    monkeypatch.setattr(sglang_rollout, "generate_rollout", generate)
    monkeypatch.setattr(mopd_eval, "_score_all_tasks", score)
    path = reference_bank.prepare_bank(args, 0, None, args.mopd_reference_bank, initial=True)
    reference_bank.prepare_bank(args, 499, None, path)
    assert calls == ["generate", "teacher"]
    assert len(torch.load(path, weights_only=False)["samples"]) == 256
    record = reference_bank.loss_record(path, [0.5] * 256)
    assert record["weighted_loss"] == 0.5
    assert all(len(values) == 64 for values in record["prompt_losses"].values())
    protocol.write_text('{"version":6}')
    with pytest.raises(ValueError, match="different protocol"):
        reference_bank.prepare_bank(args, 0, None, path, initial=True)


def _asymmetric_args(tmp_path):
    from slime_plugins.mopd.prompting import EMPTY_THINKING_SUFFIX, PROMPT_FORMAT

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": 6, "prompt_format": PROMPT_FORMAT}))
    router = tmp_path / "teachers.json"
    router.write_text(
        json.dumps({"teachers": {task: {"url": task, "prompt_suffix": EMPTY_THINKING_SUFFIX} for task in TASKS}})
    )
    return SimpleNamespace(
        experiment_data_index=str(protocol),
        mopd_reference_bank=str(tmp_path / "bank.pt"),
        mopd_loss="teacher_topk",
        chat_template_suffix_to_remove=EMPTY_THINKING_SUFFIX,
        opd_teacher_router_config=str(router),
    )


@pytest.mark.parametrize("mismatch", ["student", "teacher", "old_bank"])
def test_reference_bank_checks_active_prefixes_before_reusing_cached_targets(tmp_path, monkeypatch, mismatch):
    from pathlib import Path

    from slime.rollout import sglang_rollout
    from slime_plugins.m2rl import opd
    from slime_plugins.mopd import reference_bank

    args = _asymmetric_args(tmp_path)
    identity = hashlib.sha256(Path(args.experiment_data_index).read_bytes()).hexdigest()
    torch.save({"protocol_sha256": identity, "samples": []}, args.mopd_reference_bank)
    monkeypatch.setattr(sglang_rollout, "generate_rollout", lambda *_a, **_kw: pytest.fail("cached bank generated"))
    assert reference_bank.prepare_bank(args, 0, None, args.mopd_reference_bank) == args.mopd_reference_bank
    if mismatch == "student":
        args.chat_template_suffix_to_remove = None
    elif mismatch == "teacher":
        router = json.loads(Path(args.opd_teacher_router_config).read_text())
        router["teachers"]["math"].pop("prompt_suffix")
        monkeypatch.setattr(opd, "load_teacher_router", lambda _path: router)
    else:
        torch.save({"protocol_sha256": "legacy", "samples": []}, args.mopd_reference_bank)
    with pytest.raises(ValueError, match="differs|different protocol"):
        reference_bank.prepare_bank(args, 0, None, args.mopd_reference_bank)


def test_diagnostic_rejects_legacy_rendered_manifest_before_generation(tmp_path, monkeypatch):
    from slime.utils.async_utils import run
    from slime_plugins.mopd import diagnostic, rollout

    args = _asymmetric_args(tmp_path)
    manifest = tmp_path / "diagnostic.json"
    manifest.write_text(json.dumps({"protocol": "qwen3_1.7b_base_4t_common_checkpoint", "sources": []}))
    args.mopd_diagnostic_data = str(manifest)
    monkeypatch.setattr(rollout, "_generate_samples", lambda *_a: pytest.fail("legacy diagnostic generated"))
    with pytest.raises(ValueError, match="Regenerate"):
        run(diagnostic.generate_samples(args, "evaluation", [16] * 4, 0))


def _tp_topk_worker(rank, rendezvous, output, objective):
    from pathlib import Path
    from slime_plugins.mopd.loss import _VocabParallelSelectedLogProbs, corrected_reverse_kl, student_topk_advantage

    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    full = torch.tensor([[1.0, 0.2, -0.8, 1.3, -0.4, 9.0], [0.1, -0.3, 2.0, 0.0, -0.1, 7.0]])
    logits = full[:, rank * 3 : (rank + 1) * 3].clone().requires_grad_()
    ids = torch.tensor([[0, 3], [2, 4]])
    selected = _VocabParallelSelectedLogProbs.apply(logits, ids, torch.distributed.group.WORLD, rank, 5)
    teacher = torch.tensor([[0.4, 0.3], [0.5, 0.2]]).log()
    loss = (-(student_topk_advantage(selected, teacher) * selected).sum()
            if objective == "student_topk" else corrected_reverse_kl(selected, teacher).sum())
    loss.backward()
    torch.save({"selected": selected.detach(), "gradient": logits.grad}, Path(output) / f"{rank}.pt")
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("objective", ["teacher_topk", "student_topk"])
def test_tensor_parallel_topk_normalization_and_gradient_match_full_vocab(tmp_path, objective):
    import torch.multiprocessing as mp
    from slime_plugins.mopd.loss import corrected_reverse_kl, student_topk_advantage

    mp.spawn(_tp_topk_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), objective), nprocs=2, join=True)
    logits = torch.tensor([[1.0, 0.2, -0.8, 1.3, -0.4, 9.0], [0.1, -0.3, 2.0, 0.0, -0.1, 7.0]], requires_grad=True)
    ids = torch.tensor([[0, 3], [2, 4]])
    selected = logits[:, :5].log_softmax(dim=-1).gather(-1, ids)
    teacher = torch.tensor([[0.4, 0.3], [0.5, 0.2]]).log()
    loss = (-(student_topk_advantage(selected, teacher) * selected).sum()
            if objective == "student_topk" else corrected_reverse_kl(selected, teacher).sum())
    loss.backward()
    outputs = [torch.load(tmp_path / f"{rank}.pt", weights_only=True) for rank in range(2)]
    for output in outputs:
        torch.testing.assert_close(output["selected"], selected)
    torch.testing.assert_close(torch.cat([output["gradient"] for output in outputs], dim=-1), logits.grad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
