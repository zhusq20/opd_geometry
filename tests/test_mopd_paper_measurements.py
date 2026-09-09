import json
from types import SimpleNamespace

import torch
import pytest

from slime_plugins.mopd.optimizer_views import OptimizerParameterView
from slime_plugins.mopd.paper_measurements import PaperMeasurements, export_hf_optimizer_state

NUM_GPUS = 0


def _args(tmp_path):
    return SimpleNamespace(
        mopd_output_dir=str(tmp_path), mopd_profile="qwen3", mopd_loss="teacher_topk",
        mopd_reduction="domain_response", mopd_total_steps=1, mopd_checkpoint_steps="1",
        hf_checkpoint="base", clip_grad=1.0, vocab_size=4, num_query_groups=1,
        num_attention_heads=2, kv_channels=2, hidden_size=4, q_lora_rank=None,
        use_wandb=False, use_tensorboard=False, metrics_output_dir=str(tmp_path / "metrics"),
    )


@pytest.mark.parametrize("loss,k", [("teacher_topk", None), ("student_topk", 16), ("student_topk", 64),
                                   ("topk_intersection", 64)])
def test_measurements_keep_master_updates_separate_from_bf16_rounding(tmp_path, loss, k):
    # The fifth row is Megatron vocabulary padding, omitted from HF and plots.
    model = torch.nn.Parameter(torch.ones((5, 4), dtype=torch.bfloat16))
    master = torch.nn.Parameter(model.detach().float().clone())
    optimizer = torch.optim.AdamW([master], lr=1e-6, weight_decay=0.0)
    view = OptimizerParameterView(
        "chunk0.embedding.word_embeddings.weight", model, master, optimizer,
        optimizer.param_groups[0], "adam",
    )
    args = _args(tmp_path)
    args.mopd_loss = loss
    if k is not None:
        args.mopd_topk = k
    measurements = PaperMeasurements(args, [view], 0)
    master.grad = torch.ones_like(master)
    measurements.before_step()
    optimizer.step()
    model.data.copy_(master.data)
    measurements.after_step()

    rows = [json.loads(line) for line in (tmp_path / "paper/measurements.jsonl").read_text().splitlines()]
    final = {row["quantity"]: row["metrics"] for row in rows if row["step"] == 1 and row["layer"] == "all"}
    assert final["delta_fp32"]["parameters"] == 16
    assert final["delta_fp32"]["l2"] > 0
    assert final["delta_bf16"]["l2"] == 0
    assert final["update"]["l2"] == final["delta_fp32"]["l2"]
    exported = torch.load(tmp_path / "paper/checkpoint_step_0001.pt", weights_only=True)
    assert exported["topk"] == k
    assert all(row["topk"] == k for row in rows)
    state = exported["parameters"]["model.embed_tokens.weight"]
    torch.testing.assert_close(state["value"], master.data[:4])
    torch.testing.assert_close(state["exp_avg"], optimizer.state[master]["exp_avg"][:4])
    torch.testing.assert_close(state["exp_avg_sq"], optimizer.state[master]["exp_avg_sq"][:4])
    assert state["step"] == 1


def test_optimizer_export_permutates_qkv_moments_with_weights(tmp_path):
    (tmp_path / "paper").mkdir()
    master = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    model = torch.nn.Parameter(master.data.bfloat16())
    optimizer = torch.optim.AdamW([master], lr=1e-3)
    master.grad = torch.arange(32, dtype=torch.float32).reshape(8, 4) + 1
    optimizer.step()
    view = OptimizerParameterView(
        "chunk0.decoder.layers.0.self_attention.linear_qkv.weight", model, master,
        optimizer, optimizer.param_groups[0], "adam",
    )
    args = _args(tmp_path)
    args.num_query_groups, args.num_attention_heads, args.kv_channels = 2, 4, 1
    path = export_hf_optimizer_state(args, [view], 1)
    parameters = torch.load(path, weights_only=True)["parameters"]
    for field, expected in (("value", master.data), ("exp_avg", optimizer.state[master]["exp_avg"]),
                            ("exp_avg_sq", optimizer.state[master]["exp_avg_sq"])):
        parts = [parameters[f"model.layers.0.self_attn.{name}_proj.weight"][field] for name in ("q", "k", "v")]
        torch.testing.assert_close(torch.cat(parts), expected[[0, 1, 4, 5, 2, 6, 3, 7]])


def test_measurement_resume_preserves_initial_delta_and_next_learning_rate(tmp_path):
    args = _args(tmp_path)
    args.mopd_total_steps, args.mopd_checkpoint_steps = 3, "1,3"
    model = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    master = torch.nn.Parameter(model.detach().float().clone())
    optimizer = torch.optim.AdamW([master], lr=0.1, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    view = OptimizerParameterView("chunk0.embedding.word_embeddings.weight", model, master,
                                  optimizer, optimizer.param_groups[0], "adam")
    measurements = PaperMeasurements(args, [view], 0)
    for step in range(1, 4):
        master.grad = torch.ones_like(master)
        measurements.before_step()
        optimizer.step()
        model.data.copy_(master.data)
        scheduler.step()
        measurements.after_step()
        if step == 1:
            first = torch.load(tmp_path / "paper/checkpoint_step_0001.pt", weights_only=True)
            assert first["parameters"]["model.embed_tokens.weight"]["lr"] == 0.05
            measurements = PaperMeasurements(args, [view], 1)
    rows = [json.loads(line) for line in (tmp_path / "paper/measurements.jsonl").read_text().splitlines()]
    assert {row["step"] for row in rows} == {0, 1, 3}
    assert not (tmp_path / "paper/checkpoint_step_0002.pt").exists()
    final = next(row for row in rows if row["step"] == 3 and row["layer"] == "all" and row["quantity"] == "delta_fp32")
    assert abs(final["metrics"]["l2"] - float((master.detach() - 1).norm())) < 1e-6
    exported = torch.load(tmp_path / "paper/checkpoint_step_0003.pt", weights_only=True)
    assert exported["parameters"]["model.embed_tokens.weight"]["lr"] == 0.0125


def test_step_zero_retry_reuses_identical_snapshots_without_writing(tmp_path, monkeypatch):
    args = _args(tmp_path)
    model = torch.nn.Parameter(torch.ones((5, 4), dtype=torch.bfloat16))
    master = torch.nn.Parameter(model.detach().float().clone())
    optimizer = torch.optim.AdamW([master], lr=0.01, weight_decay=0.0)
    view = OptimizerParameterView("chunk0.embedding.word_embeddings.weight", model, master,
                                  optimizer, optimizer.param_groups[0], "adam")
    PaperMeasurements(args, [view], 0)
    paths = [tmp_path / "paper" / name for name in ("initial_parameters.pt", "checkpoint_step_0000.pt")]
    before = [(path.stat().st_ino, path.stat().st_mtime_ns) for path in paths]
    monkeypatch.setattr("slime_plugins.mopd.paper_measurements._atomic_save",
                        lambda *_: pytest.fail("Identical snapshots should not be rewritten"))
    retry = PaperMeasurements(args, [view], 0)
    assert [(path.stat().st_ino, path.stat().st_mtime_ns) for path in paths] == before
    master.grad = torch.ones_like(master)
    retry.before_step()
    optimizer.step()
    model.data.copy_(master.data)
    assert torch.equal(retry.initial["fp32"][view.name], torch.ones(16))


@pytest.mark.parametrize("mismatch", ["fp32", "bf16", "config", "adam", "moment", "exported_value"])
def test_step_zero_retry_rejects_changed_state(tmp_path, mismatch):
    args = _args(tmp_path)
    model = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.bfloat16))
    master = torch.nn.Parameter(model.detach().float().clone())
    optimizer = torch.optim.AdamW([master], lr=0.01, weight_decay=0.0)
    view = OptimizerParameterView("chunk0.embedding.word_embeddings.weight", model, master,
                                  optimizer, optimizer.param_groups[0], "adam")
    PaperMeasurements(args, [view], 0)
    if mismatch in {"fp32", "bf16"}:
        (master if mismatch == "fp32" else model).data[0, 0] += 0.5
    elif mismatch == "config":
        args.mopd_reduction = "global_token"
    elif mismatch == "adam":
        optimizer.param_groups[0]["lr"] = 0.02
    else:
        path = tmp_path / "paper/checkpoint_step_0000.pt"
        snapshot = torch.load(path, weights_only=True)
        field = "exp_avg" if mismatch == "moment" else "value"
        snapshot["parameters"]["model.embed_tokens.weight"][field][0, 0] += 1
        torch.save(snapshot, path)
    with pytest.raises(ValueError, match="Cannot reuse step-zero"):
        PaperMeasurements(args, [view], 0)
