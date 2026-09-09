import math

import pytest
import torch

from slime_plugins.mopd.paper_diagnostics import (
    js_divergence, local_distillation_loss, normalization_decomposition,
    prefix_weights, support_overlap, tensor_metrics,
)

NUM_GPUS = 0


def test_full_vocab_gradient_matches_expected_fresh_sampled_pg():
    logits = torch.tensor([[0.2, -0.1, 0.8]], requires_grad=True)
    teacher = torch.tensor([[0.4, 0.2, 0.4]]).log()
    full = local_distillation_loss(logits, teacher, loss="full_vocab", weights=torch.ones(1))
    expected = torch.autograd.grad(full, logits)[0]
    gradient = torch.zeros_like(logits)
    for token, probability in enumerate(logits.detach().softmax(-1)[0]):
        pg = local_distillation_loss(logits, teacher, loss="sampled_pg", weights=torch.ones(1), action_ids=[token])
        gradient += probability * torch.autograd.grad(pg, logits)[0]
    torch.testing.assert_close(gradient, expected)


def test_topk_uses_full_vocabulary_probabilities_and_correction():
    logits = torch.tensor([[0.4, -0.1, 0.0]], requires_grad=True)
    teacher = torch.tensor([[0.7, 0.2, 0.1]]).log()
    loss = local_distillation_loss(logits, teacher, loss="teacher_topk", weights=torch.ones(1), topk=1)
    p = logits.softmax(-1)[0, 0]
    expected = p * (p.log() - math.log(0.7)) - p + 0.7
    torch.testing.assert_close(loss, expected)
    assert torch.autograd.grad(loss, logits)[0][0, 1] != 0


def test_reductions_have_distinct_global_domain_response_weights():
    masks = [torch.ones(1), torch.ones(3), torch.ones(4)]
    domains = ["a", "a", "b"]
    assert prefix_weights(masks, domains, "global_token").tolist() == pytest.approx([1 / 8] * 8)
    assert prefix_weights(masks, domains, "domain_token").tolist() == pytest.approx([1 / 8] * 8)
    assert prefix_weights(masks, domains, "domain_response").tolist() == pytest.approx([1 / 4] + [1 / 12] * 3 + [1 / 8] * 4)


def test_length_gradient_covariance_is_exact():
    record = normalization_decomposition([torch.tensor([1., 3.]), torch.tensor([4., -2.])], [2, 10])
    assert record["residual_l2"] < 1e-12
    torch.testing.assert_close(record["token_mean"], torch.tensor([3.5, -7 / 6], dtype=torch.float64))


def test_js_distance_and_degenerate_support_conventions():
    teacher = torch.tensor([[0.7, 0.2, 0.1]]).log()
    assert js_divergence(teacher, teacher) == pytest.approx(0.0, abs=1e-14)
    assert 0 < js_divergence(teacher, torch.tensor([[0.1, 0.2, 0.7]]).log()) < math.log(2)
    assert support_overlap(torch.zeros(4), torch.zeros(4), 0.5)["jaccard"] is None
    comparison = support_overlap(torch.tensor([1., 2., 0., 0.]), torch.tensor([0., -2., 3., 0.]), 0.5)
    assert comparison["jaccard"] == pytest.approx(1 / 3)
    assert comparison["random_jaccard"] == pytest.approx(1 / 3)
    assert tensor_metrics(torch.tensor([3., 0., 0., 0.]))["energy90_fraction"] == 0.25
    assert tensor_metrics(torch.zeros(4))["energy90_fraction"] is None


def test_exported_state_proposal_matches_actual_adamw_with_clipping():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("paper_probe", Path(__file__).resolve().parents[1] / "examples/mopd_gpas/probe_checkpoint.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.03, betas=(0.8, 0.9), eps=1e-4, weight_decay=0.1)
    parameter.grad = torch.tensor([0.3, -0.1])
    optimizer.step()
    before = parameter.detach().clone()
    state = optimizer.state[parameter]
    exported = {"p": {"value": before, "exp_avg": state["exp_avg"].clone(), "exp_avg_sq": state["exp_avg_sq"].clone(),
                      "step": int(state["step"]), "lr": 0.03, "betas": (0.8, 0.9), "eps": 1e-4, "weight_decay": 0.1}}
    gradients = {"p": torch.tensor([3., -4.])}
    clipped, update = probe.proposed_adamw_step(exported, gradients, 1.0)
    parameter.grad = gradients["p"].clone()
    torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    optimizer.step()
    torch.testing.assert_close(clipped["p"], parameter.grad)
    torch.testing.assert_close(update["p"], parameter.detach() - before)
    torch.testing.assert_close(exported["p"]["value"], before, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["qwen3", "smollm3"])
@pytest.mark.parametrize("k", [16, 64])
def test_local_probe_runs_on_real_tiny_profile_checkpoint(tmp_path, family, k):
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import yaml
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, SmolLM3Config

    spec = importlib.util.spec_from_file_location("paper_probe", Path(__file__).resolve().parents[1] / "examples/mopd_gpas/probe_checkpoint.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>")
    config_class = Qwen3Config if family == "qwen3" else SmolLM3Config
    extra = {"head_dim": 8} if family == "qwen3" else {"no_rope_layers": [0], "tie_word_embeddings": True}
    config = config_class(vocab_size=80, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                          eos_token_id=1, pad_token_id=1, **extra)
    torch.manual_seed(8)
    model = AutoModelForCausalLM.from_config(config)
    student_dir = tmp_path / "student"
    model.save_pretrained(student_dir)
    tokenizer.save_pretrained(student_dir)
    parameters = {name: {"value": value.detach().clone(), "model_value": value.detach().bfloat16(),
                         "exp_avg": torch.zeros_like(value), "exp_avg_sq": torch.zeros_like(value), "step": 0,
                         "lr": 1e-3, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.01}
                  for name, value in model.named_parameters()}
    torch.save({"step": 0, "profile": family, "hf_checkpoint": str(student_dir), "parameters": parameters,
                "loss": "student_topk", "topk": k,
                "clip_grad": 1.0, "tasks": ["math", "code"]}, tmp_path / "snapshot.pt")
    sources, routes = [], {}
    for index, task in enumerate(("math", "code")):
        torch.manual_seed(12 + index)
        teacher = AutoModelForCausalLM.from_config(config)
        teacher_dir = tmp_path / task
        teacher.save_pretrained(teacher_dir)
        tokenizer.save_pretrained(teacher_dir)
        routes[task] = {"model_path": str(teacher_dir), "prompt_suffix": ""}
        data = tmp_path / f"{task}.jsonl"
        data.write_text("\n".join(json.dumps({"prompt": prompt}) for prompt in ("a b", "c d")))
        sources.append({"name": task, "path": str(data)})
    (tmp_path / "diagnostic.yaml").write_text(yaml.safe_dump({"sources": sources}))
    (tmp_path / "heldout.yaml").write_text(yaml.safe_dump({"sources": sources}))
    (tmp_path / "teachers.yaml").write_text(yaml.safe_dump({"teachers": routes}))
    args = SimpleNamespace(snapshot=str(tmp_path / "snapshot.pt"), manifest=str(tmp_path / "diagnostic.yaml"),
                           teachers=str(tmp_path / "teachers.yaml"), output=str(tmp_path / "probe"),
                           tasks=None, batches=1, responses_per_domain=2, prefixes_per_response=1,
                           max_new_tokens=2, pg_advantage_clip=0., seed=42, device="cpu", wandb_mode="disabled")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        records = probe.run_probe(args)
    finally:
        torch.set_num_threads(previous)
    assert {row["kind"] for row in records} == {"sparsity", "normalization", "overlap", "teacher_distance", "supervision", "cost"}
    assert all(row["probe_forward_precision"] == "fp32_master" for row in records)
    assert all(row["topk"] == k for row in records)
    assert any(row.get("heldout_metric") == f"student_top{k}_normalized_logratio" for row in records)
    normalization = [row for row in records if "decomposition_residual_l2" in row["metrics"]]
    assert max(row["metrics"]["decomposition_residual_l2"] for row in normalization) < 1e-10
    assert (tmp_path / "probe/prefix_bank_00.pt").is_file()
    assert any(row.get("branch") == "common_prefix" for row in records)


def test_online_sampled_pg_detaches_advantage_and_clips_before_loss(monkeypatch):
    from types import SimpleNamespace
    from slime.backends.megatron_utils import loss as megatron_loss
    from slime_plugins.mopd import loss as mopd_loss
    logits = torch.tensor([[1., -1., 0.], [0., 0.5, -0.5]], requires_grad=True)
    actions = torch.tensor([0, 2])
    teacher = torch.tensor([-5., -0.1])
    monkeypatch.setattr(megatron_loss, "get_responses", lambda *a, **kw: iter([(logits, actions)]))
    monkeypatch.setattr(mopd_loss, "selected_log_probs", lambda values, ids, **kw: values.log_softmax(-1).gather(-1, ids))
    metadata = {"teacher_sampled_log_probs": teacher}
    batch = {"unconcat_tokens": [], "total_lengths": [], "response_lengths": [],
             "metadata": [metadata], "loss_masks": [torch.ones(2)]}
    objective, metrics = mopd_loss.paper_loss(SimpleNamespace(mopd_loss="sampled_reverse_kl", vocab_size=3, mopd_pg_advantage_clip=1.), batch, logits, torch.mean)
    objective.backward()
    probabilities = logits.detach().softmax(-1)
    sampled_logp = logits.detach().log_softmax(-1).gather(-1, actions[:, None]).squeeze(-1)
    advantages = (teacher - sampled_logp).clamp(-1, 1)
    expected = -advantages[:, None] * (torch.nn.functional.one_hot(actions, 3) - probabilities) / 2
    torch.testing.assert_close(logits.grad, expected)
    assert metrics["pg_advantage_clipped_fraction"] == 1.
    assert metadata["mopd_teacher_loss"] == pytest.approx(float((sampled_logp - teacher).mean()))


@pytest.mark.parametrize("distribution", ["normal", "ties", "zeros"])
def test_partition_energy_metrics_match_exact_sorted_reference(distribution):
    generator = torch.Generator().manual_seed(51)
    values = torch.randn(997, generator=generator)
    if distribution == "ties":
        values = values.round()
    elif distribution == "zeros":
        values.zero_()
    observed = tensor_metrics(values)
    ranked = values.abs().sort(descending=True).values.double().square()
    total = float(ranked.sum())
    expected_count = int(torch.searchsorted(ranked.cumsum(0), 0.9 * total)) + 1
    assert observed["energy90_fraction"] == (expected_count / values.numel() if total else None)
    for fraction in (0.01, 0.05, 0.1):
        expected = float(ranked[:math.ceil(values.numel() * fraction)].sum()) / total if total else None
        assert observed[f"energy_at_{fraction:g}"] == pytest.approx(expected)
