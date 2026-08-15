"""Tests for the official LiveCodeBench conversion path."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from types import SimpleNamespace

import pytest

from examples.optimizer_geometry import prepare_livecodebench_eval
from examples.optimizer_geometry.prepare_livecodebench_eval import convert, decode_json, write_configs


NUM_GPUS = 0


def _encoded_pickle(value) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(value))).decode("ascii")


@pytest.mark.unit
def test_decode_livecodebench_legacy_private_cases():
    cases = [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]

    assert decode_json(_encoded_pickle(json.dumps(cases))) == cases


@pytest.mark.unit
def test_livecodebench_pickle_decoder_forbids_globals():
    class Unsafe:
        def __reduce__(self):
            return eval, ("1 + 1",)

    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        decode_json(_encoded_pickle(Unsafe()))


@pytest.mark.unit
def test_convert_preserves_private_tests_for_sandboxfusion():
    row = {
        "question_id": "q1",
        "question_content": "Double the input.",
        "starter_code": "",
        "public_test_cases": json.dumps([{"input": "1\n", "output": "2\n"}]),
        "private_test_cases": _encoded_pickle(json.dumps([{"input": "2\n", "output": "4\n"}])),
        "metadata": json.dumps({}),
        "difficulty": "easy",
        "platform": "test",
    }

    converted = convert(row)
    sandbox_row = json.loads(converted["metadata"]["sandboxfusion_row"])
    tests = json.loads(json.loads(sandbox_row["test"])["input_output"])

    assert tests["inputs"] == ["1\n", "2\n"]
    assert tests["outputs"] == ["2\n", "4\n"]


@pytest.mark.unit
def test_eval_configs_use_online_greedy_and_final_pass_at_k_protocol(tmp_path):
    write_configs(tmp_path, tmp_path / "online.parquet", tmp_path / "final.parquet")

    import yaml

    online = yaml.safe_load((tmp_path / "code_eval.yaml").read_text())
    final = yaml.safe_load((tmp_path / "code_eval_final.yaml").read_text())

    assert online["eval"]["defaults"]["n_samples_per_eval_prompt"] == 1
    assert online["eval"]["defaults"]["max_response_len"] == 16384
    assert online["eval"]["defaults"]["temperature"] == 0.0
    assert online["eval"]["defaults"]["top_k"] == -1
    assert final["eval"]["defaults"]["n_samples_per_eval_prompt"] == 10
    assert final["eval"]["defaults"]["max_response_len"] == 16384
    assert final["eval"]["defaults"]["temperature"] == 0.2
    assert final["eval"]["defaults"]["top_p"] == 0.95
    assert final["eval"]["defaults"]["top_k"] == -1


@pytest.mark.unit
def test_prepare_updates_single_task_index_with_code_eval_provenance(tmp_path, monkeypatch):
    output = tmp_path / "single_task" / "code"
    output.parent.mkdir(parents=True)
    single_index = output.parent / "single_task_index.json"
    single_index.write_text(json.dumps({"tasks": {"code": {"eval_kind": "disabled"}}}) + "\n")
    rows = [
        {"prompt": "p1", "label": "", "metadata": {"difficulty": "easy", "question_id": "1"}},
        {"prompt": "p2", "label": "", "metadata": {"difficulty": "hard", "question_id": "2"}},
    ]
    monkeypatch.setitem(prepare_livecodebench_eval.EXPECTED_ROWS, "v5", 2)
    monkeypatch.setattr(prepare_livecodebench_eval, "training_texts", lambda _path: [])
    monkeypatch.setattr(
        prepare_livecodebench_eval,
        "load_version",
        lambda _version, _raw, _prompts, _cache: (rows, [], [{"filename": "test", "sha256": "0" * 64}]),
    )

    def fake_parquet(values, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values))

    monkeypatch.setattr(prepare_livecodebench_eval, "atomic_parquet", fake_parquet)
    prepare_livecodebench_eval.prepare(
        SimpleNamespace(
            output_dir=output,
            version="v5",
            online_version="v5",
            online_samples=1,
            seed=42,
            training_data=None,
        )
    )

    code = json.loads(single_index.read_text())["tasks"]["code"]
    assert code["eval_kind"] == "external_benchmark"
    assert code["eval_rows"] == 1
    assert code["final_eval_rows"] == 2
    assert code["benchmark_index_sha256"] == prepare_livecodebench_eval.sha256_file(
        output / "livecodebench_index.json"
    )
