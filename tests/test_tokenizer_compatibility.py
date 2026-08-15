"""Tests for the OPD tokenizer compatibility preflight."""

import runpy
from pathlib import Path

import pytest


NUM_GPUS = 0
MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "examples" / "optimizer_geometry" / "validate_tokenizers.py")
)


class FakeTokenizer:
    def __init__(self, vocab, *, eos=2):
        self._vocab = vocab
        self.special_tokens_map = {"eos_token": "</s>"}
        self.bos_token_id = 1
        self.eos_token_id = eos
        self.pad_token_id = 0

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return {"</s>": 2}


@pytest.mark.unit
def test_identical_token_id_maps_are_compatible():
    compare = MODULE["compare_tokenizers"]
    assert compare(FakeTokenizer({"a": 3, "</s>": 2}), FakeTokenizer({"a": 3, "</s>": 2}))["compatible"]


@pytest.mark.unit
def test_same_tokens_with_different_ids_are_rejected():
    compare = MODULE["compare_tokenizers"]
    result = compare(FakeTokenizer({"a": 3, "</s>": 2}), FakeTokenizer({"a": 4, "</s>": 2}))
    assert not result["compatible"]
    assert result["mismatched_id_count"] == 1
