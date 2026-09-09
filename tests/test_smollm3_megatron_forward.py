"""A four-layer SmolLM3 exercises both RoPE and the fourth no-RoPE layer."""
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 1


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron/TE forward requires a GPU")
def test_smollm3_forward_matches_huggingface(tmp_path, monkeypatch):
    # Compare FP32 arithmetic; TE's TF32 GEMMs otherwise add expected rounding
    # differences large enough to obscure a misplaced no-RoPE layer.
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    pytest.importorskip("megatron.core")
    pytest.importorskip("transformer_engine")
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers import SmolLM3Config, SmolLM3ForCausalLM

    from slime.backends.megatron_utils.hf_to_megatron.qwen import qwen_hf_tensor

    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)
    try:
        hf_config = SmolLM3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                 num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                                 no_rope_layer_interval=4, tie_word_embeddings=True,
                                 pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                 max_position_embeddings=128, rms_norm_eps=1e-6,
                                 rope_parameters={"rope_type": "default", "rope_theta": 5000000.0})
        torch.manual_seed(42)
        hf_model = SmolLM3ForCausalLM(hf_config).cuda().float().eval()
        config = TransformerConfig(num_layers=4, hidden_size=64, ffn_hidden_size=128,
                                   num_attention_heads=4, num_query_groups=2, kv_channels=16,
                                   normalization="RMSNorm", layernorm_epsilon=1e-6,
                                   activation_func=torch.nn.functional.silu, gated_linear_unit=True,
                                   add_bias_linear=False, no_rope_freq=4,
                                   attention_dropout=0.0, hidden_dropout=0.0,
                                   params_dtype=torch.float32, attention_backend=AttnBackend.unfused)
        megatron = GPTModel(config, get_gpt_layer_with_transformer_engine_spec(),
                            vocab_size=128, max_sequence_length=128,
                            share_embeddings_and_output_weights=True, position_embedding_type="rope",
                            rotary_base=5000000.0, parallel_output=False).cuda().eval()
        tensors = hf_model.state_dict()
        reader = SimpleNamespace(get_tensor=lambda name: tensors[name])
        with torch.no_grad():
            for name, parameter in megatron.named_parameters():
                parameter.copy_(qwen_hf_tensor(name, reader, hf_config))
            inputs = torch.tensor([[12, 43, 7, 18, 92, 2, 5, 6]], device="cuda")
            positions = torch.arange(inputs.shape[1], device="cuda").unsqueeze(0)
            expected = hf_model(inputs).logits
            actual = megatron(inputs, positions, None)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
