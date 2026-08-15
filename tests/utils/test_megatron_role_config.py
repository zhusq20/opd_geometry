"""Unit tests for Megatron role config parsing and application."""

import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
import yaml


def _write_yaml(data: dict) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, handle)
    handle.flush()
    return handle.name


def _base_args(**overrides):
    args = dict(
        lr=2e-6,
        tensor_model_parallel_size=1,
        kl_coef=0.1,
        use_kl_loss=False,
        use_opd=True,
        opd_type="megatron",
        custom_advantage_function_path="slime.test.adv",
        untie_embeddings_and_output_weights=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        critic_num_nodes=1,
        critic_num_gpus_per_node=1,
        use_critic=False,
        megatron_config_path=None,
        start_rollout_id=None,
        rollout_global_dataset=False,
    )
    args.update(overrides)
    return Namespace(**args)


class TestMegatronRoleConfig:
    def test_parse_actor_and_critic_role_overrides(self):
        from slime.utils.arguments import parse_megatron_role_args

        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {"lr": "1e-5", "tensor_model_parallel_size": 2},
                    },
                    {"name": "default", "role": "actor", "overrides": {"lr": "1e-6", "tensor_model_parallel_size": 4}},
                ]
            }
        )
        args = _base_args()

        actor_args = parse_megatron_role_args(args, path, role="actor")
        critic_args = parse_megatron_role_args(args, path, role="critic")

        assert actor_args.lr == 1e-6
        assert actor_args.tensor_model_parallel_size == 4
        assert actor_args.kl_coef == args.kl_coef
        assert actor_args.use_opd is args.use_opd

        assert critic_args.lr == 1e-5
        assert critic_args.tensor_model_parallel_size == 2
        assert critic_args.kl_coef == 0
        assert critic_args.use_opd is False
        assert critic_args.custom_advantage_function_path is None
        assert critic_args.untie_embeddings_and_output_weights is True

    def test_missing_role_inherits_base_args(self):
        from slime.utils.arguments import parse_megatron_role_args

        path = _write_yaml(
            {
                "megatron": [
                    {"name": "default", "role": "actor", "overrides": {"lr": "1e-6"}},
                ]
            }
        )
        args = _base_args()

        critic_args = parse_megatron_role_args(args, path, role="critic")

        assert critic_args is not args
        assert critic_args.lr == args.lr
        assert critic_args.kl_coef == 0
        assert critic_args.use_opd is False

    def test_role_config_expands_environment_variables(self, monkeypatch):
        from slime.utils.arguments import parse_megatron_role_args

        monkeypatch.setenv("SLIME_TEST_CRITIC_LOAD", "/checkpoints/value")
        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {"load": "${SLIME_TEST_CRITIC_LOAD}"},
                    }
                ]
            }
        )

        critic_args = parse_megatron_role_args(_base_args(), path, role="critic")

        assert critic_args.load == "/checkpoints/value"

    def test_optimizer_geometry_ppo_critic_uses_batch_profile_values(self, monkeypatch):
        from slime.utils.arguments import parse_megatron_role_args

        critic_values = {
            "OPTIMIZER_GEOMETRY_CRITIC_LOAD": "/checkpoints/actor",
            "OPTIMIZER_GEOMETRY_CRITIC_SAVE": "/checkpoints/critic",
            "OPTIMIZER_GEOMETRY_CRITIC_LR": "2.5e-6",
            "OPTIMIZER_GEOMETRY_CRITIC_WEIGHT_DECAY": "0.025",
            "OPTIMIZER_GEOMETRY_CRITIC_BETA2": "0.9987381276",
        }
        for name, value in critic_values.items():
            monkeypatch.setenv(name, value)
        config = Path(__file__).parents[2] / "examples/optimizer_geometry/configs/ppo_roles.yaml"

        critic_args = parse_megatron_role_args(_base_args(), str(config), role="critic")

        assert critic_args.optimizer == "adam"
        assert critic_args.load == "/checkpoints/actor"
        assert critic_args.save == "/checkpoints/critic"
        assert critic_args.lr == pytest.approx(2.5e-6)
        assert critic_args.weight_decay == pytest.approx(0.025)
        assert critic_args.adam_beta1 == pytest.approx(0.9)
        assert critic_args.adam_beta2 == pytest.approx(0.9987381276)

    def test_role_config_rejects_unresolved_environment_variables(self, monkeypatch):
        from slime.utils.arguments import parse_megatron_role_args

        monkeypatch.delenv("SLIME_TEST_MISSING_LOAD", raising=False)
        path = _write_yaml(
            {
                "megatron": [
                    {
                        "name": "default",
                        "role": "critic",
                        "overrides": {"load": "${SLIME_TEST_MISSING_LOAD}"},
                    }
                ]
            }
        )

        with pytest.raises(ValueError, match="SLIME_TEST_MISSING_LOAD"):
            parse_megatron_role_args(_base_args(), path, role="critic")

    @pytest.mark.parametrize(
        "config",
        [
            {"critic": [{"name": "default", "overrides": {"lr": "1e-5"}}]},
            {"lr": "1e-5"},
            {},
        ],
    )
    def test_requires_top_level_megatron_key(self, config):
        from slime.utils.arguments import parse_megatron_role_args

        path = _write_yaml(config)
        args = _base_args()

        with pytest.raises(AssertionError, match="top-level 'megatron' list"):
            parse_megatron_role_args(args, path, role="critic")

    def test_create_training_models_applies_actor_override_without_critic(self, monkeypatch):
        from slime.ray import placement_group as placement_group_module

        path = _write_yaml(
            {
                "megatron": [
                    {"name": "default", "role": "actor", "overrides": {"lr": "1e-6"}},
                ]
            }
        )
        args = _base_args(megatron_config_path=path, use_critic=False)

        class DummyModel:
            def __init__(self, model_args, with_ref=False, with_opd_teacher=False):
                self.args = model_args
                self.with_ref = with_ref
                self.with_opd_teacher = with_opd_teacher
                self.create_calls = []
                self.rollout_manager = None

            def create(self, rollout_manager=None):
                self.rollout_manager = rollout_manager
                self.create_calls.append(
                    {
                        "args": self.args,
                        "with_ref": self.with_ref,
                        "with_opd_teacher": self.with_opd_teacher,
                        "rollout_manager": rollout_manager,
                    }
                )
                return [7]

        def fake_allocate_train_group(
            args,
            num_nodes,
            num_gpus_per_node,
            pg,
            role="actor",
            with_ref=False,
            with_opd_teacher=False,
        ):
            return DummyModel(args, with_ref=with_ref, with_opd_teacher=with_opd_teacher)

        monkeypatch.setattr(placement_group_module, "allocate_train_group", fake_allocate_train_group)
        monkeypatch.setattr(placement_group_module.ray, "get", lambda value: value)

        actor_model, critic_model = placement_group_module.create_training_models(
            args,
            {"actor": None, "critic": None},
            object(),
        )

        assert critic_model is None
        assert actor_model.args.lr == 1e-6
        assert actor_model.create_calls[0]["args"].lr == 1e-6
        assert args.start_rollout_id == 7
