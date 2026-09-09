import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.placement_group import _create_placement_group, _get_placement_group_layout

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "rollout_external": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (16, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])


@pytest.mark.unit
@pytest.mark.parametrize("profile", ["qwen3", "smollm3", None])
def test_paper_roles_follow_reversed_visible_gpu_order(monkeypatch, profile):
    from slime.ray import placement_group as module

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    pg = object()
    # The scheduler allocated physical GPU 3 to bundle 5 and GPU 1 to bundle 7;
    # _create_placement_group returns those bundles sorted by physical GPU ID.
    monkeypatch.setattr(module, "_create_placement_group", lambda count: (pg, [7, 5], [1.0, 3.0]))
    groups = module.create_placement_groups(
        _args(actor_num_nodes=1, actor_num_gpus_per_node=1, rollout_num_gpus=1,
              use_critic=False, mopd_profile=profile)
    )
    if profile:
        assert groups["actor"] == (pg, [5, 7], [3.0, 1.0])
        assert groups["rollout"] == (pg, [7], [1.0])
    else:
        assert groups["actor"] == (pg, [7, 5], [1.0, 3.0])
        assert groups["rollout"] == (pg, [5], [3.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
