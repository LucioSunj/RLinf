"""CPU-only tests for the gate decision/environment-step contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml

RLINF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RLINF_ROOT))

from rlinf.data.embodied_io_struct import (  # noqa: E402
    ChunkStepResult,
    EmbodiedRolloutResult,
)


def _load_contract_module():
    path = (
        RLINF_ROOT
        / "rlinf"
        / "workers"
        / "env"
        / "gate_contract.py"
    )
    spec = importlib.util.spec_from_file_location("gate_contract_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load_contract_module()


@pytest.mark.parametrize("exec_horizon", [10, 24])
def test_generation_and_execution_horizons_are_separate(exec_horizon):
    assert (
        contract.validate_gate_horizons(
            generation_horizon=32, exec_horizon=exec_horizon
        )
        == exec_horizon
    )
    generated = torch.arange(2 * 32 * 3).reshape(2, 32, 3)
    executed = contract.slice_gate_action_chunk(
        generated,
        generation_horizon=32,
        exec_horizon=exec_horizon,
    )
    assert executed.shape == (2, exec_horizon, 3)
    torch.testing.assert_close(executed, generated[:, :exec_horizon])


def test_invalid_execution_horizon_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        contract.validate_gate_horizons(generation_horizon=32, exec_horizon=33)


def test_generated_chunk_must_match_configured_horizon():
    with pytest.raises(ValueError, match="expected configured"):
        contract.slice_gate_action_chunk(
            torch.zeros(2, 31, 7),
            generation_horizon=32,
            exec_horizon=10,
        )


@pytest.mark.parametrize("exec_horizon", [10, 24])
def test_bootstrap_dones_stack_with_environment_chunks(exec_horizon):
    bootstrap = contract.make_bootstrap_dones(
        batch_size=4, exec_horizon=exec_horizon
    )
    environment = torch.zeros(4, exec_horizon, dtype=torch.bool)
    stacked = torch.stack([bootstrap, environment], dim=0)
    assert stacked.shape == (2, 4, exec_horizon)


def test_actual_trajectory_keeps_decision_and_environment_shapes_separate():
    batch, exec_horizon = 3, 10
    result = EmbodiedRolloutResult(max_episode_length=480)
    zero_dones = contract.make_bootstrap_dones(
        batch_size=batch, exec_horizon=exec_horizon
    )
    for step in range(2):
        mode = torch.full((batch, 1), step, dtype=torch.long)
        result.append_step_result(
            ChunkStepResult(
                actions=mode,
                prev_logprobs=torch.zeros(batch, 1),
                forward_inputs={"action": mode, "mode": mode},
                dones=zero_dones.clone(),
                terminations=zero_dones.clone(),
                truncations=zero_dones.clone(),
                rewards=None if step == 0 else torch.zeros(batch, exec_horizon),
            )
        )
    result.append_step_result(
        ChunkStepResult(
            dones=zero_dones.clone(),
            terminations=zero_dones.clone(),
            truncations=zero_dones.clone(),
            rewards=torch.zeros(batch, exec_horizon),
        )
    )

    trajectory = result.to_trajectory()
    assert trajectory.actions.shape == (2, batch, 1)
    assert trajectory.prev_logprobs.shape == (2, batch, 1)
    assert trajectory.rewards.shape == (2, batch, exec_horizon)
    assert trajectory.dones.shape == (3, batch, exec_horizon)
    assert trajectory.forward_inputs["action"].shape == (2, batch, 1)


def test_pending_cost_aligns_with_executed_action_and_charges_final_action():
    pending = contract.PendingGateDecisions(num_stages=1)
    first = {"mode": torch.tensor([[0]]), "mode_cost": torch.tensor([[0.2]])}
    second = {"mode": torch.tensor([[1]]), "mode_cost": torch.tensor([[1.0]])}

    # Bootstrap observation has no preceding action/reward.
    assert pending.consume_reward(0) is None
    pending.mark_executed(0, first)

    # Reward returned with the second observation belongs to the first action.
    reward_inputs = pending.consume_reward(0)
    assert reward_inputs is first
    assert reward_inputs["mode_cost"].item() == pytest.approx(0.2)
    pending.mark_executed(0, second)

    # The final bootstrap receives the last environment reward and must still
    # consume the second action's cost rather than an unexecuted bootstrap action.
    final_inputs = pending.consume_reward(0)
    assert final_inputs is second
    assert final_inputs["mode_cost"].item() == pytest.approx(1.0)
    assert pending.consume_reward(0) is None


@pytest.mark.parametrize(
    ("max_env_steps", "exec_horizon", "expected_decisions", "global_batch"),
    [(480, 10, 24576, 2048), (216, 24, 4608, 512)],
)
def test_gate_rollout_batch_is_divisible(
    max_env_steps, exec_horizon, expected_decisions, global_batch
):
    decisions = contract.rollout_decision_count(
        max_env_steps=max_env_steps,
        exec_horizon=exec_horizon,
        rollout_epoch=8,
        total_num_envs=64,
    )
    assert decisions == expected_decisions
    assert decisions % global_batch == 0


def test_episode_cost_normalization_uses_ceil_for_partial_final_chunk():
    assert contract.max_gate_decisions(max_episode_steps=200, exec_horizon=24) == 9
    costs = torch.ones(9)
    normalized = contract.normalize_episode_mode_cost(
        costs,
        max_episode_steps=200,
        exec_horizon=24,
    )
    assert normalized.sum().item() == pytest.approx(1.0)


def test_partial_final_chunk_executes_only_remaining_steps_and_pads_rollout():
    actions = torch.zeros(2, 24, 14)
    executed = contract.limit_gate_action_chunk_to_episode(
        actions,
        elapsed_steps=torch.tensor([192, 192]),
        max_episode_steps=200,
    )
    assert executed.shape == (2, 8, 14)

    rewards = contract.pad_gate_env_chunk(
        torch.ones(2, 8), target_horizon=24
    )
    assert rewards.shape == (2, 24)
    assert torch.equal(rewards[:, :8], torch.ones(2, 8))
    assert torch.equal(rewards[:, 8:], torch.zeros(2, 16))

    truncations = torch.zeros(2, 8, dtype=torch.bool)
    truncations[:, -1] = True
    padded_truncations = contract.pad_gate_env_chunk(
        truncations, target_horizon=24, move_final_flag=True
    )
    assert not padded_truncations[:, 7].any()
    assert padded_truncations[:, -1].all()


def test_episode_limiter_rejects_asynchronous_vector_env_clocks():
    with pytest.raises(RuntimeError, match="different remaining episode budgets"):
        contract.limit_gate_action_chunk_to_episode(
            torch.zeros(2, 24, 14),
            elapsed_steps=torch.tensor([192, 180]),
            max_episode_steps=200,
        )


def test_eval_active_mask_excludes_post_success_decisions_and_resets():
    active = torch.tensor([True, True, True])
    active = contract.update_gate_eval_active_mask(
        active,
        success_once=torch.tensor([True, False, False]),
        dones=torch.tensor([[False], [False], [True]]),
    )
    assert torch.equal(active, torch.tensor([False, True, False]))

    done_stays_inactive = contract.update_gate_eval_active_mask(
        active,
        success_once=torch.tensor([True, False, False]),
        dones=torch.tensor([[True], [False], [True]]),
    )
    assert torch.equal(done_stays_inactive, torch.tensor([False, True, False]))


def test_gate_configs_encode_verified_horizon_and_camera_contracts():
    config_dir = RLINF_ROOT / "examples" / "embodiment" / "config"
    libero = yaml.safe_load((config_dir / "libero_10_grpo_gate.yaml").read_text())
    robotwin = yaml.safe_load((config_dir / "robotwin_grpo_gate.yaml").read_text())

    assert libero["actor"]["model"]["wam"]["generation_horizon"] == 32
    assert libero["actor"]["model"]["wam"]["exec_horizon"] == 10
    assert libero["actor"]["global_batch_size"] == 2048
    assert libero["env"]["eval"]["total_num_envs"] == 496
    assert libero["env"]["eval"]["max_episode_steps"] == 700
    assert libero["env"]["eval"]["max_steps_per_rollout_epoch"] == 700

    for split in ("train", "eval"):
        env = robotwin["env"][split]
        assert env["center_crop"] is False
        assert env["max_episode_steps"] == 200
        assert env["max_steps_per_rollout_epoch"] == 216
        assert env["task_config"]["camera"]["collect_wrist_camera"] is True
    assert robotwin["actor"]["model"]["wam"]["generation_horizon"] == 32
    assert robotwin["actor"]["model"]["wam"]["exec_horizon"] == 24
    assert robotwin["actor"]["micro_batch_size"] == 64
    assert robotwin["actor"]["global_batch_size"] == 512
    assert robotwin["env"]["train"]["use_fixed_reset_state_ids"] is False
    assert robotwin["env"]["eval"]["total_num_envs"] == 96
    assert str(robotwin["env"]["train"]["seeds_path"]).endswith(
        "/seeds/train_seeds.json"
    )
    assert str(robotwin["env"]["eval"]["seeds_path"]).endswith(
        "/seeds/eval_seeds.json"
    )


def test_partial_execution_chunk_is_rejected():
    with pytest.raises(ValueError, match="partial action chunk"):
        contract.rollout_decision_count(
            max_env_steps=481,
            exec_horizon=10,
            rollout_epoch=8,
            total_num_envs=64,
        )


def test_gate_distributed_contract_preserves_whole_grpo_groups():
    contract.validate_gate_distribution_contract(
        algorithm_group_size=8,
        env_group_size=8,
        train_total_envs=64,
        eval_total_envs=496,
        env_world_size=8,
        rollout_world_size=8,
        actor_world_size=8,
        stage_num=1,
        actor_split_num=1,
        use_training_pipeline=False,
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"env_group_size": 4}, "env.train.group_size"),
        ({"train_total_envs": 60}, "env.train.total_num_envs"),
        ({"eval_total_envs": 500}, "env.eval.total_num_envs"),
        (
            {
                "train_total_envs": 64,
                "env_world_size": 4,
                "actor_world_size": 8,
                "actor_split_num": 4,
            },
            "trajectory env batch",
        ),
    ],
)
def test_gate_distributed_contract_rejects_split_groups(override, match):
    kwargs = dict(
        algorithm_group_size=8,
        env_group_size=8,
        train_total_envs=64,
        eval_total_envs=496,
        env_world_size=8,
        rollout_world_size=8,
        actor_world_size=8,
        stage_num=1,
        actor_split_num=1,
        use_training_pipeline=False,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        contract.validate_gate_distribution_contract(**kwargs)
