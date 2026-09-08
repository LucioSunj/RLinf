# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_route_neutral_online import _compose

from rlinf.envs.libero.task_sampler import BalancedLiberoTaskSampler
from rlinf.models.embodiment.wam_policy.route_neutral_online.config import (
    validate_route_neutral_online_idm_bc_training_config,
    validate_shared_gpu_device_plan,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.task_metrics import (
    summarize_task_rollout,
)
from rlinf.utils.nested_dict_process import map_nested_tensors
from rlinf.workers.actor.fsdp_actor_worker import flatten_nested_tensor_time_batch


@pytest.mark.parametrize("n", [42, 49, 56, 63, 70, 84, 126])
def test_balanced_quotas_and_rank_assignment(n):
    sampler = BalancedLiberoTaskSampler(total_envs=n, reset_pool_sizes=[50] * 10)
    allocations = Counter()
    for step in range(sampler.cycle_length * 3):
        plan = sampler.next_plan(step)
        assert max(plan["quotas"]) - min(plan["quotas"]) <= 1
        assert sum(plan["quotas"]) == n
        assert [len(slots) for slots in plan["ranks"]] == [n // 7] * 7
        slots = [slot for rank in plan["ranks"] for slot in rank]
        counts = Counter(slot["task_id"] for slot in slots)
        assert [counts[i] for i in range(10)] == plan["quotas"]
        assert len({slot["episode_slot_id"] for slot in slots}) == n
        assert all(
            slot["reset_state_id"] == 50 * slot["task_id"] + slot["trial_id"]
            for slot in slots
        )
        allocations.update(counts)
        if (step + 1) % sampler.cycle_length == 0:
            assert len(set(allocations.values())) == 1


@pytest.mark.parametrize("saved_step", [1, 4, 5, 9, 10, 11])
def test_resume_continues_plan_and_both_rng_streams(saved_step):
    sampler = BalancedLiberoTaskSampler(
        total_envs=42, reset_pool_sizes=list(range(40, 50))
    )
    for step in range(saved_step):
        sampler.next_plan(step)
    restored = BalancedLiberoTaskSampler(
        total_envs=42, reset_pool_sizes=list(range(40, 50))
    )
    restored.load_state_dict(
        json.loads(json.dumps(sampler.state_dict())), runner_step=saved_step
    )
    for step in range(saved_step, saved_step + 12):
        assert sampler.next_plan(step) == restored.next_plan(step)
    with pytest.raises(ValueError, match="expects runner step"):
        restored.next_plan(0)


def test_capacity_candidates_share_common_task_reset_ordinals():
    a = BalancedLiberoTaskSampler(total_envs=42, reset_pool_sizes=[50] * 10)
    b = BalancedLiberoTaskSampler(total_envs=84, reset_pool_sizes=[50] * 10)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    for step in range(15):
        maps = []
        for sampler in (a, b):
            slots = [slot for rank in sampler.next_plan(step)["ranks"] for slot in rank]
            maps.append(
                {
                    (slot["task_id"], slot["task_episode_ordinal"]): slot["trial_id"]
                    for slot in slots
                }
            )
        assert all(maps[1][key] == value for key, value in maps[0].items())
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


@pytest.mark.parametrize("n", [42, 84, 126])
def test_legal_balanced_configuration_preserves_k15(monkeypatch, n):
    cfg = _compose(
        monkeypatch,
        "libero_10_ppo_fastwam_route_neutral_online_all",
        [f"env.train.total_num_envs={n}", f"actor.global_batch_size={196 * n // 42}"],
    )
    validate_route_neutral_online_idm_bc_training_config(cfg)
    assert 70 * n // cfg.actor.global_batch_size == 15
    assert cfg.actor.model.route_neutral_online.critic_warmup.runner_updates == 10
    assert cfg.rollout.model.route_neutral_online.critic_warmup.runner_updates == 10
    assert cfg.algorithm.fixed_branch_cost.controller.critic_warmup.runner_updates == 10
    assert cfg.actor.model.decision_telemetry_enabled is False


@pytest.mark.parametrize(
    "override",
    [
        "actor.global_batch_size=294",
        "env.train.total_num_envs=56",
        "cluster.component_placement.rollout=0-7",
        "env.train.task_id_filter=[8]",
        "env.train.auto_reset=true",
        "actor.micro_batch_size=1",
    ],
)
def test_rejects_unvalidated_balanced_geometry(monkeypatch, override):
    cfg = _compose(
        monkeypatch, "libero_10_ppo_fastwam_route_neutral_online_all", [override]
    )
    with pytest.raises(ValueError):
        validate_route_neutral_online_idm_bc_training_config(cfg)


def _dedicated_balanced_config(monkeypatch):
    return _compose(
        monkeypatch,
        "libero_10_ppo_fastwam_route_neutral_online_all",
        [
            "cluster.component_placement.env=1-7",
            "cluster.component_placement.rollout=1-7",
            "route_neutral_online_implementation.shared_gpu_rollout_rank=null",
            "actor.enable_offload=false",
        ],
    )


def test_dedicated_eight_gpu_balanced_configuration_preserves_scientific_geometry(
    monkeypatch,
):
    cfg = _dedicated_balanced_config(monkeypatch)
    validate_route_neutral_online_idm_bc_training_config(cfg)
    assert cfg.env.train.total_num_envs == 42
    assert cfg.actor.global_batch_size == 196
    assert cfg.actor.micro_batch_size == 4
    assert cfg.actor.enable_offload is False
    assert cfg.rollout.enable_offload is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor.enable_offload", True),
        ("rollout.enable_offload", True),
        ("route_neutral_online_implementation.shared_gpu_rollout_rank", 0),
        ("runner.overlap_env_bootstrap", True),
        ("env.train.total_num_envs", 84),
    ],
)
def test_dedicated_balanced_configuration_rejects_incompatible_lifecycle(
    monkeypatch, field, value
):
    from omegaconf import OmegaConf

    cfg = _dedicated_balanced_config(monkeypatch)
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError):
        validate_route_neutral_online_idm_bc_training_config(cfg)


@pytest.mark.parametrize("wrong_physical_plan", [False, True])
def test_dedicated_balanced_device_plan_preserves_physical_indices(
    monkeypatch, wrong_physical_plan
):
    cfg = _dedicated_balanced_config(monkeypatch)
    devices = {"actor": [0], "env": list(range(1, 8)), "rollout": list(range(1, 8))}
    if wrong_physical_plan:
        devices["rollout"] = list(range(7))

    def strategy(component):
        return SimpleNamespace(
            get_placement=lambda cluster: [
                SimpleNamespace(
                    rank=rank, cluster_node_rank=0, visible_accelerators=[str(device)]
                )
                for rank, device in enumerate(devices[component])
            ]
        )

    placement = SimpleNamespace(get_strategy=strategy)
    if wrong_physical_plan:
        with pytest.raises(ValueError, match="physical rollout placement"):
            validate_shared_gpu_device_plan(cfg, None, placement)
    else:
        report = validate_shared_gpu_device_plan(cfg, None, placement)
        assert report["schema"] == "route-neutral-dedicated-gpu-device-plan-v1"
        assert report["rollout"][0] == (0, 0, ["1"])
        assert report["rollout"][-1] == (6, 0, ["7"])


def test_task_identity_stays_with_language_and_trajectory_after_flatten_shuffle():
    task_ids = torch.arange(10).repeat(3, 1)
    batch = {
        "forward_inputs": {
            "multitask_task_id": task_ids,
            "multitask_episode_slot_id": task_ids + 42,
            "fastwam_context": task_ids[..., None, None].float().expand(-1, -1, 2, 3),
        },
        "trajectory": task_ids * 100 + torch.arange(3)[:, None],
    }
    flattened = flatten_nested_tensor_time_batch(batch)
    order = torch.randperm(30, generator=torch.Generator().manual_seed(8))
    shuffled = map_nested_tensors(flattened, lambda value: value[order])
    tasks = shuffled["forward_inputs"]["multitask_task_id"]
    assert torch.equal(tasks, shuffled["trajectory"] // 100)
    assert torch.equal(
        tasks, shuffled["forward_inputs"]["multitask_episode_slot_id"] - 42
    )
    assert torch.equal(
        tasks.float(), shuffled["forward_inputs"]["fastwam_context"][:, 0, 0]
    )


def test_task_rollout_uses_valid_masks_and_original_episode_counts():
    tasks = torch.arange(10).repeat(3, 1)
    valid = torch.ones(3, 10, 1, dtype=torch.bool)
    valid[2, :5] = False
    routes = (tasks % 2).long()
    forward = {
        "multitask_task_id": tasks,
        "multitask_episode_slot_id": tasks + 42,
        "multitask_episode_success": (tasks % 2 == 0),
        "multitask_episode_failed": torch.zeros_like(tasks, dtype=torch.bool),
        "multitask_episode_truncated": (tasks % 2 == 1),
        "online_idm_bc_teacher_present": routes == 0,
    }
    batch = {
        "forward_inputs": forward,
        "route_info": SimpleNamespace(route_used=routes),
        "emitted_gate": SimpleNamespace(behavior_probability=torch.full((3, 10), 0.5)),
        "loss_mask": valid,
        "gate_valid_mask": valid[..., 0],
        "returns": torch.ones(3, 10, 1),
        "prev_values": torch.zeros(4, 10, 1),
    }
    metrics = summarize_task_rollout(batch)
    assert metrics["task_global/episode_slots"] == 10
    assert metrics["task_global/valid_chunks"] == 25
    assert metrics["task/0/valid_chunks"] == 2
    assert metrics["task/9/valid_chunks"] == 3
    assert metrics["task_macro/training_success"] == 0.5
    assert metrics["task/0/critic_preupdate_mse"] == 1


def test_metadata_never_enters_gate_features_or_changes_logits():
    from test_route_neutral_online import _RecordingReplayGate, _replay_gate_features

    from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
        serialize_route_neutral_features,
    )
    from rlinf.models.embodiment.wam_policy.route_neutral_online.policy import (
        RouteNeutralOnlineIDMBCFastWAMPolicy,
    )

    policy = SimpleNamespace(
        gate=_RecordingReplayGate(),
        runtime=SimpleNamespace(
            route_neutral_visual=SimpleNamespace(layer_indices=[14])
        ),
    )
    inputs = serialize_route_neutral_features(_replay_gate_features(4))
    features = RouteNeutralOnlineIDMBCFastWAMPolicy._gate_features_from_forward_inputs(
        policy, inputs
    )
    before = policy.gate(features)
    inputs.update(
        {
            "multitask_task_id": torch.tensor([9, 2, 4, 7]),
            "multitask_episode_slot_id": torch.arange(4),
        }
    )
    features = RouteNeutralOnlineIDMBCFastWAMPolicy._gate_features_from_forward_inputs(
        policy, inputs
    )
    after = policy.gate(features)
    assert torch.equal(before, after)
    assert not any(key.startswith("multitask") for key in policy.gate.rows[-1])


def test_requested_base_entropy_regularizer_has_base_entropy_gradient():
    from rlinf.algorithms.fastwam_dual_ppo import compute_gate_ppo_loss
    from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
        base_entropy_loss_correction,
    )

    logits = torch.tensor([-2.0, 1.5, 0.8], requires_grad=True)
    p = logits.sigmoid()
    q = 0.9 * p + 0.05
    valid = torch.tensor([True, False, True])
    legacy, _ = compute_gate_ppo_loss(
        logprobs=q.log(),
        old_logprobs=q.detach().log(),
        advantages=torch.zeros(3),
        valid_mask=valid,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        base_probabilities=p,
        behavior_probabilities=q,
        entropy_coefficient=0.01,
        selected_loss_scale=0.25,
    )
    corrected = legacy + base_entropy_loss_correction(
        base=p,
        behavior=q,
        valid=valid,
        coefficient=0.01,
        selected_loss_scale=0.25,
    )
    expected = (
        -0.01 * torch.distributions.Bernoulli(probs=p).entropy()[valid].sum() * 0.25
    )
    torch.testing.assert_close(corrected, expected)
    torch.testing.assert_close(
        torch.autograd.grad(corrected, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_empty_owner_skips_existing_adam_momentum(tmp_path):
    from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
        RouteNeutralOnlineIDMBCFSDPActor,
    )

    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    names = ("gate", "uncond_lora", "value_head")
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in names]
    actor.optimizer = torch.optim.Adam(
        [{"name": name, "params": [value]} for name, value in zip(names, parameters)],
        lr=0.01,
    )
    for value in parameters:
        value.grad = torch.ones_like(value)
    actor.optimizer.step()
    previous_lora = parameters[1].detach().clone()
    for value in parameters:
        value.grad = torch.zeros_like(value)
    actor.optimizer_steps = 1
    actor.version = 10
    actor.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    actor.model = None
    actor._strategy = SimpleNamespace(clip_grad_norm_=lambda **kwargs: 0.0)
    actor._task_owner_counts = {"gate": 3, "uncond_lora": 0, "value_head": 3}
    actor._route_neutral_warmup_active = False
    actor._fastwam_update_resolution_checked = True
    actor._online_idm_bc_gradient_audit_complete = True
    actor.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            logger=SimpleNamespace(log_path=str(tmp_path), experiment_name="run")
        )
    )
    actor.optimizer_step()
    assert torch.equal(parameters[1], previous_lora)
    assert [actor.optimizer.state[p]["step"].item() for p in parameters] == [2, 1, 2]
    record = json.loads((tmp_path / "run/audits/owner_steps.jsonl").read_text())
    assert record["owners"][1]["skip_reason"] == "no_effective_samples"
    assert record["owners"][1]["step_before"] == record["owners"][1]["step_after"]
