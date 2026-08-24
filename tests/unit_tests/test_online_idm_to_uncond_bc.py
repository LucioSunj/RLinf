# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts for the opt-in online IDM-to-UNCOND BC extension."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicy,
    FastWAMAdaptivePolicyConfig,
    FastWAMChunkSample,
)
from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord, WAMRoute
from rlinf.models.embodiment.wam_policy.libero_runtime import LiberoFastWAMRuntime
from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import (
    assemble_online_idm_bc_loss,
    audit_online_idm_bc_gradient_ownership,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.config import (
    ONLINE_IDM_BC_ACTOR_TARGET,
    ONLINE_IDM_BC_FLOW_VALID,
    ONLINE_IDM_BC_POLICY_TARGET,
    ONLINE_IDM_BC_RUNTIME_TARGET,
    ONLINE_IDM_BC_SAMPLE_IDENTITIES,
    ONLINE_IDM_BC_TEACHER_ACTIONS,
    ONLINE_IDM_BC_TEACHER_BYTES,
    ONLINE_IDM_BC_TEACHER_PRESENT,
    ONLINE_IDM_BC_TEACHER_SECONDS,
    OnlineIDMBCConfig,
    validate_online_idm_bc_training_config,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.policy import (
    OnlineIDMBCFastWAMPolicy,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.runtime import (
    OnlineIDMTeacherLiberoRuntime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "examples" / "embodiment" / "config"


class _TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.lora = nn.Parameter(torch.tensor(0.5))


class _TinyLoRAAdapter:
    def __init__(self, actor: _TinyActor) -> None:
        self.actor = actor

    def lora_parameters(self):
        return iter((self.actor.lora,))


class _FlowScheduler:
    num_train_timesteps = 1000
    shift = 1.0

    @staticmethod
    def _phi(value, _shift):
        return value

    @staticmethod
    def add_noise(action, noise, _timestep):
        return action + noise

    @staticmethod
    def training_target(action, noise, _timestep):
        return action - noise

    @staticmethod
    def training_weight(timestep):
        return torch.ones_like(timestep)


def _set_asset_environment(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT_SHA256", "b" * 64)


def _route_record(routes: torch.Tensor) -> ChunkRouteRecord:
    batch_size = int(routes.numel())
    return ChunkRouteRecord(
        route_used=routes,
        route_was_forced=torch.ones(batch_size, dtype=torch.bool),
        chunk_ids=torch.arange(batch_size),
        episode_ids=torch.arange(batch_size),
        route_source_chunk_ids=torch.full((batch_size,), -1, dtype=torch.long),
        actor_versions=torch.zeros(batch_size, dtype=torch.long),
    )


def _loss_outputs(*, loss_sum: torch.Tensor, selected_count: float) -> dict:
    return {
        "online_idm_bc_loss_sum": loss_sum,
        "online_idm_bc_raw_loss": loss_sum.detach(),
        "online_idm_bc_selected_count": torch.tensor(selected_count),
        "online_idm_bc_expected_count": torch.tensor(selected_count),
        "online_idm_bc_present_count": torch.tensor(selected_count),
        "online_idm_bc_mse_per_dimension": torch.arange(7, dtype=torch.float32),
        "online_idm_bc_mse_pose": torch.tensor(2.5),
        "online_idm_bc_mse_gripper": torch.tensor(6.0),
        "online_idm_bc_mse_by_timestep_bin": torch.arange(10, dtype=torch.float32),
        "online_idm_bc_timestep_bin_count": torch.tensor(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ),
        "online_idm_bc_valid_action_count": torch.tensor(32),
        "online_idm_bc_full_action_mse": torch.tensor(3.0),
        "online_idm_bc_executed_prefix_mse": torch.tensor(4.0),
        "online_idm_bc_teacher_seconds_sum": torch.tensor(0.5),
        "online_idm_bc_teacher_bytes_sum": torch.tensor(448),
    }


def test_enabled_and_control_configs_are_exact_additive_variants(monkeypatch) -> None:
    _set_asset_environment(monkeypatch)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        enabled = compose(
            config_name="libero_10_ppo_fastwam_adaptive_formal",
            overrides=["+online_idm_bc=enabled"],
        )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        control = compose(
            config_name="libero_10_ppo_fastwam_adaptive_formal",
            overrides=["+online_idm_bc=control"],
        )

    config = validate_online_idm_bc_training_config(enabled)
    assert config == OnlineIDMBCConfig(enabled=True, loss_weight=0.2)
    assert enabled.online_idm_bc_implementation.actor_target == (
        ONLINE_IDM_BC_ACTOR_TARGET
    )
    assert enabled.online_idm_bc_implementation.policy_target == (
        ONLINE_IDM_BC_POLICY_TARGET
    )
    assert enabled.actor.model.runtime._target_ == ONLINE_IDM_BC_RUNTIME_TARGET
    assert enabled.rollout.model.runtime._target_ == ONLINE_IDM_BC_RUNTIME_TARGET
    assert control.actor.model.runtime._target_.endswith("LiberoFastWAMRuntime")
    assert control.algorithm.uncond_idm_bc.enabled is False
    assert control.algorithm.uncond_idm_bc.loss_weight == 0.0

    enabled_payload = OmegaConf.to_container(enabled, resolve=True)
    control_payload = OmegaConf.to_container(control, resolve=True)
    enabled_payload.pop("online_idm_bc_implementation")
    for payload in (enabled_payload, control_payload):
        payload["algorithm"].pop("uncond_idm_bc")
        for owner in ("actor", "rollout"):
            payload[owner]["model"]["runtime"].pop("_target_")
    assert enabled_payload == control_payload


def test_online_config_validator_rejects_a_wrong_runtime(monkeypatch) -> None:
    _set_asset_environment(monkeypatch)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        config = compose(
            config_name="libero_10_ppo_fastwam_adaptive_formal",
            overrides=["+online_idm_bc=enabled"],
        )
    invalid = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    invalid.rollout.model.runtime._target_ = (
        "rlinf.models.embodiment.wam_policy.libero_runtime.LiberoFastWAMRuntime"
    )
    with pytest.raises(ValueError, match="rollout runtime"):
        validate_online_idm_bc_training_config(invalid)


def test_rollout_teacher_targets_only_uncond_and_reuses_student_noise(
    monkeypatch,
) -> None:
    actor = _TinyActor()
    runtime = OnlineIDMTeacherLiberoRuntime(
        actor=actor,
        lora_adapter=_TinyLoRAAdapter(actor),
        processor=None,
        generation_horizon=32,
        execution_horizon=10,
    )
    batch_size = 3
    chains = torch.arange(
        batch_size * 2 * 32 * 7,
        dtype=torch.float32,
    ).reshape(batch_size, 2, 32, 7)

    def base_sample(_self, **_kwargs):
        return FastWAMChunkSample(
            actions=torch.zeros(batch_size, 10, 7),
            old_flow_logprobs=torch.zeros(batch_size, 10, 7),
            flow_chains=chains,
            denoise_indices=torch.zeros(batch_size, dtype=torch.long),
            gate_snapshots=(SimpleNamespace(batch_size=batch_size),),
            forward_inputs={
                "fastwam_images": torch.zeros(batch_size, 3, 4, 4),
                "fastwam_context": torch.zeros(batch_size, 2, 4),
                "fastwam_context_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            },
        )

    calls = []

    def teacher(_self, **kwargs):
        calls.append(
            (
                kwargs["initial_action_noise"].clone(),
                kwargs["idm_noise_seed"],
            )
        )
        return torch.full_like(kwargs["initial_action_noise"], 0.25)

    monkeypatch.setattr(LiberoFastWAMRuntime, "sample_action_batch", base_sample)
    runtime._sample_idm_teacher_action = MethodType(teacher, runtime)
    routes = torch.tensor([int(WAMRoute.UNCOND), int(WAMRoute.IDM), 0])
    env_obs = {
        "_fastwam_action_noise_seeds": torch.tensor([101, 102, 103]),
        "_fastwam_idm_noise_seeds": torch.tensor([201, 202, 203]),
    }
    sample = runtime.sample_action_batch(
        env_obs=env_obs,
        routes=routes,
        mode="train",
        actor_version=4,
    )

    assert len(calls) == 2
    assert torch.equal(calls[0][0], chains[0:1, 0])
    assert torch.equal(calls[1][0], chains[2:3, 0])
    assert [item[1] for item in calls] == [201, 203]
    assert sample.forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS].dtype is torch.bfloat16
    assert sample.forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS].shape == (3, 32, 7)
    assert sample.forward_inputs[ONLINE_IDM_BC_TEACHER_PRESENT].tolist() == [
        True,
        False,
        True,
    ]
    assert sample.forward_inputs[ONLINE_IDM_BC_SAMPLE_IDENTITIES].tolist() == [
        101,
        102,
        103,
    ]
    assert sample.forward_inputs[ONLINE_IDM_BC_TEACHER_BYTES].tolist() == [
        448,
        0,
        448,
    ]

    calls.clear()
    evaluation = runtime.sample_action_batch(
        env_obs=env_obs,
        routes=routes,
        mode="eval",
        actor_version=4,
        collect_replay=False,
    )
    assert calls == []
    assert ONLINE_IDM_BC_TEACHER_ACTIONS not in evaluation.forward_inputs


def test_actor_replay_masks_bc_and_keeps_gradient_on_lora_only() -> None:
    actor = _TinyActor()
    actor.train_action_scheduler = _FlowScheduler()
    runtime = OnlineIDMTeacherLiberoRuntime(
        actor=actor,
        lora_adapter=_TinyLoRAAdapter(actor),
        processor=None,
        generation_horizon=32,
        execution_horizon=10,
    )

    def prepare(_self, **_kwargs):
        return SimpleNamespace(), None

    def velocity(_self, _condition, **_kwargs):
        def run(action, _timestep):
            return SimpleNamespace(velocity=action * actor.lora)

        return run

    runtime._prepare_action_condition = MethodType(prepare, runtime)
    runtime._velocity = MethodType(velocity, runtime)
    routes = torch.tensor([int(WAMRoute.UNCOND), int(WAMRoute.IDM), 0])
    forward_inputs = {
        ONLINE_IDM_BC_FLOW_VALID: torch.tensor([True, True, False]),
        ONLINE_IDM_BC_TEACHER_ACTIONS: torch.full(
            (3, 32, 7),
            0.25,
            dtype=torch.bfloat16,
        ),
        ONLINE_IDM_BC_TEACHER_PRESENT: torch.tensor([True, False, True]),
        ONLINE_IDM_BC_SAMPLE_IDENTITIES: torch.tensor([11, 12, 13]),
        ONLINE_IDM_BC_TEACHER_SECONDS: torch.tensor([0.1, 0.0, 0.2]),
        ONLINE_IDM_BC_TEACHER_BYTES: torch.tensor([448, 0, 448]),
        "flow_chains": torch.full((3, 2, 32, 7), 0.5),
        "fastwam_images": torch.zeros(3, 3, 4, 4),
        "fastwam_context": torch.zeros(3, 2, 4),
        "fastwam_context_mask": torch.ones(3, 2, dtype=torch.bool),
    }
    result = runtime.compute_online_idm_bc_loss(
        forward_inputs=forward_inputs,
        route_info=_route_record(routes),
    )
    assert result.selected_count.item() == 1
    assert result.expected_count.item() == 1
    assert result.present_count.item() == 2
    assert result.valid_action_count.item() == 32
    result.loss_sum.backward()
    assert actor.lora.grad is not None
    assert torch.isfinite(actor.lora.grad)
    assert actor.lora.grad.item() != 0.0
    assert actor.base.grad is None

    missing_target = deepcopy(forward_inputs)
    missing_target[ONLINE_IDM_BC_TEACHER_PRESENT] = torch.tensor([False, False, True])
    with pytest.raises(RuntimeError, match="lack IDM teacher targets"):
        runtime.compute_online_idm_bc_loss(
            forward_inputs=missing_target,
            route_info=_route_record(routes),
        )


def test_loss_assembly_reuses_flow_scale_and_reports_conserved_counts() -> None:
    current_loss = torch.tensor(3.0, requires_grad=True)
    loss_sum = torch.tensor(4.0, requires_grad=True)
    total, metrics = assemble_online_idm_bc_loss(
        current_loss=current_loss,
        output_dict=_loss_outputs(loss_sum=loss_sum, selected_count=1.0),
        config=OnlineIDMBCConfig(enabled=True, loss_weight=0.2),
        selected_loss_scale=2.0,
        metric_scale_numerator=4.0,
        flow_metric_loss=2.0,
    )
    assert total.item() == pytest.approx(4.6)
    assert metrics["online_idm_bc/raw_loss"] == pytest.approx(8.0)
    assert metrics["online_idm_bc/weighted_loss"] == pytest.approx(1.6)
    assert metrics["online_idm_bc/globally_normalized_count"] == 2.0
    assert metrics["online_idm_bc/teacher_call_count"] == 4.0
    assert metrics["online_idm_bc/transported_bytes"] == 1792.0
    total.backward()
    assert current_loss.grad.item() == 1.0
    assert loss_sum.grad.item() == pytest.approx(0.4)


def test_policy_rewrap_preserves_state_and_bc_gradient_ownership() -> None:
    actor = _TinyActor()
    adapter = _TinyLoRAAdapter(actor)
    runtime = OnlineIDMTeacherLiberoRuntime(
        actor=actor,
        lora_adapter=adapter,
        processor=None,
    )
    base = FastWAMAdaptivePolicy(
        actor=actor,
        runtime=runtime,
        lora_adapter=adapter,
        gate=nn.Linear(1, 1),
        critic=nn.Linear(1, 1),
        config=FastWAMAdaptivePolicyConfig(),
    )
    base.set_global_step(3)
    names_before = tuple(base.state_dict())
    route_tracker = base.route_tracker
    wrapped = OnlineIDMBCFastWAMPolicy.from_base_policy(
        base,
        config=OnlineIDMBCConfig(enabled=True, loss_weight=0.2),
    )
    assert tuple(wrapped.state_dict()) == names_before
    assert wrapped.route_tracker is route_tracker
    assert wrapped.actor_version == 3

    metrics = audit_online_idm_bc_gradient_ownership(
        bc_loss=(wrapped.actor.lora * 2.0).square(),
        policy=wrapped,
    )
    assert metrics["online_idm_bc/gradient_audit_pass"] == 1.0
    assert wrapped.gate.weight.grad is None
    assert wrapped.critic.weight.grad is None
    assert wrapped.actor.base.requires_grad is False

    escaped_loss = (wrapped.actor.lora + wrapped.gate.weight.sum()).square()
    with pytest.raises(RuntimeError, match="escaped"):
        audit_online_idm_bc_gradient_ownership(
            bc_loss=escaped_loss,
            policy=wrapped,
        )
