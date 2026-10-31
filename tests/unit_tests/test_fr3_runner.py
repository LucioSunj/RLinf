# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real tiny parent → dual BC → mixed RL → native resume → deploy chain."""

# Imported pytest fixtures are intentionally named by test function parameters.
# ruff: noqa: F811

import copy
import importlib.util
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastwam.adapters import RegimeLoRAConfig, sha256_file
from fastwam.uncond_bc import FastWAMUncondBCConfig, FastWAMUncondBCPolicy
from fastwam.uncond_bc_checkpoint import save_uncond_bc_sidecar
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from test_fr3_mixed_policy import ROOT, _observation, policy  # noqa: F401

from rlinf.models.embodiment.wam_policy import _load_strict_fastwam_parent
from rlinf.models.embodiment.wam_policy.fr3_inference import (
    Fr3InferenceSession,
    load_deployment_state,
)
from rlinf.models.embodiment.wam_policy.fr3_runner import Fr3SynchronousRunner


class MeasuredRobot:
    """Hardware edge only: deterministic measurements and exact prefix receipts."""

    def __init__(self, mixed, *, interrupt=False):
        self.prompt = mixed.runtime.task_prompt
        self.task = mixed.runtime.task_name
        self.task_id = mixed.runtime.task_id
        self.horizon = mixed.runtime.action_protocol.generation_horizon
        self.episode_id = None
        self.episodes = []
        self.versions = []
        self.interrupt = interrupt
        self.stopped = False

    def status(self):
        return {
            "scene_ready": True,
            "state": "ready",
            "action_dt": 0.1,
            "execution_horizon": self.horizon,
            "max_episode_actions": self.horizon + 5,
            "max_episode_seconds": 120.0,
        }

    def observation(self):
        obs = _observation(self.prompt, batch_size=1)
        obs["states"][:, 0] += self.chunk * 0.01
        return {
            key: value for key, value in obs.items() if not key.startswith("_fastwam_")
        }

    def reset(self, *, options):
        assert not options.get("scene_ready", False)
        self.episode_id = options["episode_id"]
        self.version, self.chunk = options["policy_version"], 0
        self.episodes.append(self.episode_id)
        return self.observation(), {}

    def step_chunk(self, actions, *, policy_version, chunk_id, route, eligible, forced):
        assert actions.shape == (1, self.horizon, 8)
        assert chunk_id == self.chunk and policy_version == self.version
        self.versions.append(policy_version)
        if self.interrupt and self.chunk == 1:
            raise ConnectionError("lost physical receipt; never retry")
        done = self.chunk == 1
        count = 5 if done else self.horizon
        info = {
            "episode_id": self.episode_id,
            "chunk_id": self.chunk,
            "policy_version": self.version,
            "execution_mask": [True] * count + [False] * (self.horizon - count),
            "chunk_valid": True,
            "gate_valid": eligible and not forced,
            "final_observation_valid": True,
            "bootstrap_allowed": done,
            "terminated": False,
            "truncated": done,
            "reward_env": 0.0,
            "stop_reason": "timeout" if done else None,
        }
        self.chunk += 1
        return self.observation(), 0.0, False, done, info

    def stop(self):
        self.stopped = True

    def close(self):
        self.stopped = True


def _native_to_bc(mixed, path):
    horizon = mixed.runtime.action_protocol.generation_horizon
    spec = importlib.util.spec_from_file_location(
        "runner_bc_helpers", ROOT / "tests/test_fr3_bc_pipeline.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    from export_easywam_fastwam_parent import export_parent

    native = helpers._tiny_model(native=True)
    source, parent = path / "native.pt", path / "parent.pt"
    native.save_checkpoint(source, step=1)
    export_parent(checkpoint=source, output=parent)
    actor = helpers._tiny_model()
    _load_strict_fastwam_parent(actor, str(parent))
    actor.vae = mixed.actor.vae
    bc = FastWAMUncondBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(rank=128, alpha=128),
        video_lora_config=RegimeLoRAConfig(rank=128, alpha=128),
        config=FastWAMUncondBCConfig(
            action_horizon=horizon,
            action_dim=8,
            proprio_dim=8,
            gripper_dimension=7,
            expected_video_frames=1,
            expected_video_height=224,
            expected_video_width=448,
        ),
    )
    observation = _observation(mixed.runtime.task_prompt, batch_size=2)
    prepared = mixed.runtime.fr3_processor.prepare_batch(
        mixed.runtime._rows(observation)
    )
    batch = {
        "video": prepared["input_image"][:, :, None],
        "action": torch.randn(2, horizon, 8),
        "proprio": prepared["proprio"],
        "context": torch.randn(2, 4, 8),
        "context_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        "action_is_pad": torch.tensor(
            [[False] * (horizon - 5) + [True] * 5, [False] * horizon]
        ),
    }
    optimizer = torch.optim.AdamW(bc.lora_parameters(), lr=1e-4)
    initial = {
        k: copy.deepcopy(v.lora_state_dict()) for k, v in bc.lora_adapters.items()
    }
    loss = bc(
        batch,
        noise=torch.randn_like(batch["action"]),
        timestep=torch.tensor([200.0, 700.0]),
    )["loss_action_bc"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for branch, adapter in bc.lora_adapters.items():
        assert any(
            not torch.equal(v, initial[branch][k])
            for k, v in adapter.lora_state_dict().items()
        )
    mixed.actor = mixed.runtime.actor = bc.actor
    mixed.lora_adapter = mixed.runtime.lora_adapter = bc.lora_adapter
    mixed.video_lora_adapter = mixed.runtime.video_lora_adapter = bc.video_lora_adapter
    mixed.runtime.clear_session()
    mixed.config = replace(mixed.config, formal_training_sampling_seed=7)
    sidecar = path / "bc_sidecar.pt"
    save_uncond_bc_sidecar(
        sidecar,
        adapter=bc.lora_adapter,
        video_adapter=bc.video_lora_adapter,
        parent_checkpoint_sha256=sha256_file(parent),
        extra_metadata={
            "bc_step": 1,
            "action_horizon": horizon,
            "bc_config_sha256": "b" * 64,
            "statistics_sha256": sha256_file(mixed.runtime.processor_stats_path),
        },
    )
    return parent, sidecar


def _config(mixed, path, parent, sidecar):
    horizon = mixed.runtime.action_protocol.generation_horizon
    with initialize_config_dir(
        config_dir=str(ROOT / "RLinf/examples/embodiment/config"), version_base="1.3"
    ):
        cfg = compose(config_name=f"realworld_fr3_fastwam_{mixed.runtime.task_name}")
    with open_dict(cfg):
        cfg.actor.seed = 7
        cfg.fr3.update(
            {
                "device": "cpu",
                "updates": 2,
                "episodes_per_update": 1,
                "warmup_updates": 1,
                "max_episode_actions": horizon + 5,
                "action_horizon": horizon,
                "max_episode_seconds": 120.0,
                "output_dir": str(path / "run"),
                "data_contract_confirmed": True,
                "ready": True,
                "statistics": mixed.runtime.processor_stats_path,
                "statistics_sha256": sha256_file(mixed.runtime.processor_stats_path),
                "training_config": str(
                    ROOT
                    / f"deployments/fr3_wam/config_assets/{mixed.runtime.task_name}/config.yaml"
                ),
                "parent_checkpoint": str(parent),
                "parent_checkpoint_sha256": sha256_file(parent),
            }
        )
        cfg.actor.model.update(
            {
                "precision": "fp32",
                "num_action_chunks": horizon,
                "actor_checkpoint": str(parent),
                "actor_checkpoint_sha256": sha256_file(parent),
                "model_path": str(parent),
            }
        )
        cfg.actor.model.runtime.update(
            {
                "generation_horizon": horizon,
                "execution_horizon": horizon,
                "num_video_frames": horizon // 4 + 1,
                "processor_stats_path": cfg.fr3.statistics,
                "training_config": cfg.fr3.training_config,
                "easywam_root": str(ROOT / "EasyWAM"),
                "task_name": cfg.fr3.task_name,
                "task_prompt": cfg.fr3.prompt,
            }
        )
        cfg.actor.model.critic = {"kind": "fastwam_current_frame_value"}
        cfg.actor.model.fastwam.action_dit_config.action_dim = 8
        cfg.env.train.task_id_filter = [cfg.fr3.task_id]
        cfg.env.train.execution_horizon = horizon
        cfg.runner.bootstrap_uncond_lora_sidecar = str(sidecar)
        cfg.runner.bootstrap_uncond_lora_sidecar_sha256 = sha256_file(sidecar)
        cfg.algorithm.fixed_branch_cost.task_budget = {
            "profile": "fr3",
            "suite": "fr3",
            "targets": {str(cfg.fr3.task_id): 0.37},
            "task_names": {str(cfg.fr3.task_id): cfg.fr3.task_name},
            "calibration": {
                "status": "COMPLETE",
                "parameters": {
                    "awe_threshold": 0.005,
                    "sail_radius_m": 0.04,
                    "sail_min_samples": 3,
                    "execution_horizon": horizon,
                },
                "source": "synthetic-test-calibration",
                "data_version": "synthetic-test-only",
            },
        }
    return cfg


def _parameters(mixed):
    return {k: p.detach().clone() for k, p in mixed.named_parameters()}


@pytest.mark.parametrize(
    "policy", ["drawer", "insert_tubes", ("insert_tubes", 32)], indirect=True
)
def test_real_parent_bc_warmup_joint_resume_and_deploy(policy, tmp_path):  # noqa: F811
    parent, sidecar = _native_to_bc(policy, tmp_path)
    cfg = _config(policy, tmp_path, parent, sidecar)
    env = MeasuredRobot(policy)
    runner = Fr3SynchronousRunner(cfg, policy=policy, env=env)
    if runner.action_horizon == 32:
        original = torch.load(sidecar, weights_only=True)
        wrong = copy.deepcopy(original)
        for branch in ("action", "video"):
            wrong[branch]["metadata"]["extra"]["action_horizon"] = 16
        torch.save(wrong, sidecar)
        cfg.runner.bootstrap_uncond_lora_sidecar_sha256 = sha256_file(sidecar)
        with pytest.raises(ValueError, match="action horizon differs"):
            runner.bootstrap_bc()
        torch.save(original, sidecar)
        cfg.runner.bootstrap_uncond_lora_sidecar_sha256 = sha256_file(sidecar)
    runner.bootstrap_bc()
    checkpoint0 = runner.save_checkpoint()
    assert checkpoint0.is_file()
    initial = _parameters(policy)
    warmup = runner.run_update()
    assert warmup["warmup"] == 1
    after_warmup = _parameters(policy)
    for key in initial:
        if "value_head" not in key:
            assert torch.equal(initial[key], after_warmup[key]), key
    assert any(
        not torch.equal(v, initial[k])
        for k, v in after_warmup.items()
        if "value_head" in k
    )
    assert all(
        not runner.optimizer.state[p]
        for name in ("gate", "uncond_lora")
        for p in runner.groups[name]
    )
    # Formal per-episode/chunk seed 7 samples both real branches in this update.
    joint = runner.run_update()
    assert joint["warmup"] == 0 and joint["uncond_chunks"] > 0
    assert joint["online_idm_bc/weighted_loss"] > 0
    assert joint["gate/ratio"] == pytest.approx(1.0, abs=1e-6)
    assert joint["uncond_flow/ratio"] == pytest.approx(1.0, abs=1e-6)
    assert joint["gate/log_ratio_max_abs"] < 1e-6
    assert joint["uncond_flow/log_ratio_max_abs"] < 1e-6
    after_joint = _parameters(policy)
    for pattern in ("gate.", ".lora_A", ".lora_B", "value_head"):
        assert any(
            not torch.equal(v, after_warmup[k])
            for k, v in after_joint.items()
            if pattern in k
        ), pattern
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None and torch.equal(parameter, initial[name])
    assert env.versions == [0, 0, 1, 1]
    rows = torch.load(runner.output / "rollout_000002.pt", weights_only=False)
    assert rows[0]["flow_execution_mask"].sum() == cfg.fr3.action_horizon
    assert rows[-1]["flow_execution_mask"].sum() == 5
    assert rows[-1]["forward_inputs"]["fr3_execution_mask"].sum() == 5
    audit = rows[0]["task_cost_audit"]
    assert audit["cost_identity_max_abs_error"] == 0
    assert audit["cost_tensor_max_abs_error"] == 0
    assert audit["actual_cost_sum"] == pytest.approx(audit["expected_cost_sum"])
    checkpoint = runner.save_checkpoint()
    source = torch.load(checkpoint, weights_only=False)
    expected_rng = (random.random(), np.random.random(), torch.rand(2))
    saved_controller = copy.deepcopy(runner.controller.state_dict())
    saved_optimizer = copy.deepcopy(runner.optimizer.state_dict())
    with torch.no_grad():
        for p in runner.groups["gate"]:
            p.add_(3)
    runner.load_checkpoint(checkpoint)
    assert random.random() == expected_rng[0] and np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(2), expected_rng[2], atol=0, rtol=0)
    assert runner.controller.state_dict() == saved_controller
    assert (
        runner.optimizer.state_dict()["param_groups"] == saved_optimizer["param_groups"]
    )
    for key, value in saved_optimizer["state"].items():
        for field, expected in value.items():
            torch.testing.assert_close(
                runner.optimizer.state_dict()["state"][key][field],
                expected,
                atol=0,
                rtol=0,
            )
    assert not policy.runtime.physical_history.state_dict()["states"]
    old_episode = env.episodes[-1]
    decision = runner.controller.decisions(runner.step)
    resumed_rows = runner.collect_episode(decision["tasks"][str(cfg.fr3.task_id)])
    assert env.episodes[-1] != old_episode
    assert resumed_rows[0]["route_info"].chunk_ids.item() == 0
    assert (
        resumed_rows[0]["route_info"].episode_ids.item()
        > rows[0]["route_info"].episode_ids.item()
    )
    from tools.fr3_deployment.exporting import deployment_payload

    deploy = deployment_payload(source, runner.fr3_contract, kind="rl")
    assert set(deploy["state"]) == {"gate", "lora", "video_lora"}
    for p in runner.groups["gate"]:
        p.data.zero_()
    load_deployment_state(policy, deploy, runner.fr3_contract)
    for name, value in policy.gate.state_dict().items():
        torch.testing.assert_close(
            value, source["policy"]["gate"][name], atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="contract differs"):
        deployment_payload(source, runner.fr3_contract | {"task_id": 9}, kind="rl")
    changed = copy.deepcopy(runner.contract)
    changed["fr3"]["max_episode_actions"] += 1
    runner.contract = changed
    with pytest.raises(ValueError, match="contract"):
        runner.load_checkpoint(checkpoint)
    policy.critic = None
    session = Fr3InferenceSession(
        policy, runner.fr3_contract, mode="rl_mixed", artifact_kind="rl"
    )
    observation = _observation(policy.runtime.task_prompt, batch_size=1)
    observation = {
        key: value
        for key, value in observation.items()
        if not key.startswith("_fastwam_")
    }
    observation.update(
        task=policy.runtime.task_prompt,
        episode_id="new-deployment-episode",
        chunk_id=0,
        policy_version=source["step"],
    )
    inference = session.infer(observation)
    assert inference["actions"].shape == (cfg.fr3.action_horizon, 8)
    assert inference["policy_version"] == source["step"]
    assert inference["route"] in {"idm", "uncond"}
    assert "forward_inputs" not in inference and policy.critic is None


def test_transport_failure_seals_pre_action_fragment_without_retry(policy, tmp_path):  # noqa: F811
    parent, sidecar = _native_to_bc(policy, tmp_path)
    cfg = _config(policy, tmp_path, parent, sidecar)
    env = MeasuredRobot(policy, interrupt=True)
    runner = Fr3SynchronousRunner(cfg, policy=policy, env=env)
    decision = runner.controller.decisions(0)["tasks"][str(cfg.fr3.task_id)]
    rows = runner.collect_episode(decision)
    assert len(rows) == 1 and env.versions == [0, 0]
    assert rows[0]["receipt"]["chunk_id"] == 0
    expected = rows[0]["reward_train"] + float(
        cfg.algorithm.gamma
    ) * runner._bootstrap_value(env.observation())
    torch.testing.assert_close(
        rows[0]["returns"].reshape(1), expected.reshape(1), atol=2e-5, rtol=2e-5
    )
    assert runner.optimizer_steps == 0


@pytest.mark.parametrize(
    "outcome,reward", [("success", 1.0), ("failure", 0.0), ("stop", 0.0), (None, 0.0)]
)
@pytest.mark.parametrize(
    "policy", ["insert_tubes", ("insert_tubes", 32)], indirect=True
)
def test_real_websocket_idle_outcome_seals_previous_transition(
    policy,
    tmp_path,
    outcome,
    reward,  # noqa: F811
):
    import threading

    from rlinf.envs.real.fr3_polymetis.env import Fr3PolymetisEnv

    scripts = ROOT / "deployments/fr3_wam/sources/irl_polymetis/polymetis/tests"
    spec = importlib.util.spec_from_file_location(
        "runner_bridge_helpers", scripts / "python/test_fr3_rl_bridge.py"
    )
    hardware_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hardware_helpers)
    from fr3_fastwam_client import validate_config
    from fr3_rl_bridge import Fr3Bridge, make_server

    parent, sidecar = _native_to_bc(policy, tmp_path)
    cfg = _config(policy, tmp_path, parent, sidecar)
    clock, hardware = hardware_helpers.Clock(), hardware_helpers.Hardware()
    cameras = hardware_helpers.Cameras(clock)
    receipt_ready, operator_done = threading.Event(), threading.Event()
    events = []

    def record(event, **data):
        events.append((event, data))
        if (
            outcome is not None
            and event == "execution_receipt"
            and data["chunk_id"] == 0
        ):
            receipt_ready.set()
            assert operator_done.wait(timeout=10)

    bridge = Fr3Bridge(
        validate_config(
            {
                "task_name": cfg.fr3.task_name,
                "task": cfg.fr3.prompt,
                "robot": {
                    "data_contract_confirmed": True,
                    "execution_horizon": cfg.fr3.action_horizon,
                },
                "episode": {
                    "max_actions": cfg.fr3.max_episode_actions,
                    "max_duration_s": 120,
                    "success_definition": "Test operator confirms task complete.",
                    "manual_reset_procedure": "Test operator restores scene before confirming ready.",
                },
            }
        ),
        hardware,
        cameras,
        record,
        clock=clock,
        wait=clock.wait,
    )
    server = make_server(bridge, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg.env.train.bridge_url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
    env = Fr3PolymetisEnv(cfg.env.train)
    env._rpc("scene_ready", confirmed=True)
    errors = []

    def operator():
        try:
            assert receipt_ready.wait(timeout=10)
            env._rpc("outcome", episode_id=env.episode_id, outcome=outcome)
        except Exception as error:
            errors.append(error)
        finally:
            operator_done.set()

    operator_thread = threading.Thread(target=operator, daemon=True)
    if outcome is not None:
        operator_thread.start()
    try:
        runner = Fr3SynchronousRunner(cfg, policy=policy, env=env)
        decision = runner.controller.decisions(0)["tasks"][str(cfg.fr3.task_id)]
        rows = runner.collect_episode(decision)
        assert not errors
        if outcome is None:
            assert len(rows) == 2
            assert rows[-1]["flow_execution_mask"].sum() == 5
            assert rows[-1]["receipt"]["bootstrap_allowed"] is True
            assert rows[-1]["receipt"]["stop_reason"] == "time_limit"
            assert (
                len(hardware.commands) == cfg.fr3.max_episode_actions
                and bridge.next_chunk == 2
            )
            return
        assert len(rows) == 1
        assert rows[0]["reward_env"] == reward
        assert rows[0]["flow_execution_mask"].sum() == cfg.fr3.action_horizon
        assert rows[0]["episode_seal"]["bootstrap_allowed"] is False
        assert rows[0]["episode_seal"]["stop_reason"] == outcome
        assert rows[0]["returns"].item() == pytest.approx(rows[0]["reward_train"])
        assert len(hardware.commands) == cfg.fr3.action_horizon
        assert bridge.next_chunk == 1
        assert any(event == "episode_sealed" for event, _ in events)
    finally:
        operator_done.set()
        if outcome is not None:
            operator_thread.join(timeout=2)
        env.close()
        server.shutdown()
        thread.join(timeout=2)
