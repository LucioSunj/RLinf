# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Batch-boundary learning, checkpointing and explicit CPU/mock entrypoint."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import torch
import yaml

from rlinf.envs.pad_realworld.action_codec import ActionCodec
from rlinf.envs.pad_realworld.env import SynchronousRobotEnv
from rlinf.envs.pad_realworld.mock_backend import MockRobotBackend
from rlinf.envs.pad_realworld.observation import ObservationAdapter
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_budget import (
    PadCriticWarmupReversalDampedController,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_gate import (
    PhysicalStateHistoryTracker,
)
from rlinf.models.embodiment.wam_policy.pad_rv.route_neutral_policy import (
    RouteNeutralRoutingState,
)
from rlinf.runners.fastwam_idm_cost_control import FastWAMIDMCostObservation
from rlinf.utils.utils import get_rng_state, set_rng_state

from .actor import RealRobotPADActor
from .builder import build_real_robot_policy, route_neutral_profile
from .collector import (
    SequentialRobotCollector,
    complete_auxiliary_inference,
    prepare_training_batch,
)
from .config import action_protocol


def build_controller():
    """Reuse the implemented B50 total-change clamp and critic-only warm-up."""
    return PadCriticWarmupReversalDampedController(
        {
            "type": "pad_critic_warmup_reversal_damped",
            "constraint": "two_sided_band",
            "rate": {
                "scope": "eligible_gate_decisions",
                "feedback": "expected_behavior_probability",
                "target_idm_fraction": 0.5,
                "half_width": 0.03,
            },
            "charge_scope": "eligible_nonforced",
            "signed_price": {
                "initial_value": 0.0,
                "learning_rate": 0.0025,
                "ema_beta": 0.0,
                "update_interval": 1,
                "max_abs_value": 0.1,
                "max_delta_per_update": 0.005,
                "reversal": {"mode": "opposing_decay", "factor": 0.0},
            },
            "critic_warmup": route_neutral_profile()["critic_warmup"],
        }
    )


def feedback_observation(batch, step):
    emitted = batch["emitted_gate"]
    eligible = batch["alignment"].gate_valid_mask
    count = int(eligible.sum())
    idm = int((batch["route_info"].route_used[eligible] == 1).sum())
    return FastWAMIDMCostObservation(
        runner_step=step,
        eligible_gate_decision_count=count,
        eligible_idm_decision_count=idm,
        eligible_realized_fraction=idm / count,
        eligible_expected_fraction=float(emitted.behavior_probability[eligible].mean()),
        valid_chunk_count=count,
        valid_idm_chunk_count=idm,
        executed_realized_fraction=idm / count,
        forced_fraction=0.0,
        break_even_idm_cost=None,
        configured_idm_cost=None,
    )


def make_mock_env(cfg, *, run_id, backend=None):
    if cfg["backend"] != "mock":
        raise ValueError("The offline runner accepts only an explicitly mock backend.")
    backend = backend or MockRobotBackend(
        camera_names=[c["name"] for c in cfg["observation"]["cameras"]],
        episode_steps=cfg["mock"]["episode_steps"],
        target_x=cfg["mock"]["target_x"],
        gripper_wait_seconds=cfg["mock"]["gripper_wait_seconds"],
    )
    return SynchronousRobotEnv(
        backend,
        ObservationAdapter.from_config(cfg["observation"]),
        ActionCodec(**cfg["limits"]),
        action_protocol(cfg),
        run_id=run_id,
        calibration_id=cfg["hardware"]["calibration_id"],
        allow_motion=False,
        episode_timeout_seconds=cfg["execution"]["episode_timeout_seconds"],
    )


def save_checkpoint(path, *, policy, learner, controller, cfg, batch_id):
    """Atomically publish one completed update with the existing adaptive payload."""
    if policy.actor_version != controller.observed_runner_steps:
        raise RuntimeError("Actor and B50 versions must commit together.")
    payload = {
        "schema": "real-robot-pad-checkpoint-v1",
        "committed_update": policy.actor_version,
        "batch_id": batch_id,
        "policy": policy.trainable_state_dict(),
        "optimizer": learner.optimizer.state_dict(),
        "scheduler": learner.lr_scheduler.state_dict(),
        "scaler": learner.scaler.state_dict(),
        "optimizer_steps": learner.optimizer_steps,
        "controller": controller.state_dict(),
        "rng": get_rng_state(),
        "sampling_streams": {
            key: generator.get_state() for key, generator in policy.streams.items()
        },
        "config": cfg,
        "initialization": policy.initialization,
    }
    if cfg["model"]["kind"] == "tiny":
        payload["fixture_backbones"] = {
            "actor": policy.actor.state_dict(),
            "critic": policy.critic.backbone.prefix_encoder.state_dict(),
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path, *, policy, learner=None, controller=None, cfg, evaluation_method=None
):
    """Restore learning state only; every new attempt still requires a new start."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "real-robot-pad-checkpoint-v1":
        raise ValueError("Unsupported real PAD checkpoint.")
    for key in ("model", "observation", "action", "limits", "execution", "training"):
        if payload["config"][key] != cfg[key]:
            raise ValueError(
                f"Resume/evaluation {key} differs from the saved physical/model specification."
            )
    if cfg["model"]["kind"] == "tiny":
        policy.actor.load_state_dict(payload["fixture_backbones"]["actor"])
        if policy.critic is not None:
            policy.critic.backbone.prefix_encoder.load_state_dict(
                payload["fixture_backbones"]["critic"]
            )
    state = payload["policy"]
    if evaluation_method is None:
        policy.load_trainable_state_dict(state)
    else:
        policy.lora_adapter.load_lora_state_dict(state["lora"])
        if evaluation_method == "learned":
            policy.gate.load_state_dict(state["gate"])
        policy.actor_version = state["actor_version"]
    policy.initialization = payload["initialization"]
    history = PhysicalStateHistoryTracker(policy.runtime.route_neutral_input)
    policy.runtime.physical_history = history
    policy.route_tracker = RouteNeutralRoutingState(physical_history=history)
    policy.begin_episode()
    if learner is not None:
        learner.optimizer.load_state_dict(payload["optimizer"])
        learner.lr_scheduler.load_state_dict(payload["scheduler"])
        learner.scaler.load_state_dict(payload["scaler"])
        learner.optimizer_steps = payload["optimizer_steps"]
        controller.load_state_dict(payload["controller"])
        if (
            controller.observed_runner_steps != policy.actor_version
            or policy.actor_version != payload["committed_update"]
        ):
            raise RuntimeError("Saved actor/controller commit counters disagree.")
        set_rng_state(payload["rng"])
        for name, state in payload["sampling_streams"].items():
            policy.streams[name].set_state(state)
    return payload


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def run_training_batches(
    cfg,
    run_dir,
    *,
    policy,
    env,
    attempts_for_update,
    resume=None,
    updates=None,
    wait_for_outcome=None,
):
    """Generic synchronous runner usable with an explicitly provided robot binding."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "STATUS.json").exists() and resume is None:
        raise ValueError(
            "Existing run requires explicit --resume; use a fresh output directory otherwise."
        )
    learner = RealRobotPADActor(policy, cfg["training"])
    controller = build_controller()
    if resume is not None:
        load_checkpoint(
            resume, policy=policy, learner=learner, controller=controller, cfg=cfg
        )
    target = cfg["training"]["total_runner_updates"] if updates is None else updates
    if target < policy.actor_version:
        raise ValueError("Target update cannot precede the restored checkpoint.")
    session = uuid.uuid4().hex[:12]
    collector = SequentialRobotCollector(env, policy, wait_for_outcome=wait_for_outcome)
    _write_json(run_dir / "resolved_config.json", cfg)
    records = []
    checkpoint = str(resume) if resume is not None else None
    for step in range(policy.actor_version, target):
        batch_id = f"{session}-update-{step + 1:06d}"
        status = {
            "status": "COLLECTING",
            "committed_update": policy.actor_version,
            "pending_batch_id": batch_id,
            "latest_checkpoint": checkpoint,
            "physical_num_envs": 1,
            "model": cfg["model"]["kind"],
            "real_assets": "REAL-ASSET-NOT-RUN"
            if cfg["model"]["kind"] == "tiny"
            else "USER_PROVIDED_ASSETS",
            "hardware": "HARDWARE-NOT-RUN"
            if env.backend.is_mock
            else "EXPLICIT_BINDING",
        }
        _write_json(run_dir / "STATUS.json", status)
        decision = controller.decision_for_step(step)
        try:
            episodes = collector.collect(attempts_for_update(step, session))
            replay_path = run_dir / f"{batch_id}.rollout.pt"
            torch.save(episodes, replay_path)
            status["status"] = "COLLECTED_NOT_COMMITTED"
            status["rollout"] = str(replay_path)
            _write_json(run_dir / "STATUS.json", status)
            auxiliary = complete_auxiliary_inference(episodes, policy)
            batch = prepare_training_batch(
                episodes, decision=decision, training=cfg["training"]
            )
            policy.runtime.synchronize()
            started = time.perf_counter()
            metrics = learner.update(batch)
            policy.runtime.synchronize()
            update_seconds = time.perf_counter() - started
            feedback = controller.observe_rollout(feedback_observation(batch, step))
            policy.actor_version = step + 1
            checkpoint = str(run_dir / "checkpoints" / f"update_{step + 1:06d}.pt")
            torch.save(
                {
                    "episodes": episodes,
                    "returns": batch["returns"],
                    "advantages": batch["advantages"],
                    "dones": batch["dones"],
                    "chunk_valid": batch["chunk_valid"],
                    "executed_prefix_mask": batch["executed_prefix_mask"],
                },
                run_dir / f"{batch_id}.completed.pt",
            )
            save_checkpoint(
                checkpoint,
                policy=policy,
                learner=learner,
                controller=controller,
                cfg=cfg,
                batch_id=batch_id,
            )
            status.update(
                status="COMMITTED",
                committed_update=policy.actor_version,
                pending_batch_id=None,
                latest_checkpoint=checkpoint,
            )
            record = {
                "update": step + 1,
                "batch_id": batch_id,
                "episodes": len(episodes),
                "eligible_episodes": sum(e.trainable for e in episodes),
                "episode_chunks": [len(e.proposals) for e in episodes],
                "outcomes": [e.outcome.kind for e in episodes],
                "chunks": len(batch["rows"]),
                "auxiliary": auxiliary,
                "update_seconds": update_seconds,
                "learner": metrics,
                "controller": feedback,
                "checkpoint": checkpoint,
            }
            with (run_dir / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
            status.update(
                status="COMMITTED",
                committed_update=policy.actor_version,
                pending_batch_id=None,
                latest_checkpoint=checkpoint,
            )
            _write_json(run_dir / "STATUS.json", status)
        except Exception as error:
            status.update(
                status="FAILED_RETAINED", error=f"{type(error).__name__}: {error}"
            )
            _write_json(run_dir / "STATUS.json", status)
            raise
    report = {
        "status": "COMPLETE",
        "committed_update": policy.actor_version,
        "latest_checkpoint": checkpoint,
        "run_dir": str(run_dir),
        "physical_num_envs": 1,
        "real_assets": "REAL-ASSET-NOT-RUN"
        if cfg["model"]["kind"] == "tiny"
        else "USER_PROVIDED_ASSETS",
        "hardware": "HARDWARE-NOT-RUN" if env.backend.is_mock else "EXPLICIT_BINDING",
    }
    _write_json(run_dir / "STATUS.json", report)
    return report


def run_mock_training(cfg, run_dir, *, tasks_path, resume=None, updates=None):
    if cfg["backend"] != "mock":
        raise ValueError("Mock training cannot instantiate a live backend.")
    tasks = yaml.safe_load(Path(tasks_path).read_text())
    policy = build_real_robot_policy(cfg, initialize=resume is None)
    env = make_mock_env(cfg, run_id=Path(run_dir).name)

    def attempts(step, session):
        return [
            {
                "episode_id": f"{session}:u{step + 1}:{repeat}:{task}",
                "task_id": task,
                "instruction": tasks[task]["instruction"],
                "layout_id": f"mock_layout_{repeat}",
                "operator_ready": True,
            }
            for repeat in range(cfg["collection"]["episodes_per_task_per_update"])
            for task in cfg["collection"]["task_ids"]
        ]

    return run_training_batches(
        cfg,
        run_dir,
        policy=policy,
        env=env,
        attempts_for_update=attempts,
        resume=resume,
        updates=updates,
    )
