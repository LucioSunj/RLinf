# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Test actual task-plan dispatch, outcome capture and native sampler resume."""

import asyncio
import importlib.util
import json
import weakref
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_fastwam_runner_checkpoint import _bare_runner, _runner_cfg, _WorkerGroup
from test_fastwam_uncond_rl import _compose

from rlinf.models.embodiment.wam_policy.uncond_rl_lifecycle import (
    UncondRLEnvWorker,
    UncondRLRunner,
)
from rlinf.runners.embodied_runner import EmbodiedRunner


class _Handle:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value


class _TaskGroup(_WorkerGroup):
    def __init__(self, events, owner, *, loaded_steps=None):
        super().__init__(loaded_steps=loaded_steps)
        self.events = events
        self.owner = owner
        self.rank = None
        self.worker_info_list = [SimpleNamespace(rank=i) for i in range(6)]
        self.plans = []

    def execute_on(self, rank):
        self.rank = rank
        return self

    def init_worker(self):
        self.events.append(("init", self.owner, self.rank))
        return super().init_worker()

    def sync_model_from_actor(self):
        self.events.append(("sync", self.owner))
        return _Handle()

    sync_model_to_rollout = sync_model_from_actor

    def save_checkpoint(self, path, step):
        self.events.append(("save", self.owner, step))
        return super().save_checkpoint(path, step)

    def describe_task_pools(self):
        return _Handle(
            [{"reset_pool_sizes": [50] * 10, "task_names": list(map(str, range(10)))}]
            * 6
        )

    def set_global_task_plan(self, plan):
        self.plans.append(plan)
        self.events.append(("plan", plan["runner_step"]))
        return _Handle()

    def set_global_step(self, step):
        self.events.append(("version", self.owner, step))
        return _Handle()


def _runner(tmp_path, resume=None):
    events = []
    cfg = _runner_cfg(tmp_path, resume_dir=None if resume is None else str(resume))
    cfg.env = OmegaConf.create({"train": {"total_num_envs": 84, "seed": 42}})
    actor = _TaskGroup(events, "actor", loaded_steps=[5])
    rollout = _TaskGroup(events, "rollout", loaded_steps=[5] * 6)
    env = _TaskGroup(events, "env")
    runner = _bare_runner(cfg, actor=actor, rollout=rollout, env=env)
    runner.__class__ = UncondRLRunner
    return runner, events


def test_native_zero_follows_serial_initialization_and_bc_sync(tmp_path):
    runner, events = _runner(tmp_path)
    runner.init_workers()
    assert events[:6] == [("init", "rollout", i) for i in range(6)]
    assert events[6:8] == [("init", "env", None), ("init", "actor", None)]
    assert events[8:] == [
        ("sync", "rollout"),
        ("sync", "actor"),
        ("save", "actor", 0),
        ("save", "rollout", 0),
    ]
    state = json.loads(
        (tmp_path / "l12/checkpoints/global_step_0/task_sampler.json").read_text()
    )
    assert state["next_update"] == 0 and state["num_ranks"] == 6


def test_dispatched_plans_cover_all_tasks_and_resume_next_reset_exactly(tmp_path):
    runner, events = _runner(tmp_path)
    runner.init_workers()
    for step in range(5):
        runner.global_step = step
        runner._set_worker_global_step()
        assert events[-3:] == [
            ("plan", step),
            ("version", "actor", step),
            ("version", "rollout", step),
        ]
    counts = Counter(
        row["task_id"]
        for plan in runner.env.plans
        for rank in plan["ranks"]
        for row in rank
    )
    assert counts == dict.fromkeys(range(10), 42)
    assert all(len(rank) == 14 for plan in runner.env.plans for rank in plan["ranks"])
    assert runner.env.plans[0] != runner.env.plans[1]
    runner.global_step = 5
    runner._save_checkpoint()
    checkpoint = tmp_path / "l12/checkpoints/global_step_5"
    restored, restored_events = _runner(tmp_path / "resumed", checkpoint)
    restored.init_workers()
    assert not any(event[0] in {"sync", "save"} for event in restored_events)
    restored._set_worker_global_step()
    runner._set_worker_global_step()
    assert restored.env.plans[-1] == runner.env.plans[-1]


def test_n84_sampler_rejects_n42_native_state(tmp_path):
    from rlinf.envs.libero.task_sampler import BalancedLiberoTaskSampler

    previous = BalancedLiberoTaskSampler(
        total_envs=42, reset_pool_sizes=[50] * 10, seed=42, num_ranks=6
    )
    runner, _ = _runner(tmp_path)
    runner.init_workers()
    with pytest.raises(ValueError, match="Sampler resume changed total_envs"):
        runner.task_sampler.load_state_dict(previous.state_dict(), runner_step=0)


def test_environment_captures_actual_success_before_reset_and_checks_identity():
    worker = object.__new__(UncondRLEnvWorker)
    worker._rank = 0
    worker.stage_num = 1
    slots = [
        {"task_id": 2, "trial_id": 9, "reset_state_id": 109, "episode_slot_id": 0},
        {"task_id": 7, "trial_id": 1, "reset_state_id": 351, "episode_slot_id": 1},
    ]
    received = []
    env = SimpleNamespace(
        task_ids=np.array([2, 7]),
        trial_ids=np.array([9, 1]),
        success_once=np.array([True, False]),
        set_global_task_plan=lambda rows, step: received.append((rows, step)),
    )
    worker.env_list = [env]
    worker.set_global_task_plan({"runner_step": 0, "ranks": [slots]})
    worker.capture_task_outcomes(0)
    env.success_once[:] = False
    assert received == [(slots, 0)]
    assert [row["success"] for row in worker.task_outcomes()["episodes"]] == [
        True,
        False,
    ]
    env.trial_ids[0] = 10
    with pytest.raises(ValueError, match="Executed task/reset"):
        worker.capture_task_outcomes(0)


def test_environment_returns_replay_pages_after_transport_finishes(monkeypatch):
    """Trim after the awaited sender and its final trajectory view are gone."""

    import rlinf.models.embodiment.wam_policy.uncond_rl_lifecycle as lifecycle

    worker = object.__new__(UncondRLEnvWorker)
    worker._rank = 2
    worker._global_task_plan = {"runner_step": 5}
    result = {"replay": torch.ones(2, 3)}
    replay = weakref.ref(result["replay"])
    channel = object()
    events, logs = [], []
    worker.log_info = logs.append
    worker.capture_task_outcomes = lambda stage: events.append(("outcomes", stage))

    async def send(self, rollout_result, target, *, stage_id):
        assert self is worker and rollout_result is result and target is channel
        last_trajectory = rollout_result.pop("replay")
        await asyncio.sleep(0)
        assert replay() is last_trajectory
        events.append(("sent", stage_id))

    def release(**kwargs):
        assert replay() is None and not result
        events.append((kwargs["phase"], kwargs["rank"]))
        return kwargs

    monkeypatch.setattr(lifecycle.EnvWorker, "send_rollout_trajectories", send)
    monkeypatch.setattr(lifecycle, "release_pad_host_memory", release)
    asyncio.run(worker.send_rollout_trajectories(result, channel, stage_id=0))
    assert events == [("outcomes", 0), ("sent", 0), ("post_trajectory_send", 2)]
    assert '"runner_step": 5' in logs[0]


def test_outcome_log_uses_executed_episodes_and_rejects_previous_update(
    tmp_path, monkeypatch
):
    runner, _ = _runner(tmp_path)
    runner.init_workers()
    runner._set_worker_global_step()
    plan = runner.env.plans[-1]
    outcomes = [
        {
            "runner_step": 0,
            "episodes": [{**row, "success": row["task_id"] == 2} for row in rank],
        }
        for rank in plan["ranks"]
    ]
    runner.env.task_outcomes = lambda: _Handle(outcomes)
    runner.global_step = 1
    logged = []
    runner.metric_logger = SimpleNamespace(log=lambda **kw: logged.append(kw))
    monkeypatch.setattr(EmbodiedRunner, "_log_step_metrics", lambda *_a, **_k: None)
    kwargs = {
        "step": 0,
        "env_handle": _Handle(),
        "actor_rollout_metrics": [],
        "actor_training_metrics": [],
    }
    runner._log_step_metrics(**kwargs)
    record = json.loads((runner._task_audit_dir / "updates.jsonl").read_text())
    assert len(record["episodes"]) == 84
    assert logged[0]["data"]["task_global/success_count"] == plan["quotas"][2]
    runner.global_step = 2
    with pytest.raises(ValueError, match="does not match this update"):
        runner._log_step_metrics(**kwargs)


def test_standard_entrypoint_dispatches_uncond_lifecycle(monkeypatch):
    import rlinf.config as config_module
    import rlinf.models.embodiment.wam_policy.uncond_rl_lifecycle as lifecycle
    from rlinf.models.embodiment.wam_policy.uncond_rl_actor import UncondRLFSDPActor

    path = Path(__file__).parents[2] / "examples/embodiment/train_embodied_agent.py"
    spec = importlib.util.spec_from_file_location("uncond_training_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    cfg = _compose(monkeypatch)
    launched = []

    class Placement:
        def __init__(self, *_args):
            pass

        def get_world_size(self, owner):
            return 1 if owner == "actor" else 6

        def get_strategy(self, owner):
            return owner

    class Group:
        def __init__(self, owner):
            self.owner = owner

        def launch(self, *_args, **_kwargs):
            launched.append(self.owner)
            return self

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs["actor"].owner == "uncond_actor"
            assert kwargs["env"].owner == "uncond_env"
            assert kwargs["rollout"].owner == "uncond_rollout"
            assert kwargs["cfg"].runner.weight_sync_interval == 1

        def init_workers(self):
            launched.append("init_balanced_runner")

        def run(self):
            launched.append("run")

    for module in (config_module, entry):
        monkeypatch.setattr(module, "Cluster", lambda **_kwargs: None)
        monkeypatch.setattr(module, "HybridComponentPlacement", Placement)
    for worker, owner in (
        (UncondRLFSDPActor, "uncond_actor"),
        (lifecycle.UncondRLEnvWorker, "uncond_env"),
        (lifecycle.UncondRLRolloutWorker, "uncond_rollout"),
    ):
        monkeypatch.setattr(
            worker, "create_group", lambda _cfg, name=owner: Group(name)
        )
    monkeypatch.setattr(lifecycle, "UncondRLRunner", Runner)
    entry.main.__wrapped__(cfg)
    assert launched == [
        "uncond_actor",
        "uncond_rollout",
        "uncond_env",
        "init_balanced_runner",
        "run",
    ]


def test_pure_native_zero_rollout_restores_sampling_and_rng(tmp_path, monkeypatch):
    import torch
    from test_fastwam_rollout_checkpoint import _worker, worker_module
    from test_fastwam_uncond_rl import _policy

    worker = _worker()
    worker.model_cfg.uncond_rl = {"critic_warmup_updates": 10}
    worker.hf_model = _policy(monkeypatch)
    rng = {"cpu": torch.get_rng_state()}
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng)
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _value: None)
    worker.save_checkpoint(str(tmp_path), step=0)
    worker.version = 99
    worker.hf_model.actor_version = 99
    assert worker.load_checkpoint(str(tmp_path)) == 0
    assert worker.version == worker.hf_model.actor_version == 0
    assert worker.hf_model.route_tracker.episode_chunks == {}


def test_terminal_mask_is_cumulative_and_resets_without_shortening_rollout(monkeypatch):
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(UncondRLEnvWorker)
    worker.cfg = OmegaConf.create(
        {"env": {"train": {"auto_reset": False, "ignore_terminations": False}}}
    )
    monkeypatch.setattr(
        EnvWorker,
        "_build_rollout_input_data",
        lambda _self, batch, **_kwargs: {
            "obs": batch["obs"],
            "fastwam_reset_mask": torch.zeros(3, dtype=torch.bool),
        },
    )
    obs = {"states": torch.randn(3, 8)}
    build = worker._build_rollout_input_data
    initial = build({"obs": obs}, stage_id=0, force_reset=True)
    assert initial["obs"]["_uncond_rl_active"].tolist() == [True] * 3
    assert "_uncond_rl_active" not in obs
    done = torch.tensor([[False, False], [False, True], [False, False]])
    partial = build({"obs": obs, "dones": done}, stage_id=0)
    assert partial["obs"]["_uncond_rl_active"].tolist() == [True, False, True]
    persistent = build({"obs": obs, "dones": torch.zeros_like(done)}, stage_id=0)
    assert persistent["obs"]["_uncond_rl_active"].tolist() == [True, False, True]
    all_done = build({"obs": obs, "dones": torch.ones_like(done)}, stage_id=0)
    assert not all_done["obs"]["_uncond_rl_active"].any()
    assert all_done["obs"]["states"] is obs["states"]
    reset = build({"obs": obs}, stage_id=0, force_reset=True)
    assert reset["obs"]["_uncond_rl_active"].all()
    evaluation = build({"obs": obs}, stage_id=0, eval_mode=True)
    assert "_uncond_rl_active" not in evaluation["obs"]


def test_helpers_run_between_rollout_completion_and_next_sync(monkeypatch):
    import rlinf.models.embodiment.wam_policy.uncond_rl_lifecycle as lifecycle
    from rlinf.models.embodiment.wam_policy.uncond_rl_helpers import (
        unpack_helper_message,
    )

    runner = object.__new__(UncondRLRunner)
    runner.cfg = _compose(monkeypatch)
    runner.global_step = 10
    events = []
    commands = []

    class Queue:
        def put(self, message, *, key):
            commands.append((key, message))
            events.append(("finish", key))

    class HelperHandle:
        def wait(self):
            assert len(commands) == 6
            events.append("helpers_finished")
            return [{"status": "PASS"}] * 6

    def execute(*ranks):
        assert ranks == tuple(range(6))
        return runner.rollout

    def serve(**kwargs):
        assert kwargs["version"] == 10
        events.append("helpers_started")
        return HelperHandle()

    runner.rollout = SimpleNamespace(
        worker_info_list=[SimpleNamespace(rank=rank) for rank in range(6)],
        execute_on=execute,
        serve_actor_training=serve,
    )
    runner.actor = SimpleNamespace(
        configure_training_helpers=lambda **_kwargs: _Handle()
    )
    runner.logger = SimpleNamespace(info=lambda *_args: None)
    monkeypatch.setattr(lifecycle.Channel, "create", lambda *_args, **_kwargs: Queue())

    def train(_self, *, next_step):
        assert next_step
        events.append("owner_trained")
        return ([], _Handle(), None)

    monkeypatch.setattr(EmbodiedRunner, "_run_actor_training", train)
    runner._run_actor_training(next_step=True)
    assert events[:2] == ["helpers_started", "owner_trained"]
    assert events[-1] == "helpers_finished"
    assert [key for key, _message in commands] == list(range(6))
    for _, message in commands:
        assert (message.kind, message.version) == ("finish_update", 10)
        assert unpack_helper_message(message) == {"optimizer_opportunities": 15}


@pytest.mark.parametrize("ranks", [[0, 1], list(range(7)), [True], [0.0]])
def test_helper_configuration_preserves_native_six_rank_topology(monkeypatch, ranks):
    from rlinf.models.embodiment.wam_policy.uncond_rl_config import (
        uncond_training_helper_ranks,
    )

    cfg = _compose(monkeypatch)
    cfg.actor.uncond_rl_execution.helper_ranks = ranks
    with pytest.raises(ValueError, match="helper ranks"):
        uncond_training_helper_ranks(cfg)
