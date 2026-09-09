# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Physical PAD integration against the production PPO, GAE, BC and LoRA math."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from rlinf.models.embodiment.wam_policy.online_idm_bc.config import (
    ONLINE_IDM_BC_TEACHER_ACTIONS,
    ONLINE_IDM_BC_TEACHER_PRESENT,
)
from rlinf.models.embodiment.wam_policy.real_robot.actor import RealRobotPADActor
from rlinf.models.embodiment.wam_policy.real_robot.builder import (
    build_real_robot_policy,
)
from rlinf.models.embodiment.wam_policy.real_robot.collector import (
    SequentialRobotCollector,
    complete_auxiliary_inference,
    prepare_training_batch,
)
from rlinf.models.embodiment.wam_policy.real_robot.config import load_config
from rlinf.models.embodiment.wam_policy.real_robot.evaluation import run_mock_evaluation
from rlinf.models.embodiment.wam_policy.real_robot.runner import (
    build_controller,
    load_checkpoint,
    make_mock_env,
    run_mock_training,
)
from rlinf.runners.fastwam_idm_cost_control import FastWAMIDMCostObservation

ROOT = Path(__file__).resolve().parents[3]


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_nested_equal(a, b)
    else:
        assert first == second


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    cfg = load_config(ROOT / "configs/real_robot/pad_mock.yaml")
    output = tmp_path_factory.mktemp("real_pad_learning")
    result = run_mock_training(
        cfg, output, tasks_path=ROOT / "configs/real_robot/tasks.yaml"
    )
    assert result["committed_update"] == 6
    return cfg, output


def test_five_critic_updates_joint_ratios_ownership_and_frozen_idm(completed_run):
    cfg, output = completed_run
    records = [
        json.loads(line)
        for line in (output / "training.jsonl").read_text().splitlines()
    ]
    initial = torch.load(output / "checkpoints/update_000001.pt", weights_only=False)
    for step, row in enumerate(records, 1):
        payload = torch.load(
            output / f"checkpoints/update_{step:06d}.pt", weights_only=False
        )
        for name, value in initial["fixture_backbones"]["actor"].items():
            if not name.endswith(("lora_A", "lora_B")):
                assert torch.equal(payload["fixture_backbones"]["actor"][name], value)
        if step <= 5:
            assert_nested_equal(payload["policy"]["gate"], initial["policy"]["gate"])
            assert_nested_equal(payload["policy"]["lora"], initial["policy"]["lora"])
            for group in payload["optimizer"]["param_groups"]:
                if group["name"] in {"gate", "uncond_lora"}:
                    assert all(
                        index not in payload["optimizer"]["state"]
                        for index in group["params"]
                    )
            assert row["learner"]["parameter_delta_norms"]["gate"] == 0
            assert row["learner"]["parameter_delta_norms"]["uncond_lora"] == 0
            assert payload["controller"]["signed_price"] == 0
        assert row["learner"]["parameter_delta_norms"]["value_head"] > 0
    joint = records[-1]["learner"]
    assert all(value > 0 for value in joint["parameter_delta_norms"].values())
    assert joint["preupdate_ratios"]["sample_counts"]["flow"] > 0
    assert max(joint["preupdate_ratios"]["max_abs_log_ratio"].values()) < 1e-4
    assert {outcome for row in records for outcome in row["outcomes"]} == {
        "success",
        "failure",
    }


def test_sequential_lengths_shapes_masks_and_terminal_gae(completed_run):
    _, output = completed_run
    saved = torch.load(
        next(output.glob("*update-000006.completed.pt")), weights_only=False
    )
    assert [len(e.proposals) for e in saved["episodes"]][:2] == [2, 3]
    assert saved["chunk_valid"].shape == (15, 1)
    assert saved["executed_prefix_mask"].shape == (15, 1, 10)
    row = 0
    for episode in saved["episodes"]:
        for proposal, receipt in zip(episode.proposals, episode.receipts, strict=True):
            fwd = proposal.replay["forward_inputs"]
            assert fwd["flow_chains"].shape == (1, 11, 32, 7)
            assert proposal.replay["prev_logprobs"].shape == (1, 10, 7)
            assert fwd[ONLINE_IDM_BC_TEACHER_ACTIONS].shape == (1, 32, 7)
            assert bool(fwd[ONLINE_IDM_BC_TEACHER_PRESENT].item()) == (
                proposal.route == 0
            )
            assert receipt.chunk_valid
            assert saved["chunk_valid"][row, 0]
            row += 1
        assert episode.receipts[-1].executed_count == 3
        assert bool(saved["dones"][row, 0])
        assert saved["returns"][row - 1, 0] == pytest.approx(
            float(episode.outcome.kind == "success")
        )
    # Compute the same episode independently: the following reset cannot alter returns.
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns

    episode = saved["episodes"][0]
    values = torch.cat(
        [p.replay["prev_values"] for p in episode.proposals] + [torch.zeros(1, 1)]
    )
    rewards = torch.zeros(2, 1)
    rewards[-1] = float(episode.outcome.kind == "success")
    _, returns = compute_gae_advantages_and_returns(
        rewards=rewards,
        values=values,
        dones=torch.tensor([[False], [False], [True]]),
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    torch.testing.assert_close(returns, saved["returns"][:2])


def collect_small(cfg, policy, count=8):
    cfg = deepcopy(cfg)
    cfg["mock"]["episode_steps"] = [3]
    env = make_mock_env(cfg, run_id="auxiliary-test")
    attempts = [
        {
            "episode_id": f"aux-{i}",
            "task_id": "fixture",
            "instruction": "Move the block.",
            "layout_id": "fixed",
            "operator_ready": True,
        }
        for i in range(count)
    ]
    return env, SequentialRobotCollector(env, policy).collect(attempts)


def test_auxiliary_uses_saved_snapshot_and_never_advances_history(completed_run):
    cfg, output = completed_run
    policy = build_real_robot_policy(cfg, initialize=False)
    load_checkpoint(output / "checkpoints/update_000006.pt", policy=policy, cfg=cfg)
    env, episodes = collect_small(cfg, policy)
    decision = build_controller()._build_decision(6)
    with pytest.raises(RuntimeError, match="zero value placeholders"):
        prepare_training_batch(episodes, decision=decision, training=cfg["training"])
    history = deepcopy(policy.route_tracker.state_dict())
    observations = env.backend.observe_calls
    commands = len(env.backend.commands)
    # Latest physical state is different from every saved current snapshot.
    env.backend.state["tcp_pose"][:3] = 999
    complete_auxiliary_inference(episodes, policy)
    assert_nested_equal(policy.route_tracker.state_dict(), history)
    assert env.backend.observe_calls == observations
    assert len(env.backend.commands) == commands
    uncond = next(p for e in episodes for p in e.proposals if p.route == 0)
    request = uncond.replay["teacher_request"]
    expected = policy.runtime.complete_teacher(
        sample=request["sample"],
        seeds=request["seeds"],
        route=uncond.replay["route_info"].route_used,
        actor_version=uncond.actor_version,
    )
    torch.testing.assert_close(
        expected.forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS],
        uncond.replay["forward_inputs"][ONLINE_IDM_BC_TEACHER_ACTIONS],
        rtol=0,
        atol=0,
    )
    uncond.replay["forward_inputs"][ONLINE_IDM_BC_TEACHER_PRESENT].fill_(False)
    batch = prepare_training_batch(
        episodes, decision=decision, training=cfg["training"]
    )
    learner = RealRobotPADActor(policy, cfg["training"])
    with pytest.raises(RuntimeError, match="lack IDM teacher"):
        learner.update(batch)
    assert learner.optimizer_steps == 0


def test_no_u_minibatch_does_not_advance_adam_momentum(completed_run):
    cfg, output = completed_run
    policy = build_real_robot_policy(cfg, initialize=False)
    learner = RealRobotPADActor(policy, cfg["training"])
    controller = build_controller()
    load_checkpoint(
        output / "checkpoints/update_000006.pt",
        policy=policy,
        learner=learner,
        controller=controller,
        cfg=cfg,
    )
    _, episodes = collect_small(cfg, policy)
    episodes = [e for e in episodes if e.proposals[0].route == 1]
    assert episodes
    complete_auxiliary_inference(episodes, policy)
    batch = prepare_training_batch(
        episodes, decision=controller._build_decision(6), training=cfg["training"]
    )
    parameters = learner.groups["uncond_lora"]
    before = [
        (p.detach().clone(), deepcopy(learner.optimizer.state[p])) for p in parameters
    ]
    assert any(state for _, state in before)
    report = learner.update(batch)
    assert report["parameter_delta_norms"]["uncond_lora"] == 0
    for parameter, (weight, state) in zip(parameters, before, strict=True):
        assert torch.equal(parameter, weight)
        assert_nested_equal(learner.optimizer.state[parameter], state)


def test_checkpoint_resume_commits_next_update_once(completed_run, tmp_path):
    cfg, output = completed_run
    checkpoint = output / "checkpoints/update_000006.pt"
    result = run_mock_training(
        cfg,
        tmp_path,
        tasks_path=ROOT / "configs/real_robot/tasks.yaml",
        resume=checkpoint,
        updates=7,
    )
    assert result["committed_update"] == 7
    record = json.loads((tmp_path / "training.jsonl").read_text())
    assert record["update"] == 7 and not record["learner"]["warmup"]
    assert record["learner"]["optimizer_steps"] == 7
    assert record["learner"]["scheduler_steps"] == 7
    saved = torch.load(result["latest_checkpoint"], weights_only=False)
    assert saved["controller"]["observed_runner_steps"] == 7
    assert saved["batch_id"] != torch.load(checkpoint, weights_only=False)["batch_id"]


def test_b50_total_price_change_clamp():
    controller = build_controller()
    controller.signed_price = 0.01
    controller.observed_runner_steps = 5
    controller.decision_for_step(5)
    observation = FastWAMIDMCostObservation(
        runner_step=5,
        eligible_gate_decision_count=10,
        eligible_idm_decision_count=4,
        eligible_realized_fraction=0.4,
        eligible_expected_fraction=0.4,
        valid_chunk_count=10,
        valid_idm_chunk_count=4,
        executed_realized_fraction=0.4,
        forced_fraction=0.0,
        break_even_idm_cost=None,
        configured_idm_cost=None,
    )
    controller.observe_rollout(observation)
    assert controller.signed_price == pytest.approx(0.005)


def test_four_eval_paths_load_same_u_and_skip_unused_owners(completed_run, tmp_path):
    cfg, output = completed_run
    report = run_mock_evaluation(
        cfg,
        tmp_path,
        methods=["always_idm", "always_uncond", "random", "learned"],
        tasks_path=ROOT / "configs/real_robot/tasks.yaml",
        checkpoint=output / "checkpoints/update_000006.pt",
    )
    rows = [
        json.loads(line)
        for line in (Path(report["evaluation_dir"]) / "episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 24
    for row in rows:
        assert row["model_calls"]["teacher"] == row["model_calls"]["critic"] == 0
        if row["method"] != "learned":
            assert row["model_calls"]["gate"] == row["model_calls"]["feature"] == 0
        else:
            assert row["model_calls"]["feature"] == row["chunk_count"]
            assert row["routing"]["mode"] == "learned_behavior_sampling"
            assert row["routing"]["epsilon"] == 0.1
        assert row["timings"]["rpc_acknowledgement_including_gripper"] > 0
        assert row["wallclock_seconds"] >= row["prediction_seconds"]
    for layout in cfg["evaluation"]["layouts"]:
        matched = [
            row
            for row in rows
            if row["layout_id"] == layout
            and row["task_id"] == cfg["collection"]["task_ids"][0]
        ]
        seeds = [
            torch.load(row["replay_path"], weights_only=False).proposals[0].noise
            for row in matched
        ]
        assert all(seed == seeds[0] for seed in seeds)


def test_short_final_minibatch_and_microbatch_denominators_match(completed_run):
    cfg, output = completed_run
    policies = [build_real_robot_policy(cfg, initialize=False) for _ in range(2)]
    for policy in policies:
        load_checkpoint(output / "checkpoints/update_000005.pt", policy=policy, cfg=cfg)
    env = make_mock_env(cfg, run_id="tail")
    episodes = SequentialRobotCollector(env, policies[0]).collect(
        [
            {
                "episode_id": f"tail-{i}",
                "task_id": "test",
                "instruction": "Move.",
                "layout_id": "a",
                "operator_ready": True,
            }
            for i in range(2)
        ]
    )
    complete_auxiliary_inference(episodes, policies[0])
    batch = prepare_training_batch(
        episodes,
        decision=build_controller()._build_decision(5),
        training=cfg["training"],
    )
    learners = [
        RealRobotPADActor(
            policy,
            {**cfg["training"], "optimizer_batch_size": 4, "micro_batch_size": micro},
        )
        for policy, micro in zip(policies, (1, 4), strict=True)
    ]
    reports = [learner.update(batch) for learner in learners]
    assert all(
        [r["chunks"] for r in report["minibatches"]] == [4, 1] for report in reports
    )
    for (name, first), (second_name, second) in zip(
        policies[0].named_parameters(), policies[1].named_parameters(), strict=True
    ):
        assert name == second_name
        torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-7)


def test_unknown_camera_attempt_is_retained_and_excluded_from_gae():
    from rlinf.envs.pad_realworld.mock_backend import MockRobotBackend

    cfg = load_config(ROOT / "configs/real_robot/pad_mock.yaml")

    class InterruptedCamera(MockRobotBackend):
        def observe(self, **identity):
            if self.episode_index == 0 and self.steps >= 10:
                raise ConnectionError("Recorded camera disconnect fixture")
            return super().observe(**identity)

    policy = build_real_robot_policy(cfg, initialize=False)
    env = make_mock_env(cfg, run_id="interruption", backend=InterruptedCamera())
    episodes = SequentialRobotCollector(env, policy).collect(
        [
            {
                "episode_id": f"disconnect-{i}",
                "task_id": "test",
                "instruction": "Move.",
                "layout_id": "a",
                "operator_ready": True,
            }
            for i in range(2)
        ]
    )
    assert episodes[0].outcome.kind == "censored" and len(episodes[0].proposals) == 1
    complete_auxiliary_inference(episodes, policy)
    batch = prepare_training_batch(
        episodes,
        decision=build_controller()._build_decision(0),
        training=cfg["training"],
    )
    assert len(batch["rows"]) == len(episodes[1].proposals)
    assert all(
        row.snapshot.episode_id == episodes[1].episode_id for row in batch["rows"]
    )


def test_stopped_collector_waits_for_score_without_extra_commands():
    from rlinf.envs.pad_realworld.contracts import EpisodeOutcome
    from rlinf.envs.pad_realworld.mock_backend import MockRobotBackend

    cfg = load_config(ROOT / "configs/real_robot/pad_mock.yaml")

    class OperatorStop(MockRobotBackend):
        def send(self, command):
            result = super().send(command)
            if self.steps == 3:
                env.request_stop("operator_stop")
            return result

    policy = build_real_robot_policy(cfg, initialize=False)
    env = make_mock_env(cfg, run_id="manual", backend=OperatorStop())

    def score(episode_id):
        assert env.backend.steps == 3
        return EpisodeOutcome(episode_id, "success")

    episode = SequentialRobotCollector(env, policy, wait_for_outcome=score).collect(
        [
            {
                "episode_id": "score",
                "task_id": "test",
                "instruction": "Move.",
                "layout_id": "a",
                "operator_ready": True,
            }
        ]
    )[0]
    assert env.backend.steps == 3
    assert episode.trainable and episode.receipts[0].chunk_valid
