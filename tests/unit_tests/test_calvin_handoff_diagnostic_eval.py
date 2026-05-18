import json

import numpy as np
import pytest

from toolkits.eval_scripts_openpi.calvin_handoff_diagnostic_eval import (
    advance_symbolic_state,
    compute_symbolic_state_prefixes,
    parse_pair_positions,
    run_handoff_diagnostic,
    summarize_records,
)


def test_advance_symbolic_state_applies_single_matching_effect():
    task_specs = {
        "open_drawer": [
            {
                "condition": {"drawer": "closed", "grasped": 0},
                "effect": {"drawer": "open"},
            },
            {
                "condition": {"drawer": "open", "grasped": 0},
                "effect": {"drawer": "closed"},
            },
        ]
    }
    state = {"drawer": "closed", "grasped": 0, "slider": "left"}

    next_state = advance_symbolic_state(state, "open_drawer", task_specs)

    assert next_state == {"drawer": "open", "grasped": 0, "slider": "left"}
    assert state == {"drawer": "closed", "grasped": 0, "slider": "left"}


def test_advance_symbolic_state_rejects_ambiguous_effects():
    task_specs = {
        "toggle": [
            {"condition": {"led": 0}, "effect": {"led": 1}},
            {"condition": {"led": 0}, "effect": {"lightbulb": 1}},
        ]
    }

    with pytest.raises(ValueError, match="exactly one symbolic effect"):
        advance_symbolic_state({"led": 0}, "toggle", task_specs)


def test_compute_prefixes_and_parse_pair_positions():
    task_specs = {
        "a": [{"condition": {"pos": 0}, "effect": {"pos": 1}}],
        "b": [{"condition": {"pos": 1}, "effect": {"pos": 2}}],
        "c": [{"condition": {"pos": 2}, "effect": {"pos": 3}}],
    }

    assert compute_symbolic_state_prefixes({"pos": 0}, ["a", "b", "c"], task_specs) == [
        {"pos": 0},
        {"pos": 1},
        {"pos": 2},
        {"pos": 3},
    ]
    assert parse_pair_positions("all", 4) == [0, 1, 2]
    assert parse_pair_positions("0,2", 4) == [0, 2]

    with pytest.raises(ValueError, match="out of range"):
        parse_pair_positions("3", 4)


def test_summarize_records_uses_conditional_policy_terminal_denominator():
    records = [
        {
            "pair_position": 0,
            "current_task": "a",
            "next_task": "b",
            "current_success": True,
            "policy_terminal_next_attempted": True,
            "policy_terminal_next_success": False,
            "oracle_next_success": True,
        },
        {
            "pair_position": 0,
            "current_task": "a",
            "next_task": "b",
            "current_success": False,
            "policy_terminal_next_attempted": False,
            "policy_terminal_next_success": None,
            "oracle_next_success": True,
        },
        {
            "pair_position": 1,
            "current_task": "b",
            "next_task": "c",
            "current_success": True,
            "policy_terminal_next_attempted": True,
            "policy_terminal_next_success": True,
            "oracle_next_success": False,
        },
    ]

    summary = summarize_records(records)
    overall = summary["overall"]

    assert overall["num_pairs"] == 3
    assert overall["current_success_rate"] == pytest.approx(2 / 3)
    assert overall["policy_terminal_next_attempts"] == 2
    assert overall["next_success_from_policy_terminal"] == pytest.approx(1 / 2)
    assert overall["next_success_from_oracle_start"] == pytest.approx(2 / 3)
    assert overall["handoff_gap"] == pytest.approx((2 / 3) - (1 / 2))
    assert summary["by_pair_position"]["0"]["num_pairs"] == 2
    assert summary["by_current_task"]["a"]["current_success_rate"] == pytest.approx(0.5)
    assert summary["by_next_task"]["b"]["next_success_from_oracle_start"] == pytest.approx(1.0)


class FakePolicy:
    def __init__(self):
        self.prompts = []

    def infer(self, element):
        self.prompts.append(element["prompt"])
        actions = np.zeros((2, 7), dtype=np.float32)
        return {"actions": actions}


class FakeEnv:
    def __init__(self):
        self.state = None
        self.step_count = 0

    def reset(self, robot_obs=None, scene_obs=None):
        del scene_obs
        self.state = dict(robot_obs)
        self.step_count = 0

    def get_info(self):
        return {
            "state": dict(self.state),
            "step_count": self.step_count,
        }

    def get_obs(self):
        image_value = self.step_count % 255
        return {
            "rgb_obs": {
                "rgb_static": np.full((4, 4, 3), image_value, dtype=np.uint8),
                "rgb_gripper": np.full((2, 2, 3), image_value, dtype=np.uint8),
            },
            "robot_obs": np.zeros((7,), dtype=np.float32),
        }

    def step(self, action):
        del action
        self.step_count += 1
        return self.get_obs(), 0.0, False, self.get_info()


class FakeTaskReward:
    def __init__(self, succeed_after):
        self.succeed_after = succeed_after

    def get_task_info_for_set(self, start_info, current_info, subtasks):
        subtask = next(iter(subtasks))
        elapsed = current_info["step_count"] - start_info["step_count"]
        if elapsed >= self.succeed_after[subtask]:
            return {subtask: True}
        return {}


def _fake_env_state_for_initial_condition(symbolic_state):
    return dict(symbolic_state), {"unused": True}


def test_run_handoff_diagnostic_writes_jsonl_summary_and_skips_failed_current(tmp_path):
    task_specs = {
        "a": [{"condition": {"pos": 0}, "effect": {"pos": 1}}],
        "b": [{"condition": {"pos": 1}, "effect": {"pos": 2}}],
        "c": [{"condition": {"pos": 2}, "effect": {"pos": 3}}],
        "d": [{"condition": {"pos": 3}, "effect": {"pos": 4}}],
    }
    task_definitions = [({"pos": 0}, ["a", "b", "c", "d"])]
    task_instructions = {
        "a": ["do a"],
        "b": ["do b"],
        "c": ["do c"],
        "d": ["do d"],
    }
    policy = FakePolicy()

    summary = run_handoff_diagnostic(
        env=FakeEnv(),
        policy=policy,
        task_definitions=task_definitions,
        task_instructions=task_instructions,
        task_reward=FakeTaskReward({"a": 1, "b": 1, "c": 10, "d": 1}),
        task_specs=task_specs,
        get_env_state_for_initial_condition=_fake_env_state_for_initial_condition,
        output_dir=tmp_path,
        pair_positions="0,2",
        max_steps_current=2,
        max_steps_next=2,
        action_chunk=1,
        num_save_videos=0,
        video_temp_subsample=1,
    )

    records_path = tmp_path / "handoff_trials.jsonl"
    summary_path = tmp_path / "handoff_summary.json"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(records) == 2
    assert records[0]["current_success"] is True
    assert records[0]["policy_terminal_next_success"] is True
    assert records[0]["oracle_next_success"] is True
    assert records[1]["current_success"] is False
    assert records[1]["policy_terminal_next_attempted"] is False
    assert records[1]["policy_terminal_next_success"] is None
    assert records[1]["oracle_next_success"] is True
    assert policy.prompts.count("do d") == 1

    overall = summary["overall"]
    assert overall["num_pairs"] == 2
    assert overall["current_success_rate"] == pytest.approx(0.5)
    assert overall["policy_terminal_next_attempts"] == 1
    assert overall["next_success_from_policy_terminal"] == pytest.approx(1.0)
    assert overall["next_success_from_oracle_start"] == pytest.approx(1.0)
    assert persisted_summary["overall"] == summary["overall"]
