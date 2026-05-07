import json

import numpy as np
import torch

from rlinf.envs.robotwin.subtask_collection import (
    CollectedChunk,
    RoboTwinStackThreeSubtaskEnv,
    SubtaskSegment,
    build_trajectory,
    load_episode_prompt,
    load_seed_list,
    load_subtask_segments,
    pad_or_trim_actions,
)


class _FakeTask:
    def __init__(self, stall_after: int | None = None):
        self.take_action_cnt = 0
        self.eval_success = False
        self.step_lim = 999
        self.stall_after = stall_after
        self.actions: list[np.ndarray] = []
        self.action_types: list[str] = []

    def take_action(self, action, action_type="qpos"):
        if self.stall_after is not None and self.take_action_cnt >= self.stall_after:
            return
        self.actions.append(np.asarray(action).copy())
        self.action_types.append(action_type)
        self.take_action_cnt += 1


def _make_env_with_fake_task(
    task: _FakeTask | None = None,
) -> RoboTwinStackThreeSubtaskEnv:
    env = RoboTwinStackThreeSubtaskEnv.__new__(RoboTwinStackThreeSubtaskEnv)
    env.task = task if task is not None else _FakeTask()
    return env


def _segment(start_step: int = 0, num_steps: int = 10) -> SubtaskSegment:
    return SubtaskSegment(
        episode_idx=0,
        subtask_id=0,
        name="place_red",
        arm="left",
        start_step=start_step,
        end_step=start_step + num_steps,
        action_slice=(1, 1 + num_steps),
        num_steps=num_steps,
        hdf5_file="data/episode0.hdf5",
    )


def test_load_seed_list_preserves_episode_to_seed_mapping(tmp_path):
    seed_path = tmp_path / "seed.txt"
    seed_path.write_text("0 1 2 3 5")

    assert load_seed_list(seed_path) == [0, 1, 2, 3, 5]


def test_load_subtask_segments_filters_and_sorts(tmp_path):
    manifest = {
        "episodes": {
            "episode_2": {
                "file": "data/episode2.hdf5",
                "valid": True,
                "subtasks": [
                    {
                        "subtask_id": 0,
                        "name": "first",
                        "arm": "left",
                        "start_step": 0,
                        "end_step": 10,
                        "action_slice": [1, 11],
                        "num_steps": 10,
                    }
                ],
            },
            "episode_0": {
                "file": "data/episode0.hdf5",
                "valid": True,
                "subtasks": [
                    {
                        "subtask_id": 1,
                        "name": "second",
                        "arm": "right",
                        "start_step": 10,
                        "end_step": 20,
                        "action_slice": [11, 21],
                        "num_steps": 10,
                    }
                ],
            },
            "episode_1": {
                "file": "data/episode1.hdf5",
                "valid": False,
                "subtasks": [],
            },
        }
    }
    manifest_path = tmp_path / "subtask_segments.json"
    manifest_path.write_text(json.dumps(manifest))

    segments = load_subtask_segments(
        manifest_path, episode_ids=[0, 2], subtask_ids=[1]
    )

    assert len(segments) == 1
    assert segments[0].episode_idx == 0
    assert segments[0].subtask_id == 1
    assert segments[0].action_slice == (11, 21)


def test_load_episode_prompt_uses_split_and_index(tmp_path):
    instruction_dir = tmp_path / "instructions"
    instruction_dir.mkdir()
    (instruction_dir / "episode3.json").write_text(
        json.dumps({"seen": ["a", "b"], "unseen": ["c"]})
    )

    assert load_episode_prompt(tmp_path, 3, split="seen", prompt_index=3) == "b"
    assert load_episode_prompt(tmp_path, 3, split="unseen", prompt_index=0) == "c"


def test_pad_or_trim_actions_normalizes_to_chunk_shape():
    actions = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)

    padded = pad_or_trim_actions(actions, action_chunk_size=5, action_dim=4)

    assert padded.shape == (2, 5, 4)
    assert torch.equal(padded[:, 0, :], actions[:, 0, :4])
    assert torch.equal(padded[:, -1, :], actions[:, 2, :4])


def test_execute_qpos_actions_uses_robotwin_take_action_interface():
    task = _FakeTask()
    env = _make_env_with_fake_task(task)
    actions = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)

    executed = env._execute_qpos_actions(
        actions, stop_on_subtask_success=False
    )

    assert not hasattr(task, "gen_sparse_reward_data")
    assert executed == 3
    assert task.take_action_cnt == 3
    assert task.action_types == ["qpos", "qpos", "qpos"]
    np.testing.assert_array_equal(np.stack(task.actions), actions)


def test_execute_qpos_actions_stops_when_take_action_count_stalls():
    task = _FakeTask(stall_after=1)
    env = _make_env_with_fake_task(task)
    actions = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)

    executed = env._execute_qpos_actions(
        actions, stop_on_subtask_success=False
    )

    assert executed == 1
    assert task.take_action_cnt == 1
    np.testing.assert_array_equal(np.stack(task.actions), actions[:1])


def test_replay_expert_prefix_executes_exact_prefix_actions():
    task = _FakeTask()
    env = _make_env_with_fake_task(task)
    env.replay_chunk_size = 2
    env.expert_vectors = np.arange(6 * 14, dtype=np.float64).reshape(6, 14)

    env._replay_expert_prefix(_segment(start_step=3))

    np.testing.assert_array_equal(
        np.stack(task.actions),
        env.expert_vectors[1:4],
    )


def test_step_chunk_marks_success_at_actual_executed_step():
    task = _FakeTask()
    env = _make_env_with_fake_task(task)
    env.current_segment = _segment()
    env.max_steps = 10
    env.elapsed_steps = 0
    env.check_subtask_success = lambda: env.task.take_action_cnt >= 2
    env.get_obs = lambda: {
        "main_images": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 14),
    }
    env.get_diagnostics = lambda **kwargs: kwargs
    actions = torch.arange(4 * 14, dtype=torch.float32).reshape(1, 4, 14)

    _obs, rewards, terminations, truncations, info = env.step_chunk(actions)

    assert env.elapsed_steps == 2
    assert len(task.actions) == 2
    assert rewards[0, 1].item() == 1.0
    assert terminations[0, 1].item() is True
    assert not terminations[0, 2:].any().item()
    assert not rewards[0, 2:].any().item()
    assert not truncations.any().item()
    assert info["success"] is True


def test_build_trajectory_from_collected_chunks():
    obs = {
        "main_images": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 14),
    }
    chunk = CollectedChunk(
        actions=torch.zeros(1, 700),
        rewards=torch.zeros(1, 50),
        terminations=torch.zeros(1, 50, dtype=torch.bool),
        truncations=torch.zeros(1, 50, dtype=torch.bool),
        dones=torch.zeros(1, 50, dtype=torch.bool),
        curr_obs=obs,
        next_obs=obs,
        versions=torch.zeros(1, 1),
        forward_inputs={"action": torch.zeros(1, 700)},
    )

    trajectory = build_trajectory(
        [chunk], max_episode_length=50, model_weights_id="model"
    )

    assert trajectory.model_weights_id == "model"
    assert trajectory.actions.shape == (1, 1, 700)
    assert trajectory.rewards.shape == (1, 1, 50)
    assert trajectory.curr_obs["states"].shape == (1, 1, 14)
