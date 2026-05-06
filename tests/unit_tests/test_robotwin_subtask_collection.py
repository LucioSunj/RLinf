import json

import torch

from rlinf.envs.robotwin.subtask_collection import (
    CollectedChunk,
    build_trajectory,
    load_episode_prompt,
    load_seed_list,
    load_subtask_segments,
    pad_or_trim_actions,
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
