import numpy as np
import torch

from rlinf.envs.calvin.subtask_collection import (
    CalvinSubtaskEnv,
    CalvinSubtaskSegment,
    CollectedChunk,
    build_trajectory,
    compute_dense_handoff_reward,
    load_calvin_subtask_segments,
    pad_or_trim_actions,
)


def _make_fake_split(tmp_path):
    split_dir = tmp_path / "training"
    (split_dir / "lang_annotations").mkdir(parents=True)
    np.save(split_dir / "ep_start_end_ids.npy", np.asarray([[0, 20]], dtype=np.int64))
    np.save(split_dir / "scene_info.npy", {"calvin_scene_D": [0, 20]})
    annotations = {
        "info": {
            "indx": [(0, 3), (4, 6), (7, 9)],
            "episodes": [],
        },
        "language": {
            "ann": [
                "pull the drawer open",
                "open up the drawer",
                "slide the door left",
            ],
            "task": ["open_drawer", "open_drawer", "move_slider_left"],
            "emb": np.zeros((3, 4), dtype=np.float32),
        },
    }
    np.save(split_dir / "lang_annotations" / "auto_lang_ann.npy", annotations)
    for frame_id in range(12):
        image = np.full((4, 4, 3), frame_id, dtype=np.uint8)
        np.savez(
            split_dir / f"episode_{frame_id:07d}.npz",
            rgb_static=image,
            rgb_gripper=image,
            robot_obs=np.full(15, frame_id, dtype=np.float32),
            scene_obs=np.full(24, frame_id, dtype=np.float32),
            rel_actions=np.full(7, frame_id, dtype=np.float32),
            actions=np.full(7, -frame_id, dtype=np.float32),
        )
    return split_dir


def test_load_calvin_subtask_segments_groups_by_canonical_task_not_prompt(tmp_path):
    _make_fake_split(tmp_path)

    segments = load_calvin_subtask_segments(
        tmp_path,
        split="training",
        max_segments_per_task=1,
        target_window_size=2,
        next_window_size=2,
        resolve_scene_from_scene_info=True,
    )

    assert [segment.canonical_task for segment in segments] == [
        "open_drawer",
        "move_slider_left",
    ]
    assert [segment.prompt_text for segment in segments] == [
        "pull the drawer open",
        "slide the door left",
    ]
    assert segments[0].target_window == (2, 3)
    assert segments[0].next_subtask_start_window == (4, 5)
    assert segments[0].scene == "calvin_scene_D"


def test_load_calvin_subtask_segments_filters_canonical_task(tmp_path):
    _make_fake_split(tmp_path)

    segments = load_calvin_subtask_segments(
        tmp_path,
        split="training",
        canonical_tasks=["open_drawer"],
    )

    assert len(segments) == 2
    assert {segment.prompt_text for segment in segments} == {
        "pull the drawer open",
        "open up the drawer",
    }


def test_get_expert_action_chunk_uses_frame_actions_and_pads(tmp_path):
    split_dir = _make_fake_split(tmp_path)
    env = CalvinSubtaskEnv.__new__(CalvinSubtaskEnv)
    env.split_dir = split_dir
    env.current_segment = CalvinSubtaskSegment(
        split="training",
        segment_index=0,
        owner_long_episode=0,
        start=0,
        end=1,
        prompt_text="open",
        canonical_task="open_drawer",
        target_window=(1,),
    )
    env.elapsed_steps = 0
    env.action_chunk_size = 3
    env.action_key = "rel_actions"

    actions = env.get_expert_action_chunk()

    assert actions.shape == (1, 3, 7)
    assert torch.equal(actions[0, 0], torch.zeros(7))
    assert torch.equal(actions[0, 1], torch.ones(7))
    assert torch.equal(actions[0, 2], torch.ones(7))


def test_pad_or_trim_actions_accepts_unbatched_action_chunks():
    actions = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)

    normalized = pad_or_trim_actions(actions, action_chunk_size=5, action_dim=7)

    assert normalized.shape == (1, 5, 7)
    assert torch.equal(normalized[0, 0], actions[0, :7])
    assert torch.equal(normalized[0, -1], actions[-1, :7])


def test_dense_handoff_reward_is_larger_near_target():
    target = {
        "robot_obs": np.zeros(15, dtype=np.float32),
        "rgb_static": np.zeros((4, 4, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    near = {
        "robot_obs": np.full(15, 0.05, dtype=np.float32),
        "rgb_static": np.zeros((4, 4, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    far = {
        "robot_obs": np.full(15, 3.0, dtype=np.float32),
        "rgb_static": np.full((4, 4, 3), 255, dtype=np.uint8),
        "rgb_gripper": np.full((4, 4, 3), 255, dtype=np.uint8),
    }

    near_reward = compute_dense_handoff_reward(
        near,
        [target],
        beta=0.2,
        state_weight=1.0,
        visual_weight=0.1,
    )
    far_reward = compute_dense_handoff_reward(
        far,
        [target],
        beta=0.2,
        state_weight=1.0,
        visual_weight=0.1,
    )

    assert near_reward > far_reward
    assert 0.0 <= far_reward <= near_reward <= 1.0


def test_build_trajectory_from_collected_calvin_chunks():
    obs = {
        "main_images": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 7),
        "robot_obs": torch.zeros(1, 15),
        "scene_obs": torch.zeros(1, 24),
    }
    chunk = CollectedChunk(
        actions=torch.zeros(1, 35),
        rewards=torch.zeros(1, 5),
        terminations=torch.zeros(1, 5, dtype=torch.bool),
        truncations=torch.zeros(1, 5, dtype=torch.bool),
        dones=torch.zeros(1, 5, dtype=torch.bool),
        curr_obs=obs,
        next_obs=obs,
        versions=torch.zeros(1, 1),
        forward_inputs={"action": torch.zeros(1, 35)},
    )

    trajectory = build_trajectory(
        [chunk],
        max_episode_length=5,
        model_weights_id="model",
    )

    assert trajectory.model_weights_id == "model"
    assert trajectory.actions.shape == (1, 1, 35)
    assert trajectory.rewards.shape == (1, 1, 5)
    assert trajectory.curr_obs["states"].shape == (1, 1, 7)
