# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for truthful same-chunk prediction and terminal-frame recording."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from rlinf.models.embodiment.wam_policy.evaluation_video import (
    EvaluationPredictionRecorder,
    recording_chunk_dir,
)
from rlinf.models.embodiment.wam_policy.libero_runtime import LiberoFastWAMRuntime
from rlinf.runners.fastwam_libero_eval_collector import (
    EvaluationEpisodeSlot,
    EvaluationIdentityBatch,
    FastWAMLiberoEvalCollector,
)
from rlinf.runners.fastwam_libero_video_collector import FastWAMLiberoVideoCollector


def _obs(value=10):
    return {
        "main_images": torch.full((1, 256, 256, 3), value, dtype=torch.uint8),
        "wrist_images": torch.full((1, 256, 256, 3), value + 1, dtype=torch.uint8),
        "states": torch.zeros(1, 8),
        "_fastwam_task_ids": torch.tensor([106]),
        "_fastwam_trial_ids": torch.tensor([0]),
        "_fastwam_reset_state_ids": torch.tensor([5300]),
        "_fastwam_recording_chunk_ids": torch.tensor([2]),
        "_fastwam_action_noise_seeds": torch.tensor([101]),
        "_fastwam_idm_noise_seeds": torch.tensor([102]),
    }


def test_prediction_is_detached_actual_latents_and_decode_draws_no_noise(tmp_path):
    capture = EvaluationPredictionRecorder(str(tmp_path))
    capture.begin(env_obs=_obs(), routes=torch.tensor([1]))
    original = torch.full((1, 2, 3, 4, 5), 7.0, requires_grad=True)
    capture.observe(original)
    assert not capture.latents.requires_grad
    assert capture.latents.data_ptr() != original.data_ptr()
    calls = []

    def decode(latents, *, tiled):
        calls.append(tiled)
        assert torch.equal(latents, original)
        return [
            Image.fromarray(np.full((224, 448, 3), i + 7, np.uint8)) for i in range(9)
        ]

    rng = torch.get_rng_state().clone()
    capture.finish(
        actor=SimpleNamespace(_decode_latents=decode), actor_version=55, tiled=False
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert calls == [False] and torch.all(original == 7)
    directory = recording_chunk_dir(tmp_path, task_id=106, trial_id=0, chunk_id=2)
    with np.load(directory / "prediction.npz") as archive:
        assert archive["frames"].shape == (9, 224, 448, 3)
        assert archive["frames"][8, 0, 0, 0] == 15
    metadata = json.loads((directory / "inference.json").read_text())
    assert metadata["actor_version"] == 55 and metadata["route"] == "idm"
    assert capture.latents is None


def test_uncond_has_no_decode_no_prediction_and_clears_previous_capture(tmp_path):
    capture = EvaluationPredictionRecorder(str(tmp_path))
    capture.latents = torch.ones(1)
    capture.begin(env_obs=_obs(), routes=torch.tensor([0]))
    actor = SimpleNamespace(
        _decode_latents=lambda *a, **kw: pytest.fail("UNCOND decoded a prediction")
    )
    capture.finish(actor=actor, actor_version=55, tiled=False)
    directory = recording_chunk_dir(tmp_path, task_id=106, trial_id=0, chunk_id=2)
    assert not (directory / "prediction.npz").exists()
    assert (
        json.loads((directory / "inference.json").read_text())["prediction_source"]
        == "not_called"
    )


def test_missing_or_duplicate_idm_prediction_fails_before_mislabeling(tmp_path):
    capture = EvaluationPredictionRecorder(str(tmp_path))
    capture.begin(env_obs=_obs(), routes=torch.tensor([1]))
    with pytest.raises(RuntimeError, match="disagree"):
        capture.finish(actor=None, actor_version=55, tiled=False)
    capture.observe(torch.ones(1))
    with pytest.raises(RuntimeError, match="exactly one"):
        capture.observe(torch.ones(1))


def test_standard_hooks_have_no_tensor_or_filesystem_effect():
    tensor = torch.ones(1)
    LiberoFastWAMRuntime._observe_idm_prediction(None, tensor)
    FastWAMLiberoEvalCollector.record_step_observations(None, 0, ())
    assert torch.equal(tensor, torch.ones(1))


def test_video_collector_preserves_input_and_seed_fields(monkeypatch, tmp_path):
    collector = object.__new__(FastWAMLiberoVideoCollector)
    collector.video_recording_dir = tmp_path
    collector._video_snapshots = {}
    monkeypatch.setattr(
        FastWAMLiberoEvalCollector,
        "augment_rollout_input",
        lambda self, data, snap: {**data, "obs": dict(data["obs"])},
    )
    obs = _obs()
    data = {"obs": obs}
    entry = {"task_id": 106, "trial_id": 0, "reset_state_id": 5300}
    snapshot = EvaluationIdentityBatch(
        slots=(
            EvaluationEpisodeSlot(
                stage_id=0, local_env_index=0, env_id=0, chunk_id=2, entry=entry
            ),
        ),
        action_contract=None,
    )
    result = collector.augment_rollout_input(data, snapshot)
    assert result["obs"] is not obs
    for key in (
        "main_images",
        "wrist_images",
        "states",
        "_fastwam_action_noise_seeds",
        "_fastwam_idm_noise_seeds",
    ):
        assert result["obs"][key] is obs[key]
    directory = recording_chunk_dir(tmp_path, task_id=106, trial_id=0, chunk_id=2)
    with np.load(directory / "observation.npz") as archive:
        frame = archive["frame"]
        assert np.all(frame[:, :224] == 10) and np.all(frame[:, 224:] == 11)


def test_terminal_frame_not_replaced_by_reset_frame(tmp_path):
    collector = object.__new__(FastWAMLiberoVideoCollector)
    collector.video_recording_dir = tmp_path
    entry = {"task_id": 106, "trial_id": 0, "reset_state_id": 5300}
    collector._video_snapshots = {
        0: EvaluationIdentityBatch(
            slots=(
                EvaluationEpisodeSlot(
                    stage_id=0, local_env_index=0, env_id=0, chunk_id=2, entry=entry
                ),
            ),
            action_contract=None,
        )
    }
    directory = recording_chunk_dir(tmp_path, task_id=106, trial_id=0, chunk_id=2)
    directory.mkdir(parents=True)
    reset = _obs(200)
    terminal = _obs(50)
    result = (
        [_obs(20), reset],
        torch.zeros(1, 2),
        torch.tensor([[False, True]]),
        torch.zeros(1, 2, dtype=torch.bool),
        [{}, {"final_observation": terminal}],
    )
    collector.record_step_observations(0, result)
    with np.load(directory / "execution.npz") as archive:
        frames = archive["frames"]
        assert np.all(frames[0, :, :224] == 20)
        assert np.all(frames[1, :, :224] == 50)
    assert result[0][-1] is reset
