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

"""Focused tests for FastWAM rollout-owned runtime checkpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.rollout.hf import huggingface_worker as worker_module
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _Policy:
    def __init__(self) -> None:
        self.actor_version = 0
        self.route_state = {
            "next_episode_id": 1,
            "states": {
                0: {
                    "episode_id": 0,
                    "chunk_id": 2,
                    "force_next_idm": False,
                    "pending": {
                        "route": 0,
                        "source_chunk_id": 1,
                        "episode_id": 0,
                        "actor_version": 0,
                    },
                }
            },
        }

    def rollout_runtime_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-adaptive-rollout-policy-runtime-v1",
            "actor_version": self.actor_version,
            "route_tracker": self.route_state,
        }

    def load_rollout_runtime_state_dict(self, payload: dict[str, Any]) -> None:
        if payload["schema"] != "fastwam-adaptive-rollout-policy-runtime-v1":
            raise ValueError("bad policy runtime schema")
        self.actor_version = int(payload["actor_version"])
        self.route_state = payload["route_tracker"]


def _worker() -> MultiStepRolloutWorker:
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker.model_cfg = SimpleNamespace(
        model_type="fastwam_adaptive",
        actor_checkpoint_sha256="a" * 64,
        critic=SimpleNamespace(backbone_checkpoint_sha256="b" * 64),
    )
    worker.hf_model = _Policy()
    worker.version = 0
    worker._rank = 0
    worker._world_size = 1
    worker._fastwam_checkpoint_contract = lambda: {"kind": "unit"}
    return worker


def test_rollout_runtime_checkpoint_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng_state = {"cpu": torch.tensor([7], dtype=torch.uint8)}
    restored_rng: list[dict[str, Any]] = []
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(
        worker_module,
        "set_rng_state",
        lambda state: restored_rng.append(state),
    )
    worker = _worker()
    checkpoint_dir = tmp_path / "rollout"

    worker.save_checkpoint(str(checkpoint_dir), step=1)

    checkpoint_path = checkpoint_dir / "rank_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "fastwam-adaptive-rollout-runtime-v1"
    assert payload["step"] == 1
    assert payload["rollout_actor_version"] == 0
    assert set(payload["policy_runtime"]) == {
        "schema",
        "actor_version",
        "route_tracker",
    }
    assert (
        "gate" not in payload and "lora" not in payload and "value_head" not in payload
    )
    assert not (checkpoint_dir / "rank_0.pt.tmp").exists()

    worker.version = 99
    worker.hf_model.actor_version = 99
    worker.hf_model.route_state = {"next_episode_id": 0, "states": {}}

    assert worker.load_checkpoint(str(checkpoint_dir)) == 1
    assert worker.version == 0
    assert worker.hf_model.actor_version == 0
    assert worker.hf_model.route_state["next_episode_id"] == 1
    assert len(restored_rng) == 1
    assert torch.equal(restored_rng[0]["cpu"], rng_state["cpu"])


def test_rollout_runtime_checkpoint_rejects_extra_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )
    worker = _worker()
    checkpoint_dir = tmp_path / "rollout"
    worker.save_checkpoint(str(checkpoint_dir), step=1)
    checkpoint_path = checkpoint_dir / "rank_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["frozen_backbone"] = {"weight": torch.ones(1)}
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="keys changed"):
        worker.load_checkpoint(str(checkpoint_dir))


def test_rollout_runtime_failed_save_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )

    def fail_after_partial_write(_payload: Any, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated rollout save failure")

    monkeypatch.setattr(worker_module.torch, "save", fail_after_partial_write)
    worker = _worker()
    checkpoint_dir = tmp_path / "rollout"

    with pytest.raises(RuntimeError, match="simulated rollout save failure"):
        worker.save_checkpoint(str(checkpoint_dir), step=1)

    assert not (checkpoint_dir / "rank_0.pt").exists()
    assert not (checkpoint_dir / "rank_0.pt.tmp").exists()


def test_eval_checkpoint_contract_fails_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_sha256 = "a" * 64
    live_model = OmegaConf.create(
        {
            "model_type": "fastwam_adaptive",
            "precision": "bf16",
            "init_device": "cpu",
            "actor_checkpoint": "/parents/fastwam.pt",
            "actor_checkpoint_sha256": parent_sha256,
            "model_path": "/parents/fastwam.pt",
            "fastwam": {
                "load_text_encoder": False,
                "action_dit_config": {"num_layers": 30},
            },
            "uncond_lora": {"rank": 16, "alpha": 16.0},
            "gate": {
                "hidden_dim": 256,
                "share_blocks": False,
                "denoise_last_n": 1,
                "layer_taps": {
                    "mode": "all",
                    "last_n": None,
                    "indices": None,
                },
            },
            "gate_epsilon": 0.0,
            "gate_temperature": 1.0,
            "runtime": {"text_embedding_cache_dir": "/cache"},
            "critic": {"load_for_eval": False},
        }
    )
    checkpoint_model = OmegaConf.to_container(live_model, resolve=True)
    checkpoint_model["gate"]["hidden_dim"] = 128
    checkpoint_path = tmp_path / "rank_0.pt"
    torch.save(
        {
            "schema": "fastwam-adaptive-rl-checkpoint-v1",
            "parent_checkpoint_sha256": parent_sha256,
            "contract": {"model": checkpoint_model},
        },
        checkpoint_path,
    )
    cfg = OmegaConf.create(
        {
            "runner": {"ckpt_path": str(checkpoint_path)},
            "rollout": {"model": live_model},
        }
    )
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker.cfg = cfg
    worker.model_cfg = cfg.rollout.model
    worker._rank = 0
    model_construction_calls = []
    monkeypatch.setattr(
        worker_module,
        "get_model",
        lambda _cfg: model_construction_calls.append(True),
    )

    with pytest.raises(ValueError, match=r"gate\.hidden_dim"):
        worker.init_worker()

    assert model_construction_calls == []
