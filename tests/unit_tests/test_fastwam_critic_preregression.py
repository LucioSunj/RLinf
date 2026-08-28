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

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.fastwam_critic_preregression import (
    DATASET_SCHEMA,
    explained_variance,
    run_critic_preregression,
)
from rlinf.workers.actor import fsdp_actor_worker as worker_module
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class _RouteTracker:
    def __init__(self) -> None:
        self.state = {"next_episode_ids": {0: 1}, "states": {}}

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = copy.deepcopy(state)


class _PreregressionPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_version = 0
        self.gate_parameter = torch.nn.Parameter(torch.tensor([11.0]))
        self.lora_parameter = torch.nn.Parameter(torch.tensor([12.0]))
        self.value_head = ValueHead(
            input_dim=2,
            hidden_sizes=(),
            output_dim=1,
            activation="gelu",
            bias_last=True,
        )
        self.route_tracker = _RouteTracker()

    def set_global_step(self, step: int) -> None:
        self.actor_version = int(step)

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-adaptive-policy-v1",
            "actor_version": self.actor_version,
            "gate": {"parameter": self.gate_parameter.detach().clone()},
            "lora": {"parameter": self.lora_parameter.detach().clone()},
            "value_head": copy.deepcopy(self.value_head.state_dict()),
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        assert state["schema"] == "fastwam-adaptive-policy-v1"
        self.actor_version = int(state["actor_version"])
        self.gate_parameter.data.copy_(state["gate"]["parameter"])
        self.lora_parameter.data.copy_(state["lora"]["parameter"])
        self.value_head.load_state_dict(state["value_head"], strict=True)
        self.route_tracker.load_state_dict(state["route_tracker"])


class _GradScaler:
    def __init__(self) -> None:
        self.state = {"scale": 1.0}

    def state_dict(self) -> dict[str, float]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.state = dict(state)


def _native_worker() -> Any:
    class Worker:
        _checkpoint_cpu_clone = staticmethod(EmbodiedFSDPActor._checkpoint_cpu_clone)
        _fastwam_policy_module = EmbodiedFSDPActor._fastwam_policy_module
        save_checkpoint = EmbodiedFSDPActor.save_checkpoint
        load_checkpoint = EmbodiedFSDPActor.load_checkpoint

        def _fastwam_checkpoint_contract(self) -> dict[str, str]:
            return {"kind": "critic-preregression-unit"}

    worker = Worker()
    worker.model = _PreregressionPolicy()
    worker.cfg = SimpleNamespace(
        runner=SimpleNamespace(resume_dir=None),
        actor=SimpleNamespace(
            model=SimpleNamespace(
                model_type="fastwam_adaptive",
                actor_checkpoint_sha256="a" * 64,
                critic=SimpleNamespace(backbone_checkpoint_sha256="b" * 64),
            )
        ),
    )
    worker.optimizer = torch.optim.AdamW(worker.model.parameters(), lr=1e-3)
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    worker.grad_scaler = _GradScaler()
    worker.optimizer_steps = 0
    worker.version = 0
    worker._rank = 0
    worker._world_size = 1
    worker.is_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    return worker


def _write_dataset(path: Path, *, learnable: bool = True) -> None:
    generator = torch.Generator().manual_seed(19)
    features = torch.randn(320, 2, generator=generator)
    if learnable:
        returns = 1.75 * features[:, 0] - 0.5 * features[:, 1] + 0.25
    else:
        returns = torch.randn(320, generator=generator)
    validation_mask = torch.zeros(320, dtype=torch.bool)
    validation_mask[::5] = True
    torch.save(
        {
            "schema": DATASET_SCHEMA,
            "feature_kind": "value_head_input",
            "features": features,
            "returns": returns,
            "validation_mask": validation_mask,
        },
        path,
    )


def test_preregression_ev_matches_centered_production_definition() -> None:
    targets = torch.tensor([-1.0, 0.0, 1.0])

    assert explained_variance(targets + 7.0, targets) == pytest.approx(1.0)


def test_preregression_reaches_ev_gate_and_native_training_loader_accepts_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _native_worker()
    input_actor = tmp_path / "input/actor"
    worker.save_checkpoint(str(input_actor), step=0)
    input_payload = torch.load(
        input_actor / "rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    dataset = tmp_path / "rollout_features.pt"
    _write_dataset(dataset)
    output_actor = tmp_path / "output/actor"

    manifest = run_critic_preregression(
        dataset_path=dataset,
        input_actor_dir=input_actor,
        output_actor_dir=output_actor,
        epochs=80,
        batch_size=64,
        learning_rate=0.05,
        minimum_heldout_explained_variance=0.35,
    )

    assert manifest["heldout_explained_variance"] >= 0.35
    assert manifest["strict_checkpoint_reload"] is True
    output_payload = torch.load(
        output_actor / "rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert torch.equal(
        output_payload["policy"]["gate"]["parameter"],
        input_payload["policy"]["gate"]["parameter"],
    )
    assert torch.equal(
        output_payload["policy"]["lora"]["parameter"],
        input_payload["policy"]["lora"]["parameter"],
    )
    assert output_payload["optimizer"] == input_payload["optimizer"]

    for parameter in worker.model.value_head.parameters():
        parameter.data.zero_()
    assert worker.load_checkpoint(str(output_actor)) == 0
    loaded = worker.model.value_head.state_dict()
    for name, tensor in output_payload["policy"]["value_head"].items():
        assert torch.equal(loaded[name], tensor)


def test_preregression_misses_ev_gate_without_writing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _native_worker()
    input_actor = tmp_path / "input/actor"
    worker.save_checkpoint(str(input_actor), step=0)
    dataset = tmp_path / "unlearnable.pt"
    _write_dataset(dataset, learnable=False)
    output_actor = tmp_path / "rejected/actor"

    with pytest.raises(RuntimeError, match="missed its gate"):
        run_critic_preregression(
            dataset_path=dataset,
            input_actor_dir=input_actor,
            output_actor_dir=output_actor,
            epochs=2,
            batch_size=64,
            learning_rate=1e-3,
            minimum_heldout_explained_variance=0.99,
        )

    assert not output_actor.exists()
