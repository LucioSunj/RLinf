# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused tests for the adaptive FastWAM actor checkpoint I/O path."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.workers.actor import fsdp_actor_worker as worker_module
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class _RouteTracker:
    def __init__(self) -> None:
        self.state = {"next_episode_ids": {0: 1}, "states": {}}

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = copy.deepcopy(state)


def test_p8_formal_runner_step_requires_exactly_ten_optimizer_updates() -> None:
    audit = worker_module._validate_p8_formal_optimizer_geometry(
        enabled=True,
        optimizer_steps_before=20,
        optimizer_steps_after=30,
        expected_updates=10,
    )

    assert audit == {
        "optimizer_steps_before": 20,
        "optimizer_steps_after": 30,
        "optimizer_updates": 10,
    }
    assert (
        worker_module._validate_p8_formal_optimizer_geometry(
            enabled=False,
            optimizer_steps_before=20,
            optimizer_steps_after=21,
            expected_updates=-1,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="optimizer geometry changed"):
        worker_module._validate_p8_formal_optimizer_geometry(
            enabled=True,
            optimizer_steps_before=20,
            optimizer_steps_after=29,
            expected_updates=10,
        )


def test_fsdp_optimizer_step_can_fail_before_nonfinite_update() -> None:
    calls = []

    class _Scaler:
        def unscale_(self, optimizer) -> None:
            calls.append(("unscale", optimizer))

        def step(self, optimizer) -> None:
            calls.append(("step", optimizer))

        def update(self) -> None:
            calls.append(("update", None))

    optimizer = SimpleNamespace(param_groups=[])
    manager = SimpleNamespace(
        optimizer_steps=0,
        grad_scaler=_Scaler(),
        optimizer=optimizer,
        _strategy=SimpleNamespace(
            clip_grad_norm_=lambda *, model: torch.tensor(float("nan"))
        ),
        model=object(),
    )

    with pytest.raises(FloatingPointError, match="before optimizer step"):
        FSDPModelManager.optimizer_step(manager, fail_on_nonfinite=True)

    assert calls == [("unscale", optimizer)]
    assert manager.optimizer_steps == 0


def test_p8_formal_route_audit_requires_idm_first_and_uncond_later() -> None:
    route = worker_module.ChunkRouteRecord(
        route_used=torch.tensor([[1, 1], [0, 0], [0, 0]]),
        route_was_forced=torch.tensor([[True, True], [False, False], [False, False]]),
        chunk_ids=torch.tensor([[0, 0], [1, 1], [2, 2]]),
        episode_ids=torch.tensor([[4, 9], [4, 9], [4, 9]]),
        route_source_chunk_ids=torch.tensor([[-1, -1], [0, 0], [1, 1]]),
        actor_versions=torch.zeros(3, 2, dtype=torch.long),
    )

    audit = worker_module._audit_p8_formal_fixed_route(
        route=route,
        loss_mask=torch.ones(3, 2, 10, dtype=torch.bool),
    )

    assert audit["status"] == "PASS"
    assert audit["valid_chunk_count"] == 6
    assert audit["first_chunk_count"] == 2
    assert audit["later_chunk_count"] == 4
    tampered = worker_module.ChunkRouteRecord(
        route_used=route.route_used.clone(),
        route_was_forced=route.route_was_forced,
        chunk_ids=route.chunk_ids,
        episode_ids=route.episode_ids,
        route_source_chunk_ids=route.route_source_chunk_ids,
        actor_versions=route.actor_versions,
    )
    tampered.route_used[1, 0] = int(worker_module.WAMRoute.IDM)
    with pytest.raises(RuntimeError, match="every later chunk"):
        worker_module._audit_p8_formal_fixed_route(
            route=tampered,
            loss_mask=torch.ones(3, 2, 10, dtype=torch.bool),
        )


class _Policy:
    def __init__(self) -> None:
        self.actor_version = 0
        self.weight = torch.tensor([1.0])
        self.route_tracker = _RouteTracker()

    def set_global_step(self, step: int) -> None:
        self.actor_version = int(step)

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "schema": "fastwam-adaptive-policy-v1",
            "actor_version": self.actor_version,
            "weight": self.weight.clone(),
            "route_tracker": self.route_tracker.state_dict(),
        }

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        self.actor_version = int(state["actor_version"])
        self.weight.copy_(state["weight"])
        self.route_tracker.load_state_dict(state["route_tracker"])


class _Stateful:
    def __init__(self, value: float) -> None:
        self.value = torch.tensor([value])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.value.copy_(state["value"])


class _SidecarAdapter:
    def __init__(self, parameter: torch.Tensor) -> None:
        self.parameter = parameter

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {"blocks.0.self_attn.q.lora_B": self.parameter.clone()}

    def load_lora_state_dict(
        self,
        state: dict[str, torch.Tensor],
        *,
        strict: bool,
    ) -> None:
        expected = {"blocks.0.self_attn.q.lora_B"}
        if strict and set(state) != expected:
            raise ValueError("LoRA state key mismatch")
        self.parameter.copy_(state["blocks.0.self_attn.q.lora_B"])

    def load_sidecar(
        self,
        path: str,
        *,
        expected_parent_checkpoint_sha256: str,
        strict: bool,
    ) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if metadata["parent_checkpoint_sha256"] != (expected_parent_checkpoint_sha256):
            raise ValueError("LoRA sidecar parent checkpoint mismatch")
        self.load_lora_state_dict(payload["state_dict"], strict=strict)
        return metadata


class _BootstrapPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self.lora_adapter = _SidecarAdapter(self.weight)
        self.gate = _Stateful(11.0)
        self.critic = SimpleNamespace(value_head=_Stateful(12.0))


_FAKE_MISSING = object()


class _FakeFSDP:
    def __init__(self, initial, *children) -> None:
        self.children = list(children)
        if initial is not _FAKE_MISSING:
            self._is_root = initial

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()


class _RootMutatingPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self.missing = _FakeFSDP(_FAKE_MISSING)
        self.nested = _FakeFSDP(False, self.missing)
        self.root = _FakeFSDP(None, self.nested)

    def modules(self):
        yield self
        yield from self.root.modules()

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        super().load_trainable_state_dict(state)
        self.root._is_root = True
        self.nested._is_root = True
        self.missing._is_root = True


def _checkpoint_worker() -> Any:
    class CheckpointWorker:
        _checkpoint_cpu_clone = staticmethod(EmbodiedFSDPActor._checkpoint_cpu_clone)
        _fastwam_policy_module = EmbodiedFSDPActor._fastwam_policy_module
        bootstrap_fastwam_uncond_lora = EmbodiedFSDPActor.bootstrap_fastwam_uncond_lora
        save_checkpoint = EmbodiedFSDPActor.save_checkpoint
        load_checkpoint = EmbodiedFSDPActor.load_checkpoint

        def _fastwam_checkpoint_contract(self) -> dict[str, str]:
            return {"kind": "unit"}

    worker = CheckpointWorker()
    worker.model = _Policy()
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
    worker.optimizer = _Stateful(2.0)
    worker.lr_scheduler = _Stateful(3.0)
    worker.grad_scaler = _Stateful(4.0)
    worker.optimizer_steps = 1
    worker.version = 0
    worker._rank = 0
    worker._world_size = 1
    worker.is_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    return worker


def _write_bc_sidecar(
    path: Path,
    *,
    parent_sha256: str = "a" * 64,
    bc_step: int = 17,
    bc_config_sha256: str = "c" * 64,
) -> str:
    torch.save(
        {
            "metadata": {
                "schema": "fastwam-regime-lora-v1",
                "parent_checkpoint_sha256": parent_sha256,
                "active_regime": "uncond",
                "rank": 16,
                "alpha": 16.0,
                "dropout": 0.0,
                "target_groups": [],
                "target_names": ["blocks.0.self_attn.q"],
                "extra": {
                    "bc_step": bc_step,
                    "bc_config_sha256": bc_config_sha256,
                },
            },
            "state_dict": {
                "blocks.0.self_attn.q.lora_B": torch.tensor([7.0]),
            },
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fastwam_actor_checkpoint_rank_file_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rng_state = {"cpu": torch.tensor([7], dtype=torch.uint8)}
    restored_rng = []
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(
        worker_module,
        "set_rng_state",
        lambda state: restored_rng.append(state),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    checkpoint_dir = tmp_path / "actor"

    worker.save_checkpoint(str(checkpoint_dir), step=7)

    checkpoint_path = checkpoint_dir / "rank_0.pt"
    assert checkpoint_path.is_file()
    assert not (checkpoint_dir / "rank_0.pt.tmp").exists()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "fastwam-adaptive-rl-checkpoint-v1"
    assert payload["step"] == 7
    assert payload["policy"]["actor_version"] == 7
    assert payload["optimizer_steps"] == 1

    worker.model.weight.fill_(9.0)
    worker.model.set_global_step(99)
    worker.optimizer.value.fill_(9.0)
    worker.lr_scheduler.value.fill_(9.0)
    worker.grad_scaler.value.fill_(9.0)
    worker.optimizer_steps = 99

    assert worker.load_checkpoint(str(checkpoint_dir)) == 7
    assert worker.version == 7
    assert worker.model.actor_version == 7
    assert torch.equal(worker.model.weight, torch.tensor([1.0]))
    assert torch.equal(worker.optimizer.value, torch.tensor([2.0]))
    assert torch.equal(worker.lr_scheduler.value, torch.tensor([3.0]))
    assert torch.equal(worker.grad_scaler.value, torch.tensor([4.0]))
    assert worker.optimizer_steps == 1
    assert len(restored_rng) == 1
    assert torch.equal(restored_rng[0]["cpu"], rng_state["cpu"])
    audit_output = capsys.readouterr().out
    assert "FASTWAM_ACTOR_RESUME_AUDIT" in audit_output
    assert '"route_state_sha256"' in audit_output


def test_p8_actor_checkpoint_uses_v2_and_rejects_old_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([8], dtype=torch.uint8)},
    )
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    worker.cfg.actor.model.uncond_visual_sidecar = SimpleNamespace(enabled=True)
    checkpoint_dir = tmp_path / "actor-p8"
    worker.save_checkpoint(str(checkpoint_dir), step=7)

    checkpoint_path = checkpoint_dir / "rank_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "fastwam-adaptive-rl-checkpoint-v2-p8-a0-kv"

    payload["schema"] = "fastwam-adaptive-rl-checkpoint-v1"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="Unsupported FastWAM adaptive RL"):
        worker.load_checkpoint(str(checkpoint_dir))


def test_fastwam_actor_checkpoint_round_trips_native_step_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng_state = {"cpu": torch.tensor([3], dtype=torch.uint8)}
    restored_rng = []
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(
        worker_module,
        "set_rng_state",
        lambda state: restored_rng.append(state),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    worker.optimizer_steps = 0
    checkpoint_dir = tmp_path / "actor"

    worker.save_checkpoint(str(checkpoint_dir), step=0)

    checkpoint_path = checkpoint_dir / "rank_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["optimizer_steps"] == 0
    assert payload["policy"]["actor_version"] == 0

    worker.model.weight.fill_(9.0)
    worker.model.set_global_step(9)
    worker.optimizer_steps = 9

    assert worker.load_checkpoint(str(checkpoint_dir)) == 0
    assert worker.version == 0
    assert worker.model.actor_version == 0
    assert worker.optimizer_steps == 0
    assert torch.equal(worker.model.weight, torch.tensor([1.0]))
    assert len(restored_rng) == 1
    assert torch.equal(restored_rng[0]["cpu"], rng_state["cpu"])


def _bootstrap_worker(
    tmp_path: Path,
    *,
    loaded_step: int = 0,
    resume_dir: str | None = None,
) -> Any:
    worker = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "runner": {
                "ckpt_path": str(tmp_path / "actor"),
                "resume_dir": resume_dir,
            },
            "actor": {"model": {"model_type": "fastwam_adaptive"}},
        }
    )
    worker.enable_offload = False
    worker.calls = []
    worker.setup_model_and_optimizer = lambda: worker.calls.append(("setup",))

    def load_checkpoint(path: str) -> int:
        worker.calls.append(("load_checkpoint", path))
        return loaded_step

    worker.load_checkpoint = load_checkpoint
    return worker


def test_fastwam_actor_init_bootstraps_native_step_zero_after_setup(
    tmp_path: Path,
) -> None:
    worker = _bootstrap_worker(tmp_path)

    EmbodiedFSDPActor.init_worker(worker)

    assert worker.calls == [
        ("setup",),
        ("load_checkpoint", str(tmp_path / "actor")),
    ]


def _step_zero_export_worker(
    tmp_path: Path,
    *,
    seed: int,
    idm_cost: float,
) -> Any:
    worker = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "runner": {
                "bootstrap_project_checkpoint_dir": str(tmp_path / "step0"),
                "ckpt_path": None,
                "resume_dir": None,
            },
            "actor": {
                "seed": seed,
                "model": {"model_type": "fastwam_adaptive"},
            },
            "algorithm": {
                "fixed_branch_cost": {"idm_cost": idm_cost},
            },
        }
    )
    worker.enable_offload = False
    worker._rank = 0

    def setup_model_and_optimizer() -> None:
        worker.initial_adaptive_state = {
            "gate": torch.rand(8),
            "lora_a": torch.rand(8),
            "lora_b": torch.zeros(8),
            "value_head": torch.rand(8),
        }

    worker.setup_model_and_optimizer = setup_model_and_optimizer
    return worker


def test_step_zero_export_seeds_adaptive_initialization_before_setup(
    tmp_path: Path,
) -> None:
    low_cost = _step_zero_export_worker(tmp_path, seed=42, idm_cost=0.001)
    high_cost = _step_zero_export_worker(tmp_path, seed=42, idm_cost=0.01)

    EmbodiedFSDPActor.init_worker(low_cost)
    EmbodiedFSDPActor.init_worker(high_cost)

    assert low_cost.initial_adaptive_state.keys() == (
        high_cost.initial_adaptive_state.keys()
    )
    for key, low_tensor in low_cost.initial_adaptive_state.items():
        assert torch.equal(low_tensor, high_cost.initial_adaptive_state[key])


def test_step_zero_export_different_seeds_change_random_initialization(
    tmp_path: Path,
) -> None:
    first = _step_zero_export_worker(tmp_path, seed=11, idm_cost=0.001)
    second = _step_zero_export_worker(tmp_path, seed=29, idm_cost=0.001)

    EmbodiedFSDPActor.init_worker(first)
    EmbodiedFSDPActor.init_worker(second)

    assert not torch.equal(
        first.initial_adaptive_state["gate"],
        second.initial_adaptive_state["gate"],
    )


def test_fastwam_actor_init_rejects_nonzero_bootstrap(tmp_path: Path) -> None:
    worker = _bootstrap_worker(tmp_path, loaded_step=1)

    with pytest.raises(ValueError, match="step-zero"):
        EmbodiedFSDPActor.init_worker(worker)


def test_fastwam_actor_init_rejects_bootstrap_plus_resume(tmp_path: Path) -> None:
    worker = _bootstrap_worker(
        tmp_path,
        resume_dir=str(tmp_path / "global_step_1"),
    )

    with pytest.raises(ValueError, match="ckpt_path.*resume_dir"):
        EmbodiedFSDPActor.init_worker(worker)


def test_fastwam_actor_checkpoint_load_restores_fsdp_lazy_root_state(
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
    original_snapshot = worker_module._snapshot_fastwam_fsdp_lazy_root_state
    monkeypatch.setattr(
        worker_module,
        "_snapshot_fastwam_fsdp_lazy_root_state",
        lambda model: original_snapshot(model, fsdp_cls=_FakeFSDP),
    )
    worker = _checkpoint_worker()
    policy = _RootMutatingPolicy()
    worker.model = policy
    checkpoint_dir = tmp_path / "actor"
    worker.save_checkpoint(str(checkpoint_dir), step=7)

    assert worker.load_checkpoint(str(checkpoint_dir)) == 7

    assert policy.root._is_root is None
    assert policy.nested._is_root is False
    assert not hasattr(policy.missing, "_is_root")


def test_fastwam_actor_checkpoint_failed_save_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([7], dtype=torch.uint8)},
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    def fail_after_partial_write(_payload: Any, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(worker_module.torch, "save", fail_after_partial_write)
    worker = _checkpoint_worker()
    checkpoint_dir = tmp_path / "actor"

    with pytest.raises(RuntimeError, match="simulated save failure"):
        worker.save_checkpoint(str(checkpoint_dir), step=7)

    assert not (checkpoint_dir / "rank_0.pt").exists()
    assert not (checkpoint_dir / "rank_0.pt.tmp").exists()


def test_fastwam_actor_checkpoint_rejects_extra_frozen_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng_state = {"cpu": torch.tensor([7], dtype=torch.uint8)}
    monkeypatch.setattr(worker_module, "get_rng_state", lambda: rng_state)
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    checkpoint_dir = tmp_path / "actor"
    worker.save_checkpoint(str(checkpoint_dir), step=7)
    checkpoint_path = checkpoint_dir / "rank_0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["frozen_backbone"] = {"weight": torch.ones(1)}
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="keys changed"):
        worker.load_checkpoint(str(checkpoint_dir))


def test_collective_checkpoint_failure_reaches_healthy_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather_failures(output, local_message) -> None:
        output[:] = [local_message, "ValueError: corrupt rank checkpoint"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_failures)
    with pytest.raises(RuntimeError, match="failed collectively.*corrupt rank"):
        worker_module._raise_fastwam_collective_checkpoint_error(
            None,
            context="actor checkpoint load",
        )


def test_collective_checkpoint_success_is_one_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    calls = []

    def gather_success(output, local_message) -> None:
        calls.append(local_message)
        output[:] = [None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_success)
    worker_module._raise_fastwam_collective_checkpoint_error(
        None,
        context="actor checkpoint save",
    )
    assert calls == [None]


def test_fastwam_actor_bc_bootstrap_preserves_fresh_gate_and_value_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_rng_state",
        lambda: {"cpu": torch.tensor([5], dtype=torch.uint8)},
    )
    monkeypatch.setattr(worker_module, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    worker = _checkpoint_worker()
    worker.optimizer_steps = 0
    worker.model = _BootstrapPolicy()
    sidecar = tmp_path / "bc-lora.pt"
    digest = _write_bc_sidecar(sidecar)
    gate_before = worker.model.gate.value.clone()
    value_before = worker.model.critic.value_head.value.clone()

    provenance = worker.bootstrap_fastwam_uncond_lora(str(sidecar), digest)

    assert torch.equal(worker.model.weight, torch.tensor([7.0]))
    assert torch.equal(worker.model.gate.value, gate_before)
    assert torch.equal(worker.model.critic.value_head.value, value_before)
    assert provenance == {
        "schema": "fastwam-uncond-bc-bootstrap-v1",
        "bc_step": 17,
        "bc_config_sha256": "c" * 64,
        "sidecar_sha256": digest,
        "parent_checkpoint_sha256": "a" * 64,
    }

    checkpoint_dir = tmp_path / "actor"
    worker.save_checkpoint(str(checkpoint_dir), step=0)
    payload = torch.load(
        checkpoint_dir / "rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["step"] == 0
    assert payload["optimizer_steps"] == 0
    assert payload["bc_bootstrap"] == provenance

    worker.model.weight.fill_(3.0)
    worker._fastwam_bc_bootstrap = None
    assert worker.load_checkpoint(str(checkpoint_dir)) == 0
    assert torch.equal(worker.model.weight, torch.tensor([7.0]))
    assert worker._fastwam_bc_bootstrap == provenance


def test_fastwam_actor_bc_bootstrap_failure_is_atomic(tmp_path: Path) -> None:
    worker = _checkpoint_worker()
    worker.optimizer_steps = 0
    worker.model = _BootstrapPolicy()
    sidecar = tmp_path / "bad-metadata.pt"
    digest = _write_bc_sidecar(sidecar, bc_config_sha256="invalid")

    with pytest.raises(ValueError, match="bc_config_sha256"):
        worker.bootstrap_fastwam_uncond_lora(str(sidecar), digest)

    assert torch.equal(worker.model.weight, torch.tensor([1.0]))
    assert not hasattr(worker, "_fastwam_bc_bootstrap")


@pytest.mark.parametrize(
    ("version", "optimizer_steps", "resume_dir", "message"),
    [
        (1, 0, None, "RL step/version"),
        (0, 1, None, "optimizer_steps"),
        (0, 0, "/resume", "resume"),
    ],
)
def test_fastwam_actor_bc_bootstrap_requires_pristine_rl_state(
    tmp_path: Path,
    version: int,
    optimizer_steps: int,
    resume_dir: str | None,
    message: str,
) -> None:
    worker = _checkpoint_worker()
    worker.model = _BootstrapPolicy()
    worker.version = version
    worker.model.actor_version = version
    worker.optimizer_steps = optimizer_steps
    worker.cfg.runner.resume_dir = resume_dir
    sidecar = tmp_path / "bc-lora.pt"
    digest = _write_bc_sidecar(sidecar)

    with pytest.raises(ValueError, match=message):
        worker.bootstrap_fastwam_uncond_lora(str(sidecar), digest)


def test_frozen_gate_p8_optimizer_metrics_require_only_three_families() -> None:
    worker = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "fastwam_adaptive",
                    "gate_trainable": False,
                    "uncond_visual_sidecar": {"enabled": True},
                }
            }
        }
    )
    worker.optimizer = SimpleNamespace(
        param_groups=[
            {"name": "uncond_lora", "lr": 1.0e-5},
            {"name": "wan_current_refiner", "lr": 1.0e-5},
            {"name": "value_head", "lr": 1.0e-4},
        ]
    )

    metrics = worker._optimizer_metrics(grad_norm=1.25, lr_list=[])

    assert metrics == {
        "actor/grad_norm": 1.25,
        "uncond_flow/lora_lr": 1.0e-5,
        "uncond_flow/refiner_lr": 1.0e-5,
        "critic/lr": 1.0e-4,
    }


def test_p8_formal_first_update_requires_finite_nonzero_three_family_delta() -> None:
    before = {
        "uncond_lora": (torch.zeros(2),),
        "wan_current_refiner": (torch.zeros(3),),
        "value_head": (torch.zeros(1),),
    }
    after = {
        name: tuple(tensor + float(index + 1) for tensor in tensors)
        for index, (name, tensors) in enumerate(before.items())
    }

    summary = EmbodiedFSDPActor._summarize_p8_formal_optimizer_update(
        before,
        after,
    )

    assert set(summary) == set(before)
    assert all(item["update_l2_norm"] > 0 for item in summary.values())
    unchanged = dict(after)
    unchanged["value_head"] = before["value_head"]
    with pytest.raises(RuntimeError, match="did not update"):
        EmbodiedFSDPActor._summarize_p8_formal_optimizer_update(
            before,
            unchanged,
        )

    nonfinite = dict(after)
    nonfinite["uncond_lora"] = (torch.tensor([float("nan"), 1.0]),)
    with pytest.raises(FloatingPointError, match="non-finite"):
        EmbodiedFSDPActor._summarize_p8_formal_optimizer_update(
            before,
            nonfinite,
        )

    unchanged_summary = EmbodiedFSDPActor._summarize_p8_formal_optimizer_update(
        before,
        before,
        require_nonzero=False,
    )
    assert all(item["finite"] is True for item in unchanged_summary.values())
    assert all(item["update_l2_norm"] == 0.0 for item in unchanged_summary.values())


def test_p8_formal_frozen_manifest_detects_in_place_parameter_change() -> None:
    model = torch.nn.Module()
    model.trainable = torch.nn.Parameter(torch.ones(2))
    model.frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    worker = object.__new__(EmbodiedFSDPActor)
    worker.model = model
    worker.optimizer = SimpleNamespace(
        param_groups=[{"name": "trainable", "params": [model.trainable]}]
    )

    before = worker._capture_p8_formal_frozen_state_manifest()
    with torch.no_grad():
        model.frozen.add_(1.0)
    after = worker._capture_p8_formal_frozen_state_manifest()

    assert before["state_manifest_sha256"] != after["state_manifest_sha256"]
