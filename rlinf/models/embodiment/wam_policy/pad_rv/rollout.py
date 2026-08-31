# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PAD rollout worker with bounded post-build host memory."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.models.embodiment.wam_policy import (
    resolve_fastwam_adaptive_eval_checkpoint,
)
from rlinf.utils.checkpoint_state import (
    FASTWAM_RESUME_AUDIT_SCHEMA,
    FASTWAM_ROLLOUT_RESUME_AUDIT_SENTINEL,
    checkpoint_state_sha256,
)
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.rollout.hf.huggingface_worker import (
    MultiStepRolloutWorker,
    _fastwam_checkpoint_cpu_clone,
)

from .checkpoint import (
    PAD_FROZEN_CHECKPOINT_SCHEMA,
    build_pad_frozen_checkpoint_contract,
    validate_pad_frozen_checkpoint_contract,
    validate_pad_frozen_eval_checkpoint,
)
from .memory import release_pad_host_memory


class PadFrozenRolloutWorker(MultiStepRolloutWorker):
    """Release temporary CPU model-build storage after device materialization."""

    def _load_fastwam_eval_checkpoint_payload(
        self,
        rollout_model_config: Any,
    ) -> dict[str, Any]:
        """Load the Stage 1 actor schema through the rollout interface."""

        checkpoint_path = resolve_fastwam_adaptive_eval_checkpoint(
            self.cfg.runner.ckpt_path,
            rank=self._rank,
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return validate_pad_frozen_eval_checkpoint(payload, rollout_model_config)

    def _restore_fastwam_eval_checkpoint_payload(
        self,
        payload: dict[str, Any],
    ) -> int:
        """Restore PAD Gate weights without inheriting training route state."""

        return self.hf_model.load_eval_checkpoint(
            payload,
            expected_parent_checkpoint_sha256=str(
                self.model_cfg.actor_checkpoint_sha256
            ),
            expected_critic_parent_checkpoint_sha256=None,
        )

    def init_worker(self) -> None:
        """Build the normal rollout policy, then return dead CPU pages to Linux."""

        super().init_worker()
        if not bool(
            self.cfg.pad_rv_implementation.release_host_memory_after_rollout_init
        ):
            raise ValueError("PAD rollout host-memory release was disabled.")
        report = release_pad_host_memory(
            schema="pad-rollout-host-memory-release-v1",
            rank=int(self._rank),
            phase="post_model_initialization",
        )
        print(
            "PAD_ROLLOUT_HOST_MEMORY_RELEASE=" + json.dumps(report, sort_keys=True),
            flush=True,
        )

    def _build_rollout_result(
        self,
        actions: torch.Tensor,
        result: dict[str, Any],
        *,
        final_obs: dict[str, Any] | None = None,
    ) -> RolloutResult:
        """Keep versions batch-shaped while omitting Action-PPO logprobs."""

        if result.get("prev_logprobs") is not None:
            raise ValueError("PAD-Frozen rollout cannot carry Action-PPO logprobs.")
        intervene_flags = result.get("intervene_flags")
        if intervene_flags is None and result.get("expert_label_flag", False):
            intervene_flags = torch.full(
                (actions.shape[0], self.model_cfg.num_action_chunks),
                True,
                dtype=torch.bool,
                device=actions.device,
            )
        return RolloutResult(
            actions=actions,
            prev_logprobs=None,
            prev_values=(result["prev_values"] if self.collect_prev_infos else None),
            bootstrap_values=self.get_bootstrap_values(final_obs),
            intervene_flags=intervene_flags,
            forward_inputs=result["forward_inputs"],
            versions=torch.full(
                (actions.shape[0], 1),
                float(self.version),
                dtype=torch.float32,
                device=actions.device,
            ),
            route_info=result.get("route_info"),
            emitted_gate=result.get("emitted_gate"),
            action_execution_trace=result.get("action_execution_trace"),
        )

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save current-step route and RNG state under the Stage 1 schema."""

        if not hasattr(self.hf_model, "rollout_runtime_state_dict"):
            raise TypeError("PAD-Frozen rollout policy has no runtime-state API.")
        step = int(step)
        rollout_actor_version = int(self.version)
        if step < 1 or rollout_actor_version not in {step - 1, step}:
            raise ValueError(
                "PAD rollout checkpoint version must be the checkpoint step or "
                "its immediately preceding behavior version."
            )
        policy_runtime = self.hf_model.rollout_runtime_state_dict()
        if int(policy_runtime.get("actor_version", -1)) != rollout_actor_version:
            raise ValueError("PAD rollout worker and policy versions disagree at save.")
        payload = {
            "schema": PAD_FROZEN_CHECKPOINT_SCHEMA,
            "owner": "rollout",
            "rank": int(self._rank),
            "world_size": int(self._world_size),
            "step": step,
            "rollout_actor_version": rollout_actor_version,
            "stage_contract": build_pad_frozen_checkpoint_contract(
                self.cfg,
                world_size=int(self._world_size),
            ),
            "policy_runtime": policy_runtime,
            "rng": get_rng_state(),
        }
        payload = _fastwam_checkpoint_cpu_clone(payload)
        os.makedirs(save_path, exist_ok=True)
        target = os.path.join(save_path, f"rank_{self._rank}.pt")
        temporary = f"{target}.tmp"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, load_path: str) -> int:
        """Restore only an exact PAD-Frozen rollout continuation."""

        if not hasattr(self.hf_model, "load_rollout_runtime_state_dict"):
            raise TypeError("PAD-Frozen rollout policy has no runtime-state loader.")
        checkpoint_path = os.path.join(load_path, f"rank_{self._rank}.pt")
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_keys = {
            "schema",
            "owner",
            "rank",
            "world_size",
            "step",
            "rollout_actor_version",
            "stage_contract",
            "policy_runtime",
            "rng",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                f"PAD-Frozen rollout checkpoint keys changed: {sorted(payload)}."
            )
        if (
            payload.get("schema") != PAD_FROZEN_CHECKPOINT_SCHEMA
            or payload.get("owner") != "rollout"
        ):
            raise ValueError("Unsupported PAD-Frozen rollout checkpoint.")
        if int(payload.get("rank", -1)) != int(self._rank) or int(
            payload.get("world_size", -1)
        ) != int(self._world_size):
            raise ValueError("PAD rollout checkpoint rank/world-size mismatch.")
        validate_pad_frozen_checkpoint_contract(
            payload.get("stage_contract"),
            self.cfg,
            world_size=int(self._world_size),
        )
        step = int(payload.get("step", -1))
        rollout_actor_version = int(payload.get("rollout_actor_version", -1))
        if step < 1 or rollout_actor_version not in {step - 1, step}:
            raise ValueError("PAD rollout checkpoint step/version mismatch.")
        policy_runtime = payload.get("policy_runtime")
        if (
            not isinstance(policy_runtime, dict)
            or int(policy_runtime.get("actor_version", -1)) != rollout_actor_version
            or "route_tracker" not in policy_runtime
        ):
            raise ValueError("PAD rollout checkpoint policy runtime is malformed.")
        expected_route_sha256 = checkpoint_state_sha256(policy_runtime["route_tracker"])
        source_rng_sha256 = checkpoint_state_sha256(payload["rng"])
        self.hf_model.load_rollout_runtime_state_dict(policy_runtime)
        self.version = rollout_actor_version
        restored_runtime = self.hf_model.rollout_runtime_state_dict()
        restored_route_sha256 = checkpoint_state_sha256(
            restored_runtime["route_tracker"]
        )
        if restored_route_sha256 != expected_route_sha256:
            raise ValueError(
                "PAD current-step route state changed during rollout load."
            )
        set_rng_state(payload["rng"])
        restored_rng_sha256 = checkpoint_state_sha256(get_rng_state())
        if restored_rng_sha256 != source_rng_sha256:
            raise ValueError("PAD rollout RNG state changed during load.")
        print(
            f"{FASTWAM_ROLLOUT_RESUME_AUDIT_SENTINEL} "
            + json.dumps(
                {
                    "schema": FASTWAM_RESUME_AUDIT_SCHEMA,
                    "checkpoint_schema": PAD_FROZEN_CHECKPOINT_SCHEMA,
                    "stage": "gate_only_frozen_pair",
                    "owner": "rollout",
                    "rank": int(self._rank),
                    "step": step,
                    "actor_version": rollout_actor_version,
                    "route_state_sha256": restored_route_sha256,
                    "rng_sha256": restored_rng_sha256,
                    "status": "PASS",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return step
