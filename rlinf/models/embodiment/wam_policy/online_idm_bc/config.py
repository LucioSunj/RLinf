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

"""Configuration and tensor-key contract for online IDM-to-UNCOND BC."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

ONLINE_IDM_BC_TEACHER_ACTIONS = "online_idm_bc_teacher_actions"
ONLINE_IDM_BC_TEACHER_PRESENT = "online_idm_bc_teacher_present"
ONLINE_IDM_BC_SAMPLE_IDENTITIES = "online_idm_bc_sample_identities"
ONLINE_IDM_BC_TEACHER_SECONDS = "online_idm_bc_teacher_seconds"
ONLINE_IDM_BC_TEACHER_BYTES = "online_idm_bc_teacher_bytes"
ONLINE_IDM_BC_FLOW_VALID = "online_idm_bc_flow_valid"

ONLINE_IDM_BC_FORWARD_KEYS = (
    ONLINE_IDM_BC_TEACHER_ACTIONS,
    ONLINE_IDM_BC_TEACHER_PRESENT,
    ONLINE_IDM_BC_SAMPLE_IDENTITIES,
    ONLINE_IDM_BC_TEACHER_SECONDS,
    ONLINE_IDM_BC_TEACHER_BYTES,
)

ONLINE_IDM_BC_RUNTIME_TARGET = (
    "rlinf.models.embodiment.wam_policy.online_idm_bc.runtime."
    "OnlineIDMTeacherLiberoRuntime"
)
ONLINE_IDM_BC_POLICY_TARGET = (
    "rlinf.models.embodiment.wam_policy.online_idm_bc.policy.OnlineIDMBCFastWAMPolicy"
)
ONLINE_IDM_BC_ACTOR_TARGET = (
    "rlinf.models.embodiment.wam_policy.online_idm_bc.actor.OnlineIDMBCFSDPActor"
)


@dataclass(frozen=True, slots=True)
class OnlineIDMBCConfig:
    """Loss configuration for the opt-in online teacher objective."""

    enabled: bool
    loss_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Online IDM BC `enabled` must be a boolean.")
        if not math.isfinite(float(self.loss_weight)) or self.loss_weight < 0.0:
            raise ValueError(
                "Online IDM BC `loss_weight` must be finite and nonnegative."
            )
        if self.enabled and self.loss_weight <= 0.0:
            raise ValueError("Enabled online IDM BC requires a positive loss weight.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OnlineIDMBCConfig":
        """Materialize the exact two-field public configuration."""

        if not isinstance(value, Mapping):
            raise TypeError("`algorithm.uncond_idm_bc` must be a mapping.")
        unknown = sorted(set(value) - {"enabled", "loss_weight"})
        if unknown:
            raise ValueError(f"Unknown online IDM BC configuration fields: {unknown}.")
        if "enabled" not in value or "loss_weight" not in value:
            raise ValueError(
                "Online IDM BC configuration requires `enabled` and `loss_weight`."
            )
        return cls(
            enabled=value["enabled"],
            loss_weight=float(value["loss_weight"]),
        )


def validate_online_idm_bc_training_config(cfg: Any) -> OnlineIDMBCConfig:
    """Fail closed at the dedicated training-entrypoint boundary."""

    raw_config = OmegaConf.select(cfg, "algorithm.uncond_idm_bc")
    config = OnlineIDMBCConfig.from_mapping(raw_config)
    if not config.enabled:
        raise ValueError("Online IDM BC training requires `enabled: true`.")
    if bool(OmegaConf.select(cfg, "actor.enable_sft_co_train", default=False)):
        raise ValueError("Online IDM BC must not use the existing SFT co-train path.")
    if bool(OmegaConf.select(cfg, "reward.use_reward_model", default=False)):
        raise ValueError("Online IDM BC preserves the environment-only reward path.")
    if str(OmegaConf.select(cfg, "algorithm.loss_type")) != "fastwam_dual_ppo":
        raise ValueError("Online IDM BC requires the existing FastWAM dual-PPO loss.")
    micro_batch_size = int(OmegaConf.select(cfg, "actor.micro_batch_size"))
    if micro_batch_size not in {1, 4}:
        raise ValueError(
            "The approved online IDM BC diagnostic requires microbatch 1 or 4."
        )
    for field, expected in {
        "online_idm_bc_implementation.actor_target": ONLINE_IDM_BC_ACTOR_TARGET,
        "online_idm_bc_implementation.policy_target": ONLINE_IDM_BC_POLICY_TARGET,
    }.items():
        actual = str(OmegaConf.select(cfg, field))
        if actual != expected:
            raise ValueError(f"Online IDM BC {field} must be {expected}, got {actual}.")

    actor_seed = OmegaConf.select(cfg, "actor.seed")
    if isinstance(actor_seed, bool) or not isinstance(actor_seed, int):
        raise TypeError("Online IDM BC requires an integer actor seed.")
    for owner in ("actor", "rollout"):
        target = str(OmegaConf.select(cfg, f"{owner}.model.runtime._target_"))
        if target != ONLINE_IDM_BC_RUNTIME_TARGET:
            raise ValueError(
                f"Online IDM BC {owner} runtime must be "
                f"{ONLINE_IDM_BC_RUNTIME_TARGET}, got {target}."
            )
        sampling_seed = OmegaConf.select(
            cfg,
            f"{owner}.model.formal_training_sampling_seed",
        )
        if sampling_seed != actor_seed:
            raise ValueError(
                f"Online IDM BC {owner} formal sampling seed must equal "
                f"actor.seed ({actor_seed}), got {sampling_seed}."
            )
    return config
