# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration boundary for the opt-in PAD-RV implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from omegaconf import OmegaConf

from .budget import PAD_PREDICTION_BUDGET_CONTROLLER_TARGET

PAD_FROZEN_BUILDER_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.builder.build_pad_frozen_model"
)
PAD_FROZEN_RUNTIME_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.runtime.PadFrozenLiberoRuntime"
)
PAD_FROZEN_POLICY_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.policy.PadFrozenPolicy"
)
PAD_FROZEN_ACTOR_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.actor.PadFrozenFSDPActor"
)
PAD_FROZEN_ROLLOUT_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.rollout.PadFrozenRolloutWorker"
)
PAD_FROZEN_RUNNER_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.runner.PadFrozenRunner"
)
PAD_FROZEN_ENV_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.env.PadFrozenEnvWorker"
)
PAD_FROZEN_EVAL_RUNNER_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.eval_runner.PadFrozenEvalRunner"
)
PAD_FROZEN_EVAL_COLLECTOR_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.eval_collector.PadFrozenEvalCollector"
)
PAD_FROZEN_TEXT_CACHE_PREFLIGHT_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.preflight."
    "validate_pad_text_cache_coverage"
)
PAD_FROZEN_EGL_INSTANTIATION_TARGET = (
    "rlinf.models.embodiment.wam_policy.pad_rv.egl.instantiate_with_physical_egl"
)


class PadRVStage(str, Enum):
    """Explicit ownership stages; conversion is never an in-place resume."""

    FROZEN = "gate_only_frozen_pair"
    COADAPT = "coadaptive_uncond_delta"

    @classmethod
    def parse(cls, value: "PadRVStage | str") -> "PadRVStage":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as error:
            allowed = ", ".join(stage.value for stage in cls)
            raise ValueError(
                f"Unsupported PAD-RV stage {value!r}; expected {allowed}."
            ) from error


@dataclass(frozen=True, slots=True)
class PadFrozenConfig:
    """Small public switch for the independently implemented Stage 1 path."""

    enabled: bool
    stage: PadRVStage | str
    routing_semantics: str

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("PAD-RV `enabled` must be boolean.")
        object.__setattr__(self, "stage", PadRVStage.parse(self.stage))
        if self.stage is not PadRVStage.FROZEN:
            raise NotImplementedError(
                "PAD-CoAdapt remains locked until matched-budget Gate G3 passes."
            )
        if self.routing_semantics != "current_step":
            raise ValueError("PAD-Frozen requires current_step routing semantics.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PadFrozenConfig":
        if not isinstance(value, Mapping):
            raise TypeError("`algorithm.pad_rv` must be a mapping.")
        required = {"enabled", "stage", "routing_semantics"}
        unknown = sorted(set(value) - required)
        missing = sorted(required - set(value))
        if unknown or missing:
            raise ValueError(
                f"PAD-RV config mismatch: missing={missing}, unknown={unknown}."
            )
        return cls(**{key: value[key] for key in required})


def validate_pad_frozen_training_config(
    cfg: Any,
    *,
    only_eval: bool = False,
) -> PadFrozenConfig:
    """Validate config-selected ownership before any model allocation."""

    config = PadFrozenConfig.from_mapping(OmegaConf.select(cfg, "algorithm.pad_rv"))
    if not config.enabled:
        raise ValueError("PAD-Frozen entrypoint requires `enabled: true`.")
    expected = {
        "rollout.model.builder_target": PAD_FROZEN_BUILDER_TARGET,
        "rollout.model.runtime._target_": PAD_FROZEN_RUNTIME_TARGET,
        "pad_rv_implementation.rollout_worker_target": PAD_FROZEN_ROLLOUT_TARGET,
        "pad_rv_implementation.env_worker_target": PAD_FROZEN_ENV_TARGET,
        "pad_rv_implementation.text_cache_preflight_target": (
            PAD_FROZEN_TEXT_CACHE_PREFLIGHT_TARGET
        ),
        "pad_rv_implementation.policy_target": PAD_FROZEN_POLICY_TARGET,
        "pad_rv_implementation.rollout_init_mode": "serial_rank",
        "env.eval.egl_instantiation_target": PAD_FROZEN_EGL_INSTANTIATION_TARGET,
    }
    if only_eval:
        expected["pad_rv_implementation.evaluation_runner_target"] = (
            PAD_FROZEN_EVAL_RUNNER_TARGET
        )
        expected["runner.evaluation_collector._target_"] = (
            PAD_FROZEN_EVAL_COLLECTOR_TARGET
        )
    else:
        expected.update(
            {
                "actor.model.builder_target": PAD_FROZEN_BUILDER_TARGET,
                "actor.model.runtime._target_": PAD_FROZEN_RUNTIME_TARGET,
                "pad_rv_implementation.actor_target": PAD_FROZEN_ACTOR_TARGET,
                "pad_rv_implementation.runner_target": PAD_FROZEN_RUNNER_TARGET,
                "env.train.egl_instantiation_target": (
                    PAD_FROZEN_EGL_INSTANTIATION_TARGET
                ),
            }
        )
    for field, target in expected.items():
        actual = str(OmegaConf.select(cfg, field))
        if actual != target:
            raise ValueError(f"PAD-Frozen {field} must be {target}, got {actual}.")
    model_roles = ("rollout",) if only_eval else ("actor", "rollout")
    for role in model_roles:
        prefix = f"{role}.model"
        if str(OmegaConf.select(cfg, f"{prefix}.policy_target")) != (
            PAD_FROZEN_POLICY_TARGET
        ):
            raise ValueError(f"PAD-Frozen {prefix} must select its policy target.")
        if str(OmegaConf.select(cfg, f"{prefix}.kv_replay.backend")) != "condition":
            raise ValueError(f"PAD-Frozen {prefix} requires condition-only replay.")
        if (
            OmegaConf.select(cfg, f"{prefix}.kv_replay.gate_kv_sample_budget")
            is not None
        ):
            raise ValueError(f"PAD-Frozen {prefix} cannot sample Action-KV replay.")
        if bool(OmegaConf.select(cfg, f"{prefix}.flow_sde.enabled")):
            raise ValueError(f"PAD-Frozen {prefix} disables Flow-SDE replay.")
        if (
            bool(OmegaConf.select(cfg, f"{prefix}.flow_sde.joint_logprob"))
            or float(OmegaConf.select(cfg, f"{prefix}.flow_sde.noise_level")) != 0.0
        ):
            raise ValueError(f"PAD-Frozen {prefix} has no action-policy logprob path.")
        if OmegaConf.select(cfg, f"{prefix}.uncond_lora") is not None:
            raise ValueError(f"PAD-Frozen {prefix} must erase dynamic Warm-U LoRA.")
        if OmegaConf.select(cfg, f"{prefix}.online_uncond_delta_lora") is not None:
            raise ValueError(f"PAD-Frozen {prefix} cannot instantiate Stage 2 delta.")
        experts = OmegaConf.select(cfg, f"{prefix}.route_action_experts")
        if experts is None or str(experts.get("idm_source", "")) != "parent_checkpoint":
            raise ValueError(f"PAD-Frozen {prefix} requires parent IDM expert.")
        if not str(experts.get("uncond_merged_checkpoint", "")).strip():
            raise ValueError(f"PAD-Frozen {prefix} requires merged Warm-U artifact.")
        if not str(
            OmegaConf.select(cfg, f"{prefix}.runtime.text_embedding_cache_dir")
        ).strip():
            raise ValueError(f"PAD-Frozen {prefix} requires cached text contexts.")
    if only_eval:
        model_cfg = cfg.rollout.model
        if int(cfg.runner.evaluation_collector.get("ledger_shard_count", 0)) != 7:
            raise ValueError("PAD-Frozen GPU 0-6 evaluation requires 7 ledger shards.")
        ordered_reset_ids = OmegaConf.select(
            cfg, "env.eval.ordered_reset_state_ids", default=None
        )
        if ordered_reset_ids is not None and (
            OmegaConf.select(cfg, "env.eval.specific_reset_id", default=None)
            is not None
            or OmegaConf.select(cfg, "env.eval.task_id_filter", default=None)
            is not None
        ):
            raise ValueError(
                "PAD-Frozen ordered_reset_state_ids cannot be combined with "
                "specific_reset_id or task_id_filter."
            )
        if not bool(model_cfg.get("eval_without_critic", False)) or bool(
            model_cfg.critic.get("load_for_eval", False)
        ):
            raise ValueError("PAD-Frozen evaluation must omit the training critic.")
        if float(model_cfg.get("gate_epsilon", -1.0)) != 0.0:
            raise ValueError("PAD-Frozen evaluation requires epsilon=0.")
        from rlinf.models.embodiment.wam_policy.evaluation import (
            EvaluationRoutingConfig,
        )

        EvaluationRoutingConfig(
            mode=str(model_cfg.get("eval_routing_mode", "learned_threshold")),
            idm_threshold=float(model_cfg.get("eval_idm_threshold", 0.5)),
            random_idm_probability=model_cfg.get("eval_random_idm_probability", None),
            random_lag1_autocorrelation=model_cfg.get(
                "eval_random_lag1_autocorrelation", None
            ),
            routing_seed=model_cfg.get("eval_routing_seed", 0),
        )
    if str(OmegaConf.select(cfg, "algorithm.loss_type")) != ("fastwam_gate_only_ppo"):
        raise ValueError("PAD-Frozen requires its dedicated Gate-only loss type.")
    budget = OmegaConf.select(cfg, "algorithm.prediction_budget")
    if budget is None or set(budget) != {
        "enabled",
        "controller_target",
        "target_idm_fraction",
        "dual_lr",
        "proportional_gain",
    }:
        raise ValueError("PAD-Frozen prediction-budget fields changed.")
    if not bool(budget.enabled):
        raise ValueError("PAD-Frozen requires the prediction-budget dual.")
    if str(budget.controller_target) != PAD_PREDICTION_BUDGET_CONTROLLER_TARGET:
        raise ValueError("PAD-Frozen prediction-budget controller target changed.")
    if float(budget.proportional_gain) != 0.0:
        raise ValueError("PAD-Frozen uses the projected dual without a P term.")
    if (
        OmegaConf.select(cfg, "algorithm.fixed_branch_cost.controller", default=None)
        is not None
    ):
        raise ValueError(
            "PAD-Frozen must not compose the generic fastwam_idm_cost_control "
            "group; algorithm.prediction_budget is its sole controller source."
        )
    pi = OmegaConf.select(cfg, "algorithm.fixed_branch_cost.fair_cost.pi")
    if pi is None or bool(pi.get("enabled", True)):
        raise ValueError("PAD-Frozen disables the inherited fair-cost PI controller.")
    if not bool(OmegaConf.select(cfg, "algorithm.fixed_branch_cost.fair_cost.enabled")):
        raise ValueError("PAD-Frozen prediction-budget controller is disabled.")
    if bool(OmegaConf.select(cfg, "actor.enable_sft_co_train", default=False)):
        raise ValueError("PAD-Frozen does not use SFT co-training.")
    if not bool(
        OmegaConf.select(
            cfg,
            "pad_rv_implementation.release_host_memory_after_rollout_init",
            default=False,
        )
    ):
        raise ValueError(
            "PAD-Frozen requires host model-build memory release after rollout init."
        )
    for field in (
        "release_host_memory_after_trajectory_send",
        "release_host_memory_after_trajectory_receive",
    ):
        if not bool(
            OmegaConf.select(
                cfg,
                f"pad_rv_implementation.{field}",
                default=False,
            )
        ):
            raise ValueError(f"PAD-Frozen requires {field}.")
    trajectory_send_mode = str(
        OmegaConf.select(
            cfg,
            "pad_rv_implementation.trajectory_send_mode",
            default="",
        )
    )
    if trajectory_send_mode not in {"concurrent", "serialized"}:
        raise ValueError(
            "PAD-Frozen trajectory_send_mode must be concurrent or serialized."
        )
    if float(OmegaConf.select(cfg, "algorithm.uncond_flow_ppo.loss_weight")) != 0.0:
        raise ValueError("PAD-Frozen requires zero UNCOND Flow-PPO weight.")
    online_bc = OmegaConf.select(cfg, "algorithm.online_idm_bc", default={})
    if (
        bool(online_bc.get("enabled", False))
        or float(online_bc.get("loss_weight", 0.0)) != 0.0
    ):
        raise ValueError("PAD-Frozen disables online IDM-to-UNCOND BC.")
    for field in (
        "runner.bootstrap_uncond_lora_sidecar",
        "runner.bootstrap_uncond_lora_sidecar_sha256",
    ):
        if OmegaConf.select(cfg, field) is not None:
            raise ValueError(f"PAD-Frozen forbids legacy Warm-U bootstrap: {field}.")
    if not only_eval:
        placement = OmegaConf.select(cfg, "cluster.component_placement")
        expected_placement = {"actor": "0-0", "env": "1-6", "rollout": "1-6"}
        actual_placement = OmegaConf.to_container(placement, resolve=True)
        if actual_placement != expected_placement:
            raise ValueError(
                "PAD-Frozen GPU 0-6 placement changed: "
                f"expected {expected_placement}, got {actual_placement}."
            )
    return config
