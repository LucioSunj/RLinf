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

"""Small fail-fast contracts shared by RLinf configuration entry points."""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def validate_pi05_critic_artifact_config(
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> None:
    path = str(checkpoint_path).strip()
    digest = str(checkpoint_sha256).strip().lower()
    if not path:
        raise ValueError("The pi0.5 critic requires a non-empty checkpoint path.")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            "The pi0.5 critic requires a 64-character hexadecimal SHA-256."
        )


def validate_fastwam_kv_weight_sync(backend: str, weight_sync_interval: int) -> None:
    backend = str(backend).lower()
    interval = int(weight_sync_interval)
    if interval < 1:
        raise ValueError("`runner.weight_sync_interval` must be positive.")
    if backend == "recompute" and interval != 1:
        raise ValueError(
            "FastWAM K/V recomputation requires `runner.weight_sync_interval: 1` "
            "so the rollout actor version can be reconstructed exactly."
        )


def validate_libero_terminal_reward_config(
    *,
    ignore_terminations: bool,
    use_rel_reward: bool,
) -> None:
    if bool(ignore_terminations) and not bool(use_rel_reward):
        raise ValueError(
            "FastWAM adaptive forbids `ignore_terminations: true` with "
            "`use_rel_reward: false` because terminal success rewards would repeat."
        )


def _resolved_checkpoint_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _selected_checkpoint_values(owner: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: _resolved_checkpoint_value(OmegaConf.select(owner, key, default=None))
        for key in keys
    }


_FASTWAM_EVAL_RUNTIME_ONLY_PATHS = (
    "actor_checkpoint",
    "model_path",
    "init_device",
    "fastwam.load_text_encoder",
    "runtime.text_embedding_cache_dir",
    "gate_epsilon",
    "eval_routing_mode",
    "eval_idm_threshold",
    "eval_random_idm_probability",
    "eval_random_lag1_autocorrelation",
    "eval_routing_seed",
    "eval_microbatch_size",
    "eval_timing_cuda_synchronize",
    "decision_telemetry_enabled",
    "eval_detailed_timing_output",
    "eval_without_critic",
)


def _resolved_mapping(value: Any, *, name: str) -> dict[str, Any]:
    resolved = _resolved_checkpoint_value(value)
    if not isinstance(resolved, Mapping):
        raise TypeError(f"{name} must resolve to a mapping.")
    return copy.deepcopy(dict(resolved))


def _remove_nested_path(mapping: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    owner: Any = mapping
    for part in parts[:-1]:
        if not isinstance(owner, Mapping) or part not in owner:
            return
        owner = owner[part]
    if isinstance(owner, dict):
        owner.pop(parts[-1], None)


def build_fastwam_eval_model_contract(
    model_cfg: Any,
    *,
    load_critic: bool,
) -> dict[str, Any]:
    """Project the resolved model config onto standalone-eval semantics.

    New model fields are compared by default. Only explicitly enumerated
    runtime/evaluation fields are removed. A standalone evaluator without a
    critic intentionally excludes the complete critic subtree; when the critic
    is loaded, only its artifact path and load switch may differ.
    """

    model = _resolved_mapping(model_cfg, name="FastWAM model config")
    for dotted_path in _FASTWAM_EVAL_RUNTIME_ONLY_PATHS:
        _remove_nested_path(model, dotted_path)
    if load_critic:
        _remove_nested_path(model, "critic.load_for_eval")
        _remove_nested_path(model, "critic.backbone.model_path")
    else:
        model.pop("critic", None)
    return {
        "schema": "fastwam-adaptive-eval-model-contract-v1",
        "model": model,
    }


def _flatten_contract(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value, key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_contract(value[key], prefix=child_prefix))
        return flattened
    return {prefix: value}


def validate_fastwam_eval_model_contract(
    checkpoint_model_cfg: Any,
    live_model_cfg: Any,
    *,
    load_critic: bool,
) -> dict[str, Any]:
    """Require exact standalone-eval structural compatibility."""

    checkpoint_contract = build_fastwam_eval_model_contract(
        checkpoint_model_cfg,
        load_critic=load_critic,
    )
    live_contract = build_fastwam_eval_model_contract(
        live_model_cfg,
        load_critic=load_critic,
    )
    if checkpoint_contract == live_contract:
        return live_contract

    checkpoint_flat = _flatten_contract(checkpoint_contract["model"])
    live_flat = _flatten_contract(live_contract["model"])
    missing = object()
    differences = []
    for path in sorted(set(checkpoint_flat) | set(live_flat)):
        checkpoint_value = checkpoint_flat.get(path, missing)
        live_value = live_flat.get(path, missing)
        if checkpoint_value != live_value:
            checkpoint_display = (
                "<missing>" if checkpoint_value is missing else repr(checkpoint_value)
            )
            live_display = "<missing>" if live_value is missing else repr(live_value)
            differences.append(
                f"{path}: checkpoint={checkpoint_display}, live={live_display}"
            )
    raise ValueError(
        "FastWAM evaluation model contract mismatch: " + "; ".join(differences[:16])
    )


def validate_fastwam_eval_checkpoint_contract(
    payload: Any,
    live_model_cfg: Any,
    *,
    expected_parent_checkpoint_sha256: str,
    load_critic: bool,
) -> dict[str, Any]:
    """Validate a project checkpoint before allocating the FastWAM model."""

    if not isinstance(payload, Mapping):
        raise TypeError("FastWAM evaluation checkpoint payload must be a mapping.")
    if payload.get("schema") != "fastwam-adaptive-rl-checkpoint-v1":
        raise ValueError("Unsupported FastWAM adaptive evaluation checkpoint.")
    expected_parent = str(expected_parent_checkpoint_sha256).strip().lower()
    if len(expected_parent) != 64 or any(
        character not in "0123456789abcdef" for character in expected_parent
    ):
        raise ValueError("Expected FastWAM parent SHA-256 is invalid.")
    if payload.get("parent_checkpoint_sha256") != expected_parent:
        raise ValueError("FastWAM evaluation checkpoint parent hash mismatch.")
    contract = payload.get("contract")
    checkpoint_model = contract.get("model") if isinstance(contract, Mapping) else None
    if not isinstance(checkpoint_model, Mapping):
        raise ValueError("FastWAM evaluation checkpoint is missing its model contract.")
    if (
        str(checkpoint_model.get("actor_checkpoint_sha256", "")).lower()
        != expected_parent
    ):
        raise ValueError(
            "FastWAM evaluation checkpoint contract has the wrong parent hash."
        )
    if load_critic:
        from rlinf.models.embodiment.wam_policy.critic import (
            CriticKind,
            critic_parent_checkpoint_sha256,
        )

        live_model = _resolved_mapping(
            live_model_cfg,
            name="FastWAM live evaluation model config",
        )
        live_critic = live_model.get("critic")
        if not isinstance(live_critic, Mapping):
            raise ValueError("FastWAM evaluation critic config is missing.")
        critic_kind = CriticKind.parse(
            live_critic.get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
        )
        expected_critic_parent = critic_parent_checkpoint_sha256(live_critic)
        if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM and (
            expected_critic_parent is None
            or len(expected_critic_parent) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_critic_parent
            )
        ):
            raise ValueError("Expected pi0.5 critic parent SHA-256 is invalid.")
        if payload.get("critic_parent_checkpoint_sha256") != expected_critic_parent:
            raise ValueError("FastWAM evaluation critic parent mismatch.")
        checkpoint_critic = checkpoint_model.get("critic")
        if not isinstance(checkpoint_critic, Mapping):
            raise ValueError(
                "FastWAM evaluation checkpoint contract has no critic config."
            )
        checkpoint_kind = CriticKind.parse(
            checkpoint_critic.get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
        )
        if checkpoint_kind is not critic_kind:
            raise ValueError("FastWAM evaluation checkpoint critic backend mismatch.")
        if critic_parent_checkpoint_sha256(checkpoint_critic) != expected_critic_parent:
            raise ValueError(
                "FastWAM evaluation checkpoint contract has the wrong critic "
                "parent identity."
            )
    return validate_fastwam_eval_model_contract(
        checkpoint_model,
        live_model_cfg,
        load_critic=load_critic,
    )


def _normalized_fastwam_actor_contract(cfg: Any) -> dict[str, Any]:
    actor_keys = (
        "seed",
        "micro_batch_size",
        "global_batch_size",
        "training_backend",
        "enable_offload",
        "fsdp_config",
        "optim",
    )
    actor = _selected_checkpoint_values(cfg.actor, actor_keys)
    fsdp = _resolved_mapping(actor["fsdp_config"], name="FSDP config")
    amp_autocast = fsdp.get("amp_autoc") or {}
    grad_scaler = fsdp.get("grad_scaler") or {}
    if not isinstance(amp_autocast, Mapping) or not isinstance(
        grad_scaler,
        Mapping,
    ):
        raise TypeError("FSDP AMP and GradScaler configs must be mappings.")
    fsdp["amp_autocast"] = {
        "enabled": amp_autocast.get("enabled", False),
        "precision": amp_autocast.get("precision", "bf16"),
    }
    fsdp["grad_scaler"] = {
        "enabled": grad_scaler.get("enabled", False),
        "init_scale": grad_scaler.get("init_scale"),
        "growth_interval": grad_scaler.get("growth_interval"),
    }
    actor["fsdp_config"] = fsdp
    return actor


def build_fastwam_checkpoint_contract(cfg: Any, *, world_size: int) -> dict[str, Any]:
    """Build the resolved continuation contract shared by actor and rollout.

    Output and scheduling limits are intentionally excluded so a checkpoint made
    by a one-step interruption job can resume in a two-step continuation job.
    Every field that can change model construction, rollout sampling, trajectory
    semantics, optimizer behavior, FSDP ownership, or initial weight sync is
    included and compared exactly on load.
    """

    rollout_keys = (
        "generation_backend",
        "recompute_logprobs",
        "unnorm_key",
        "enable_offload",
        "pipeline_stage_num",
        "collect_prev_infos",
        "enable_cuda_graph",
    )
    env_keys = (
        "env_type",
        "task_suite_name",
        "total_num_envs",
        "rollout_epoch",
        "group_size",
        "auto_reset",
        "ignore_terminations",
        "max_steps_per_rollout_epoch",
        "max_episode_steps",
        "specific_reset_id",
        "use_fixed_reset_state_ids",
        "use_ordered_reset_state_ids",
        "use_rel_reward",
        "reward_coef",
        "use_step_penalty",
        "reset_gripper_open",
        "seed",
        "init_params",
    )
    runner_keys = (
        "weight_sync_interval",
        "overlap_env_bootstrap",
        "use_training_pipeline",
        "fastwam_training_guard",
    )
    runner = _selected_checkpoint_values(cfg.runner, runner_keys)
    if runner["weight_sync_interval"] is None:
        runner["weight_sync_interval"] = 1
    overlap = runner["overlap_env_bootstrap"]
    if overlap is None:
        overlap = False
    env_offload = bool(OmegaConf.select(cfg, "env.train.enable_offload", default=False))
    runner["overlap_env_bootstrap"] = bool(overlap) and not env_offload
    formal_execution_profile = OmegaConf.select(
        cfg.runner,
        "formal_execution_profile",
        default=None,
    )
    if formal_execution_profile is not None:
        runner["formal_execution_profile"] = _resolved_checkpoint_value(
            formal_execution_profile
        )
    env_train = _selected_checkpoint_values(cfg.env.train, env_keys)
    task_id_filter = OmegaConf.select(
        cfg.env.train,
        "task_id_filter",
        default=None,
    )
    if task_id_filter is not None:
        env_train["task_id_filter"] = _resolved_checkpoint_value(task_id_filter)
    if formal_execution_profile is not None:
        env_train["stage_invariant_fixed_reset_ids"] = bool(
            OmegaConf.select(
                cfg.env.train,
                "stage_invariant_fixed_reset_ids",
                default=False,
            )
        )
        env_train["libero_variant"] = str(
            OmegaConf.select(cfg.env.train, "libero_variant", default="standard")
        )
    return {
        "schema": "fastwam-adaptive-checkpoint-contract-v2",
        "model": _resolved_checkpoint_value(cfg.actor.model),
        "algorithm": _resolved_checkpoint_value(cfg.algorithm),
        "actor": _normalized_fastwam_actor_contract(cfg),
        "rollout": _selected_checkpoint_values(cfg.rollout, rollout_keys),
        "env_train": env_train,
        "runner": runner,
        "weight_syncer": _resolved_checkpoint_value(cfg.weight_syncer),
        "component_placement": _resolved_checkpoint_value(
            cfg.cluster.component_placement
        ),
        "world_size": int(world_size),
    }


FASTWAM_RESUME_MODE_EXACT = "exact"
FASTWAM_RESUME_MODE_N4_TO_THREE_ROLLOUT = "n4_to_three_rollout"

_FASTWAM_N4_EXPANSION_GEOMETRY_PATHS = {
    "actor.global_batch_size",
    "component_placement.env",
    "component_placement.rollout",
    "env_train.total_num_envs",
    "world_size",
}
_FASTWAM_N4_EXPANSION_RUNTIME_PATHS = {
    "model.kv_replay.cold_capacity_bytes_per_rollout_rank",
    "model.kv_replay.hot_capacity_bytes_per_rollout_rank",
    "model.kv_replay.hot_min_free_bytes",
    "model.kv_replay.nvme_capacity_bytes_per_rollout_rank",
    "model.kv_replay.nvme_path",
    "model.kv_replay.prefetch_depth",
    "model.kv_replay.transport",
    "weight_syncer.patch.transport_device",
}


def _contract_difference_paths(
    checkpoint_contract: Mapping[str, Any],
    live_contract: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    checkpoint_flat = _flatten_contract(checkpoint_contract)
    live_flat = _flatten_contract(live_contract)
    paths = []
    descriptions = []
    missing = object()
    for path in sorted(set(checkpoint_flat) | set(live_flat)):
        checkpoint_value = checkpoint_flat.get(path, missing)
        live_value = live_flat.get(path, missing)
        if checkpoint_value == live_value:
            continue
        paths.append(path)
        checkpoint_display = (
            "<missing>" if checkpoint_value is missing else repr(checkpoint_value)
        )
        live_display = "<missing>" if live_value is missing else repr(live_value)
        descriptions.append(
            f"{path}: checkpoint={checkpoint_display}, live={live_display}"
        )
    return paths, descriptions


def validate_fastwam_training_checkpoint_contract(
    checkpoint_contract: Any,
    live_contract: Any,
    *,
    allow_n4_to_three_rollout_expansion: bool,
    owner: str,
) -> dict[str, Any]:
    """Validate exact resume or the explicit N=4 capacity expansion.

    The expansion changes only batch geometry and bounded K/V transport
    capacities. All model, sampling, reward, optimizer, and loss fields remain
    exact because every non-enumerated contract difference is rejected.
    """

    checkpoint = _resolved_mapping(
        checkpoint_contract,
        name="FastWAM checkpoint continuation contract",
    )
    live = _resolved_mapping(
        live_contract,
        name="FastWAM live continuation contract",
    )
    if checkpoint == live:
        return {
            "mode": FASTWAM_RESUME_MODE_EXACT,
            "source_world_size": int(checkpoint.get("world_size", -1)),
            "target_world_size": int(live.get("world_size", -1)),
        }

    difference_paths, difference_descriptions = _contract_difference_paths(
        checkpoint,
        live,
    )
    if not allow_n4_to_three_rollout_expansion:
        raise ValueError(
            "FastWAM adaptive checkpoint/config contract mismatch: "
            + "; ".join(difference_descriptions[:16])
        )
    if owner not in {"actor", "rollout"}:
        raise ValueError(f"Unsupported FastWAM checkpoint owner {owner!r}.")
    if checkpoint.get("schema") != "fastwam-adaptive-checkpoint-contract-v2" or (
        live.get("schema") != "fastwam-adaptive-checkpoint-contract-v2"
    ):
        raise ValueError("FastWAM capacity resume requires checkpoint contract v2.")

    allowed_paths = (
        _FASTWAM_N4_EXPANSION_GEOMETRY_PATHS | _FASTWAM_N4_EXPANSION_RUNTIME_PATHS
    )
    forbidden = [path for path in difference_paths if path not in allowed_paths]
    if forbidden:
        details = [
            description
            for path, description in zip(
                difference_paths,
                difference_descriptions,
                strict=True,
            )
            if path in forbidden
        ]
        raise ValueError(
            "FastWAM N=4 capacity resume changed scientific config: "
            + "; ".join(details[:16])
        )

    source_actor = _resolved_mapping(checkpoint.get("actor"), name="source actor")
    target_actor = _resolved_mapping(live.get("actor"), name="target actor")
    source_env = _resolved_mapping(
        checkpoint.get("env_train"),
        name="source train environment",
    )
    target_env = _resolved_mapping(
        live.get("env_train"),
        name="target train environment",
    )
    source_placement = _resolved_mapping(
        checkpoint.get("component_placement"),
        name="source component placement",
    )
    target_placement = _resolved_mapping(
        live.get("component_placement"),
        name="target component placement",
    )
    source_environment_count = int(source_env.get("total_num_envs", -1))
    target_environment_count = int(target_env.get("total_num_envs", -1))
    source_geometry = (
        source_environment_count == 4
        and int(source_actor.get("global_batch_size", -1)) == 28
        and source_placement.get("actor") == "0-0"
        and source_placement.get("env") == "1-1"
        and source_placement.get("rollout") == "1-1"
        and int(checkpoint.get("world_size", -1)) == 1
    )
    target_world_size = 1 if owner == "actor" else 3
    target_geometry = (
        target_environment_count >= 12
        and target_environment_count % 3 == 0
        and int(target_actor.get("global_batch_size", -1))
        == 7 * target_environment_count
        and target_placement.get("actor") == "0-0"
        and target_placement.get("env") == "1-3"
        and target_placement.get("rollout") == "1-3"
        and int(live.get("world_size", -1)) == target_world_size
    )
    if not source_geometry or not target_geometry:
        raise ValueError(
            "FastWAM capacity resume requires N=4/gbs=28/one-rollout source "
            "and N>=12/gbs=7*N/three-rollout target geometry."
        )

    live_model = _resolved_mapping(live.get("model"), name="target model")
    kv_replay = _resolved_mapping(
        live_model.get("kv_replay"),
        name="target K/V replay config",
    )
    if kv_replay.get("backend") != "stored":
        raise ValueError("FastWAM capacity resume requires stored K/V replay.")
    if kv_replay.get("transport") != "host_staging":
        raise ValueError("FastWAM capacity resume requires host-staging K/V transport.")
    for field in (
        "hot_capacity_bytes_per_rollout_rank",
        "cold_capacity_bytes_per_rollout_rank",
        "nvme_capacity_bytes_per_rollout_rank",
        "hot_min_free_bytes",
    ):
        value = kv_replay.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"FastWAM capacity resume requires nonnegative {field}.")
    prefetch_depth = kv_replay.get("prefetch_depth")
    if (
        isinstance(prefetch_depth, bool)
        or not isinstance(prefetch_depth, int)
        or prefetch_depth < 1
    ):
        raise ValueError("FastWAM capacity resume requires positive prefetch_depth.")
    weight_syncer = _resolved_mapping(
        live.get("weight_syncer"),
        name="target weight syncer",
    )
    patch = _resolved_mapping(weight_syncer.get("patch"), name="patch syncer")
    if patch.get("transport_device") != "cpu":
        raise ValueError("FastWAM capacity resume requires CPU patch transport.")

    return {
        "mode": FASTWAM_RESUME_MODE_N4_TO_THREE_ROLLOUT,
        "source_world_size": 1,
        "target_world_size": target_world_size,
        "source_environment_count": source_environment_count,
        "target_environment_count": target_environment_count,
    }


def validate_fastwam_resume_steps(
    loaded_steps,
    resume_dir: str,
) -> int:
    """Use the cross-rank checkpoint payload step and audit the directory label."""

    values = (
        list(loaded_steps)
        if isinstance(loaded_steps, (list, tuple))
        else [loaded_steps]
    )
    if not values or any(value is None or isinstance(value, bool) for value in values):
        raise ValueError("FastWAM checkpoint workers did not return payload steps.")
    steps = {int(value) for value in values}
    if len(steps) != 1:
        raise ValueError(
            f"FastWAM checkpoint ranks disagree on payload step: {sorted(steps)}."
        )
    payload_step = steps.pop()
    directory_name = Path(str(resume_dir)).name
    prefix = "global_step_"
    if not directory_name.startswith(prefix):
        raise ValueError(
            "FastWAM resume directory must be named `global_step_<payload step>`."
        )
    try:
        directory_step = int(directory_name.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(
            "FastWAM resume directory must end in an integer payload step."
        ) from exc
    if directory_step != payload_step:
        raise ValueError(
            "FastWAM resume directory/payload step mismatch: "
            f"directory={directory_step}, payload={payload_step}."
        )
    return payload_step
