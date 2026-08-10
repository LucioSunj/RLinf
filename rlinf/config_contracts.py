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
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

P6_PARENT_SHA256 = "e979511a2d7a1310009496c6b2f06957171bba28b96aac0d513992c6ed21ca5a"
P6_VAE_SHA256 = "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
P6_DINO_SOURCE_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
P6_DINO_WEIGHTS_SHA256 = (
    "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
)
P6_DINO_PREPROCESS_SHA256 = (
    "0a70d846042c4bb29893ead5d9433e97d1ec5089875704e472dd821632e24dc0"
)
P6_DINO_OUTPUT_SHA256 = (
    "631bb13876a3e9c79476eb1ac089bced12aae93302f117cfeb106dd8acfe9f18"
)
P6_CAMERA_INPUT_SHA256 = (
    "336d06c39a488c89b1404154b6ed8a9ee25969507611d8b06b884f8c2945e8e7"
)
P6_SPATIAL_CONTRACT_SHA256 = (
    "c58c1548b4b0ac6ddcc58266d4e4caf4ec9e5652aadfe29c17f4eff31195ac4e"
)
P6_TRANSPORT_SHA256 = "89ce4864e45350380fa1e9aab12522035fe1686bba6870b4e911d4a1c90b54ab"


def validate_p6_readiness_mechanics_contract(
    *,
    visual_sidecar: object,
    precision: str,
    actor_checkpoint_sha256: str,
    text_embedding_cache_dir: object,
) -> None:
    """Validate the frozen P6 mechanics preset without touching assets."""

    if OmegaConf.is_config(visual_sidecar):
        visual_sidecar = OmegaConf.to_container(visual_sidecar, resolve=True)
    if not isinstance(visual_sidecar, Mapping):
        raise TypeError("P6 readiness sidecar must resolve to a mapping.")
    payload = dict(visual_sidecar)
    dino = payload.get("dino", {})
    router = payload.get("router", {})
    transport = payload.get("transport", {})
    optimizer = payload.get("optimizer", {})
    wan = payload.get("wan_value", {})
    spatial = wan.get("spatial_metadata", {}) if isinstance(wan, Mapping) else {}
    injection = payload.get("injection", {})
    replay = payload.get("replay", {})
    if not all(
        isinstance(item, Mapping)
        for item in (dino, router, transport, optimizer, spatial, injection, replay)
    ):
        raise TypeError("P6 readiness sidecar contains a non-mapping section.")
    expected = {
        "precision": (str(precision).lower(), "bf16"),
        "actor_checkpoint_sha256": (
            str(actor_checkpoint_sha256).lower(),
            P6_PARENT_SHA256,
        ),
        "dino.source_revision": (
            str(dino.get("source_revision", "")),
            P6_DINO_SOURCE_REVISION,
        ),
        "dino.model_name": (str(dino.get("model_name", "")), "dinov3_vits16"),
        "dino.weights_sha256": (
            str(dino.get("weights_sha256", "")).lower(),
            P6_DINO_WEIGHTS_SHA256,
        ),
        "dino.preprocess_sha256": (
            str(dino.get("preprocess_sha256", "")).lower(),
            P6_DINO_PREPROCESS_SHA256,
        ),
        "dino.output_contract_sha256": (
            str(dino.get("output_contract_sha256", "")).lower(),
            P6_DINO_OUTPUT_SHA256,
        ),
        "dino.camera_input_contract_sha256": (
            str(dino.get("camera_input_contract_sha256", "")).lower(),
            P6_CAMERA_INPUT_SHA256,
        ),
        "dino.compute_dtype": (str(dino.get("compute_dtype", "")), "bfloat16"),
        "dino.license_id": (str(dino.get("license_id", "")), "DINOv3 License"),
        "router.query_projection": (
            str(router.get("query_projection", "")),
            "low_rank",
        ),
        "router.query_rank": (int(router.get("query_rank", -1)), 32),
        "router.temperature": (float(router.get("temperature", float("nan"))), 0.07),
        "router.camera_mass_values": (
            list(router.get("camera_mass_values", [])),
            [0.5, 0.5],
        ),
        "transport.contract_sha256": (
            str(transport.get("contract_sha256", "")).lower(),
            P6_TRANSPORT_SHA256,
        ),
        "optimizer.lr": (float(optimizer.get("lr", float("nan"))), 1.0e-5),
        "optimizer.weight_decay": (
            float(optimizer.get("weight_decay", float("nan"))),
            0.0,
        ),
        "optimizer.scheduler": (str(optimizer.get("scheduler", "")), "cosine"),
        "spatial.vae_weights_sha256": (
            str(spatial.get("vae_weights_sha256", "")).lower(),
            P6_VAE_SHA256,
        ),
        "spatial.video_dit_weights_sha256": (
            str(spatial.get("video_dit_weights_sha256", "")).lower(),
            P6_PARENT_SHA256,
        ),
        "spatial.spatial_transport_contract_sha256": (
            str(spatial.get("spatial_transport_contract_sha256", "")).lower(),
            P6_SPATIAL_CONTRACT_SHA256,
        ),
        "injection.layer_indices": (
            list(injection.get("layer_indices", [])),
            [9, 15, 21],
        ),
        "injection.beta_max": (
            float(injection.get("beta_max", float("nan"))),
            1.0,
        ),
        "replay.backend": (str(replay.get("backend", "")), "stored_native"),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    cache_path = Path(str(text_embedding_cache_dir or "").strip())
    if not cache_path.is_absolute():
        mismatches.append("runtime.text_embedding_cache_dir must be absolute")
    if mismatches:
        raise ValueError(
            "P6 readiness mechanics preset changed: " + "; ".join(mismatches)
        )


def validate_p6_readiness_gate_ownership(
    *,
    p6_enabled: bool,
    gate_trainable: bool,
    readiness_endpoint: bool,
    stage2_systems_endpoint: bool,
    gate_lr: float,
    gate_loss_weight: float,
) -> None:
    """Validate the two opt-in frozen-Gate P6 engineering endpoints."""

    if not isinstance(gate_trainable, bool):
        raise TypeError("FastWAM `gate_trainable` must be a boolean.")
    if not isinstance(readiness_endpoint, bool):
        raise TypeError("`runner.p6_readiness_endpoint` must be a boolean.")
    if not isinstance(stage2_systems_endpoint, bool):
        raise TypeError("`runner.p6_stage2_systems_endpoint` must be a boolean.")
    if readiness_endpoint and stage2_systems_endpoint:
        raise ValueError("P6 readiness endpoints are mutually exclusive.")
    gate_lr = float(gate_lr)
    gate_loss_weight = float(gate_loss_weight)
    if not math.isfinite(gate_lr) or not math.isfinite(gate_loss_weight):
        raise ValueError("FastWAM Gate LR and loss weight must be finite.")
    endpoint_enabled = readiness_endpoint or stage2_systems_endpoint
    if gate_trainable:
        if endpoint_enabled:
            raise ValueError("P6 frozen-Gate endpoints require a frozen Gate.")
        if gate_lr <= 0 or gate_loss_weight <= 0:
            raise ValueError("Trainable FastWAM Gate requires positive LR and loss.")
        return
    if not p6_enabled or not endpoint_enabled:
        raise ValueError(
            "Frozen Gate is restricted to an explicit enabled P6 readiness endpoint."
        )
    if gate_lr != 0 or gate_loss_weight != 0:
        raise ValueError("Frozen P6 readiness Gate requires exact zero LR and loss.")


def _normalized_task_ids(task_id_filter: object) -> list[int] | None:
    if OmegaConf.is_config(task_id_filter):
        task_id_filter = OmegaConf.to_container(task_id_filter, resolve=True)
    if not isinstance(task_id_filter, (list, tuple)):
        return None
    return [int(item) for item in task_id_filter]


def validate_p6_readiness_endpoint_contract(
    *,
    max_steps: int,
    max_epochs: int,
    actor_total_training_steps: int,
    actor_seed: int,
    global_batch_size: int,
    env_seed: int,
    total_num_envs: int,
    task_id_filter: object,
    specific_reset_id: int,
    use_fixed_reset_state_ids: bool,
    training_route_override: str,
    load_text_encoder: bool,
    formal_training_authorized: bool,
    final_ledger_path: object,
    replay_backend: str,
) -> None:
    """Restrict P6 B0/B1 readiness to its frozen development identity."""

    exact_values = {
        "runner.max_steps": (int(max_steps), 2),
        "runner.max_epochs": (int(max_epochs), 1),
        "actor.optim.total_training_steps": (int(actor_total_training_steps), 2),
        "actor.seed": (int(actor_seed), 424242),
        "actor.global_batch_size": (int(global_batch_size), 1),
        "env.train.seed": (int(env_seed), 424242),
        "env.train.total_num_envs": (int(total_num_envs), 1),
        "env.train.specific_reset_id": (int(specific_reset_id), 1),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in exact_values.items()
        if actual != expected
    ]
    if _normalized_task_ids(task_id_filter) != [0]:
        mismatches.append(f"env.train.task_id_filter={task_id_filter!r} (expected [0])")
    if use_fixed_reset_state_ids is not True:
        mismatches.append("env.train.use_fixed_reset_state_ids must be true")
    if training_route_override != "forced_uncond_after_initial":
        mismatches.append(
            "training_route_override must be `forced_uncond_after_initial`"
        )
    if load_text_encoder is not False:
        mismatches.append("fastwam.load_text_encoder must be false")
    if formal_training_authorized is not False:
        mismatches.append("runner.formal_training_authorized must be false")
    if final_ledger_path is not None:
        mismatches.append("runner.final_ledger_path must be null")
    if replay_backend != "stored_native":
        mismatches.append("P6 readiness replay.backend must be `stored_native`")
    if mismatches:
        raise ValueError(
            "P6 readiness endpoint is restricted to the two-update B0/B1 "
            "fixture: " + "; ".join(mismatches)
        )


def validate_p6_stage2_systems_endpoint_contract(
    *,
    max_steps: int,
    max_epochs: int,
    actor_total_training_steps: int,
    actor_seed: int,
    global_batch_size: int,
    env_seed: int,
    total_num_envs: int,
    task_id_filter: object,
    specific_reset_id: int,
    use_fixed_reset_state_ids: bool,
    training_route_override: str,
    load_text_encoder: bool,
    formal_training_authorized: bool,
    final_ledger_path: object,
    replay_backend: str,
    route_seed: int,
) -> None:
    """Restrict the P6 WS1/WS2 fixture to one non-scientific update."""

    total_num_envs = int(total_num_envs)
    exact_values = {
        "runner.max_steps": (int(max_steps), 1),
        "runner.max_epochs": (int(max_epochs), 1),
        "actor.optim.total_training_steps": (int(actor_total_training_steps), 1),
        "actor.seed": (int(actor_seed), 20260731),
        "actor.global_batch_size": (int(global_batch_size), 2 * total_num_envs),
        "env.train.seed": (int(env_seed), 20260801),
        "env.train.specific_reset_id": (int(specific_reset_id), 0),
        "runner.l11_route_seed": (int(route_seed), 20260801),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in exact_values.items()
        if actual != expected
    ]
    if total_num_envs not in {1, 2}:
        mismatches.append(
            f"env.train.total_num_envs={total_num_envs!r} (expected 1 or 2)"
        )
    if _normalized_task_ids(task_id_filter) != [0]:
        mismatches.append(f"env.train.task_id_filter={task_id_filter!r} (expected [0])")
    if use_fixed_reset_state_ids is not True:
        mismatches.append("env.train.use_fixed_reset_state_ids must be true")
    if training_route_override != "forced_uncond_after_initial":
        mismatches.append(
            "training_route_override must be `forced_uncond_after_initial`"
        )
    if load_text_encoder is not False:
        mismatches.append("fastwam.load_text_encoder must be false")
    if formal_training_authorized is not False:
        mismatches.append("runner.formal_training_authorized must be false")
    if final_ledger_path is not None:
        mismatches.append("runner.final_ledger_path must be null")
    if replay_backend != "stored_native":
        mismatches.append("P6 Stage2 replay.backend must be `stored_native`")
    if mismatches:
        raise ValueError(
            "P6 Stage2 systems endpoint is restricted to the one-update reset-0 "
            "fixture: " + "; ".join(mismatches)
        )


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


def is_fastwam_p6_enabled(model_cfg: Any) -> bool:
    """Read the optional P6 flag from DictConfig, mapping, or test namespace."""

    if OmegaConf.is_config(model_cfg):
        return bool(
            OmegaConf.select(
                model_cfg,
                "uncond_visual_sidecar.enabled",
                default=False,
            )
        )
    if isinstance(model_cfg, Mapping):
        visual = model_cfg.get("uncond_visual_sidecar", {})
    else:
        visual = getattr(model_cfg, "uncond_visual_sidecar", None)
    if isinstance(visual, Mapping):
        return bool(visual.get("enabled", False))
    return bool(getattr(visual, "enabled", False))


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
    "uncond_visual_sidecar.dino.source_root",
    "uncond_visual_sidecar.dino.weights_path",
    "gate_epsilon",
    "eval_routing_mode",
    "eval_idm_threshold",
    "eval_random_idm_probability",
    "eval_routing_seed",
    "eval_microbatch_size",
    "eval_timing_cuda_synchronize",
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
    live_model = _resolved_mapping(
        live_model_cfg,
        name="FastWAM live evaluation model config",
    )
    visual_enabled = is_fastwam_p6_enabled(live_model)
    expected_schema = (
        "fastwam-adaptive-rl-checkpoint-v2-p6"
        if visual_enabled
        else "fastwam-adaptive-rl-checkpoint-v1"
    )
    if payload.get("schema") != expected_schema:
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
        live_critic = live_model.get("critic")
        expected_critic_parent = (
            str(
                live_critic.get("backbone_checkpoint_sha256", "")
                if isinstance(live_critic, Mapping)
                else ""
            )
            .strip()
            .lower()
        )
        if len(expected_critic_parent) != 64 or any(
            character not in "0123456789abcdef" for character in expected_critic_parent
        ):
            raise ValueError("Expected pi0.5 critic parent SHA-256 is invalid.")
        if payload.get("critic_parent_checkpoint_sha256") != expected_critic_parent:
            raise ValueError("pi0.5 evaluation checkpoint parent hash mismatch.")
        checkpoint_critic = checkpoint_model.get("critic")
        if not isinstance(checkpoint_critic, Mapping) or (
            str(checkpoint_critic.get("backbone_checkpoint_sha256", "")).lower()
            != expected_critic_parent
        ):
            raise ValueError(
                "FastWAM evaluation checkpoint contract has the wrong critic "
                "parent hash."
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
    p6_sidecar = OmegaConf.select(
        cfg,
        "actor.model.uncond_visual_sidecar",
        default=None,
    )
    p6_enabled = bool(
        p6_sidecar.get("enabled", False) if hasattr(p6_sidecar, "get") else False
    )
    if p6_enabled:
        runner.update(
            {
                "p6_readiness_endpoint": bool(
                    OmegaConf.select(
                        cfg,
                        "runner.p6_readiness_endpoint",
                        default=False,
                    )
                ),
                "p6_stage2_systems_endpoint": bool(
                    OmegaConf.select(
                        cfg,
                        "runner.p6_stage2_systems_endpoint",
                        default=False,
                    )
                ),
                "formal_training_authorized": bool(
                    OmegaConf.select(
                        cfg,
                        "runner.formal_training_authorized",
                        default=False,
                    )
                ),
            }
        )
    return {
        "schema": "fastwam-adaptive-checkpoint-contract-v2",
        "model": _resolved_checkpoint_value(cfg.actor.model),
        "algorithm": _resolved_checkpoint_value(cfg.algorithm),
        "actor": _normalized_fastwam_actor_contract(cfg),
        "rollout": _selected_checkpoint_values(cfg.rollout, rollout_keys),
        "env_train": _selected_checkpoint_values(cfg.env.train, env_keys),
        "runner": runner,
        "weight_syncer": _resolved_checkpoint_value(cfg.weight_syncer),
        "component_placement": _resolved_checkpoint_value(
            cfg.cluster.component_placement
        ),
        "world_size": int(world_size),
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
