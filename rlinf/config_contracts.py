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
import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

P8_FORMAL_OUTPUT_PARENT = Path("/data0/p8-formal-training")
P8_FORMAL_AUTHORIZATION_SCHEMA = "fastwam-p8-formal-training-authorization-v1"
P8_FORMAL_STOP_RULES = (
    "nonfinite_update",
    "cuda_oom",
    "action_out_of_bounds",
    "asset_hash_drift",
    "route_contract_violation",
    "frozen_or_gate_parameter_change",
    "checkpoint_failure",
    "gpu_or_host_or_disk_cap_breach",
)


def _is_sha256(value: object) -> bool:
    digest = str(value or "").strip().lower()
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _is_git_revision(value: object) -> bool:
    revision = str(value or "").strip().lower()
    return len(revision) == 40 and all(
        character in "0123456789abcdef" for character in revision
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_p8_formal_authorization_record(
    *,
    path: Path,
    expected_sha256: str,
    output_root: Path,
) -> None:
    """Read and validate the small, hash-bound P8 launch authorization."""

    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("P8 formal authorization record does not exist.") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or metadata.st_size < 2
        or metadata.st_size > 1024 * 1024
    ):
        raise ValueError("P8 formal authorization record is not a safe regular file.")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("P8 formal authorization record SHA-256 mismatch.")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("P8 formal authorization record is not valid JSON.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("P8 formal authorization record must be a JSON object.")
    if (
        payload.get("schema") != P8_FORMAL_AUTHORIZATION_SCHEMA
        or payload.get("status") != "READY-AUTHORIZED"
        or payload.get("candidate") != "P8-A0/KV"
        or payload.get("formal_training_authorized") is not True
        or payload.get("final_ledger_used") is not False
        or payload.get("output_root") != str(output_root)
    ):
        raise ValueError("P8 formal authorization record identity is invalid.")
    budget = payload.get("authorized_budget")
    if not isinstance(budget, Mapping) or dict(budget) != {
        "runner_steps": 100,
        "optimizer_updates_per_runner_step": 10,
        "optimizer_updates": 1000,
        "environments": 4,
        "global_batch_size": 28,
        "seed": 42,
    }:
        raise ValueError("P8 formal authorization record budget is invalid.")
    evidence = payload.get("candidate_evidence_sha256")
    if not isinstance(evidence, Mapping) or set(evidence) != {"p8", "p6", "p7"}:
        raise ValueError("P8 formal authorization record evidence set is invalid.")
    if any(not _is_sha256(value) for value in evidence.values()):
        raise ValueError("P8 formal authorization evidence SHA-256 is invalid.")
    revisions = payload.get("code_revisions")
    if not isinstance(revisions, Mapping) or set(revisions) != {
        "outer",
        "FastWAM",
        "RLinf",
    }:
        raise ValueError("P8 formal authorization code revisions are invalid.")
    if any(not _is_git_revision(value) for value in revisions.values()):
        raise ValueError("P8 formal authorization revision is invalid.")
    if payload.get("resource_caps") != {
        "gpu_used_bytes_per_device": 38 * 1024**3,
        "process_tree_rss_bytes": 128 * 1024**3,
        "output_bytes": 16 * 1024**3,
        "minimum_free_fraction": 0.20,
    }:
        raise ValueError("P8 formal authorization resource caps are invalid.")
    if payload.get("stop_rules") != list(P8_FORMAL_STOP_RULES) or payload.get(
        "stop_rules_sha256"
    ) != _canonical_sha256(P8_FORMAL_STOP_RULES):
        raise ValueError("P8 formal authorization stop rules are invalid.")
    for name in (
        "authorization_text_sha256",
        "formal_config_sha256",
        "asset_manifest_sha256",
        "stop_rules_sha256",
    ):
        if not _is_sha256(payload.get(name)):
            raise ValueError(f"P8 formal authorization {name} is invalid.")


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


def validate_p8_readiness_gate_ownership(
    *,
    p8_enabled: bool,
    gate_trainable: bool,
    readiness_endpoint: bool,
    gate_lr: float,
    gate_loss_weight: float,
    stage2_systems_endpoint: bool = False,
    formal_stage2_endpoint: bool = False,
) -> None:
    """Validate opt-in frozen-Gate P8 engineering endpoint ownership."""

    if not isinstance(gate_trainable, bool):
        raise TypeError("FastWAM `gate_trainable` must be a boolean.")
    if not isinstance(readiness_endpoint, bool):
        raise TypeError("`runner.p8_readiness_endpoint` must be a boolean.")
    if not isinstance(stage2_systems_endpoint, bool):
        raise TypeError("`runner.p8_stage2_systems_endpoint` must be a boolean.")
    if not isinstance(formal_stage2_endpoint, bool):
        raise TypeError("`runner.p8_formal_stage2_endpoint` must be a boolean.")
    endpoints = (
        readiness_endpoint,
        stage2_systems_endpoint,
        formal_stage2_endpoint,
    )
    if sum(endpoints) > 1:
        raise ValueError(
            "P8 readiness, Stage2 systems, and formal Stage2 endpoints are "
            "mutually exclusive."
        )
    gate_lr = float(gate_lr)
    gate_loss_weight = float(gate_loss_weight)
    if not math.isfinite(gate_lr) or not math.isfinite(gate_loss_weight):
        raise ValueError("FastWAM Gate LR and loss weight must be finite.")
    if gate_trainable:
        if any(endpoints):
            raise ValueError(
                "P8 frozen-Gate endpoints require an explicitly frozen Gate."
            )
        if gate_lr <= 0 or gate_loss_weight <= 0:
            raise ValueError("Trainable FastWAM Gate requires positive LR and loss.")
        return
    if not p8_enabled or not any(endpoints):
        raise ValueError(
            "Frozen Gate is restricted to an explicit enabled P8 endpoint."
        )
    if gate_lr != 0 or gate_loss_weight != 0:
        raise ValueError("Frozen P8 readiness Gate requires exact zero LR and loss.")


def validate_p8_formal_stage2_endpoint_contract(
    *,
    max_steps: int,
    max_epochs: int,
    save_interval: int,
    optimizer_updates_per_runner_step: int,
    actor_total_training_steps: int,
    actor_seed: int,
    micro_batch_size: int,
    global_batch_size: int,
    env_seed: int,
    total_num_envs: int,
    task_id_filter: object,
    specific_reset_id: object,
    use_fixed_reset_state_ids: bool,
    training_route_override: str,
    preserve_fixed_route_across_actor_updates: bool,
    load_text_encoder: bool,
    formal_training_authorized: bool,
    authorization_record_path: object,
    authorization_record_sha256: object,
    final_ledger_path: object,
    replay_backend: str,
    compile_enabled: bool,
    update_epoch: int,
    precision: str,
    storage_dtype: str,
    refiner_layer_indices: object,
    refiner_query_rank: int,
    refiner_output_rank: int,
    refiner_temperature: float,
    refiner_alpha: float,
    lora_lr: float,
    refiner_lr: float,
    refiner_weight_decay: float,
    value_lr: float,
    component_placement: object,
    output_root: object,
    formal_stage2_mode: str,
    checkpoint_path: object,
    bootstrap_checkpoint_dir: object,
    resume_dir: object,
    checkpoint_keep_last: int,
    checkpoint_atomic: bool,
    training_guard_enabled: bool,
    formal_action_audit: bool,
) -> None:
    """Validate the separately authorized 1000-update P8 Stage2 run."""

    exact_values = {
        "runner.max_steps": (int(max_steps), 100),
        "runner.max_epochs": (int(max_epochs), 100),
        "runner.save_interval": (int(save_interval), 10),
        "runner.formal_optimizer_updates_per_runner_step": (
            int(optimizer_updates_per_runner_step),
            10,
        ),
        "actor.optim.total_training_steps": (
            int(actor_total_training_steps),
            1000,
        ),
        "actor.seed": (int(actor_seed), 42),
        "actor.micro_batch_size": (int(micro_batch_size), 1),
        "actor.global_batch_size": (int(global_batch_size), 28),
        "env.train.seed": (int(env_seed), 42),
        "env.train.total_num_envs": (int(total_num_envs), 4),
        "algorithm.update_epoch": (int(update_epoch), 1),
        "actor.optim.lora_lr": (float(lora_lr), 1.0e-5),
        "actor.optim.refiner_lr": (float(refiner_lr), 1.0e-5),
        "actor.optim.refiner_weight_decay": (
            float(refiner_weight_decay),
            0.0,
        ),
        "actor.optim.value_lr": (float(value_lr), 1.0e-4),
        "refiner.query_rank": (int(refiner_query_rank), 32),
        "refiner.output_rank": (int(refiner_output_rank), 32),
        "refiner.temperature": (float(refiner_temperature), 0.07),
        "refiner.alpha": (float(refiner_alpha), 1.0),
        "runner.checkpoint_keep_last": (int(checkpoint_keep_last), 2),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in exact_values.items()
        if actual != expected
    ]
    if OmegaConf.is_config(task_id_filter):
        task_id_filter = OmegaConf.to_container(task_id_filter, resolve=True)
    if task_id_filter is not None:
        mismatches.append(
            "env.train.task_id_filter must be null so formal training uses "
            "the complete LIBERO-10 suite"
        )
    if specific_reset_id is not None:
        mismatches.append(
            "env.train.specific_reset_id must be null for formal training"
        )
    if use_fixed_reset_state_ids is not True:
        mismatches.append("env.train.use_fixed_reset_state_ids must be true")
    if training_route_override != "forced_uncond_after_initial":
        mismatches.append(
            "training_route_override must be `forced_uncond_after_initial`"
        )
    if preserve_fixed_route_across_actor_updates is not True:
        mismatches.append("preserve_fixed_route_across_actor_updates must be true")
    if load_text_encoder is not False:
        mismatches.append(
            "fastwam.load_text_encoder must be false so formal Stage2 uses only "
            "the pinned text cache"
        )
    if formal_training_authorized is not True:
        mismatches.append("runner.formal_training_authorized must be true")
    authorization_path = str(authorization_record_path or "").strip()
    if not authorization_path or not Path(authorization_path).is_absolute():
        mismatches.append(
            "runner.formal_training_authorization_record must be an absolute path"
        )
    authorization_sha = str(authorization_record_sha256 or "").strip().lower()
    if len(authorization_sha) != 64 or any(
        character not in "0123456789abcdef" for character in authorization_sha
    ):
        mismatches.append(
            "runner.formal_training_authorization_sha256 must be a SHA-256"
        )
    if final_ledger_path is not None:
        mismatches.append("runner.final_ledger_path must be null")
    if replay_backend != "stored_native":
        mismatches.append("P8 formal Stage2 replay.backend must be `stored_native`")
    if compile_enabled is not False:
        mismatches.append("P8 formal Stage2 compile must be false")
    if str(precision).lower() != "bf16":
        mismatches.append("P8 formal Stage2 precision must be `bf16`")
    if str(storage_dtype).lower() != "bfloat16":
        mismatches.append("P8 formal Stage2 storage dtype must be `bfloat16`")
    if OmegaConf.is_config(refiner_layer_indices):
        refiner_layer_indices = OmegaConf.to_container(
            refiner_layer_indices,
            resolve=True,
        )
    if refiner_layer_indices != [12]:
        mismatches.append("P8 formal Stage2 refiner layers must be exactly [12]")
    if checkpoint_atomic is not True:
        mismatches.append("runner.checkpoint_atomic must be true")
    if training_guard_enabled is not False:
        mismatches.append(
            "runner.fastwam_training_guard.enabled must be false for the "
            "fixed-route endpoint"
        )
    if formal_action_audit is not True:
        mismatches.append("runner.p8_formal_action_audit must be true")

    if OmegaConf.is_config(component_placement):
        component_placement = OmegaConf.to_container(
            component_placement,
            resolve=True,
        )
    expected_placement = {"actor": "0-1", "env,rollout": "2-3"}
    if component_placement != expected_placement:
        mismatches.append(
            "cluster.component_placement must dedicate actor=0-1 and env,rollout=2-3"
        )

    output_path = Path(str(output_root or "").strip())
    output_name_pattern = re.compile(r"^p8_a0_kv_stage2_seed42_[0-9]{8}T[0-9]{6}Z_v1$")
    if (
        not output_path.is_absolute()
        or output_path.parent != P8_FORMAL_OUTPUT_PARENT
        or output_name_pattern.fullmatch(output_path.name) is None
    ):
        mismatches.append(
            "runner.logger.log_path must be the exact absolute P8 formal output "
            "root under /data0/p8-formal-training"
        )
    elif authorization_path:
        expected_authorization_path = output_path / "formal_training_authorization.json"
        if Path(authorization_path) != expected_authorization_path:
            mismatches.append(
                "runner.formal_training_authorization_record must be the "
                "hash-bound record inside the formal output root"
            )
        else:
            try:
                _validate_p8_formal_authorization_record(
                    path=expected_authorization_path,
                    expected_sha256=authorization_sha,
                    output_root=output_path,
                )
            except ValueError as error:
                mismatches.append(str(error))

    mode = str(formal_stage2_mode).strip()
    checkpoint_value = str(checkpoint_path or "").strip()
    bootstrap_value = str(bootstrap_checkpoint_dir or "").strip()
    if resume_dir is not None:
        mismatches.append("runner.resume_dir must be null for a fresh formal launch")
    if mode == "training":
        expected_checkpoint = output_path / "step_zero" / "actor"
        if checkpoint_value != str(expected_checkpoint):
            mismatches.append(
                "runner.ckpt_path must load the fresh formal step-zero actor directory"
            )
        if bootstrap_checkpoint_dir is not None:
            mismatches.append(
                "runner.bootstrap_project_checkpoint_dir must be null while training"
            )
    elif mode == "step_zero_export":
        expected_bootstrap = output_path / "step_zero"
        if checkpoint_path is not None:
            mismatches.append("step-zero export requires runner.ckpt_path=null")
        if bootstrap_value != str(expected_bootstrap):
            mismatches.append(
                "step-zero export must write inside the formal output root"
            )
    else:
        mismatches.append(
            "runner.p8_formal_stage2_mode must be `training` or `step_zero_export`"
        )
    if mismatches:
        raise ValueError(
            "P8 formal Stage2 endpoint differs from its authorized 1000-update "
            "fixed-route contract: " + "; ".join(mismatches)
        )


def validate_p8_stage2_systems_endpoint_contract(
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
    compile_enabled: bool,
    route_seed: int,
) -> None:
    """Restrict the test-only Stage2 FSDP systems fixture.

    This endpoint is deliberately distinct from the P8-5 two-update canary. It
    permits only the one-update reset-0 systems identity at one or two ranks;
    it never authorizes formal training or access to a final ledger.
    """

    total_num_envs = int(total_num_envs)
    exact_values = {
        "runner.max_steps": (int(max_steps), 1),
        "runner.max_epochs": (int(max_epochs), 1),
        "actor.optim.total_training_steps": (
            int(actor_total_training_steps),
            1,
        ),
        "actor.seed": (int(actor_seed), 20260731),
        "actor.global_batch_size": (
            int(global_batch_size),
            2 * total_num_envs,
        ),
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
    if OmegaConf.is_config(task_id_filter):
        task_id_filter = OmegaConf.to_container(task_id_filter, resolve=True)
    normalized_task_ids = (
        list(task_id_filter) if isinstance(task_id_filter, (list, tuple)) else None
    )
    if normalized_task_ids != [0]:
        mismatches.append(f"env.train.task_id_filter={task_id_filter!r} (expected [0])")
    if use_fixed_reset_state_ids is not True:
        mismatches.append("env.train.use_fixed_reset_state_ids must be true")
    if training_route_override != "forced_uncond_after_initial":
        mismatches.append(
            "training_route_override must be `forced_uncond_after_initial`"
        )
    if load_text_encoder is not False:
        mismatches.append(
            "fastwam.load_text_encoder must be false so Stage2 uses the pinned cache"
        )
    if formal_training_authorized is not False:
        mismatches.append("runner.formal_training_authorized must be false")
    if final_ledger_path is not None:
        mismatches.append("runner.final_ledger_path must be null")
    if replay_backend != "stored_native":
        mismatches.append("P8 Stage2 replay.backend must be `stored_native`")
    if compile_enabled is not False:
        mismatches.append("P8 Stage2 compile must be false")
    if mismatches:
        raise ValueError(
            "P8 Stage2 systems endpoint is restricted to the one-update reset-0 "
            "FSDP fixture: " + "; ".join(mismatches)
        )


def validate_p8_readiness_endpoint_contract(
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
) -> None:
    """Restrict the P8 readiness preset to the frozen B0/B1 identity."""

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
    if OmegaConf.is_config(task_id_filter):
        task_id_filter = OmegaConf.to_container(task_id_filter, resolve=True)
    normalized_task_ids = (
        list(task_id_filter) if isinstance(task_id_filter, (list, tuple)) else None
    )
    if normalized_task_ids != [0]:
        mismatches.append(f"env.train.task_id_filter={task_id_filter!r} (expected [0])")
    if use_fixed_reset_state_ids is not True:
        mismatches.append("env.train.use_fixed_reset_state_ids must be true")
    if training_route_override != "forced_uncond_after_initial":
        mismatches.append(
            "training_route_override must be `forced_uncond_after_initial`"
        )
    if load_text_encoder is not False:
        mismatches.append(
            "fastwam.load_text_encoder must be false so the endpoint uses only "
            "the pinned text cache"
        )
    if formal_training_authorized is not False:
        mismatches.append("runner.formal_training_authorized must be false")
    if final_ledger_path is not None:
        mismatches.append("runner.final_ledger_path must be null")
    if mismatches:
        raise ValueError(
            "P8 readiness endpoint is restricted to the two-step B0/B1 harness: "
            + "; ".join(mismatches)
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
    p8_cfg = live_model.get("uncond_visual_sidecar", {})
    p8_enabled = bool(
        p8_cfg.get("enabled", False) if isinstance(p8_cfg, Mapping) else False
    )
    expected_schema = (
        "fastwam-adaptive-rl-checkpoint-v2-p8-a0-kv"
        if p8_enabled
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
    p8_sidecar = OmegaConf.select(
        cfg,
        "actor.model.uncond_visual_sidecar",
        default=None,
    )
    p8_enabled = bool(
        p8_sidecar.get("enabled", False) if hasattr(p8_sidecar, "get") else False
    )
    if p8_enabled:
        formal_stage2_endpoint = bool(
            OmegaConf.select(
                cfg,
                "runner.p8_formal_stage2_endpoint",
                default=False,
            )
        )
        runner.update(
            {
                "p8_readiness_endpoint": bool(
                    OmegaConf.select(
                        cfg,
                        "runner.p8_readiness_endpoint",
                        default=False,
                    )
                ),
                "p8_stage2_systems_endpoint": bool(
                    OmegaConf.select(
                        cfg,
                        "runner.p8_stage2_systems_endpoint",
                        default=False,
                    )
                ),
                "p8_formal_stage2_endpoint": formal_stage2_endpoint,
                "formal_training_authorized": bool(
                    OmegaConf.select(
                        cfg,
                        "runner.formal_training_authorized",
                        default=False,
                    )
                ),
                "formal_training_authorization_record": OmegaConf.select(
                    cfg,
                    "runner.formal_training_authorization_record",
                    default=None,
                ),
                "formal_training_authorization_sha256": OmegaConf.select(
                    cfg,
                    "runner.formal_training_authorization_sha256",
                    default=None,
                ),
            }
        )
        if formal_stage2_endpoint:
            runner["formal_optimizer_updates_per_runner_step"] = int(
                OmegaConf.select(
                    cfg,
                    "runner.formal_optimizer_updates_per_runner_step",
                    default=-1,
                )
            )
            runner["p8_formal_action_audit"] = bool(
                OmegaConf.select(
                    cfg,
                    "runner.p8_formal_action_audit",
                    default=False,
                )
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
