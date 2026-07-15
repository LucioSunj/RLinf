# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Paired counterfactual state-bank contract for adaptive imagination."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PAIRED_SCHEMA = "paired-v1"
PAIRED_MODE_ORDER = ("uncond", "idm")
PAIRED_NUM_FOLDS = 5

PAIRED_TENSOR_KEYS = (
    "world_feat",
    "proprio",
    "text_feat",
    "trajectory_id",
    "decision_index",
    "task_id",
    "source_mode",
    "branch_seed",
    "success_uncond",
    "success_idm",
    "progress_1_uncond",
    "progress_1_idm",
    "progress_3_uncond",
    "progress_3_idm",
    "sensitivity_mask",
    "sensitivity_success_uncond",
    "sensitivity_success_idm",
    "sensitivity_progress_3_uncond",
    "sensitivity_progress_3_idm",
)

PAIRED_META_KEYS = (
    "task",
    "backbone_kind",
    "ckpt_fingerprint",
    "ckpt_file_sha256",
    "dataset_stats_fingerprint",
    "num_video_frames",
    "inference_steps",
    "solver_fingerprint",
    "context_len",
    "model_dtype",
    "exec_horizon",
    "action_horizon",
    "world_feat_layout",
    "text_feat_layout",
    "mode_order",
    "world_feat_dim",
    "proprio_dim",
    "text_feat_dim",
    "snapshot_schema",
    "episode_manifest_sha256",
    "heldout_test_manifest_sha256",
    "libero_plus_commit",
    "manifest_split",
    "collector_seed",
    "continuation_mode",
    "max_reference_decisions",
    "max_branch_decisions",
    "sensitivity_fraction",
    "reference_policy_mix",
    "reference_policy_assignment",
    "reference_assignment_manifest_sha256",
    "reference_assignment_sha256",
    "reference_policy_episode_assignments",
)

PAIRED_RECORD_KEYS = (
    "state_id",
    "episode_uid",
    "base_task",
    "task_suite_name",
    "task_description",
    "trial_id",
    "reset_state_id",
    "env_seed",
    "factor",
    "level",
    "perturbation_id",
    "phase",
    "phase_reliable",
    "snapshot_path",
    "snapshot_sha256",
    "asset_ids",
)


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def reference_assignment_sha256(assignments: Mapping[str, str]) -> str:
    """Hash the parent-manifest-wide source-policy assignment contract."""
    return _canonical_sha256(
        {
            "schema": PAIRED_SCHEMA,
            "artifact": "reference-policy-episode-assignments",
            "assignments": dict(sorted(assignments.items())),
        }
    )


def _require_sha256(value: Any, *, field: str, source: str) -> str:
    raw = str(value)
    digest = raw.lower()
    if raw != digest or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{source}: {field} must be a lowercase SHA256")
    return digest


def _resolve_paths(paths: str | os.PathLike[str] | Sequence[str]) -> list[str]:
    values = [os.fspath(paths)] if isinstance(paths, (str, os.PathLike)) else list(paths)
    resolved: list[str] = []
    for value in values:
        matches = sorted(glob.glob(os.path.expanduser(os.fspath(value))))
        if not matches and os.path.isfile(os.path.expanduser(os.fspath(value))):
            matches = [os.path.expanduser(os.fspath(value))]
        for match in matches:
            if os.path.isdir(match):
                shards = sorted(glob.glob(os.path.join(match, "tensors", "*.pt")))
                if not shards:
                    raise FileNotFoundError(
                        f"paired-v1 directory has no tensors/*.pt shards: {match}"
                    )
                resolved.extend(os.path.realpath(shard) for shard in shards)
            else:
                resolved.append(os.path.realpath(match))
    resolved = sorted(set(resolved))
    if not resolved:
        raise FileNotFoundError(f"no paired-v1 shards match {values!r}")
    return resolved


def _positive_int(meta: Mapping[str, Any], key: str, source: str) -> int:
    try:
        value = int(meta[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: meta.{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{source}: meta.{key} must be a positive integer")
    return value


def validate_paired_payload(
    payload: Mapping[str, Any], *, source: str = "<paired-v1>"
) -> dict[str, Any]:
    """Validate shapes, outcomes, immutable WAM provenance and split identity."""
    if not isinstance(payload, Mapping) or payload.get("schema") != PAIRED_SCHEMA:
        actual = payload.get("schema") if isinstance(payload, Mapping) else None
        raise ValueError(f"{source}: expected schema={PAIRED_SCHEMA!r}, got {actual!r}")
    meta = payload.get("meta")
    data = payload.get("data")
    records = payload.get("records")
    if not isinstance(meta, Mapping):
        raise ValueError(f"{source}: missing metadata object")
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: missing tensor data object")
    if not isinstance(records, list):
        raise ValueError(f"{source}: missing records list")

    missing_meta = [key for key in PAIRED_META_KEYS if key not in meta]
    if missing_meta:
        raise ValueError(f"{source}: paired metadata is missing {missing_meta}")
    if str(meta["backbone_kind"]).lower() != "idm":
        raise ValueError(f"{source}: paired collection requires backbone_kind='idm'")
    if list(meta["mode_order"]) != list(PAIRED_MODE_ORDER):
        raise ValueError(
            f"{source}: mode_order must be {list(PAIRED_MODE_ORDER)!r}"
        )
    if str(meta["continuation_mode"]).lower() != "uncond":
        raise ValueError(
            f"{source}: primary paired outcomes require deterministic UNCOND continuation"
        )
    for key in (
        "task",
        "ckpt_fingerprint",
        "ckpt_file_sha256",
        "dataset_stats_fingerprint",
        "model_dtype",
        "world_feat_layout",
        "text_feat_layout",
        "snapshot_schema",
        "episode_manifest_sha256",
        "heldout_test_manifest_sha256",
        "libero_plus_commit",
        "manifest_split",
    ):
        if not isinstance(meta[key], str) or not str(meta[key]).strip():
            raise ValueError(f"{source}: meta.{key} must be a non-empty string")
    if str(meta["manifest_split"]).lower() not in {"train", "validation", "test"}:
        raise ValueError(f"{source}: meta.manifest_split is invalid")
    if meta["snapshot_schema"] != "libero-gate-snapshot-v1":
        raise ValueError(
            f"{source}: meta.snapshot_schema must be 'libero-gate-snapshot-v1'"
        )
    for key in (
        "episode_manifest_sha256",
        "heldout_test_manifest_sha256",
        "solver_fingerprint",
        "reference_assignment_manifest_sha256",
        "reference_assignment_sha256",
    ):
        _require_sha256(meta[key], field=f"meta.{key}", source=source)
    if meta["reference_assignment_manifest_sha256"] != meta["episode_manifest_sha256"]:
        raise ValueError(
            f"{source}: reference assignments are not bound to the collection "
            "episode manifest"
        )
    assignments = meta["reference_policy_episode_assignments"]
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError(
            f"{source}: meta.reference_policy_episode_assignments must be a "
            "non-empty mapping"
        )
    normalized_assignments: dict[str, str] = {}
    for episode_uid, assignment in assignments.items():
        episode_uid = str(episode_uid)
        assignment = str(assignment)
        if not episode_uid or assignment not in {"uncond", "idm", "random_0.5"}:
            raise ValueError(
                f"{source}: invalid reference assignment {episode_uid!r}={assignment!r}"
            )
        normalized_assignments[episode_uid] = assignment
    if reference_assignment_sha256(normalized_assignments) != str(
        meta["reference_assignment_sha256"]
    ):
        raise ValueError(f"{source}: reference assignment SHA256 mismatch")
    plus_commit = str(meta["libero_plus_commit"]).lower()
    if len(plus_commit) != 40 or any(
        character not in "0123456789abcdef" for character in plus_commit
    ):
        raise ValueError(f"{source}: meta.libero_plus_commit must be a full git SHA")
    checkpoint_sha = str(meta["ckpt_file_sha256"]).lower()
    if len(checkpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha
    ):
        raise ValueError(f"{source}: meta.ckpt_file_sha256 must be a SHA256")
    for key in (
        "num_video_frames",
        "inference_steps",
        "context_len",
        "exec_horizon",
        "action_horizon",
        "world_feat_dim",
        "proprio_dim",
        "text_feat_dim",
        "max_reference_decisions",
        "max_branch_decisions",
    ):
        _positive_int(meta, key, source)
    try:
        collector_seed = int(meta["collector_seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: meta.collector_seed must be an integer") from exc
    if collector_seed < 0:
        raise ValueError(f"{source}: meta.collector_seed must be non-negative")
    try:
        sensitivity_fraction = float(meta["sensitivity_fraction"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: meta.sensitivity_fraction must be numeric") from exc
    if not 0.0 <= sensitivity_fraction <= 1.0:
        raise ValueError(f"{source}: meta.sensitivity_fraction must be in [0,1]")
    if list(meta["reference_policy_mix"]) != ["uncond", "idm", "random_0.5"]:
        raise ValueError(f"{source}: meta.reference_policy_mix is not canonical")
    if meta["reference_policy_assignment"] != "balanced_shuffled_v1":
        raise ValueError(
            f"{source}: meta.reference_policy_assignment is not canonical"
        )

    missing_data = [key for key in PAIRED_TENSOR_KEYS if key not in data]
    if missing_data:
        raise ValueError(f"{source}: paired tensors are missing {missing_data}")
    if not all(torch.is_tensor(data[key]) for key in PAIRED_TENSOR_KEYS):
        raise ValueError(f"{source}: every paired data field must be a tensor")
    n = int(data["world_feat"].shape[0])
    if n <= 0:
        raise ValueError(f"{source}: paired shard is empty")
    if len(records) != n:
        raise ValueError(f"{source}: records length {len(records)} != tensor rows {n}")
    for key in PAIRED_TENSOR_KEYS:
        if data[key].ndim == 0 or int(data[key].shape[0]) != n:
            raise ValueError(
                f"{source}: data.{key} must have leading dimension {n}, got "
                f"{tuple(data[key].shape)}"
            )
    dimensions = {
        "world_feat": int(meta["world_feat_dim"]),
        "proprio": int(meta["proprio_dim"]),
        "text_feat": int(meta["text_feat_dim"]),
    }
    for key, expected in dimensions.items():
        value = data[key]
        if value.ndim != 2 or int(value.shape[1]) != expected:
            raise ValueError(
                f"{source}: data.{key} must be [N,{expected}], got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"{source}: data.{key} contains non-finite values")
    for key in ("trajectory_id", "decision_index", "task_id", "branch_seed"):
        value = data[key].reshape(n)
        if value.dtype == torch.bool or bool((value.long() < 0).any()):
            raise ValueError(f"{source}: data.{key} must contain non-negative integers")
    source_mode = data["source_mode"].reshape(n).long()
    if bool(((source_mode < 0) | (source_mode > 1)).any()):
        raise ValueError(f"{source}: source_mode must be UNCOND=0 or IDM=1")
    for key in (
        "success_uncond",
        "success_idm",
        "sensitivity_mask",
        "sensitivity_success_uncond",
        "sensitivity_success_idm",
    ):
        value = data[key].reshape(n)
        if bool(((value.long() < 0) | (value.long() > 1)).any()):
            raise ValueError(f"{source}: data.{key} must be binary")
    for key in (
        "progress_1_uncond",
        "progress_1_idm",
        "progress_3_uncond",
        "progress_3_idm",
        "sensitivity_progress_3_uncond",
        "sensitivity_progress_3_idm",
    ):
        if not bool(torch.isfinite(data[key].float()).all()):
            raise ValueError(f"{source}: data.{key} contains non-finite values")

    state_ids: set[str] = set()
    trajectory_episode: dict[int, str] = {}
    trajectory_ids = data["trajectory_id"].reshape(n).long().tolist()
    source_modes = data["source_mode"].reshape(n).long().tolist()
    for index, (record, trajectory_id, source_mode_value) in enumerate(
        zip(records, trajectory_ids, source_modes)
    ):
        record_source = f"{source}:records[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_source} must be an object")
        missing = [key for key in PAIRED_RECORD_KEYS if key not in record]
        if missing:
            raise ValueError(f"{record_source} is missing {missing}")
        state_id = str(record["state_id"])
        episode_uid = str(record["episode_uid"])
        if not state_id or not episode_uid:
            raise ValueError(f"{record_source}: state_id/episode_uid must be non-empty")
        if state_id in state_ids:
            raise ValueError(f"{record_source}: duplicate state_id {state_id!r}")
        state_ids.add(state_id)
        if episode_uid not in normalized_assignments:
            raise ValueError(
                f"{record_source}: episode_uid is absent from the parent reference "
                "assignment contract"
            )
        assignment = normalized_assignments[episode_uid]
        if assignment == "uncond" and int(source_mode_value) != 0:
            raise ValueError(f"{record_source}: always-UNCOND trajectory used IDM")
        if assignment == "idm" and int(source_mode_value) != 1:
            raise ValueError(f"{record_source}: always-IDM trajectory used UNCOND")
        existing = trajectory_episode.setdefault(int(trajectory_id), episode_uid)
        if existing != episode_uid:
            raise ValueError(
                f"{record_source}: trajectory_id {trajectory_id} spans multiple episodes"
            )
        if str(record["phase"]) not in {
            "approach",
            "contact_alignment",
            "transport_completion",
            "unknown",
        }:
            raise ValueError(f"{record_source}: unsupported pre-treatment phase")
        if not isinstance(record["phase_reliable"], bool):
            raise ValueError(f"{record_source}: phase_reliable must be boolean")
        if not isinstance(record["base_task"], str) or not record["base_task"]:
            raise ValueError(f"{record_source}: base_task must be non-empty")
        if not isinstance(record["task_suite_name"], str) or not record["task_suite_name"]:
            raise ValueError(f"{record_source}: task_suite_name must be non-empty")
        if not isinstance(record["task_description"], str) or not record["task_description"]:
            raise ValueError(f"{record_source}: task_description must be non-empty")
        for key in ("factor", "level"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"{record_source}: {key} must be non-empty")
        for key in ("trial_id", "reset_state_id", "env_seed"):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{record_source}: {key} must be a non-negative integer")
        if not isinstance(record["perturbation_id"], str) or not record["perturbation_id"]:
            raise ValueError(f"{record_source}: perturbation_id must be non-empty")
        if not isinstance(record["asset_ids"], list) or not record["asset_ids"] or any(
            not isinstance(value, str) or not value for value in record["asset_ids"]
        ):
            raise ValueError(
                f"{record_source}: asset_ids must be a non-empty string list"
            )
        snapshot_sha = str(record["snapshot_sha256"]).lower()
        if len(snapshot_sha) != 64 or any(
            character not in "0123456789abcdef" for character in snapshot_sha
        ):
            raise ValueError(
                f"{record_source}: snapshot_sha256 must be a lowercase SHA256"
            )

    normalized = {
        "schema": PAIRED_SCHEMA,
        "meta": dict(meta),
        "data": {key: value for key, value in data.items()},
        "records": [dict(record) for record in records],
    }
    return normalized


def save_paired_shard(
    path: str | os.PathLike[str],
    *,
    data: Mapping[str, torch.Tensor],
    records: list[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> str:
    payload = validate_paired_payload(
        {"schema": PAIRED_SCHEMA, "meta": meta, "data": data, "records": records},
        source=os.fspath(path),
    )
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return str(output)


def _invariant_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"shard_index", "num_samples", "created_at", "source_episodes"}
    return {key: value for key, value in meta.items() if key not in ignored}


def load_paired_shards(
    paths: str | os.PathLike[str] | Sequence[str],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Load shards, enforce one provenance contract and remap trajectory ids."""
    resolved = _resolve_paths(paths)
    payloads = [
        validate_paired_payload(
            torch.load(path, map_location="cpu", weights_only=False), source=path
        )
        for path in resolved
    ]
    reference = _invariant_meta(payloads[0]["meta"])
    for path, payload in zip(resolved[1:], payloads[1:]):
        actual = _invariant_meta(payload["meta"])
        if actual != reference:
            changed = sorted(
                key
                for key in set(reference) | set(actual)
                if reference.get(key) != actual.get(key)
            )
            raise ValueError(f"{path}: paired shard provenance differs in {changed}")

    all_records: list[dict[str, Any]] = []
    all_data: dict[str, list[torch.Tensor]] = {}
    episode_to_group: dict[str, int] = {}
    remapped_groups: list[torch.Tensor] = []
    seen_states: set[str] = set()
    for payload in payloads:
        records = payload["records"]
        for record in records:
            state_id = str(record["state_id"])
            if state_id in seen_states:
                raise ValueError(f"paired shards contain duplicate state_id {state_id!r}")
            seen_states.add(state_id)
            all_records.append(record)
        groups = []
        for record in records:
            episode_uid = str(record["episode_uid"])
            if episode_uid not in episode_to_group:
                episode_to_group[episode_uid] = len(episode_to_group)
            groups.append(episode_to_group[episode_uid])
        remapped_groups.append(torch.tensor(groups, dtype=torch.int64))
        for key, value in payload["data"].items():
            if key == "trajectory_id":
                continue
            all_data.setdefault(key, []).append(value.cpu())
    data = {key: torch.cat(values, dim=0) for key, values in all_data.items()}
    data["trajectory_id"] = torch.cat(remapped_groups, dim=0)

    shard_roots = {
        Path(path).parent.parent
        for path in resolved
        if Path(path).parent.name == "tensors"
    }
    splits_sha256 = None
    if len(shard_roots) == 1:
        splits_path = next(iter(shard_roots)) / "splits.json"
        if splits_path.is_file():
            try:
                splits_payload = json.loads(splits_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid paired split contract {splits_path}: {exc}") from exc
            mapping = splits_payload.get("trajectory_to_fold")
            if (
                splits_payload.get("schema") != PAIRED_SCHEMA
                or int(splits_payload.get("num_folds", -1)) != PAIRED_NUM_FOLDS
                or not isinstance(mapping, Mapping)
            ):
                raise ValueError(f"invalid paired five-fold contract {splits_path}")
            missing_episodes = sorted(set(episode_to_group) - set(mapping))
            if missing_episodes:
                raise ValueError(
                    f"paired split contract is missing trajectories {missing_episodes}"
                )
            data["fold_id"] = torch.tensor(
                [int(mapping[str(record["episode_uid"])]) for record in all_records],
                dtype=torch.int64,
            )
            splits_sha256 = _sha256_file(splits_path)

    fingerprint_payload = {
        "schema": PAIRED_SCHEMA,
        "shards": [
            {"index": index, "sha256": _sha256_file(path)}
            for index, path in enumerate(resolved)
        ],
        "invariant_meta": reference,
        "state_ids": [record["state_id"] for record in all_records],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    meta = {
        **payloads[0]["meta"],
        "paired_dataset_fingerprint": fingerprint,
        "source_shards": [
            {"path": path, "sha256": _sha256_file(path)} for path in resolved
        ],
        "num_samples": len(all_records),
        "num_trajectories": len(episode_to_group),
    }
    if splits_sha256 is not None:
        meta["splits_sha256"] = splits_sha256
    validate_paired_payload(
        {"schema": PAIRED_SCHEMA, "meta": meta, "data": data, "records": all_records},
        source="<merged-paired-v1>",
    )
    return data, all_records, meta


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL row")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read paired-v1 JSONL {path}: {exc}") from exc
    return rows


def _path_for_contract(path: str, root: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        normalized = Path(os.path.normpath(candidate))
        if normalized.parts and normalized.parts[0] == "..":
            raise ValueError(f"paired artifact path escapes dataset root: {path}")
        return str(normalized)
    resolved = candidate.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _five_fold_split(
    records: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    trajectories = sorted({str(record["episode_uid"]) for record in records})
    if len(trajectories) < PAIRED_NUM_FOLDS:
        raise ValueError(
            f"paired-v1 requires at least {PAIRED_NUM_FOLDS} complete trajectories "
            f"for five-fold cross-fitting, got {len(trajectories)}"
        )
    shuffled = list(trajectories)
    random.Random(int(seed)).shuffle(shuffled)
    trajectory_to_fold = {
        trajectory: index % PAIRED_NUM_FOLDS
        for index, trajectory in enumerate(shuffled)
    }
    all_trajectories = set(trajectories)
    folds = []
    for fold in range(PAIRED_NUM_FOLDS):
        validation = sorted(
            trajectory
            for trajectory, assigned in trajectory_to_fold.items()
            if assigned == fold
        )
        training = sorted(all_trajectories - set(validation))
        folds.append(
            {
                "fold": fold,
                "train_episode_uids": training,
                "validation_episode_uids": validation,
            }
        )
    return {
        "schema": PAIRED_SCHEMA,
        "artifact": "trajectory-five-fold-splits",
        "num_folds": PAIRED_NUM_FOLDS,
        "seed": int(seed),
        "trajectory_to_fold": trajectory_to_fold,
        "folds": folds,
    }


def write_paired_dataset(
    path: str | os.PathLike[str],
    *,
    data: Mapping[str, torch.Tensor],
    records: list[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> str:
    """Write the required paired-v1 directory artifact and validate it."""
    root = Path(path).expanduser().resolve()
    if root.suffix == ".pt":
        raise ValueError(
            "paired collection --out must be a dataset directory, not a .pt file"
        )
    root.mkdir(parents=True, exist_ok=True)
    required_paths = (
        root / "states.jsonl",
        root / "outcomes.jsonl",
        root / "splits.json",
        root / "metadata.json",
    )
    existing = [str(value) for value in required_paths if value.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite an existing paired-v1 contract: {existing}"
        )
    payload = validate_paired_payload(
        {"schema": PAIRED_SCHEMA, "meta": meta, "data": data, "records": records},
        source=str(root),
    )
    # Store in-artifact snapshots as root-relative paths in both JSONL and the
    # tensor shard. This keeps a validated dataset portable when a logical
    # multi-suite merge is atomically renamed into place.
    normalized_records = []
    for record in payload["records"]:
        normalized = dict(record)
        if normalized.get("snapshot_path"):
            normalized["snapshot_path"] = _path_for_contract(
                str(normalized["snapshot_path"]), root
            )
        normalized_records.append(normalized)
    payload["records"] = normalized_records
    tensors = root / "tensors"
    tensors.mkdir(parents=True, exist_ok=True)
    shard = tensors / "shard_00000.pt"
    if shard.exists():
        raise FileExistsError(f"refusing to overwrite paired tensor shard {shard}")
    save_paired_shard(
        shard,
        data=payload["data"],
        records=payload["records"],
        meta=payload["meta"],
    )
    shard_relative = str(shard.relative_to(root))
    states = []
    outcomes = []
    n = len(payload["records"])
    for index, record in enumerate(payload["records"]):
        state = {
            "schema": PAIRED_SCHEMA,
            **dict(record),
            "trajectory_id": int(payload["data"]["trajectory_id"][index]),
            "decision_index": int(payload["data"]["decision_index"][index]),
            "task_id": int(payload["data"]["task_id"][index]),
            "source_mode": int(payload["data"]["source_mode"][index]),
            "branch_seed": int(payload["data"]["branch_seed"][index]),
            "tensor_shard": shard_relative,
            "tensor_row": index,
        }
        if state["snapshot_path"]:
            state["snapshot_path"] = _path_for_contract(
                str(state["snapshot_path"]), root
            )
        states.append(state)
        for mode_name, mode_id in zip(PAIRED_MODE_ORDER, (0, 1)):
            suffix = "uncond" if mode_id == 0 else "idm"
            outcome = {
                "schema": PAIRED_SCHEMA,
                "state_id": record["state_id"],
                "mode": mode_name,
                "mode_id": mode_id,
                "success": bool(payload["data"][f"success_{suffix}"][index]),
                "progress_1": float(
                    payload["data"][f"progress_1_{suffix}"][index]
                ),
                "progress_3": float(
                    payload["data"][f"progress_3_{suffix}"][index]
                ),
                "continuation_mode": "uncond",
                "branch_seed": int(payload["data"]["branch_seed"][index]),
            }
            sensitivity_key = f"sensitivity_success_{suffix}"
            if sensitivity_key in payload["data"]:
                sensitivity = bool(payload["data"]["sensitivity_mask"][index])
                outcome.update(
                    sensitivity_idm_continuation=sensitivity,
                    sensitivity_success=(
                        bool(payload["data"][sensitivity_key][index])
                        if sensitivity
                        else None
                    ),
                    sensitivity_progress_3=(
                        float(
                            payload["data"][
                                f"sensitivity_progress_3_{suffix}"
                            ][index]
                        )
                        if sensitivity
                        else None
                    ),
                )
            outcomes.append(outcome)
    if len(states) != n or len(outcomes) != 2 * n:
        raise RuntimeError("paired-v1 writer produced an invalid state/outcome count")
    splits = _five_fold_split(states, seed=int(payload["meta"]["collector_seed"]))
    _jsonl_write(root / "states.jsonl", states)
    _jsonl_write(root / "outcomes.jsonl", outcomes)
    with (root / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2, sort_keys=True)
    metadata = {
        "schema": PAIRED_SCHEMA,
        "artifact": "paired-counterfactual-dataset",
        "meta": payload["meta"],
        "num_states": n,
        "num_outcomes": 2 * n,
        "tensor_shards": [
            {"path": shard_relative, "sha256": _sha256_file(shard)}
        ],
    }
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
    validate_paired_dataset(root)
    return str(root)


def _resolve_dataset_root(
    paired: str | os.PathLike[str] | Sequence[str],
) -> Path:
    values = [paired] if isinstance(paired, (str, os.PathLike)) else list(paired)
    if len(values) == 1 and Path(values[0]).expanduser().is_dir():
        return Path(values[0]).expanduser().resolve()
    shards = [Path(value).expanduser().resolve() for value in _resolve_paths(values)]
    roots = {
        shard.parent.parent
        for shard in shards
        if shard.parent.name == "tensors"
    }
    if len(roots) != 1:
        raise ValueError(
            "paired shard inputs must all live under one <dataset>/tensors/ "
            "directory so states/outcomes/splits can be validated"
        )
    return roots.pop()


def validate_paired_dataset(
    paired: str | os.PathLike[str] | Sequence[str],
) -> dict[str, Any]:
    """Validate shards, snapshot bytes, strict U/I pairs and fold isolation."""
    root = _resolve_dataset_root(paired)
    files = {
        name: root / name
        for name in ("states.jsonl", "outcomes.jsonl", "splits.json", "metadata.json")
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"paired-v1 directory is incomplete: {missing}")
    states = _jsonl_read(files["states.jsonl"])
    outcomes = _jsonl_read(files["outcomes.jsonl"])
    try:
        splits = json.loads(files["splits.json"].read_text(encoding="utf-8"))
        metadata = json.loads(files["metadata.json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid paired-v1 JSON contract under {root}: {exc}") from exc
    if (
        metadata.get("schema") != PAIRED_SCHEMA
        or metadata.get("artifact") != "paired-counterfactual-dataset"
    ):
        raise ValueError("metadata.json is not a paired-v1 dataset contract")
    shard_entries = metadata.get("tensor_shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError("metadata.json has no tensor_shards")
    shard_paths = []
    declared_shard_names = set()
    for entry in shard_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("metadata tensor_shards entries must contain path/SHA")
        shard = (root / entry["path"]).resolve()
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"tensor shard escapes paired root: {shard}") from exc
        if not shard.is_file() or _sha256_file(shard) != entry.get("sha256"):
            raise ValueError(f"tensor shard missing or SHA-mismatched: {shard}")
        shard_paths.append(str(shard))
        declared_shard_names.add(str(entry["path"]))
    data, records, meta = load_paired_shards(shard_paths)
    declared_meta = metadata.get("meta")
    if not isinstance(declared_meta, Mapping):
        raise ValueError("metadata.json is missing the paired shard metadata")
    metadata_mismatches = {
        key: (meta.get(key), value)
        for key, value in declared_meta.items()
        if meta.get(key) != value
    }
    if metadata_mismatches:
        raise ValueError(
            "metadata.json provenance differs from tensor shard metadata: "
            f"{metadata_mismatches}"
        )
    n = len(records)
    if len(states) != n or int(metadata.get("num_states", -1)) != n:
        raise ValueError("states.jsonl count does not match paired tensor rows")
    if len(outcomes) != 2 * n or int(metadata.get("num_outcomes", -1)) != 2 * n:
        raise ValueError("outcomes.jsonl must contain exactly two rows per state")

    record_by_id = {str(record["state_id"]): record for record in records}
    state_by_id = {}
    index_by_id = {}
    for index, state in enumerate(states):
        state_id = str(state.get("state_id", ""))
        if state.get("schema") != PAIRED_SCHEMA or state_id not in record_by_id:
            raise ValueError(f"states.jsonl row {index} has unknown schema/state_id")
        if state_id in state_by_id:
            raise ValueError(f"states.jsonl duplicates state_id {state_id!r}")
        record = record_by_id[state_id]
        for key in (
            "episode_uid",
            "base_task",
            "task_suite_name",
            "task_description",
            "trial_id",
            "reset_state_id",
            "env_seed",
            "factor",
            "level",
            "perturbation_id",
            "phase",
            "phase_reliable",
            "snapshot_sha256",
            "asset_ids",
        ):
            if state.get(key) != record.get(key):
                raise ValueError(f"state {state_id!r} differs from tensor record in {key}")
        snapshot_value = state.get("snapshot_path")
        if not isinstance(snapshot_value, str) or not snapshot_value:
            raise ValueError(f"state {state_id!r} has no snapshot_path")
        snapshot = Path(snapshot_value).expanduser()
        if not snapshot.is_absolute():
            snapshot = root / snapshot
        snapshot = snapshot.resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(f"state {state_id!r} snapshot is missing: {snapshot}")
        record_snapshot = Path(str(record["snapshot_path"])).expanduser()
        if not record_snapshot.is_absolute():
            record_snapshot = root / record_snapshot
        if record_snapshot.resolve() != snapshot:
            raise ValueError(
                f"state {state_id!r} snapshot path differs from tensor record"
            )
        if _sha256_file(snapshot) != state["snapshot_sha256"]:
            raise ValueError(f"state {state_id!r} snapshot SHA256 mismatch")
        tensor_row = int(state.get("tensor_row", -1))
        if state.get("tensor_shard") not in declared_shard_names:
            raise ValueError(
                f"state {state_id!r} references an undeclared tensor shard"
            )
        if tensor_row != index:
            raise ValueError(
                f"state {state_id!r} tensor_row {tensor_row} != canonical row {index}"
            )
        if int(state.get("decision_index", -1)) != int(data["decision_index"][index]):
            raise ValueError(f"state {state_id!r} decision_index differs from tensor")
        tensor_identity = {
            "task_id": int(data["task_id"][index]),
            "source_mode": int(data["source_mode"][index]),
            "branch_seed": int(data["branch_seed"][index]),
        }
        identity_mismatches = {
            key: (state.get(key), value)
            for key, value in tensor_identity.items()
            if state.get(key) != value
        }
        if identity_mismatches:
            raise ValueError(
                f"state {state_id!r} identity differs from tensor: "
                f"{identity_mismatches}"
            )
        state_by_id[state_id] = state
        index_by_id[state_id] = index

    outcome_by_state: dict[str, dict[str, Mapping[str, Any]]] = {}
    for outcome in outcomes:
        state_id = str(outcome.get("state_id", ""))
        mode = str(outcome.get("mode", ""))
        if outcome.get("schema") != PAIRED_SCHEMA or state_id not in state_by_id:
            raise ValueError("outcomes.jsonl contains an unknown schema/state_id")
        if mode not in PAIRED_MODE_ORDER:
            raise ValueError(f"state {state_id!r} has invalid outcome mode {mode!r}")
        modes = outcome_by_state.setdefault(state_id, {})
        if mode in modes:
            raise ValueError(f"state {state_id!r} duplicates {mode} outcome")
        modes[mode] = outcome
    for state_id, index in index_by_id.items():
        modes = outcome_by_state.get(state_id, {})
        if set(modes) != set(PAIRED_MODE_ORDER):
            raise ValueError(f"state {state_id!r} does not have exactly U/I outcomes")
        for mode_name, mode_id in zip(PAIRED_MODE_ORDER, (0, 1)):
            outcome = modes[mode_name]
            suffix = "uncond" if mode_id == 0 else "idm"
            expected = {
                "mode_id": mode_id,
                "success": bool(data[f"success_{suffix}"][index]),
                "progress_1": float(data[f"progress_1_{suffix}"][index]),
                "progress_3": float(data[f"progress_3_{suffix}"][index]),
                "continuation_mode": "uncond",
                "branch_seed": int(data["branch_seed"][index]),
            }
            sensitivity = bool(data["sensitivity_mask"][index])
            expected.update(
                sensitivity_idm_continuation=sensitivity,
                sensitivity_success=(
                    bool(data[f"sensitivity_success_{suffix}"][index])
                    if sensitivity
                    else None
                ),
                sensitivity_progress_3=(
                    float(data[f"sensitivity_progress_3_{suffix}"][index])
                    if sensitivity
                    else None
                ),
            )
            mismatches = {
                key: (outcome.get(key), value)
                for key, value in expected.items()
                if outcome.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"state {state_id!r} {mode_name} outcome differs from tensor: "
                    f"{mismatches}"
                )

    if not isinstance(splits, Mapping) or splits.get("schema") != PAIRED_SCHEMA:
        raise ValueError("splits.json is not paired-v1")
    if int(splits.get("num_folds", -1)) != PAIRED_NUM_FOLDS:
        raise ValueError("splits.json must define exactly five folds")
    trajectory_to_fold = splits.get("trajectory_to_fold")
    folds = splits.get("folds")
    if (
        not isinstance(trajectory_to_fold, Mapping)
        or not isinstance(folds, list)
        or len(folds) != PAIRED_NUM_FOLDS
    ):
        raise ValueError("splits.json is missing trajectory mapping/folds")
    all_episodes = {str(state["episode_uid"]) for state in states}
    if set(trajectory_to_fold) != all_episodes:
        raise ValueError("splits.json trajectories do not match states.jsonl")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < PAIRED_NUM_FOLDS
        for value in trajectory_to_fold.values()
    ):
        raise ValueError("splits.json trajectory fold ids must be integers in [0,4]")
    seen_validation: set[str] = set()
    for expected_fold, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or int(fold.get("fold", -1)) != expected_fold:
            raise ValueError("splits.json folds must be ordered 0..4")
        training = set(fold.get("train_episode_uids", []))
        validation = set(fold.get("validation_episode_uids", []))
        if not validation or training & validation:
            raise ValueError(f"fold {expected_fold} is empty or leaks trajectories")
        if training | validation != all_episodes:
            raise ValueError(f"fold {expected_fold} does not cover all trajectories")
        expected_validation = {
            episode
            for episode, assigned in trajectory_to_fold.items()
            if int(assigned) == expected_fold
        }
        if validation != expected_validation:
            raise ValueError(f"fold {expected_fold} disagrees with trajectory_to_fold")
        if seen_validation & validation:
            raise ValueError("one trajectory appears in multiple validation folds")
        seen_validation |= validation
    if seen_validation != all_episodes:
        raise ValueError("not every trajectory appears in exactly one validation fold")
    logical_merge = declared_meta.get("logical_merge")
    logical_suites = None
    composite_source_fingerprint = None
    if logical_merge is not None:
        if not isinstance(logical_merge, Mapping):
            raise ValueError("meta.logical_merge must be an object")
        if logical_merge.get("schema") != "paired-v1-logical-merge-v1":
            raise ValueError("meta.logical_merge has an unsupported schema")
        composite_source_fingerprint = _require_sha256(
            logical_merge.get("composite_source_fingerprint"),
            field="meta.logical_merge.composite_source_fingerprint",
            source=str(root),
        )
        expected_composite = _canonical_sha256(
            {
                key: value
                for key, value in logical_merge.items()
                if key != "composite_source_fingerprint"
            }
        )
        if expected_composite != composite_source_fingerprint:
            raise ValueError("logical paired-v1 composite source fingerprint mismatch")
        parent_sha = _require_sha256(
            logical_merge.get("parent_manifest_sha256"),
            field="meta.logical_merge.parent_manifest_sha256",
            source=str(root),
        )
        if parent_sha != meta["episode_manifest_sha256"]:
            raise ValueError("logical paired-v1 changed its parent manifest identity")
        parent_path_value = logical_merge.get("parent_manifest_path")
        if not isinstance(parent_path_value, str) or not parent_path_value:
            raise ValueError("logical paired-v1 has no embedded parent manifest path")
        parent_path = (root / parent_path_value).resolve()
        try:
            parent_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("logical paired-v1 parent manifest escapes dataset root") from exc
        if not parent_path.is_file() or _sha256_file(parent_path) != parent_sha:
            raise ValueError("logical paired-v1 embedded parent manifest SHA mismatch")
        try:
            parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("logical paired-v1 embedded parent manifest is invalid") from exc
        if (
            not isinstance(parent_payload, Mapping)
            or parent_payload.get("schema") != "libero-plus-episode-manifest-v1"
            or parent_payload.get("suite_partition") is not None
        ):
            raise ValueError("logical paired-v1 did not embed a logical parent manifest")
        if (
            str(parent_payload.get("libero_plus_commit", ""))
            != str(meta["libero_plus_commit"])
            or str(parent_payload.get("split", "")).lower()
            != str(meta["manifest_split"]).lower()
        ):
            raise ValueError(
                "logical paired-v1 parent manifest changes split/checkout provenance"
            )
        parent_episodes = parent_payload.get("episodes")
        if not isinstance(parent_episodes, list) or not parent_episodes:
            raise ValueError("logical paired-v1 parent manifest has no episodes")
        parent_by_uid: dict[str, Mapping[str, Any]] = {}
        for episode in parent_episodes:
            if not isinstance(episode, Mapping):
                raise ValueError("logical paired-v1 parent episode must be an object")
            uid = str(episode.get("episode_id", ""))
            if not uid or uid in parent_by_uid:
                raise ValueError("logical paired-v1 parent has duplicate episode identity")
            parent_by_uid[uid] = episode
        if int(parent_payload.get("num_entries", len(parent_episodes))) != len(
            parent_episodes
        ):
            raise ValueError("logical paired-v1 parent manifest count is inconsistent")
        if int(logical_merge.get("parent_num_episodes", -1)) != len(parent_episodes):
            raise ValueError("logical paired-v1 parent episode count is inconsistent")
        if set(parent_by_uid) != all_episodes:
            raise ValueError(
                "logical paired-v1 states do not exactly cover parent-manifest episodes"
            )
        for row_index, record in enumerate(records):
            parent_episode = parent_by_uid[str(record["episode_uid"])]
            expected_record = {
                "base_task": str(parent_episode.get("base_task", "")),
                "task_suite_name": str(parent_episode.get("task_suite_name", "")),
                "trial_id": int(parent_episode.get("trial_id", -1)),
                "reset_state_id": int(parent_episode.get("reset_state_id", -1)),
                "env_seed": int(parent_episode.get("env_seed", -1)),
                "factor": str(parent_episode.get("factor", "")),
                "level": str(parent_episode.get("level", "")),
                "perturbation_id": str(parent_episode.get("perturbation_id", "")),
                "asset_ids": list(parent_episode.get("asset_ids", [])),
            }
            if any(record.get(key) != value for key, value in expected_record.items()):
                raise ValueError(
                    f"logical paired-v1 state {record['state_id']!r} differs from "
                    "its embedded parent episode"
                )
            if int(data["task_id"][row_index]) != int(
                parent_episode.get("task_id", -1)
            ):
                raise ValueError(
                    f"logical paired-v1 state {record['state_id']!r} task_id "
                    "differs from its embedded parent episode"
                )
        components = logical_merge.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("logical paired-v1 has no suite components")
        component_suites: set[str] = set()
        component_episodes: set[str] = set()
        component_states = 0
        for component in components:
            if not isinstance(component, Mapping):
                raise ValueError("logical paired-v1 component must be an object")
            suite = str(component.get("task_suite_name", ""))
            if not suite or suite in component_suites:
                raise ValueError("logical paired-v1 has an empty/duplicate suite component")
            component_suites.add(suite)
            for field in (
                "paired_dataset_fingerprint",
                "metadata_sha256",
                "splits_sha256",
            ):
                _require_sha256(
                    component.get(field),
                    field=f"meta.logical_merge.components.{field}",
                    source=str(root),
                )
            episode_uids = component.get("episode_uids")
            if (
                not isinstance(episode_uids, list)
                or not episode_uids
                or any(not isinstance(value, str) or not value for value in episode_uids)
            ):
                raise ValueError("logical paired-v1 component episode_uids are invalid")
            if component_episodes & set(episode_uids):
                raise ValueError("logical paired-v1 episode appears in multiple suites")
            component_episodes.update(episode_uids)
            num_component_states = int(component.get("num_states", -1))
            num_component_trajectories = int(component.get("num_trajectories", -1))
            if num_component_states <= 0 or num_component_trajectories != len(
                episode_uids
            ):
                raise ValueError(
                    "logical paired-v1 component state/trajectory counts are invalid"
                )
            component_states += num_component_states
            expected_for_suite = {
                uid
                for uid, episode in parent_by_uid.items()
                if str(episode.get("task_suite_name", "")) == suite
            }
            if set(episode_uids) != expected_for_suite:
                raise ValueError(
                    f"logical paired-v1 suite {suite!r} does not exactly cover its "
                    "parent partition"
                )
        if component_episodes != all_episodes or component_states != n:
            raise ValueError("logical paired-v1 component coverage/count is inconsistent")
        record_suites = {str(record["task_suite_name"]) for record in records}
        parent_suites = {
            str(episode.get("task_suite_name", "")) for episode in parent_episodes
        }
        if component_suites != record_suites or component_suites != parent_suites:
            raise ValueError("logical paired-v1 is missing or adding a task suite")
        assignments = meta["reference_policy_episode_assignments"]
        if set(assignments) != all_episodes:
            raise ValueError(
                "logical paired-v1 reference assignments do not cover the parent"
            )
        assignment_counts = {
            name: list(assignments.values()).count(name)
            for name in ("uncond", "idm", "random_0.5")
        }
        if len(set(assignment_counts.values())) != 1:
            raise ValueError(
                "logical paired-v1 reference trajectories are not exact one-thirds"
            )
        logical_suites = sorted(component_suites)

    observed_suites = {str(record["task_suite_name"]) for record in records}
    if len(observed_suites) > 1 and logical_merge is None:
        raise ValueError(
            "multi-suite paired-v1 data require the strict logical merge contract"
        )
    summary = {
        "schema": PAIRED_SCHEMA,
        "dataset_root": str(root),
        "paired_dataset_fingerprint": meta["paired_dataset_fingerprint"],
        "num_states": n,
        "num_outcomes": len(outcomes),
        "num_trajectories": len(all_episodes),
        "num_folds": PAIRED_NUM_FOLDS,
        "snapshots_verified": n,
        "tensor_shards_verified": len(shard_paths),
    }
    if logical_suites is not None:
        summary.update(
            logical_parent_manifest_sha256=meta["episode_manifest_sha256"],
            composite_source_fingerprint=composite_source_fingerprint,
            task_suites=logical_suites,
            logical_suite_count=len(logical_suites),
        )
    return summary


def _safe_component_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not slug:
        raise ValueError(f"task suite name has no safe path representation: {value!r}")
    return slug


def _resolve_record_snapshot(record: Mapping[str, Any], root: Path) -> Path:
    value = Path(str(record["snapshot_path"])).expanduser()
    if not value.is_absolute():
        value = root / value
    return value.resolve()


def merge_paired_suite_datasets(
    logical_manifest,
    suite_datasets: Mapping[str, str | os.PathLike[str]],
    out: str | os.PathLike[str],
) -> dict[str, Any]:
    """Merge physical per-suite paired-v1 artifacts into one logical dataset.

    The input manifest must be the complete logical parent. Every suite must be
    represented exactly once, every parent episode must contribute at least one
    decision state, and all WAM/feature/stats/solver/collector provenance must be
    identical. The output receives a fresh global trajectory-grouped five-fold
    split; physical per-suite folds are never reused.
    """
    if getattr(logical_manifest, "parent_manifest_path", None) is not None:
        raise ValueError("paired logical merge requires the complete parent manifest")
    manifest_episodes = list(getattr(logical_manifest, "episodes", ()))
    if not manifest_episodes:
        raise ValueError("paired logical merge parent manifest has no episodes")
    parent_sha = _require_sha256(
        getattr(logical_manifest, "sha256", ""),
        field="logical manifest SHA256",
        source="<paired-logical-merge>",
    )
    parent_path = Path(str(getattr(logical_manifest, "path", ""))).expanduser().resolve()
    if not parent_path.is_file() or _sha256_file(parent_path) != parent_sha:
        raise ValueError("logical parent manifest bytes do not match its SHA256")
    expected_by_suite: dict[str, dict[str, Any]] = {}
    expected_by_uid: dict[str, Any] = {}
    for episode in manifest_episodes:
        uid = str(episode.episode_id)
        suite = str(episode.task_suite_name)
        if not uid or uid in expected_by_uid or not suite:
            raise ValueError("logical parent manifest has duplicate/empty identity")
        expected_by_uid[uid] = episode
        expected_by_suite.setdefault(suite, {})[uid] = episode
    normalized_sources = {
        str(suite): Path(path).expanduser().resolve()
        for suite, path in suite_datasets.items()
    }
    if len(normalized_sources) != len(suite_datasets):
        raise ValueError("paired logical merge contains duplicate suite bindings")
    expected_suites = set(expected_by_suite)
    if set(normalized_sources) != expected_suites:
        raise ValueError(
            "paired suite dataset set does not match logical manifest: "
            f"missing={sorted(expected_suites - set(normalized_sources))}, "
            f"extra={sorted(set(normalized_sources) - expected_suites)}"
        )
    source_roots = list(normalized_sources.values())
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("one physical paired-v1 dataset is bound to multiple suites")

    loaded: list[
        tuple[
            str,
            Path,
            dict[str, torch.Tensor],
            list[dict[str, Any]],
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    reference_contract: dict[str, Any] | None = None
    common_keys = tuple(PAIRED_META_KEYS)
    seen_states: set[str] = set()
    seen_episodes: set[str] = set()
    components: list[dict[str, Any]] = []
    for suite in sorted(normalized_sources):
        root = normalized_sources[suite]
        summary = validate_paired_dataset(root)
        data, records, meta = load_paired_shards(root)
        if meta.get("logical_merge") is not None:
            raise ValueError(
                f"suite {suite!r} input is already a logical merge; nested "
                "paired-v1 merges are forbidden"
            )
        actual_contract = {key: meta[key] for key in common_keys}
        if reference_contract is None:
            reference_contract = actual_contract
        elif actual_contract != reference_contract:
            changed = sorted(
                key
                for key in common_keys
                if actual_contract.get(key) != reference_contract.get(key)
            )
            raise ValueError(
                f"suite {suite!r} paired provenance differs in {changed}"
            )
        if meta["episode_manifest_sha256"] != parent_sha:
            raise ValueError(
                f"suite {suite!r} is not bound to logical parent {parent_sha}"
            )
        if str(meta["libero_plus_commit"]) != str(logical_manifest.libero_plus_commit):
            raise ValueError(f"suite {suite!r} changes LIBERO-Plus commit")
        if str(meta["manifest_split"]) != str(logical_manifest.split):
            raise ValueError(f"suite {suite!r} changes manifest split")
        expected_episodes = set(expected_by_suite[suite])
        actual_episodes = {str(record["episode_uid"]) for record in records}
        if actual_episodes != expected_episodes:
            raise ValueError(
                f"suite {suite!r} does not exactly cover its logical partition: "
                f"missing={sorted(expected_episodes - actual_episodes)}, "
                f"extra={sorted(actual_episodes - expected_episodes)}"
            )
        if seen_episodes & actual_episodes:
            raise ValueError("one trajectory appears in multiple suite datasets")
        seen_episodes.update(actual_episodes)
        for index, record in enumerate(records):
            state_id = str(record["state_id"])
            if state_id in seen_states:
                raise ValueError(f"duplicate state_id across suite datasets: {state_id!r}")
            seen_states.add(state_id)
            if str(record["task_suite_name"]) != suite:
                raise ValueError(
                    f"suite binding {suite!r} contains record from "
                    f"{record['task_suite_name']!r}"
                )
            episode = expected_by_uid[str(record["episode_uid"])]
            expected_record = {
                "base_task": str(episode.base_task),
                "task_suite_name": str(episode.task_suite_name),
                "trial_id": int(episode.trial_id),
                "reset_state_id": int(episode.reset_state_id),
                "env_seed": int(episode.env_seed),
                "factor": str(episode.factor),
                "level": str(episode.level),
                "perturbation_id": str(episode.perturbation_id),
                "asset_ids": list(episode.asset_ids),
            }
            mismatches = {
                key: (record.get(key), value)
                for key, value in expected_record.items()
                if record.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"suite {suite!r} state {state_id!r} differs from parent "
                    f"manifest: {mismatches}"
                )
            if int(data["task_id"][index]) != int(episode.task_id):
                raise ValueError(
                    f"suite {suite!r} state {state_id!r} task_id differs from parent"
                )
        metadata_path = root / "metadata.json"
        splits_path = root / "splits.json"
        component = {
            "task_suite_name": suite,
            "paired_dataset_fingerprint": summary["paired_dataset_fingerprint"],
            "metadata_sha256": _sha256_file(metadata_path),
            "splits_sha256": _sha256_file(splits_path),
            "num_states": int(summary["num_states"]),
            "num_trajectories": int(summary["num_trajectories"]),
            "episode_uids": sorted(actual_episodes),
        }
        components.append(component)
        loaded.append((suite, root, data, records, meta, summary))
    if seen_episodes != set(expected_by_uid):
        raise ValueError("suite datasets do not cover the complete logical manifest")
    assert reference_contract is not None
    assignments = reference_contract["reference_policy_episode_assignments"]
    if set(assignments) != set(expected_by_uid):
        raise ValueError(
            "parent reference assignment contract does not exactly cover the "
            "logical manifest"
        )
    assignment_counts = [
        list(assignments.values()).count(name)
        for name in ("uncond", "idm", "random_0.5")
    ]
    if len(set(assignment_counts)) != 1:
        raise ValueError("logical reference-policy assignments are not exact one-thirds")

    output = Path(out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite logical paired-v1 output: {output}")
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"paired logical merge staging path exists: {staging}")
    staging.mkdir(parents=True)
    try:
        provenance_dir = staging / "provenance"
        provenance_dir.mkdir()
        embedded_parent = provenance_dir / "parent_manifest.json"
        shutil.copyfile(parent_path, embedded_parent)
        trajectory_group = {
            str(episode.episode_id): index
            for index, episode in enumerate(manifest_episodes)
        }
        merged_records: list[dict[str, Any]] = []
        merged_values: dict[str, list[torch.Tensor]] = {
            key: [] for key in PAIRED_TENSOR_KEYS
        }
        snapshots_root = staging / "snapshots"
        for suite, source_root, data, records, _meta, _summary in loaded:
            suite_snapshot_dir = snapshots_root / _safe_component_slug(suite)
            suite_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for index, source_record in enumerate(records):
                record = dict(source_record)
                source_snapshot = _resolve_record_snapshot(record, source_root)
                if not source_snapshot.is_file() or _sha256_file(source_snapshot) != record[
                    "snapshot_sha256"
                ]:
                    raise ValueError(
                        f"source snapshot changed after validation: {source_snapshot}"
                    )
                destination = suite_snapshot_dir / f"{record['state_id']}.pt"
                if destination.exists():
                    raise ValueError(f"duplicate logical snapshot path: {destination}")
                shutil.copyfile(source_snapshot, destination)
                if _sha256_file(destination) != record["snapshot_sha256"]:
                    raise RuntimeError("logical snapshot copy changed immutable bytes")
                record["snapshot_path"] = str(destination)
                merged_records.append(record)
                for key in PAIRED_TENSOR_KEYS:
                    if key == "trajectory_id":
                        value = torch.tensor(
                            trajectory_group[str(record["episode_uid"])],
                            dtype=torch.int64,
                        )
                    else:
                        value = data[key][index].detach().cpu()
                    merged_values[key].append(value)
        merged_data = {
            key: torch.stack(values, dim=0) for key, values in merged_values.items()
        }
        logical_merge = {
            "schema": "paired-v1-logical-merge-v1",
            "parent_manifest_path": "provenance/parent_manifest.json",
            "parent_manifest_sha256": parent_sha,
            "parent_num_episodes": len(manifest_episodes),
            "components": components,
        }
        logical_merge["composite_source_fingerprint"] = _canonical_sha256(
            logical_merge
        )
        merged_meta = {
            **reference_contract,
            "num_samples": len(merged_records),
            "logical_merge": logical_merge,
        }
        write_paired_dataset(
            staging,
            data=merged_data,
            records=merged_records,
            meta=merged_meta,
        )
        validate_paired_dataset(staging)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_paired_dataset(output)
