# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Immutable LIBERO-Plus episode manifests used by paired/final evaluation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


EPISODE_MANIFEST_SCHEMA = "libero-plus-episode-manifest-v1"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenEpisode:
    episode_id: str
    base_task: str
    task_suite_name: str
    task_id: int
    factor: str
    level: str
    bddl_path: str
    bddl_sha256: str
    reset_state_id: int
    trial_id: int
    env_seed: int
    perturbation_id: str
    asset_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenEpisodeManifest:
    path: str
    # ``sha256`` is the logical experiment-manifest identity.  For a physical
    # single-suite partition it remains the SHA256 of the complete parent
    # manifest so independently executed suites can be merged without changing
    # the preregistered benchmark contract.
    sha256: str
    file_sha256: str
    libero_plus_root: str
    libero_plus_commit: str
    split: str
    episodes: tuple[FrozenEpisode, ...]
    parent_manifest_path: str | None = None
    task_suite_partition: str | None = None

    def shard(self, process_index: int, num_processes: int) -> tuple[FrozenEpisode, ...]:
        process_index = int(process_index)
        num_processes = int(num_processes)
        if num_processes <= 0 or not 0 <= process_index < num_processes:
            raise ValueError(
                f"invalid manifest shard {process_index}/{num_processes}"
            )
        return self.episodes[process_index::num_processes]


def _require_string(entry: Mapping[str, Any], key: str, *, source: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: `{key}` must be a non-empty string")
    return value.strip()


def _require_int(entry: Mapping[str, Any], key: str, *, source: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{source}: `{key}` must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: `{key}` must be an integer") from exc
    if value < 0:
        raise ValueError(f"{source}: `{key}` must be non-negative")
    return value


def _git_head(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"LIBERO_PLUS_ROOT={root} is not a readable git checkout; exact "
            "benchmark commit cannot be verified"
        ) from exc
    return proc.stdout.strip()


def load_frozen_episode_manifest(
    path: str | os.PathLike[str],
    *,
    libero_plus_root: str | os.PathLike[str] | None = None,
    libero_plus_commit: str | None = None,
    libero_import_module: str | None = None,
    verify_git: bool = True,
    verify_import: bool = True,
) -> FrozenEpisodeManifest:
    """Load and validate the frozen episode contract.

    The root and commit default to ``LIBERO_PLUS_ROOT`` and
    ``LIBERO_PLUS_COMMIT``.  Both are mandatory.  This intentionally rejects a
    dynamic ``LIBERO_SUFFIX=all`` evaluation: every concrete BDDL file, seed and
    perturbation must appear in the manifest.
    """
    manifest_path = Path(path).expanduser().resolve()
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{manifest_path}: invalid episode manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != EPISODE_MANIFEST_SCHEMA:
        actual = payload.get("schema") if isinstance(payload, dict) else None
        raise ValueError(
            f"{manifest_path}: expected schema={EPISODE_MANIFEST_SCHEMA!r}, got {actual!r}"
        )

    root_value = libero_plus_root or os.environ.get("LIBERO_PLUS_ROOT")
    commit_value = libero_plus_commit or os.environ.get("LIBERO_PLUS_COMMIT")
    if not root_value or not commit_value:
        raise ValueError(
            "frozen LIBERO-Plus manifests require LIBERO_PLUS_ROOT and "
            "LIBERO_PLUS_COMMIT (or explicit equivalent arguments)"
        )
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"LIBERO_PLUS_ROOT is not a directory: {root}")
    declared_commit = _require_string(
        payload, "libero_plus_commit", source=str(manifest_path)
    )
    if str(commit_value) != declared_commit:
        raise ValueError(
            f"LIBERO_PLUS_COMMIT={commit_value!r} does not match manifest "
            f"commit {declared_commit!r}"
        )
    if verify_git:
        actual_commit = _git_head(root)
        if actual_commit != declared_commit:
            raise ValueError(
                f"LIBERO-Plus checkout HEAD {actual_commit!r} does not match "
                f"manifest commit {declared_commit!r}"
            )
    if verify_import:
        import_name = (
            libero_import_module
            or os.environ.get("LIBERO_PLUS_IMPORT_MODULE")
            or "libero"
        )
        try:
            imported_module = importlib.import_module(import_name)
        except ImportError as exc:
            raise ValueError(
                f"the {import_name!r} package is not importable from "
                "LIBERO_PLUS_ROOT"
            ) from exc
        imported_value = getattr(imported_module, "__file__", None)
        if not imported_value:
            raise ValueError(
                f"imported {import_name!r} is a namespace package without __file__; "
                "the frozen checkout identity cannot be verified"
            )
        imported_file = Path(imported_value).resolve()
        try:
            imported_file.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"imported {import_name!r} package {imported_file} is outside "
                f"LIBERO_PLUS_ROOT={root}; prepend the frozen checkout to PYTHONPATH"
            ) from exc

    declared_root = payload.get("libero_plus_root")
    if declared_root is not None and Path(declared_root).expanduser().resolve() != root:
        raise ValueError(
            f"manifest libero_plus_root={declared_root!r} does not match {root}"
        )
    source_sha256 = sha256_file(manifest_path)
    logical_sha256 = source_sha256
    parent_manifest_path: str | None = None
    task_suite_partition: str | None = None
    partition = payload.get("suite_partition")
    if partition is not None:
        if not isinstance(partition, Mapping):
            raise ValueError(f"{manifest_path}: suite_partition must be an object")
        parent_value = _require_string(
            partition, "parent_manifest_path", source=f"{manifest_path}:suite_partition"
        )
        expected_parent_sha = _require_string(
            partition,
            "parent_manifest_sha256",
            source=f"{manifest_path}:suite_partition",
        ).lower()
        if len(expected_parent_sha) != 64 or any(
            char not in "0123456789abcdef" for char in expected_parent_sha
        ):
            raise ValueError(
                f"{manifest_path}: suite_partition.parent_manifest_sha256 must "
                "be a lowercase SHA256"
            )
        parent_path = Path(parent_value).expanduser()
        if not parent_path.is_absolute():
            parent_path = manifest_path.parent / parent_path
        parent_path = parent_path.resolve()
        if parent_path == manifest_path:
            raise ValueError(f"{manifest_path}: suite partition cannot parent itself")
        if not parent_path.is_file():
            raise ValueError(
                f"{manifest_path}: suite partition parent does not exist: {parent_path}"
            )
        actual_parent_sha = sha256_file(parent_path)
        if actual_parent_sha != expected_parent_sha:
            raise ValueError(
                f"{manifest_path}: suite partition parent SHA256 mismatch: "
                f"{actual_parent_sha} != {expected_parent_sha}"
            )
        parent_manifest = load_frozen_episode_manifest(
            parent_path,
            libero_plus_root=root,
            libero_plus_commit=str(commit_value),
            libero_import_module=libero_import_module,
            verify_git=verify_git,
            verify_import=verify_import,
        )
        if parent_manifest.parent_manifest_path is not None:
            raise ValueError(f"{manifest_path}: nested suite partitions are forbidden")
        task_suite_partition = _require_string(
            partition,
            "task_suite_name",
            source=f"{manifest_path}:suite_partition",
        )
        declared_parent_entries = _require_int(
            partition,
            "parent_num_entries",
            source=f"{manifest_path}:suite_partition",
        )
        if declared_parent_entries != len(parent_manifest.episodes):
            raise ValueError(
                f"{manifest_path}: suite partition parent_num_entries="
                f"{declared_parent_entries} does not match parent "
                f"{len(parent_manifest.episodes)}"
            )
        expected_partition_ids = [
            episode.episode_id
            for episode in parent_manifest.episodes
            if episode.task_suite_name == task_suite_partition
        ]
        raw_partition_episodes = payload.get("episodes")
        actual_partition_ids = (
            [str(item.get("episode_id", "")) for item in raw_partition_episodes]
            if isinstance(raw_partition_episodes, list)
            and all(isinstance(item, Mapping) for item in raw_partition_episodes)
            else []
        )
        if not expected_partition_ids:
            raise ValueError(
                f"{manifest_path}: parent contains no suite {task_suite_partition!r}"
            )
        if actual_partition_ids != expected_partition_ids:
            raise ValueError(
                f"{manifest_path}: suite partition is not the complete ordered "
                f"{task_suite_partition!r} subset of its parent"
            )
        parent_by_id = {
            episode.episode_id: episode.to_dict()
            for episode in parent_manifest.episodes
        }
        for index, raw_episode in enumerate(raw_partition_episodes):
            expected = parent_by_id[str(raw_episode["episode_id"])]
            # Child BDDL paths can remain parent-relative strings, whereas the
            # loaded parent stores absolute paths. Normalize only that field.
            bddl_value = raw_episode.get("bddl_path", raw_episode.get("bddl"))
            bddl = Path(str(bddl_value)).expanduser()
            if not bddl.is_absolute():
                bddl = root / bddl
            candidate = {
                "episode_id": str(raw_episode.get("episode_id", "")),
                "base_task": str(raw_episode.get("base_task", "")),
                "task_suite_name": str(raw_episode.get("task_suite_name", "")),
                "task_id": int(raw_episode.get("task_id", -1)),
                "factor": str(raw_episode.get("factor", "")),
                "level": str(raw_episode.get("level", "")),
                "bddl_path": str(bddl.resolve()),
                "bddl_sha256": str(raw_episode.get("bddl_sha256", "")),
                "reset_state_id": int(raw_episode.get("reset_state_id", -1)),
                "trial_id": int(raw_episode.get("trial_id", -1)),
                "env_seed": int(raw_episode.get("env_seed", -1)),
                "perturbation_id": str(raw_episode.get("perturbation_id", "")),
                "asset_ids": tuple(raw_episode.get("asset_ids", ())),
            }
            if candidate != expected:
                raise ValueError(
                    f"{manifest_path}:episodes[{index}] differs from its logical parent"
                )
        logical_sha256 = parent_manifest.sha256
        parent_manifest_path = str(parent_path)

    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{manifest_path}: `episodes` must be a non-empty list")
    split = _require_string(payload, "split", source=str(manifest_path)).lower()
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            f"{manifest_path}: split must be train/validation/test, got {split!r}"
        )

    episodes: list[FrozenEpisode] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_episodes):
        source = f"{manifest_path}:episodes[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{source} must be an object")
        episode_id = _require_string(raw, "episode_id", source=source)
        if episode_id in seen_ids:
            raise ValueError(f"{source}: duplicate episode_id {episode_id!r}")
        seen_ids.add(episode_id)
        factor = _require_string(raw, "factor", source=source)
        base_task = _require_string(raw, "base_task", source=source)
        level = str(raw.get("level", "")).strip()
        perturbation_id = _require_string(raw, "perturbation_id", source=source)
        if not level:
            raise ValueError(f"{source}: `level` must be explicit")
        if any(value.lower() == "all" for value in (factor, level, perturbation_id)):
            raise ValueError(
                f"{source}: dynamic `all` perturbations are forbidden in frozen manifests"
            )
        raw_asset_ids = raw.get("asset_ids")
        if not isinstance(raw_asset_ids, list) or not raw_asset_ids or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_asset_ids
        ):
            raise ValueError(
                f"{source}: asset_ids must be an explicit non-empty string list"
            )
        asset_ids = tuple(value.strip() for value in raw_asset_ids)
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError(f"{source}: asset_ids contains duplicates")

        bddl_value = raw.get("bddl_path", raw.get("bddl"))
        if not isinstance(bddl_value, str) or not bddl_value.strip():
            raise ValueError(f"{source}: `bddl_path` must be a non-empty string")
        bddl = Path(bddl_value).expanduser()
        if not bddl.is_absolute():
            bddl = root / bddl
        bddl = bddl.resolve()
        try:
            bddl.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{source}: BDDL path escapes LIBERO_PLUS_ROOT: {bddl}") from exc
        if not bddl.is_file():
            raise ValueError(f"{source}: BDDL file does not exist: {bddl}")
        expected_sha = _require_string(raw, "bddl_sha256", source=source).lower()
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError(f"{source}: bddl_sha256 must be a lowercase SHA256")
        actual_sha = sha256_file(bddl)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{source}: BDDL SHA256 mismatch: {actual_sha} != {expected_sha}"
            )
        episodes.append(
            FrozenEpisode(
                episode_id=episode_id,
                base_task=base_task,
                task_suite_name=_require_string(
                    raw, "task_suite_name", source=source
                ),
                task_id=_require_int(raw, "task_id", source=source),
                factor=factor,
                level=level,
                bddl_path=str(bddl),
                bddl_sha256=expected_sha,
                reset_state_id=_require_int(raw, "reset_state_id", source=source),
                trial_id=_require_int(raw, "trial_id", source=source),
                env_seed=_require_int(raw, "env_seed", source=source),
                perturbation_id=perturbation_id,
                asset_ids=asset_ids,
            )
        )

    suffix = os.environ.get("LIBERO_SUFFIX", os.environ.get("LIBERO_PERTURBATION"))
    if suffix and suffix.lower().replace(".bddl", "") == "all":
        raise ValueError(
            "LIBERO_SUFFIX/LIBERO_PERTURBATION=all is incompatible with an "
            "episode_manifest_path; unset it so only manifest BDDLs are used"
        )
    return FrozenEpisodeManifest(
        path=str(manifest_path),
        sha256=logical_sha256,
        file_sha256=source_sha256,
        libero_plus_root=str(root),
        libero_plus_commit=declared_commit,
        split=split,
        episodes=tuple(episodes),
        parent_manifest_path=parent_manifest_path,
        task_suite_partition=task_suite_partition,
    )


def validate_manifest_disjoint(
    train: FrozenEpisodeManifest,
    heldout: FrozenEpisodeManifest,
) -> dict[str, tuple[Any, ...]]:
    """Require strict train-or-validation/test separation along frozen identities."""
    if train.split not in {"train", "validation"} or heldout.split != "test":
        raise ValueError(
            "disjoint audit requires a train-or-validation primary manifest "
            f"and test heldout manifest, got {train.split}/{heldout.split}"
        )
    if train.libero_plus_commit != heldout.libero_plus_commit:
        raise ValueError("train/test manifests use different LIBERO-Plus commits")
    dimensions = {
        "env_seed": (
            {entry.env_seed for entry in train.episodes},
            {entry.env_seed for entry in heldout.episodes},
        ),
        # Reset indices are local to a benchmark task.  Treating the bare
        # integer as global would make every task's reset 0 collide.
        "reset_state": (
            {
                (entry.task_suite_name, entry.task_id, entry.reset_state_id)
                for entry in train.episodes
            },
            {
                (entry.task_suite_name, entry.task_id, entry.reset_state_id)
                for entry in heldout.episodes
            },
        ),
        "perturbation_id": (
            {entry.perturbation_id for entry in train.episodes},
            {entry.perturbation_id for entry in heldout.episodes},
        ),
        "asset_id": (
            {asset for entry in train.episodes for asset in entry.asset_ids},
            {asset for entry in heldout.episodes for asset in entry.asset_ids},
        ),
    }
    overlaps = {
        name: tuple(sorted(train_values & heldout_values, key=str))
        for name, (train_values, heldout_values) in dimensions.items()
        if train_values & heldout_values
    }
    if overlaps:
        raise ValueError(
            "held-out Plus primary/test manifests overlap in preregistered "
            f"dimensions: {overlaps}"
        )
    return {
        name: tuple()
        for name in dimensions
    }
