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

"""Compact standard-LIBERO artifact collection for FastWAM evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    EnvOutput,
    EvaluationRolloutControl,
    RolloutResult,
)
from rlinf.envs.action_contract import (
    FASTWAM_LIBERO_ACTION_STAGES,
    ActionExecutionTrace,
)
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.utils import get_env_attr
from rlinf.models.embodiment.wam_policy.evaluation import (
    EvaluationRoutingConfig,
    EvaluationRoutingMode,
)
from rlinf.runners.fastwam_decision_telemetry import (
    build_fastwam_decision_telemetry_record,
)

LEDGER_SCHEMA_V1 = "fastwam-libero-eval-ledger-v1"
LEDGER_SCHEMA_V2 = "fastwam-libero-eval-ledger-v2"
LEDGER_SCHEMAS = {LEDGER_SCHEMA_V1, LEDGER_SCHEMA_V2}
CHUNK_SCHEMA = "fastwam-libero-eval-chunk-v2"
EPISODE_SCHEMA = "fastwam-libero-eval-episode-v1"
COMPLETED_EPISODE_SCHEMA = "fastwam-libero-eval-completed-episode-v1"
PROGRESS_SCHEMA = "fastwam-libero-eval-progress-v1"
NOISE_SEED_MODES = {"stateless_per_chunk", "fixed_per_episode"}
CONTRACT_VIOLATION_OUTCOMES = {"raise", "fail_episode"}
_IDENTITY_FIELDS_V1 = (
    "task_suite",
    "task_id",
    "trial_id",
    "reset_state_id",
    "environment_seed",
    "action_noise_seed",
    "idm_video_noise_seed",
    "max_primitive_steps",
    "action_horizon",
)
_IDENTITY_FIELDS_V2 = (
    "task_suite",
    "task_id",
    "trial_id",
    "reset_state_id",
    "environment_seed",
    "action_noise_seed",
    "idm_video_noise_seed",
    "max_primitive_steps",
    "generation_horizon",
    "execution_horizon",
    "prediction_video_frames",
    "reset_wait_steps",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in LEDGER_SCHEMAS:
        raise ValueError(f"Unsupported evaluation ledger schema in {path}.")
    if payload.get("kind") not in {"preflight", "validation", "final"}:
        raise ValueError(
            "Evaluation ledger kind must be preflight, validation, or final."
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evaluation ledger must contain at least one entry.")
    identities = set()
    identity_fields = (
        _IDENTITY_FIELDS_V1 if schema == LEDGER_SCHEMA_V1 else _IDENTITY_FIELDS_V2
    )
    reset_ids = set()
    for index, entry in enumerate(entries):
        missing = [
            field_name for field_name in identity_fields if field_name not in entry
        ]
        if missing:
            raise ValueError(f"Ledger entry {index} is missing {missing}.")
        identity_payload = {
            field_name: entry[field_name] for field_name in identity_fields
        }
        expected = _sha256_payload(identity_payload)
        if entry.get("episode_identity") != expected:
            raise ValueError(f"Ledger entry {index} episode identity mismatch.")
        if expected in identities:
            raise ValueError("Evaluation ledger contains duplicate episode identities.")
        reset_id = int(entry["reset_state_id"])
        if reset_id in reset_ids:
            raise ValueError("Evaluation ledger contains duplicate reset-state IDs.")
        identities.add(expected)
        if schema == LEDGER_SCHEMA_V2:
            generation = int(entry["generation_horizon"])
            execution = int(entry["execution_horizon"])
            prediction_frames = int(entry["prediction_video_frames"])
            wait_steps = int(entry["reset_wait_steps"])
            max_steps = int(entry["max_primitive_steps"])
            if min(generation, execution, prediction_frames, max_steps) < 1:
                raise ValueError(
                    "Evaluation ledger protocol horizons must be positive."
                )
            if wait_steps < 0:
                raise ValueError("Evaluation ledger reset wait must be non-negative.")
            if execution > generation or max_steps % execution:
                raise ValueError(
                    "Evaluation ledger generated/executed horizons are inconsistent."
                )
        reset_ids.add(reset_id)
    return payload


def _int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Environment {name} must be a one-dimensional sequence.")
    return [int(item) for item in value]


def _route_name(value: int) -> str:
    if int(value) == 1:
        return "idm"
    if int(value) == 0:
        return "uncond"
    raise ValueError(f"Invalid FastWAM route {value}.")


def _derive_chunk_seed(
    base_seed: int, episode_identity: str, chunk_id: int, kind: str
) -> int:
    payload = b"\0".join(
        (
            b"fastwam-libero-eval-noise-v1",
            str(int(base_seed)).encode("ascii"),
            episode_identity.encode("ascii"),
            str(int(chunk_id)).encode("ascii"),
            kind.encode("ascii"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeSlot:
    """Pre-step ledger identity for one environment slot."""

    stage_id: int
    local_env_index: int
    env_id: int
    chunk_id: int
    entry: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class EvaluationIdentityBatch:
    """Batch of identities captured before an auto-resetting environment step."""

    slots: tuple[EvaluationEpisodeSlot, ...]
    action_contract: LiberoActionContract

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Return which slots still own an unfinished frozen-ledger episode."""

        return tuple(slot.entry is not None for slot in self.slots)


@dataclass(frozen=True, slots=True)
class EvaluationArtifactShard:
    """Machine-readable output descriptor returned by one EnvWorker rank."""

    rank: int
    chunk_path: str
    episode_path: str
    action_contract_paths: tuple[str, ...]
    chunk_record_count: int
    episode_record_count: int
    chunk_sha256: str
    episode_sha256: str
    action_contract_file_sha256s: tuple[str, ...]
    action_contract_canonical_sha256s: tuple[str, ...]
    executable_action_contract_sha256: str
    canonical_content_sha256: str
    status: str = "PASS"


@dataclass(slots=True)
class _EpisodeState:
    entry: dict[str, Any]
    env_id: int
    stage_id: int
    local_env_index: int
    route_episode_id: int | None = None
    actor_version: int | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False


class FastWAMLiberoEvalCollector:
    """Collect outcome-aligned FastWAM routing records without raw model data."""

    def __init__(
        self,
        *,
        output_dir: str,
        ledger_path: str,
        run_id: str,
        rank: int,
        routing_mode: str,
        idm_threshold: float,
        random_idm_probability: float | None,
        random_lag1_autocorrelation: float | None = None,
        periodic_period: int | None = None,
        periodic_on_count: int | None = None,
        periodic_phase: int | None = None,
        routing_seed: int,
        fixed_idm_cost: float,
        decision_telemetry_enabled: bool = False,
        noise_seed_mode: str = "stateless_per_chunk",
        contract_violation_outcome: str = "raise",
        resume: bool = False,
        policy_checkpoint_sha256: str | None = None,
        evaluation_runtime_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_path).expanduser().resolve()
        self.ledger = _load_ledger(self.ledger_path)
        self.run_id = str(run_id)
        if not self.run_id:
            raise ValueError("Evaluation run_id must be non-empty.")
        self.rank = int(rank)
        routing_config = EvaluationRoutingConfig(
            mode=routing_mode,
            idm_threshold=idm_threshold,
            random_idm_probability=random_idm_probability,
            random_lag1_autocorrelation=random_lag1_autocorrelation,
            periodic_period=periodic_period,
            periodic_on_count=periodic_on_count,
            periodic_phase=periodic_phase,
            routing_seed=routing_seed,
        )
        self.routing_mode = routing_config.mode
        self.idm_threshold = routing_config.idm_threshold
        self.random_idm_probability = routing_config.random_idm_probability
        self.random_lag1_autocorrelation = routing_config.random_lag1_autocorrelation
        self.periodic_period = routing_config.periodic_period
        self.periodic_on_count = routing_config.periodic_on_count
        self.periodic_phase = routing_config.periodic_phase
        self.routing_seed = routing_config.routing_seed
        self.fixed_idm_cost = float(fixed_idm_cost)
        if not isinstance(decision_telemetry_enabled, bool):
            raise TypeError("decision_telemetry_enabled must be a boolean.")
        self.decision_telemetry_enabled = decision_telemetry_enabled
        self.noise_seed_mode = str(noise_seed_mode)
        self.contract_violation_outcome = str(contract_violation_outcome)
        self.resume = bool(resume)
        self.policy_checkpoint_sha256 = (
            None if policy_checkpoint_sha256 is None else str(policy_checkpoint_sha256)
        )
        if evaluation_runtime_identity is None:
            self.evaluation_runtime_identity = None
        elif not isinstance(evaluation_runtime_identity, Mapping):
            raise TypeError("Evaluation runtime identity must be a mapping.")
        else:
            # Canonical JSON round-tripping validates nested values and removes
            # OmegaConf container subclasses before persistence/comparison.
            identity_container = OmegaConf.to_container(
                OmegaConf.create(evaluation_runtime_identity),
                resolve=True,
                enum_to_str=True,
            )
            self.evaluation_runtime_identity = json.loads(
                _canonical_bytes(identity_container)
            )
        if self.noise_seed_mode not in NOISE_SEED_MODES:
            raise ValueError(
                f"Unsupported FastWAM evaluation noise seed mode {noise_seed_mode!r}."
            )
        if not math.isfinite(self.fixed_idm_cost) or self.fixed_idm_cost < 0:
            raise ValueError("fixed_idm_cost must be finite and non-negative.")
        if self.contract_violation_outcome not in CONTRACT_VIOLATION_OUTCOMES:
            raise ValueError(
                "contract_violation_outcome must be 'raise' or 'fail_episode'."
            )
        if self.policy_checkpoint_sha256 is not None and not _valid_sha256(
            self.policy_checkpoint_sha256
        ):
            raise ValueError("policy_checkpoint_sha256 must be a lowercase SHA256.")
        if self.resume and self.policy_checkpoint_sha256 is None:
            raise ValueError("Resumable evaluation requires policy_checkpoint_sha256.")
        if self.resume and self.evaluation_runtime_identity is None:
            raise ValueError(
                "Resumable evaluation requires evaluation_runtime_identity."
            )
        if self.resume and self.routing_mode in {
            EvaluationRoutingMode.MATCHED_RANDOM,
            EvaluationRoutingMode.AUTOCORRELATION_MATCHED_RANDOM,
        }:
            raise ValueError(
                "Resuming random-routing evaluation is not supported because a "
                "fresh rollout worker restarts its route episode ids and would "
                "therefore change the preregistered random draws."
            )
        self._entries_by_reset = {
            int(entry["reset_state_id"]): entry for entry in self.ledger["entries"]
        }
        self._states: dict[tuple[int, int], _EpisodeState] = {}
        self._chunks: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._action_contracts: dict[str, LiberoActionContract] = {}
        self._executable_action_contract_sha256: str | None = None
        self._chunk_path = self.output_dir / f"chunks.rank-{self.rank}.jsonl"
        self._episode_path = self.output_dir / f"episodes.rank-{self.rank}.jsonl"
        self._progress_path = self.output_dir / f"progress.rank-{self.rank}.json"
        self._completion_dir = self.output_dir / f"completed.rank-{self.rank}"
        self._live_contract_dir = self.output_dir / f"contracts.rank-{self.rank}"
        self._ledger_order = {
            str(entry["episode_identity"]): index
            for index, entry in enumerate(self.ledger["entries"])
        }
        self._progress_identity = self._build_progress_identity()
        self._initialize_progress()

    @property
    def continues_after_contract_violation(self) -> bool:
        """Return whether rejected Actions are recorded as failed episodes."""

        return self.contract_violation_outcome == "fail_episode"

    @property
    def pending_reset_state_ids(self) -> tuple[int, ...]:
        """Return ledger reset identities not restored as completed episodes."""

        completed = {str(item["episode_identity"]) for item in self._episodes}
        return tuple(
            int(entry["reset_state_id"])
            for entry in self.ledger["entries"]
            if str(entry["episode_identity"]) not in completed
        )

    def _build_progress_identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rank": self.rank,
            "ledger_sha256": _sha256_file(self.ledger_path),
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "evaluation_runtime_identity": self.evaluation_runtime_identity,
            "routing_mode": self.routing_mode.value,
            "idm_threshold": self.idm_threshold,
            "random_idm_probability": self.random_idm_probability,
            "random_lag1_autocorrelation": self.random_lag1_autocorrelation,
            "periodic_period": self.periodic_period,
            "periodic_on_count": self.periodic_on_count,
            "periodic_phase": self.periodic_phase,
            "routing_seed": self.routing_seed,
            "fixed_idm_cost": self.fixed_idm_cost,
            "noise_seed_mode": self.noise_seed_mode,
            "contract_violation_outcome": self.contract_violation_outcome,
        }

    def _initialize_progress(self) -> None:
        if self.resume:
            if not self._progress_path.is_file():
                raise FileNotFoundError(
                    "Evaluation resume requested without a progress manifest: "
                    f"{self._progress_path}."
                )
            payload = json.loads(self._progress_path.read_text(encoding="utf-8"))
            if payload.get("schema") != PROGRESS_SCHEMA:
                raise ValueError("Unsupported evaluation progress schema.")
            persisted_identity = payload.get("identity")
            if persisted_identity != self._progress_identity:
                persisted_mapping = (
                    persisted_identity if isinstance(persisted_identity, dict) else {}
                )
                mismatched_fields = sorted(
                    key
                    for key in set(persisted_mapping) | set(self._progress_identity)
                    if persisted_mapping.get(key) != self._progress_identity.get(key)
                )
                raise ValueError(
                    "Evaluation resume identity differs from the persisted run; "
                    f"mismatched fields: {mismatched_fields}."
                )
            self._restore_completion_units()
            self._restore_live_action_contracts()
            return

        existing_outputs = [
            path for path in (self._chunk_path, self._episode_path) if path.exists()
        ]
        existing_outputs.extend(
            self.output_dir.glob(f"action_contract.rank-{self.rank}*")
        )
        if (
            self._progress_path.exists()
            or self._completion_dir.exists()
            or existing_outputs
        ):
            raise FileExistsError(
                "Evaluation output already contains resumable progress; use a new "
                "output directory or set resume=true with matching provenance."
            )
        self._completion_dir.mkdir(parents=False)
        self._live_contract_dir.mkdir(parents=False)
        self._write_json_atomic(
            self._progress_path,
            {"schema": PROGRESS_SCHEMA, "identity": self._progress_identity},
        )

    def _restore_completion_units(self) -> None:
        if not self._completion_dir.is_dir():
            raise FileNotFoundError(
                "Evaluation progress is missing its completed-episode directory."
            )
        units: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
        seen_identities: set[str] = set()
        seen_record_ids: set[str] = set()
        for path in sorted(self._completion_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != COMPLETED_EPISODE_SCHEMA:
                raise ValueError(f"Unsupported completion unit schema in {path}.")
            if payload.get("progress_identity_sha256") != _sha256_payload(
                self._progress_identity
            ):
                raise ValueError(f"Completion unit provenance mismatch in {path}.")
            episode = payload.get("episode")
            chunks = payload.get("chunks")
            if (
                not isinstance(episode, dict)
                or not isinstance(chunks, list)
                or not chunks
            ):
                raise ValueError(f"Completion unit is incomplete in {path}.")
            identity = str(episode.get("episode_identity", ""))
            ledger_index = self._ledger_order.get(identity)
            if ledger_index is None or identity in seen_identities:
                raise ValueError(
                    f"Completion unit episode identity is invalid in {path}."
                )
            if (
                episode.get("run_id") != self.run_id
                or int(episode.get("rank", -1)) != self.rank
            ):
                raise ValueError(f"Completion unit run identity mismatch in {path}.")
            record_ids = [str(chunk.get("record_id", "")) for chunk in chunks]
            if record_ids != episode.get("chunk_record_ids"):
                raise ValueError(f"Completion unit chunk list mismatch in {path}.")
            if any(
                chunk.get("run_id") != self.run_id
                or int(chunk.get("rank", -1)) != self.rank
                or chunk.get("episode_identity") != identity
                for chunk in chunks
            ):
                raise ValueError(f"Completion unit chunk identity mismatch in {path}.")
            if any(record_id in seen_record_ids for record_id in record_ids):
                raise ValueError(f"Duplicate chunk record identity in {path}.")
            _canonical_bytes(payload)
            seen_identities.add(identity)
            seen_record_ids.update(record_ids)
            units.append((ledger_index, episode, chunks))

        for _index, episode, chunks in sorted(units):
            self._episodes.append(episode)
            self._chunks.extend(chunks)

    def _restore_live_action_contracts(self) -> None:
        if not self._live_contract_dir.is_dir():
            raise FileNotFoundError(
                "Evaluation progress is missing its live-contract directory."
            )
        for path in sorted(self._live_contract_dir.glob("*.json")):
            contract = LiberoActionContract.from_artifact(
                json.loads(path.read_text(encoding="utf-8"))
            )
            self._action_contracts[contract.canonical_sha256] = contract
            executable_sha256 = contract.executable_spec_sha256
            if self._executable_action_contract_sha256 is None:
                self._executable_action_contract_sha256 = executable_sha256
            elif self._executable_action_contract_sha256 != executable_sha256:
                raise ValueError(
                    "Persisted LIBERO executable Action contracts disagree."
                )

    def snapshot_before_step(
        self,
        stage_id: int,
        env: Any,
        env_ids: torch.Tensor,
    ) -> EvaluationIdentityBatch:
        """Capture task/trial/reset identity before LiberoEnv may auto-reset."""

        action_contract = get_env_attr(env, "action_contract")
        if not isinstance(action_contract, LiberoActionContract):
            raise TypeError(
                "Evaluation environment must expose a typed live Action contract."
            )
        executable_sha256 = action_contract.executable_spec_sha256
        if self._executable_action_contract_sha256 is None:
            self._executable_action_contract_sha256 = executable_sha256
        elif self._executable_action_contract_sha256 != executable_sha256:
            raise ValueError(
                "Live LIBERO executable Action contract changed within one run."
            )
        if action_contract.canonical_sha256 not in self._action_contracts:
            self._action_contracts[action_contract.canonical_sha256] = action_contract
            self._persist_live_action_contract(action_contract)
        task_ids = _int_list(get_env_attr(env, "task_ids"), name="task_ids")
        trial_ids = _int_list(get_env_attr(env, "trial_ids"), name="trial_ids")
        reset_ids = _int_list(
            get_env_attr(env, "reset_state_ids"),
            name="reset_state_ids",
        )
        environment_ids = _int_list(env_ids, name="env_ids")
        batch_size = len(environment_ids)
        if not (len(task_ids) == len(trial_ids) == len(reset_ids) == batch_size):
            raise ValueError(
                "Environment identity fields have inconsistent batch sizes."
            )

        slots = []
        completed_identities = {
            str(episode["episode_identity"]) for episode in self._episodes
        }
        for local_index, (env_id, task_id, trial_id, reset_id) in enumerate(
            zip(environment_ids, task_ids, trial_ids, reset_ids)
        ):
            key = (int(stage_id), local_index)
            state = self._states.get(key)
            if state is None or int(state.entry["reset_state_id"]) != reset_id:
                if state is not None and not state.completed:
                    raise RuntimeError(
                        "Environment reset before its prior episode terminated."
                    )
                entry = self._entries_by_reset.get(reset_id)
                if entry is None:
                    raise ValueError(
                        f"Reset-state id {reset_id} is absent from the frozen ledger."
                    )
                if (
                    int(entry["task_id"]) != task_id
                    or int(entry["trial_id"]) != trial_id
                ):
                    raise ValueError(
                        "Actual LIBERO task/trial/reset ledger identity mismatch: "
                        f"actual=({task_id}, {trial_id}, {reset_id}), "
                        f"ledger=({entry['task_id']}, {entry['trial_id']}, "
                        f"{entry['reset_state_id']})."
                    )
                if str(entry["episode_identity"]) in completed_identities:
                    raise ValueError(
                        "Evaluation environment replayed an episode already restored "
                        "from durable progress."
                    )
                state = _EpisodeState(
                    entry=entry,
                    env_id=env_id,
                    stage_id=int(stage_id),
                    local_env_index=local_index,
                )
                self._states[key] = state
            elif state.env_id != env_id:
                raise ValueError(
                    "Stable FastWAM env id changed within one worker slot."
                )

            slots.append(
                EvaluationEpisodeSlot(
                    stage_id=int(stage_id),
                    local_env_index=local_index,
                    env_id=env_id,
                    chunk_id=len(state.chunks),
                    entry=None if state.completed else state.entry,
                )
            )
        return EvaluationIdentityBatch(
            slots=tuple(slots),
            action_contract=action_contract,
        )

    def _persist_live_action_contract(self, contract: LiberoActionContract) -> None:
        path = self._live_contract_dir / f"{contract.canonical_sha256}.json"
        if path.exists():
            restored = LiberoActionContract.from_artifact(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if restored != contract:
                raise ValueError("Persisted live Action contract changed in place.")
            return
        self._write_json_atomic(path, contract.to_artifact())

    def augment_rollout_input(
        self,
        data: dict[str, Any],
        snapshot: EvaluationIdentityBatch,
    ) -> dict[str, Any]:
        """Attach compact per-episode/chunk seeds; never attach generated tensors."""

        result = dict(data)
        result["obs"] = dict(data["obs"])
        action_seeds = []
        idm_seeds = []
        for slot in snapshot.slots:
            entry = slot.entry
            if entry is None:
                state = self._states[(slot.stage_id, slot.local_env_index)]
                entry = state.entry
            if self.noise_seed_mode == "fixed_per_episode":
                action_seeds.append(int(entry["action_noise_seed"]))
                idm_seeds.append(int(entry["idm_video_noise_seed"]))
            else:
                identity = str(entry["episode_identity"])
                action_seeds.append(
                    _derive_chunk_seed(
                        int(entry["action_noise_seed"]),
                        identity,
                        slot.chunk_id,
                        "action",
                    )
                )
                idm_seeds.append(
                    _derive_chunk_seed(
                        int(entry["idm_video_noise_seed"]),
                        identity,
                        slot.chunk_id,
                        "idm_video",
                    )
                )
        result["obs"]["_fastwam_action_noise_seeds"] = torch.tensor(
            action_seeds,
            dtype=torch.long,
        )
        result["obs"]["_fastwam_idm_noise_seeds"] = torch.tensor(
            idm_seeds,
            dtype=torch.long,
        )
        batch_size = len(snapshot.slots)
        contract = snapshot.action_contract
        result["obs"]["_fastwam_action_contract_low"] = (
            torch.tensor(contract.low, dtype=torch.float32)
            .expand(batch_size, -1)
            .clone()
        )
        result["obs"]["_fastwam_action_contract_high"] = (
            torch.tensor(contract.high, dtype=torch.float32)
            .expand(batch_size, -1)
            .clone()
        )
        result["obs"]["_fastwam_action_gripper_indices"] = torch.full(
            (batch_size,), contract.gripper_dimension_index, dtype=torch.long
        )
        result["obs"]["_fastwam_action_contract_sha256"] = [
            contract.canonical_sha256
        ] * batch_size
        return result

    @property
    def is_complete(self) -> bool:
        """Return whether every frozen ledger episode has terminated once."""

        expected = {entry["episode_identity"] for entry in self.ledger["entries"]}
        completed = {episode["episode_identity"] for episode in self._episodes}
        return completed == expected

    def build_rollout_stop_control(
        self, *, logical_batch_size: int
    ) -> EvaluationRolloutControl:
        """Build a provenance-bound stop only after the ledger is complete."""

        if not self.is_complete:
            raise RuntimeError(
                "Cannot stop evaluation before the frozen ledger is complete."
            )
        episode_count = len(self.ledger["entries"])
        return EvaluationRolloutControl(
            logical_batch_size=int(logical_batch_size),
            completed_episode_count=len(self._episodes),
            ledger_episode_count=episode_count,
            ledger_sha256=_sha256_file(self.ledger_path),
        )

    @staticmethod
    def _batch_outcome(value: torch.Tensor | None, index: int) -> torch.Tensor:
        if value is None:
            return torch.empty(0)
        return torch.as_tensor(value[index]).reshape(-1)

    def record_chunk(
        self,
        *,
        snapshot: EvaluationIdentityBatch,
        rollout_result: RolloutResult,
        env_output: EnvOutput,
        policy_latency_seconds: float | None = None,
        environment_latency_seconds: float,
    ) -> None:
        """Record one executed chunk batch after reward/done alignment."""

        route = rollout_result.route_info
        emitted = rollout_result.emitted_gate
        selection = rollout_result.evaluation_selection
        if route is None or emitted is None or selection is None:
            raise ValueError(
                "FastWAM evaluation collector requires all typed route records."
            )
        batch_size = len(snapshot.slots)
        if route.shape != torch.Size([batch_size]) or emitted.shape != route.shape:
            raise ValueError(
                "Evaluation route records do not match identity batch size."
            )
        if selection.effective_next_route.shape != route.shape:
            raise ValueError("Evaluation selection does not match route batch size.")
        gate_latency = rollout_result.gate_latency_seconds
        gate_h2d = rollout_result.gate_h2d_seconds
        if (gate_latency is None) != (gate_h2d is None):
            raise ValueError("Gate latency and H2D timing must be provided together.")
        for name, timing in (
            ("Gate latency", gate_latency),
            ("Gate H2D latency", gate_h2d),
        ):
            if timing is None:
                continue
            if timing.shape != torch.Size([batch_size]):
                raise ValueError(f"{name} must have shape [B].")
            if not torch.isfinite(timing).all() or (timing < 0).any():
                raise ValueError(f"{name} must be finite and non-negative.")
        if env_output.dones is None or int(env_output.dones.shape[0]) != batch_size:
            raise ValueError("Evaluation outcomes do not match route batch size.")
        if not math.isfinite(float(environment_latency_seconds)):
            raise ValueError("Environment latency must be finite.")
        if policy_latency_seconds is not None and not math.isfinite(
            float(policy_latency_seconds)
        ):
            raise ValueError("Policy latency must be finite when recorded.")

        rollout_action_trace = rollout_result.action_execution_trace
        environment_action_trace = env_output.action_execution_trace
        if rollout_action_trace is None or environment_action_trace is None:
            raise ValueError(
                "Schema-v2 evaluation requires model and environment Action traces."
            )
        action_trace = ActionExecutionTrace.combine(
            rollout_action_trace,
            environment_action_trace,
        )
        if action_trace.stage_names != FASTWAM_LIBERO_ACTION_STAGES:
            raise ValueError(
                "FastWAM Action trace stages differ from the canonical pipeline."
            )
        if (
            action_trace.batch_size != batch_size
            or action_trace.action_contract_sha256
            != snapshot.action_contract.canonical_sha256
        ):
            raise ValueError("FastWAM Action trace does not match its live contract.")
        if any(slot.entry is not None for slot in snapshot.slots):
            self._action_contracts.setdefault(
                snapshot.action_contract.canonical_sha256,
                snapshot.action_contract,
            )

        for index, slot in enumerate(snapshot.slots):
            if slot.entry is None:
                continue
            state = self._states[(slot.stage_id, slot.local_env_index)]
            chunk_id = int(route.chunk_ids[index])
            if chunk_id != len(state.chunks) or chunk_id != slot.chunk_id:
                raise ValueError("Collector and policy chunk identities diverged.")
            route_episode_id = int(route.episode_ids[index])
            actor_version = int(route.actor_versions[index])
            if state.route_episode_id is None:
                state.route_episode_id = route_episode_id
                state.actor_version = actor_version
            elif (
                route_episode_id != state.route_episode_id
                or actor_version != state.actor_version
            ):
                raise ValueError(
                    "Episode id or actor version changed within an episode."
                )

            terminal = bool(torch.as_tensor(env_output.dones[index]).any().item())
            terminations = self._batch_outcome(env_output.terminations, index)
            truncations = self._batch_outcome(env_output.truncations, index)
            success = bool(terminations.any().item()) if terminations.numel() else False
            truncated = bool(truncations.any().item()) if truncations.numel() else False
            if terminal != (success or truncated):
                raise ValueError("Done, termination, and truncation outcomes disagree.")
            valid = bool(emitted.valid[index])
            if valid == terminal:
                raise ValueError(
                    "Terminal Gate emissions must be discarded and nonterminal "
                    "emissions must remain eligible."
                )
            probability = float(emitted.base_probability[index])
            if not math.isfinite(probability):
                raise ValueError("Gate probability must be finite.")
            random_draw = (
                None
                if selection.random_draws is None
                else float(selection.random_draws[index])
            )
            rewards = self._batch_outcome(env_output.rewards, index)
            reward = float(rewards.sum().item()) if rewards.numel() else 0.0
            actions = (
                rollout_result.actions[index]
                if rollout_result.actions is not None
                else torch.empty(0)
            )
            action_min = float(actions.min().item()) if actions.numel() else None
            action_max = float(actions.max().item()) if actions.numel() else None
            entry = state.entry
            identity = str(entry["episode_identity"])
            decision_telemetry = None
            if self.decision_telemetry_enabled:
                if (
                    emitted.exploration_forced is None
                    or emitted.mode_flip_delta is None
                ):
                    raise ValueError(
                        "FastWAM evaluation is missing per-decision Gate telemetry."
                    )
                decision_telemetry = build_fastwam_decision_telemetry_record(
                    phase="evaluation",
                    run_id=self.run_id,
                    rank=self.rank,
                    trajectory_id=identity,
                    env_id=int(slot.env_id),
                    episode_id=route_episode_id,
                    task_suite=str(entry["task_suite"]),
                    task_id=int(entry["task_id"]),
                    trial_id=int(entry["trial_id"]),
                    reset_state_id=int(entry["reset_state_id"]),
                    cycle_index=chunk_id,
                    update_step=actor_version,
                    actor_version=actor_version,
                    route=int(emitted.next_route[index]),
                    base_probability=probability,
                    behavior_probability=float(emitted.behavior_probability[index]),
                    forced_exploration=bool(emitted.exploration_forced[index]),
                    mode_flip_delta=float(emitted.mode_flip_delta[index]),
                    configured_idm_cost=None,
                    destination_advantage_unnormalized=None,
                    destination_advantage_normalized=None,
                    eligible_decision=valid,
                )
            record = {
                "schema": CHUNK_SCHEMA,
                "run_id": self.run_id,
                "rank": self.rank,
                "env_id": slot.env_id,
                "episode_id": route_episode_id,
                "episode_identity": identity,
                "task_suite": entry["task_suite"],
                "task_id": int(entry["task_id"]),
                "trial_id": int(entry["trial_id"]),
                "reset_state_id": int(entry["reset_state_id"]),
                "record_id": f"{identity}:{chunk_id}",
                "chunk_id": chunk_id,
                "route": _route_name(int(route.route_used[index])),
                "route_was_forced": bool(route.route_was_forced[index]),
                "route_source_chunk_id": int(route.route_source_chunk_ids[index]),
                "actor_version": actor_version,
                "gate_idm_probability": probability,
                "gate_epsilon": float(emitted.epsilon[index]),
                "gate_temperature": float(emitted.temperature[index]),
                "counterfactual_next_route": _route_name(
                    int(selection.counterfactual_next_route[index])
                ),
                "effective_next_route": _route_name(
                    int(selection.effective_next_route[index])
                ),
                "routing_mode": selection.mode.value,
                "random_draw": random_draw,
                "emitted_decision_consumed": valid,
                "emitted_decision_discarded": not valid,
                "eligible_decision": valid,
                **(
                    {"decision_telemetry": decision_telemetry}
                    if self.decision_telemetry_enabled
                    else {}
                ),
                "primitive_steps_executed": int(
                    entry.get("execution_horizon", entry.get("action_horizon", -1))
                ),
                "action_submission_status": "submitted",
                "action_contract_violation": None,
                "reward": reward,
                "success_observed": success,
                "terminal": terminal,
                "termination_type": (
                    "success" if success else "truncation" if truncated else None
                ),
                "policy_latency_seconds": (
                    None
                    if policy_latency_seconds is None
                    else float(policy_latency_seconds)
                ),
                "gate_latency_seconds": (
                    None if gate_latency is None else float(gate_latency[index])
                ),
                "gate_h2d_seconds": (
                    0.0 if gate_h2d is None else float(gate_h2d[index])
                ),
                "environment_latency_seconds": float(environment_latency_seconds),
                "action_min": action_min,
                "action_max": action_max,
                "action_contract_sha256": (snapshot.action_contract.canonical_sha256),
                "action_trace": action_trace.record_for_batch_index(index),
            }
            _canonical_bytes(record)
            state.chunks.append(record)
            self._chunks.append(record)
            if terminal:
                self._finish_episode(state, success=success, truncated=truncated)
        self._sync_live_jsonl()

    def record_contract_violation(
        self,
        *,
        snapshot: EvaluationIdentityBatch,
        rollout_result: RolloutResult,
        action_trace: ActionExecutionTrace,
        failure_audit: dict[str, Any],
        rejected_mask: tuple[bool, ...],
        policy_latency_seconds: float | None,
        environment_latency_seconds: float,
    ) -> None:
        """Persist rejected Action chunks as explicit failed episodes."""

        if rejected_mask != snapshot.active_mask:
            raise ValueError(
                "Contract-violation continuation currently requires every active "
                "evaluation slot to be rejected together."
            )
        route = rollout_result.route_info
        emitted = rollout_result.emitted_gate
        selection = rollout_result.evaluation_selection
        if route is None or emitted is None or selection is None:
            raise ValueError(
                "Contract-violation records require all typed route metadata."
            )
        batch_size = len(snapshot.slots)
        if action_trace.batch_size != batch_size:
            raise ValueError("Rejected Action trace batch size is inconsistent.")
        if action_trace.stage_names != FASTWAM_LIBERO_ACTION_STAGES[:-1]:
            raise ValueError(
                "Rejected Action trace must stop immediately before env submission."
            )
        if (
            action_trace.action_contract_sha256
            != snapshot.action_contract.canonical_sha256
        ):
            raise ValueError("Rejected Action trace uses a different live contract.")
        if route.shape != torch.Size([batch_size]) or emitted.shape != route.shape:
            raise ValueError("Rejected Action route metadata has the wrong shape.")
        if selection.effective_next_route.shape != route.shape:
            raise ValueError("Rejected Action selection has the wrong shape.")
        if not math.isfinite(float(environment_latency_seconds)):
            raise ValueError("Environment latency must be finite.")
        if policy_latency_seconds is not None and not math.isfinite(
            float(policy_latency_seconds)
        ):
            raise ValueError("Policy latency must be finite when recorded.")
        audit_violations = failure_audit.get("violations")
        if not isinstance(audit_violations, list) or not audit_violations:
            raise ValueError("Rejected Action audit contains no violations.")
        _canonical_bytes(failure_audit)

        for index, slot in enumerate(snapshot.slots):
            if slot.entry is None:
                continue
            state = self._states[(slot.stage_id, slot.local_env_index)]
            chunk_id = int(route.chunk_ids[index])
            if chunk_id != len(state.chunks) or chunk_id != slot.chunk_id:
                raise ValueError("Collector and rejected policy chunk diverged.")
            route_episode_id = int(route.episode_ids[index])
            actor_version = int(route.actor_versions[index])
            if state.route_episode_id is None:
                state.route_episode_id = route_episode_id
                state.actor_version = actor_version
            elif (
                route_episode_id != state.route_episode_id
                or actor_version != state.actor_version
            ):
                raise ValueError(
                    "Episode id or actor version changed within a rejected episode."
                )
            if bool(emitted.valid[index]):
                raise ValueError(
                    "Rejected terminal chunks must discard their emitted Gate decision."
                )
            probability = float(emitted.base_probability[index])
            if not math.isfinite(probability):
                raise ValueError("Gate probability must be finite.")
            random_draw = (
                None
                if selection.random_draws is None
                else float(selection.random_draws[index])
            )
            entry = state.entry
            identity = str(entry["episode_identity"])
            decision_telemetry = None
            if self.decision_telemetry_enabled:
                if (
                    emitted.exploration_forced is None
                    or emitted.mode_flip_delta is None
                ):
                    raise ValueError(
                        "FastWAM evaluation is missing per-decision Gate telemetry."
                    )
                decision_telemetry = build_fastwam_decision_telemetry_record(
                    phase="evaluation",
                    run_id=self.run_id,
                    rank=self.rank,
                    trajectory_id=identity,
                    env_id=int(slot.env_id),
                    episode_id=route_episode_id,
                    task_suite=str(entry["task_suite"]),
                    task_id=int(entry["task_id"]),
                    trial_id=int(entry["trial_id"]),
                    reset_state_id=int(entry["reset_state_id"]),
                    cycle_index=chunk_id,
                    update_step=actor_version,
                    actor_version=actor_version,
                    route=int(emitted.next_route[index]),
                    base_probability=probability,
                    behavior_probability=float(emitted.behavior_probability[index]),
                    forced_exploration=bool(emitted.exploration_forced[index]),
                    mode_flip_delta=float(emitted.mode_flip_delta[index]),
                    configured_idm_cost=None,
                    destination_advantage_unnormalized=None,
                    destination_advantage_normalized=None,
                    eligible_decision=False,
                )
            per_environment_audit = {
                **failure_audit,
                "violations": [
                    violation
                    for violation in audit_violations
                    if int(violation["environment_index"]) == index
                ],
            }
            if not per_environment_audit["violations"]:
                raise ValueError(
                    "Rejected evaluation slot has no matching audit violation."
                )
            final_stage = action_trace.stages[-1]
            action_min = float(final_stage.minimum[index].min().item())
            action_max = float(final_stage.maximum[index].max().item())
            record = {
                "schema": CHUNK_SCHEMA,
                "run_id": self.run_id,
                "rank": self.rank,
                "env_id": slot.env_id,
                "episode_id": route_episode_id,
                "episode_identity": identity,
                "task_suite": entry["task_suite"],
                "task_id": int(entry["task_id"]),
                "trial_id": int(entry["trial_id"]),
                "reset_state_id": int(entry["reset_state_id"]),
                "record_id": f"{identity}:{chunk_id}",
                "chunk_id": chunk_id,
                "route": _route_name(int(route.route_used[index])),
                "route_was_forced": bool(route.route_was_forced[index]),
                "route_source_chunk_id": int(route.route_source_chunk_ids[index]),
                "actor_version": actor_version,
                "gate_idm_probability": probability,
                "gate_epsilon": float(emitted.epsilon[index]),
                "gate_temperature": float(emitted.temperature[index]),
                "counterfactual_next_route": _route_name(
                    int(selection.counterfactual_next_route[index])
                ),
                "effective_next_route": _route_name(
                    int(selection.effective_next_route[index])
                ),
                "routing_mode": selection.mode.value,
                "random_draw": random_draw,
                "emitted_decision_consumed": False,
                "emitted_decision_discarded": True,
                "eligible_decision": False,
                **(
                    {"decision_telemetry": decision_telemetry}
                    if self.decision_telemetry_enabled
                    else {}
                ),
                "primitive_steps_executed": 0,
                "action_submission_status": "rejected",
                "action_contract_violation": per_environment_audit,
                "reward": 0.0,
                "success_observed": False,
                "terminal": True,
                "termination_type": "contract_violation",
                "policy_latency_seconds": (
                    None
                    if policy_latency_seconds is None
                    else float(policy_latency_seconds)
                ),
                "gate_latency_seconds": (
                    None
                    if rollout_result.gate_latency_seconds is None
                    else float(rollout_result.gate_latency_seconds[index])
                ),
                "gate_h2d_seconds": (
                    0.0
                    if rollout_result.gate_h2d_seconds is None
                    else float(rollout_result.gate_h2d_seconds[index])
                ),
                "environment_latency_seconds": float(environment_latency_seconds),
                "action_min": action_min,
                "action_max": action_max,
                "action_contract_sha256": (snapshot.action_contract.canonical_sha256),
                "action_trace": action_trace.record_for_batch_index(index),
            }
            _canonical_bytes(record)
            state.chunks.append(record)
            self._chunks.append(record)
            self._finish_episode(
                state,
                success=False,
                truncated=False,
                termination_type="contract_violation",
            )
        self._sync_live_jsonl()

    def _finish_episode(
        self,
        state: _EpisodeState,
        *,
        success: bool,
        truncated: bool,
        termination_type: str | None = None,
    ) -> None:
        chunks = state.chunks
        if not chunks or not chunks[-1]["terminal"]:
            raise ValueError("Cannot finalize an episode without a terminal chunk.")
        expected_termination = (
            "success"
            if success
            else "truncation"
            if truncated
            else "contract_violation"
        )
        termination_type = (
            expected_termination if termination_type is None else str(termination_type)
        )
        if termination_type != expected_termination:
            raise ValueError("Episode outcome and termination type disagree.")
        executed = len(chunks)
        idm_total = sum(chunk["route"] == "idm" for chunk in chunks)
        forced_initial = sum(
            chunk["route_was_forced"] and chunk["chunk_id"] == 0 for chunk in chunks
        )
        eligible_chunks = [chunk for chunk in chunks if not chunk["route_was_forced"]]
        eligible_idm = sum(chunk["route"] == "idm" for chunk in eligible_chunks)
        episode = {
            "schema": EPISODE_SCHEMA,
            "run_id": self.run_id,
            "rank": self.rank,
            "env_id": state.env_id,
            "episode_id": state.route_episode_id,
            "episode_identity": state.entry["episode_identity"],
            "task_suite": state.entry["task_suite"],
            "task_id": int(state.entry["task_id"]),
            "trial_id": int(state.entry["trial_id"]),
            "reset_state_id": int(state.entry["reset_state_id"]),
            "success": bool(success),
            "return": float(sum(chunk["reward"] for chunk in chunks)),
            "primitive_episode_length": int(
                sum(chunk["primitive_steps_executed"] for chunk in chunks)
            ),
            "chunk_episode_length": executed,
            "executed_chunk_count": executed,
            "idm_chunk_count_total": idm_total,
            "forced_initial_idm_count": forced_initial,
            "eligible_chunk_count": len(eligible_chunks),
            "eligible_idm_count": eligible_idm,
            "idm_fraction_total": idm_total / executed,
            "eligible_idm_fraction": (
                eligible_idm / len(eligible_chunks) if eligible_chunks else 0.0
            ),
            "normalized_prediction_compute": idm_total / executed,
            "fixed_prediction_cost": self.fixed_idm_cost * idm_total,
            "policy_latency_seconds": float(
                sum(chunk["policy_latency_seconds"] or 0.0 for chunk in chunks)
            ),
            "gate_latency_seconds": float(
                sum(chunk["gate_latency_seconds"] or 0.0 for chunk in chunks)
            ),
            "environment_latency_seconds": float(
                sum(chunk["environment_latency_seconds"] for chunk in chunks)
            ),
            "termination_type": termination_type,
            "contract_violation_count": int(termination_type == "contract_violation"),
            "terminal": True,
            "chunk_record_ids": [chunk["record_id"] for chunk in chunks],
        }
        _canonical_bytes(episode)
        self._persist_completed_episode(episode=episode, chunks=chunks)
        self._episodes.append(episode)
        state.completed = True

    def _persist_completed_episode(
        self,
        *,
        episode: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> None:
        """Atomically persist one complete episode as the resumable unit."""

        identity = str(episode["episode_identity"])
        ledger_index = self._ledger_order[identity]
        path = self._completion_dir / f"{ledger_index:08d}.{identity}.json"
        if path.exists():
            raise FileExistsError(
                f"Completed evaluation episode already exists: {identity}."
            )
        payload = {
            "schema": COMPLETED_EPISODE_SCHEMA,
            "progress_identity_sha256": _sha256_payload(self._progress_identity),
            "episode": episode,
            "chunks": list(chunks),
        }
        _canonical_bytes(payload)
        self._write_json_atomic(path, payload)

    def _sync_live_jsonl(self) -> None:
        """Atomically expose all records collected so far after every chunk."""

        chunks = sorted(
            self._chunks,
            key=lambda record: (
                self._ledger_order[str(record["episode_identity"])],
                int(record["chunk_id"]),
            ),
        )
        episodes = sorted(
            self._episodes,
            key=lambda record: self._ledger_order[str(record["episode_identity"])],
        )
        self._write_jsonl_atomic(self._chunk_path, chunks)
        self._write_jsonl_atomic(self._episode_path, episodes)

    @staticmethod
    def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
        temporary = Path(f"{path}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = Path(f"{path}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def finalize(self) -> EvaluationArtifactShard:
        """Atomically write completed local shards and return their hashes."""

        if (
            not self._action_contracts
            or self._executable_action_contract_sha256 is None
        ):
            raise RuntimeError("Evaluation collected no live Action contract.")
        incomplete = [
            state.entry["episode_identity"]
            for state in self._states.values()
            if not state.completed
        ]
        if incomplete:
            raise RuntimeError(
                f"Evaluation ended with incomplete episodes: {incomplete}."
            )
        completed_identities = {
            episode["episode_identity"] for episode in self._episodes
        }
        missing = [
            entry["episode_identity"]
            for entry in self.ledger["entries"]
            if entry["episode_identity"] not in completed_identities
        ]
        if missing:
            raise RuntimeError(
                f"Evaluation ended without starting ledger episodes: {missing}."
            )
        ledger_order = {
            entry["episode_identity"]: index
            for index, entry in enumerate(self.ledger["entries"])
        }
        chunks = sorted(
            self._chunks,
            key=lambda item: (
                ledger_order[item["episode_identity"]],
                item["chunk_id"],
            ),
        )
        episodes = sorted(
            self._episodes,
            key=lambda item: ledger_order[item["episode_identity"]],
        )
        chunk_path = self._chunk_path
        episode_path = self._episode_path
        contract_items = sorted(self._action_contracts.items())
        action_contract_payloads = [
            contract.to_artifact() for _, contract in contract_items
        ]
        if len(contract_items) == 1:
            action_contract_paths = (
                self.output_dir / f"action_contract.rank-{self.rank}.json",
            )
        else:
            action_contract_paths = tuple(
                self.output_dir
                / f"action_contract.rank-{self.rank}.{canonical_sha256}.json"
                for canonical_sha256, _ in contract_items
            )
        self._write_jsonl_atomic(chunk_path, chunks)
        self._write_jsonl_atomic(episode_path, episodes)
        for path, payload in zip(
            action_contract_paths,
            action_contract_payloads,
        ):
            self._write_json_atomic(path, payload)
        canonical_chunks = sorted(
            chunks,
            key=lambda item: (item["episode_identity"], item["chunk_id"]),
        )
        canonical_episodes = sorted(
            episodes,
            key=lambda item: item["episode_identity"],
        )
        canonical_records = {
            "chunks": [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "run_id",
                        "policy_latency_seconds",
                        "gate_latency_seconds",
                        "gate_h2d_seconds",
                        "environment_latency_seconds",
                    }
                }
                for record in canonical_chunks
            ],
            "episodes": [
                {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "run_id",
                        "policy_latency_seconds",
                        "gate_latency_seconds",
                        "environment_latency_seconds",
                    }
                }
                for record in canonical_episodes
            ],
            "action_contracts": action_contract_payloads,
        }
        return EvaluationArtifactShard(
            rank=self.rank,
            chunk_path=str(chunk_path),
            episode_path=str(episode_path),
            action_contract_paths=tuple(str(path) for path in action_contract_paths),
            chunk_record_count=len(chunks),
            episode_record_count=len(episodes),
            chunk_sha256=_sha256_file(chunk_path),
            episode_sha256=_sha256_file(episode_path),
            action_contract_file_sha256s=tuple(
                _sha256_file(path) for path in action_contract_paths
            ),
            action_contract_canonical_sha256s=tuple(
                canonical_sha256 for canonical_sha256, _ in contract_items
            ),
            executable_action_contract_sha256=(self._executable_action_contract_sha256),
            canonical_content_sha256=_sha256_payload(canonical_records),
        )


__all__ = [
    "EvaluationArtifactShard",
    "EvaluationIdentityBatch",
    "FastWAMLiberoEvalCollector",
]
