# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Canonical, mergeable JSONL records for matched-budget gate evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rlinf.models.embodiment.gate_policy.mode_selectors import (
    validate_reserved_modes,
)


TRACE_SCHEMA_VERSION = 2
TRACE_RECORDS_KEY = "__gate_eval_trace_records__"


def gate_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shape, dtype, and bytes in a stable order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _scalar_at(value: object, index: int, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        flattened = value.detach().cpu().reshape(-1)
        return flattened[index].item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    return value


def context_record_at(context: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Extract immutable episode metadata from one batched gate context."""
    uid_values = context.get("episode_uid", context.get("episode_key"))
    uid = str(_scalar_at(uid_values, index, ""))
    if not uid:
        raise ValueError("gate trace requires a non-empty episode_uid")
    task_value = _scalar_at(
        context.get(
            "base_task",
            context.get("task_description", context.get("task")),
        ),
        index,
        "",
    )
    raw_assets = _scalar_at(context.get("asset_ids"), index, [])
    asset_ids = (
        [str(value) for value in raw_assets]
        if isinstance(raw_assets, Sequence) and not isinstance(raw_assets, (str, bytes))
        else []
    )
    return {
        "episode_uid": uid,
        "task": str(task_value),
        "task_suite_name": str(
            _scalar_at(context.get("task_suite_name"), index, "unknown")
        ),
        "task_id": int(_scalar_at(context.get("task_id"), index, -1)),
        "factor": str(_scalar_at(context.get("factor"), index, "unknown")),
        "level": str(_scalar_at(context.get("level"), index, "unknown")),
        "perturbation_id": str(
            _scalar_at(context.get("perturbation_id"), index, "unknown")
        ),
        "asset_ids": asset_ids,
        "reset_state_id": int(
            _scalar_at(context.get("reset_state_id"), index, -1)
        ),
        "seed": int(
            _scalar_at(context.get("env_seed", context.get("seed")), index, -1)
        ),
        "episode_manifest_sha256": str(
            _scalar_at(context.get("episode_manifest_sha256"), index, "")
        )
        or None,
    }


class RolloutGateTraceBuilder:
    """Accumulate mode/cost decisions before the runner merges task outcomes."""

    def __init__(
        self,
        *,
        method: str,
        max_decisions: int,
        selector_provenance: Mapping[str, Any],
        gate_checkpoint_sha256: str,
        wam_checkpoint_sha256: str | None,
    ):
        self.method = str(method)
        self.max_decisions = int(max_decisions)
        self.selector_provenance = dict(selector_provenance)
        self.gate_checkpoint_sha256 = str(gate_checkpoint_sha256)
        self.wam_checkpoint_sha256 = (
            None if wam_checkpoint_sha256 is None else str(wam_checkpoint_sha256)
        )
        self._records: dict[str, dict[str, Any]] = {}

    def add_batch(
        self,
        *,
        context: Mapping[str, Any],
        modes: torch.Tensor,
        costs: torch.Tensor,
        active_mask: torch.Tensor,
        reserved_modes: torch.Tensor | None,
        control_artifacts: Sequence[Mapping[str, str] | None] | None = None,
    ) -> None:
        modes = modes.detach().cpu().reshape(-1).long()
        costs = costs.detach().cpu().reshape(-1).float()
        active = active_mask.detach().cpu().reshape(-1).bool()
        decision_indices = torch.as_tensor(
            context["decision_index"], dtype=torch.long
        ).reshape(-1)
        batch = modes.numel()
        if not (
            costs.numel() == active.numel() == decision_indices.numel() == batch
        ):
            raise ValueError("gate trace batch fields have inconsistent lengths")
        if reserved_modes is not None:
            reserved_modes = reserved_modes.detach().cpu().long()
            if reserved_modes.shape != (batch, self.max_decisions):
                raise ValueError(
                    "reserved_modes must have shape "
                    f"[{batch},{self.max_decisions}], got {tuple(reserved_modes.shape)}"
                )
        if control_artifacts is not None and len(control_artifacts) != batch:
            raise ValueError(
                "control_artifacts must have one entry per gate sample; "
                f"got {len(control_artifacts)} for batch {batch}"
            )

        for index in range(batch):
            metadata = context_record_at(context, index)
            uid = metadata["episode_uid"]
            slot = int(decision_indices[index])
            if not 0 <= slot < self.max_decisions:
                raise ValueError(
                    f"trace decision slot {slot} is outside [0,{self.max_decisions - 1}]"
                )
            record = self._records.setdefault(
                uid,
                {
                    **metadata,
                    "method": self.method,
                    "max_decisions": self.max_decisions,
                    "reserved_modes": None,
                    "reference": {},
                    "decisions": {},
                    "gate_checkpoint_sha256": self.gate_checkpoint_sha256,
                    "wam_checkpoint_sha256": self.wam_checkpoint_sha256,
                    "mode_manifest_sha256": self.selector_provenance.get(
                        "mode_manifest_sha256"
                    ),
                    "selector_provenance": self.selector_provenance,
                },
            )
            for key in (
                "task",
                "task_suite_name",
                "task_id",
                "factor",
                "level",
                "perturbation_id",
                "asset_ids",
                "reset_state_id",
                "seed",
                "episode_manifest_sha256",
            ):
                if record[key] != metadata[key]:
                    raise ValueError(
                        f"episode_uid {uid!r} changed immutable trace field {key}"
                    )
            if reserved_modes is not None:
                schedule = list(
                    validate_reserved_modes(
                        reserved_modes[index].tolist(),
                        max_decisions=self.max_decisions,
                    )
                )
                if record["reserved_modes"] not in (None, schedule):
                    raise ValueError(f"reserved schedule changed for episode {uid!r}")
                record["reserved_modes"] = schedule
            reference = {
                "mode": int(modes[index]),
                "cost": float(costs[index]),
                "phase": str(
                    _scalar_at(context.get("phase"), index, "unknown")
                ),
                "phase_reliable": bool(
                    _scalar_at(context.get("phase_reliable"), index, False)
                ),
            }
            previous_reference = record["reference"].get(slot)
            if previous_reference is not None and previous_reference != reference:
                raise ValueError(
                    f"conflicting reference gate mode at episode {uid!r}, slot {slot}"
                )
            record["reference"][slot] = reference
            if not bool(active[index]):
                continue
            decision = {
                "slot": slot,
                "mode": int(modes[index]),
                "cost": float(costs[index]),
                "phase": str(_scalar_at(context.get("phase"), index, "unknown")),
                "phase_reliable": bool(
                    _scalar_at(context.get("phase_reliable"), index, False)
                ),
            }
            if control_artifacts is not None and control_artifacts[index] is not None:
                artifact = dict(control_artifacts[index] or {})
                if not artifact.get("path") or not artifact.get("sha256"):
                    raise ValueError(
                        "control artifact trace entries require path and sha256"
                    )
                decision["donor_artifact"] = artifact
            previous = record["decisions"].get(slot)
            if previous is not None and previous != decision:
                raise ValueError(
                    f"conflicting gate trace at episode {uid!r}, slot {slot}"
                )
            record["decisions"][slot] = decision

    def records(self) -> list[dict[str, Any]]:
        result = []
        for uid in sorted(self._records):
            record = dict(self._records[uid])
            reference = record.pop("reference")
            if self.method == "learned" and set(reference) != set(
                range(self.max_decisions)
            ):
                missing = sorted(set(range(self.max_decisions)) - set(reference))
                raise ValueError(
                    "learned reference trace did not execute the full fixed horizon "
                    f"for episode {uid!r}; missing slots={missing[:10]}"
                )
            record["reference_modes"] = [
                None if slot not in reference else int(reference[slot]["mode"])
                for slot in range(self.max_decisions)
            ]
            record["reference_phase"] = [
                None if slot not in reference else str(reference[slot]["phase"])
                for slot in range(self.max_decisions)
            ]
            record["reference_phase_reliable"] = [
                False
                if slot not in reference
                else bool(reference[slot]["phase_reliable"])
                for slot in range(self.max_decisions)
            ]
            record["reference_cost"] = [
                None if slot not in reference else float(reference[slot]["cost"])
                for slot in range(self.max_decisions)
            ]
            decisions = record.pop("decisions")
            record["decisions"] = [decisions[key] for key in sorted(decisions)]
            result.append(record)
        return result


class EnvGateTraceBuilder:
    """Track terminal outcome and the first successful gate decision slot."""

    def __init__(self, *, max_decisions: int):
        self.max_decisions = int(max_decisions)
        self._records: dict[str, dict[str, Any]] = {}

    def register_batch(self, context: Mapping[str, Any]) -> None:
        values = context.get("episode_uid", context.get("episode_key"))
        batch = len(values) if isinstance(values, Sequence) else int(values.shape[0])
        for index in range(batch):
            metadata = context_record_at(context, index)
            uid = metadata["episode_uid"]
            self._records.setdefault(
                uid,
                {
                    **metadata,
                    "success": False,
                    "success_slot": None,
                },
            )

    def update_after_step(
        self,
        *,
        context_before_action: Mapping[str, Any],
        success_once: torch.Tensor | None,
        active_before_action: torch.Tensor,
    ) -> None:
        if success_once is None:
            return
        success = torch.as_tensor(success_once).detach().cpu().reshape(-1).bool()
        active = active_before_action.detach().cpu().reshape(-1).bool()
        slots = torch.as_tensor(
            context_before_action["decision_index"], dtype=torch.long
        ).reshape(-1)
        for index in range(success.numel()):
            if not bool(active[index]) or not bool(success[index]):
                continue
            uid = context_record_at(context_before_action, index)["episode_uid"]
            if uid not in self._records:
                raise KeyError(
                    f"active episode_uid {uid!r} was not registered before action"
                )
            record = self._records[uid]
            if not record["success"]:
                record["success"] = True
                record["success_slot"] = int(slots[index])

    def records(self) -> list[dict[str, Any]]:
        return [dict(self._records[uid]) for uid in sorted(self._records)]


def merge_gate_eval_traces(
    *,
    env_records: Sequence[Mapping[str, Any]],
    rollout_records: Sequence[Mapping[str, Any]],
    expected_max_decisions: int,
) -> list[dict[str, Any]]:
    """Merge outcome and mode traces into the canonical one-record-per-episode schema."""

    def unique(records, label):
        result = {}
        for record in records:
            uid = str(record.get("episode_uid", ""))
            if not uid:
                raise ValueError(f"{label} trace record is missing episode_uid")
            if uid in result:
                raise ValueError(f"duplicate {label} trace for episode_uid {uid!r}")
            result[uid] = dict(record)
        return result

    env_by_uid = unique(env_records, "environment")
    rollout_by_uid = unique(rollout_records, "rollout")
    if set(env_by_uid) != set(rollout_by_uid):
        missing_rollout = sorted(set(env_by_uid) - set(rollout_by_uid))
        missing_env = sorted(set(rollout_by_uid) - set(env_by_uid))
        raise ValueError(
            "gate trace episode sets do not match: "
            f"missing_rollout={missing_rollout[:5]}, missing_env={missing_env[:5]}"
        )
    for field in (
        "method",
        "gate_checkpoint_sha256",
        "wam_checkpoint_sha256",
        "mode_manifest_sha256",
    ):
        values = {
            json.dumps(record.get(field), sort_keys=True)
            for record in rollout_by_uid.values()
        }
        if len(values) > 1:
            raise ValueError(
                f"rollout trace provenance field {field} differs across workers"
            )

    merged = []
    identity_fields = (
        "task",
        "task_suite_name",
        "task_id",
        "factor",
        "level",
        "perturbation_id",
        "asset_ids",
        "reset_state_id",
        "seed",
        "episode_manifest_sha256",
    )
    for uid in sorted(env_by_uid):
        env_record = env_by_uid[uid]
        rollout_record = rollout_by_uid[uid]
        for key in identity_fields:
            if env_record.get(key) != rollout_record.get(key):
                raise ValueError(
                    f"trace identity mismatch for {uid!r}, field {key}: "
                    f"{env_record.get(key)!r} vs {rollout_record.get(key)!r}"
                )
        max_decisions = int(rollout_record.get("max_decisions", -1))
        if max_decisions != int(expected_max_decisions):
            raise ValueError(
                f"trace max_decisions mismatch for {uid!r}: {max_decisions} "
                f"vs {expected_max_decisions}"
            )
        reserved = rollout_record.get("reserved_modes")
        if reserved is None:
            # Learned/phase policies cannot preregister state-dependent future
            # decisions. Keep an explicit fixed-width null schedule rather than
            # silently equating observed calls with reserved budget.
            reserved = [None] * max_decisions
        else:
            reserved = list(
                validate_reserved_modes(reserved, max_decisions=max_decisions)
            )
        raw_reference = rollout_record.get("reference_modes")
        if not isinstance(raw_reference, Sequence) or isinstance(
            raw_reference, (str, bytes)
        ):
            raise ValueError(f"trace {uid!r} is missing fixed-width reference_modes")
        reference_modes = list(raw_reference)
        if len(reference_modes) != max_decisions:
            raise ValueError(
                f"trace {uid!r} reference_modes must have {max_decisions} slots"
            )
        if str(rollout_record.get("method")) == "learned":
            reference_modes = list(
                validate_reserved_modes(
                    reference_modes,
                    max_decisions=max_decisions,
                    label="reference_modes",
                )
            )
        elif any(value not in (None, 0, 1) for value in reference_modes):
            raise ValueError("reference_modes may contain only null/UNCOND/IDM")
        reference_phase = list(rollout_record.get("reference_phase", []))
        reference_phase_reliable = list(
            rollout_record.get("reference_phase_reliable", [])
        )
        reference_cost = list(rollout_record.get("reference_cost", []))
        if not (
            len(reference_phase)
            == len(reference_phase_reliable)
            == len(reference_cost)
            == max_decisions
        ):
            raise ValueError(
                f"trace {uid!r} reference phase arrays must have {max_decisions} slots"
            )
        for slot, value in enumerate(reference_cost):
            if value is None:
                continue
            numeric = float(value)
            if not torch.isfinite(torch.tensor(numeric)) or numeric < 0:
                raise ValueError(
                    f"trace {uid!r} has invalid reference cost at slot {slot}"
                )
        success = bool(env_record.get("success", False))
        success_slot = env_record.get("success_slot")
        if success and success_slot is None:
            raise ValueError(f"successful trace {uid!r} is missing success_slot")
        cutoff = max_decisions - 1 if success_slot is None else int(success_slot)
        decisions = sorted(
            (
                dict(decision)
                for decision in rollout_record.get("decisions", [])
                if int(decision["slot"]) <= cutoff
            ),
            key=lambda item: int(item["slot"]),
        )
        slots = [int(item["slot"]) for item in decisions]
        if len(slots) != len(set(slots)):
            raise ValueError(f"duplicate actual decision slots for episode {uid!r}")
        actual_modes = [int(item["mode"]) for item in decisions]
        actual_cost = [float(item["cost"]) for item in decisions]
        actual_phases = [str(item.get("phase", "unknown")) for item in decisions]
        actual_phase_reliable = [
            bool(item.get("phase_reliable", False)) for item in decisions
        ]
        donor_artifacts = [item.get("donor_artifact") for item in decisions]
        merged.append(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "method": str(rollout_record["method"]),
                "episode_uid": uid,
                **{key: env_record.get(key) for key in identity_fields},
                "success": success,
                "success_slot": None if success_slot is None else int(success_slot),
                "max_decisions": max_decisions,
                "reserved_modes": reserved,
                "reference_modes": reference_modes,
                "reference_phase": reference_phase,
                "reference_phase_reliable": reference_phase_reliable,
                "reference_cost": reference_cost,
                "actual_slots_before_success": slots,
                "actual_modes_before_success": actual_modes,
                "actual_cost_before_success": actual_cost,
                "actual_phase_before_success": actual_phases,
                "actual_phase_reliable_before_success": actual_phase_reliable,
                "donor_artifacts_before_success": donor_artifacts,
                "reserved_idm_calls": (
                    None if any(value is None for value in reserved) else sum(reserved)
                ),
                "actual_idm_calls": sum(actual_modes),
                "reference_idm_calls": (
                    None
                    if any(value is None for value in reference_modes)
                    else sum(reference_modes)
                ),
                "reference_reserved_idm_calls": (
                    None
                    if any(value is None for value in reference_modes)
                    else sum(reference_modes)
                ),
                "reference_total_cost": (
                    None
                    if any(value is None for value in reference_cost)
                    else float(sum(float(value) for value in reference_cost))
                ),
                "actual_total_cost": float(sum(actual_cost)),
                "gate_checkpoint_sha256": rollout_record.get(
                    "gate_checkpoint_sha256"
                ),
                "wam_checkpoint_sha256": rollout_record.get(
                    "wam_checkpoint_sha256"
                ),
                "episode_manifest_sha256": env_record.get(
                    "episode_manifest_sha256"
                ),
                "mode_manifest_sha256": rollout_record.get(
                    "mode_manifest_sha256"
                ),
                "selector_provenance": rollout_record.get(
                    "selector_provenance", {}
                ),
            }
        )
    return merged


def write_gate_eval_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write deterministic canonical JSONL output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    lines = [
        json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: str(item["episode_uid"]))
    ]
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)
