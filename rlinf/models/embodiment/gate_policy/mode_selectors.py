# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic evaluation policies for the binary FastWAM gate.

Training always samples the learned gate.  These selectors are evaluation-only
controls used to construct matched-compute baselines.  Randomness is derived from
SHA256 over the selector seed and immutable episode identity, so a schedule is
independent of worker count, batching, and evaluation order.

For a 700-step LIBERO episode with a 10-action execution horizon, scheduled
selectors reserve all 70 decision slots up front.  Early success changes the
compute actually spent, but never the registered schedule.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SUPPORTED_EVAL_POLICIES = {
    "learned",
    "forced",
    "episode_mixture",
    "bernoulli",
    "random_k",
    "manifest",
    "periodic_k",
    "phase_heuristic",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_uniform(*parts: object) -> float:
    return _stable_u64(*parts) / float(1 << 64)


def _validate_max_decisions(value: object) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"max_decisions must be positive, got {result}")
    return result


def _validate_k(value: object, *, max_decisions: int) -> int:
    if isinstance(value, bool):
        raise ValueError("k must be an integer, not bool")
    result = int(value)
    if float(value) != float(result):
        raise ValueError(f"k must be an integer, got {value!r}")
    if not 0 <= result <= max_decisions:
        raise ValueError(
            f"k must be in [0, {max_decisions}], got {result}"
        )
    return result


def build_random_k_schedule(
    *,
    episode_uid: str,
    max_decisions: int,
    k: int,
    seed: int,
) -> tuple[int, ...]:
    """Reserve exactly ``k`` IDM slots using order-independent hash scores."""
    max_decisions = _validate_max_decisions(max_decisions)
    k = _validate_k(k, max_decisions=max_decisions)
    ranked = sorted(
        range(max_decisions),
        key=lambda slot: (_stable_u64("random_k", seed, episode_uid, slot), slot),
    )
    selected = set(ranked[:k])
    return tuple(int(slot in selected) for slot in range(max_decisions))


def build_periodic_k_schedule(
    *, max_decisions: int, k: int
) -> tuple[int, ...]:
    """Reserve exactly ``k`` slots with gaps differing by at most one."""
    max_decisions = _validate_max_decisions(max_decisions)
    k = _validate_k(k, max_decisions=max_decisions)
    if k == 0:
        return (0,) * max_decisions
    # A Bresenham-style accumulator is deterministic and avoids rounding ties.
    return tuple(
        int(((slot + 1) * k) // max_decisions > (slot * k) // max_decisions)
        for slot in range(max_decisions)
    )


def validate_reserved_modes(
    modes: Sequence[object], *, max_decisions: int, label: str = "reserved_modes"
) -> tuple[int, ...]:
    if isinstance(modes, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of binary modes")
    result = tuple(int(value) for value in modes)
    if len(result) != max_decisions:
        raise ValueError(
            f"{label} must contain exactly {max_decisions} modes, got {len(result)}"
        )
    if any(value not in (0, 1) for value in result):
        raise ValueError(f"{label} may contain only UNCOND=0 and IDM=1")
    return result


@dataclass(frozen=True)
class NormalizedGateContext:
    episode_uids: tuple[str, ...]
    decision_indices: torch.Tensor
    phases: tuple[str, ...] | None = None
    phase_reliable: torch.Tensor | None = None


def _batch_sequence(
    value: object,
    *,
    batch_size: int,
    name: str,
    cast,
) -> tuple:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value] * batch_size
    if len(values) != batch_size:
        raise ValueError(
            f"gate_context.{name} must have one value per sample; "
            f"got {len(values)} for batch {batch_size}"
        )
    return tuple(cast(value) for value in values)


def normalize_gate_context(
    context: Mapping[str, Any] | None,
    *,
    batch_size: int,
    device: torch.device,
    max_decisions: int,
    require_phase: bool = False,
) -> NormalizedGateContext:
    if context is None:
        raise ValueError(
            "This eval policy requires env_obs.gate_context with immutable "
            "episode_uid and decision_index fields."
        )
    if not isinstance(context, Mapping):
        raise ValueError(
            f"gate_context must be a mapping, got {type(context).__name__}"
        )
    uid_value = context.get("episode_uid", context.get("episode_key"))
    if uid_value is None:
        raise ValueError("gate_context is missing episode_uid")
    if context.get("decision_index") is None:
        raise ValueError("gate_context is missing decision_index")
    episode_uids = _batch_sequence(
        uid_value, batch_size=batch_size, name="episode_uid", cast=str
    )
    if any(not value for value in episode_uids):
        raise ValueError("gate_context.episode_uid values must be non-empty")
    decision_values = _batch_sequence(
        context["decision_index"],
        batch_size=batch_size,
        name="decision_index",
        cast=int,
    )
    decision_indices = torch.tensor(
        decision_values, device=device, dtype=torch.long
    )
    if bool(((decision_indices < 0) | (decision_indices >= max_decisions)).any()):
        bad = decision_indices[
            (decision_indices < 0) | (decision_indices >= max_decisions)
        ].detach().cpu().tolist()
        raise ValueError(
            f"gate decision_index must be in [0, {max_decisions - 1}], got {bad}"
        )

    phases = None
    phase_reliable = None
    if context.get("phase") is not None:
        phases = _batch_sequence(
            context["phase"], batch_size=batch_size, name="phase", cast=str
        )
    if context.get("phase_reliable") is not None:
        reliability = _batch_sequence(
            context["phase_reliable"],
            batch_size=batch_size,
            name="phase_reliable",
            cast=bool,
        )
        phase_reliable = torch.tensor(reliability, device=device, dtype=torch.bool)
    if require_phase:
        if phases is None:
            raise ValueError("phase_heuristic requires gate_context.phase")
        if phase_reliable is None or not bool(phase_reliable.all()):
            raise ValueError(
                "phase_heuristic requires pre-treatment phase labels with "
                "gate_context.phase_reliable=true for every sample"
            )
    return NormalizedGateContext(
        episode_uids=episode_uids,
        decision_indices=decision_indices,
        phases=phases,
        phase_reliable=phase_reliable,
    )


@dataclass(frozen=True)
class ModeSelection:
    modes: torch.Tensor
    method: str
    episode_uids: tuple[str, ...] | None = None
    decision_indices: torch.Tensor | None = None
    reserved_modes: torch.Tensor | None = None
    manifest_sha256: str | None = None

    @property
    def reserved_idm_count(self) -> torch.Tensor | None:
        if self.reserved_modes is None:
            return None
        return self.reserved_modes.long().sum(dim=-1)


class EvalModeSelector:
    kind = "learned"
    requires_context = False

    def __init__(self, *, max_decisions: int, seed: int = 0):
        self.max_decisions = _validate_max_decisions(max_decisions)
        self.seed = int(seed)
        self.manifest_sha256: str | None = None

    def _context(
        self,
        context: Mapping[str, Any] | None,
        *,
        logits: torch.Tensor,
        require_phase: bool = False,
    ) -> NormalizedGateContext:
        return normalize_gate_context(
            context,
            batch_size=logits.shape[0],
            device=logits.device,
            max_decisions=self.max_decisions,
            require_phase=require_phase,
        )

    def select(
        self, logits: torch.Tensor, context: Mapping[str, Any] | None = None
    ) -> ModeSelection:
        del context
        return ModeSelection(modes=torch.argmax(logits, dim=-1), method=self.kind)

    def provenance(self) -> dict[str, object]:
        return {
            "method": self.kind,
            "schedule_seed": self.seed,
            "max_decisions": self.max_decisions,
            "mode_manifest_sha256": self.manifest_sha256,
        }


class ForcedModeSelector(EvalModeSelector):
    kind = "forced"

    def __init__(self, *, mode: int, **kwargs):
        super().__init__(**kwargs)
        self.mode = int(mode)
        if self.mode not in (0, 1):
            raise ValueError("forced mode must be UNCOND=0 or IDM=1")

    def select(self, logits, context=None):
        batch = logits.shape[0]
        modes = torch.full(
            (batch,), self.mode, device=logits.device, dtype=torch.long
        )
        reserved = modes[:, None].expand(batch, self.max_decisions).clone()
        normalized = None
        if context is not None:
            normalized = self._context(context, logits=logits)
        return ModeSelection(
            modes=modes,
            method=self.kind,
            episode_uids=None if normalized is None else normalized.episode_uids,
            decision_indices=(
                None if normalized is None else normalized.decision_indices
            ),
            reserved_modes=reserved,
        )

    def schedule_for(self, episode_uid: str) -> tuple[int, ...]:
        del episode_uid
        return (self.mode,) * self.max_decisions

    def provenance(self):
        return {**super().provenance(), "forced_mode": self.mode}


class _ScheduledSelector(EvalModeSelector):
    requires_context = True

    def schedule_for(self, episode_uid: str) -> tuple[int, ...]:
        raise NotImplementedError

    def select(self, logits, context=None):
        normalized = self._context(context, logits=logits)
        schedules = torch.tensor(
            [self.schedule_for(uid) for uid in normalized.episode_uids],
            device=logits.device,
            dtype=torch.long,
        )
        rows = torch.arange(logits.shape[0], device=logits.device)
        modes = schedules[rows, normalized.decision_indices]
        return ModeSelection(
            modes=modes,
            method=self.kind,
            episode_uids=normalized.episode_uids,
            decision_indices=normalized.decision_indices,
            reserved_modes=schedules,
            manifest_sha256=self.manifest_sha256,
        )


class EpisodeMixtureSelector(_ScheduledSelector):
    kind = "episode_mixture"

    def __init__(self, *, p_idm: float, **kwargs):
        super().__init__(**kwargs)
        self.p_idm = float(p_idm)
        if not 0.0 <= self.p_idm <= 1.0:
            raise ValueError("p_idm must be in [0, 1]")

    def schedule_for(self, episode_uid):
        mode = int(
            _stable_uniform("episode_mixture", self.seed, episode_uid) < self.p_idm
        )
        return (mode,) * self.max_decisions

    def provenance(self):
        return {**super().provenance(), "p_idm": self.p_idm}


class BernoulliSelector(_ScheduledSelector):
    kind = "bernoulli"

    def __init__(self, *, p_idm: float, **kwargs):
        super().__init__(**kwargs)
        self.p_idm = float(p_idm)
        if not 0.0 <= self.p_idm <= 1.0:
            raise ValueError("p_idm must be in [0, 1]")

    def schedule_for(self, episode_uid):
        return tuple(
            int(
                _stable_uniform(
                    "bernoulli", self.seed, episode_uid, decision_index
                )
                < self.p_idm
            )
            for decision_index in range(self.max_decisions)
        )

    def provenance(self):
        return {**super().provenance(), "p_idm": self.p_idm}


class RandomKSelector(_ScheduledSelector):
    kind = "random_k"

    def __init__(self, *, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = _validate_k(k, max_decisions=self.max_decisions)

    def schedule_for(self, episode_uid):
        return build_random_k_schedule(
            episode_uid=episode_uid,
            max_decisions=self.max_decisions,
            k=self.k,
            seed=self.seed,
        )

    def provenance(self):
        return {**super().provenance(), "k": self.k}


class PeriodicKSelector(_ScheduledSelector):
    kind = "periodic_k"

    def __init__(self, *, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = _validate_k(k, max_decisions=self.max_decisions)
        self._schedule = build_periodic_k_schedule(
            max_decisions=self.max_decisions, k=self.k
        )

    def schedule_for(self, episode_uid):
        del episode_uid
        return self._schedule

    def provenance(self):
        return {**super().provenance(), "k": self.k}


def _load_structured_file(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is an RLinf dependency
        raise RuntimeError("YAML mode manifests require PyYAML") from exc
    return yaml.safe_load(text)


class ManifestSelector(_ScheduledSelector):
    kind = "manifest"

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        expected_checkpoint_sha256: str | None = None,
        expected_episode_manifest_sha256: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = Path(manifest_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"mode schedule manifest not found: {self.path}")
        self.manifest_sha256 = sha256_file(self.path)
        payload = _load_structured_file(self.path)
        if not isinstance(payload, Mapping):
            raise ValueError("mode schedule manifest must contain a mapping")
        version = int(payload.get("version", 1))
        if version != 1:
            raise ValueError(f"unsupported mode schedule manifest version {version}")
        manifest_n = int(payload.get("max_decisions", self.max_decisions))
        if manifest_n != self.max_decisions:
            raise ValueError(
                "mode manifest max_decisions does not match evaluation: "
                f"{manifest_n} vs {self.max_decisions}"
            )
        provenance = payload.get("provenance", {}) or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("mode manifest provenance must be a mapping")
        self.manifest_provenance = dict(provenance)
        self.registered_method = str(payload.get("method", "manifest"))
        self.source_checkpoint_sha256 = provenance.get("checkpoint_sha256")
        self.episode_manifest_sha256 = provenance.get("episode_manifest_sha256")
        if not self.source_checkpoint_sha256 or not self.episode_manifest_sha256:
            raise ValueError(
                "mode manifest provenance requires checkpoint_sha256 and "
                "episode_manifest_sha256"
            )
        for expected, actual, name in (
            (
                expected_checkpoint_sha256,
                self.source_checkpoint_sha256,
                "checkpoint_sha256",
            ),
            (
                expected_episode_manifest_sha256,
                self.episode_manifest_sha256,
                "episode_manifest_sha256",
            ),
        ):
            if expected is not None and str(actual) != str(expected):
                raise ValueError(
                    f"mode manifest {name} mismatch: expected {expected}, got {actual}"
                )

        episodes = payload.get("episodes")
        if not isinstance(episodes, Mapping) or not episodes:
            raise ValueError("mode manifest episodes must be a non-empty mapping")
        self.schedules: dict[str, tuple[int, ...]] = {}
        for uid, entry in episodes.items():
            modes = entry.get("reserved_modes") if isinstance(entry, Mapping) else entry
            if modes is None and isinstance(entry, Mapping) and "idm_slots" in entry:
                slots = {int(slot) for slot in entry["idm_slots"]}
                if any(slot < 0 or slot >= self.max_decisions for slot in slots):
                    raise ValueError(f"episode {uid!r} has out-of-range idm_slots")
                modes = [int(slot in slots) for slot in range(self.max_decisions)]
            if modes is None:
                raise ValueError(f"episode {uid!r} has no reserved_modes")
            self.schedules[str(uid)] = validate_reserved_modes(
                modes,
                max_decisions=self.max_decisions,
                label=f"episodes[{uid!r}].reserved_modes",
            )

    def schedule_for(self, episode_uid):
        try:
            return self.schedules[str(episode_uid)]
        except KeyError as exc:
            raise KeyError(
                f"episode_uid {episode_uid!r} is missing from mode manifest "
                f"{self.path}"
            ) from exc

    def select(self, logits, context=None):
        if not isinstance(context, Mapping):
            raise ValueError("manifest selector requires gate_context provenance")
        runtime_hashes = _batch_sequence(
            context.get("episode_manifest_sha256"),
            batch_size=logits.shape[0],
            name="episode_manifest_sha256",
            cast=str,
        )
        mismatched = sorted(
            {
                value
                for value in runtime_hashes
                if value != str(self.episode_manifest_sha256)
            }
        )
        if mismatched:
            raise ValueError(
                "runtime episode manifest SHA does not match mode schedule "
                f"manifest: expected {self.episode_manifest_sha256}, got {mismatched}"
            )
        return super().select(logits, context)

    def provenance(self):
        return {
            **super().provenance(),
            "mode_manifest_path": str(self.path),
            "mode_manifest_sha256": self.manifest_sha256,
            "registered_method": self.registered_method,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "episode_manifest_sha256": self.episode_manifest_sha256,
            "manifest_provenance": self.manifest_provenance,
        }


class PhaseHeuristicSelector(EvalModeSelector):
    kind = "phase_heuristic"
    requires_context = True

    def __init__(self, *, idm_phases: Sequence[str], **kwargs):
        super().__init__(**kwargs)
        self.idm_phases = frozenset(str(value) for value in idm_phases)
        if not self.idm_phases:
            raise ValueError("phase_heuristic requires at least one idm_phase")

    def select(self, logits, context=None):
        normalized = self._context(context, logits=logits, require_phase=True)
        modes = torch.tensor(
            [int(phase in self.idm_phases) for phase in normalized.phases],
            device=logits.device,
            dtype=torch.long,
        )
        # The future phase sequence is not observable at episode start, so unlike
        # Random-K this policy has no preregistered 70-slot reservation.
        return ModeSelection(
            modes=modes,
            method=self.kind,
            episode_uids=normalized.episode_uids,
            decision_indices=normalized.decision_indices,
        )

    def provenance(self):
        return {**super().provenance(), "idm_phases": sorted(self.idm_phases)}


def _cfg_get(cfg: Mapping[str, Any] | object | None, key: str, default=None):
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def build_eval_mode_selector(
    cfg: Mapping[str, Any] | object | None,
) -> EvalModeSelector:
    kind = str(_cfg_get(cfg, "kind", "learned")).lower()
    if kind not in SUPPORTED_EVAL_POLICIES:
        raise ValueError(
            f"unknown eval_policy.kind={kind!r}; expected one of "
            f"{sorted(SUPPORTED_EVAL_POLICIES)}"
        )
    common = {
        "max_decisions": _cfg_get(cfg, "max_decisions", 70),
        "seed": _cfg_get(cfg, "seed", 0),
    }
    if kind == "learned":
        return EvalModeSelector(**common)
    if kind == "forced":
        return ForcedModeSelector(mode=_cfg_get(cfg, "mode", None), **common)
    if kind == "episode_mixture":
        return EpisodeMixtureSelector(
            p_idm=_cfg_get(cfg, "p_idm", None), **common
        )
    if kind == "bernoulli":
        return BernoulliSelector(p_idm=_cfg_get(cfg, "p_idm", None), **common)
    if kind == "random_k":
        return RandomKSelector(k=_cfg_get(cfg, "k", None), **common)
    if kind == "periodic_k":
        return PeriodicKSelector(k=_cfg_get(cfg, "k", None), **common)
    if kind == "manifest":
        return ManifestSelector(
            manifest_path=_cfg_get(cfg, "manifest_path", None),
            expected_checkpoint_sha256=_cfg_get(
                cfg, "expected_checkpoint_sha256", None
            ),
            expected_episode_manifest_sha256=_cfg_get(
                cfg, "expected_episode_manifest_sha256", None
            ),
            **common,
        )
    return PhaseHeuristicSelector(
        idm_phases=_cfg_get(cfg, "idm_phases", ()), **common
    )


def make_mode_schedule_manifest(
    *,
    selector: _ScheduledSelector,
    episode_uids: Sequence[str],
    checkpoint_sha256: str,
    episode_manifest_sha256: str,
) -> dict[str, object]:
    """Materialize auditable schedules before launching an evaluation."""
    if not checkpoint_sha256 or not episode_manifest_sha256:
        raise ValueError(
            "schedule manifests require checkpoint and episode-manifest SHA256"
        )
    return {
        "version": 1,
        "max_decisions": selector.max_decisions,
        "method": selector.kind,
        "provenance": {
            **selector.provenance(),
            "checkpoint_sha256": checkpoint_sha256,
            "episode_manifest_sha256": episode_manifest_sha256,
        },
        "episodes": {
            str(uid): {"reserved_modes": list(selector.schedule_for(str(uid)))}
            for uid in episode_uids
        },
    }


REFERENCE_MATCH_METHODS = {
    "reference_random_k",
    "reference_task_factor",
    "reference_phase",
}


def load_canonical_reference_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load the canonical learned-reference JSONL without accepting partial rows."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"canonical reference trace not found: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid reference JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(record, Mapping):
            raise ValueError(
                f"reference trace row {line_number} must contain a mapping"
            )
        records.append(dict(record))
    if not records:
        raise ValueError(f"canonical reference trace is empty: {path}")
    return records


def _ranked_quota_schedule(
    slots: Sequence[tuple[str, int]],
    *,
    quota: int,
    seed: int,
    method: str,
    cell: Sequence[str],
) -> set[tuple[str, int]]:
    if not 0 <= int(quota) <= len(slots):
        raise ValueError(
            f"quota must be in [0,{len(slots)}], got {quota} for cell {cell}"
        )
    ranked = sorted(
        slots,
        key=lambda item: (
            _stable_u64(method, seed, *cell, item[0], item[1]),
            item[0],
            item[1],
        ),
    )
    return set(ranked[: int(quota)])


def make_reference_matched_mode_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    method: str,
    episode_uids: Sequence[str],
    checkpoint_sha256: str,
    episode_manifest_sha256: str,
    reference_trace_sha256: str,
    seed: int,
    max_decisions: int = 70,
) -> dict[str, object]:
    """Randomize learned full-horizon modes while conserving registered quotas."""
    method = str(method)
    if method not in REFERENCE_MATCH_METHODS:
        raise ValueError(
            f"unknown reference matching method {method!r}; expected "
            f"{sorted(REFERENCE_MATCH_METHODS)}"
        )
    max_decisions = _validate_max_decisions(max_decisions)
    if max_decisions != 70:
        raise ValueError(
            "LIBERO learned-reference manifests require exactly 70 decision slots"
        )
    expected_uids = tuple(sorted(str(uid) for uid in episode_uids))
    if (
        not expected_uids
        or any(not uid for uid in expected_uids)
        or len(expected_uids) != len(set(expected_uids))
    ):
        raise ValueError("episode manifest must contain unique, non-empty episode IDs")
    if not checkpoint_sha256 or not episode_manifest_sha256:
        raise ValueError("reference manifests require WAM and episode-manifest SHA256")
    if not reference_trace_sha256:
        raise ValueError("reference_trace_sha256 is required")

    by_uid: dict[str, dict[str, Any]] = {}
    reference_gate_hashes = set()
    for raw in records:
        record = dict(raw)
        uid = str(record.get("episode_uid", ""))
        if not uid or uid in by_uid:
            raise ValueError(f"duplicate or empty reference episode_uid {uid!r}")
        if int(record.get("schema_version", -1)) < 2:
            raise ValueError(
                f"reference episode {uid!r} predates full-horizon trace schema v2"
            )
        if str(record.get("method")) != "learned":
            raise ValueError(
                f"reference episode {uid!r} was not collected with learned eval"
            )
        if int(record.get("max_decisions", -1)) != max_decisions:
            raise ValueError(
                f"reference episode {uid!r} does not have {max_decisions} slots"
            )
        if str(record.get("wam_checkpoint_sha256", "")) != checkpoint_sha256:
            raise ValueError(
                f"reference episode {uid!r} WAM checkpoint SHA does not match"
            )
        if (
            str(record.get("episode_manifest_sha256", ""))
            != episode_manifest_sha256
        ):
            raise ValueError(
                f"reference episode {uid!r} episode-manifest SHA does not match"
            )
        modes = validate_reserved_modes(
            record.get("reference_modes", ()),
            max_decisions=max_decisions,
            label=f"reference[{uid!r}].reference_modes",
        )
        declared_usage = record.get("reference_idm_calls", sum(modes))
        if isinstance(declared_usage, bool) or int(declared_usage) != sum(modes):
            raise ValueError(
                f"reference episode {uid!r} declared IDM quota does not match modes"
            )
        task = str(record.get("task", ""))
        factor = str(record.get("factor", ""))
        if task in {"", "unknown"} or factor in {"", "unknown"}:
            raise ValueError(
                f"reference episode {uid!r} needs explicit task and factor"
            )
        phases = record.get("reference_phase", ())
        phase_reliable = record.get("reference_phase_reliable", ())
        if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
            raise ValueError(f"reference episode {uid!r} has no phase sequence")
        if not isinstance(phase_reliable, Sequence) or isinstance(
            phase_reliable, (str, bytes)
        ):
            raise ValueError(
                f"reference episode {uid!r} has no phase-reliability sequence"
            )
        if len(phases) != max_decisions or len(phase_reliable) != max_decisions:
            raise ValueError(
                f"reference episode {uid!r} phase arrays must have 70 slots"
            )
        if any(not isinstance(value, bool) for value in phase_reliable):
            raise ValueError(
                f"reference episode {uid!r} phase reliability must be boolean"
            )
        gate_hash = str(record.get("gate_checkpoint_sha256", ""))
        if not gate_hash:
            raise ValueError(
                f"reference episode {uid!r} lacks gate checkpoint provenance"
            )
        reference_gate_hashes.add(gate_hash)
        by_uid[uid] = {
            "modes": modes,
            "task": task,
            "factor": factor,
            "phases": tuple(str(value) for value in phases),
            "phase_reliable": tuple(bool(value) for value in phase_reliable),
        }
    if tuple(sorted(by_uid)) != expected_uids:
        raise ValueError(
            "reference trace episode set does not match frozen episode manifest: "
            f"missing={sorted(set(expected_uids) - set(by_uid))[:5]}, "
            f"extra={sorted(set(by_uid) - set(expected_uids))[:5]}"
        )
    if len(reference_gate_hashes) != 1:
        raise ValueError("reference trace mixes gate checkpoint identities")

    schedules = {
        uid: [0 for _ in range(max_decisions)] for uid in expected_uids
    }
    quota_cells: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    if method == "reference_random_k":
        for uid in expected_uids:
            quota_cells[(uid,)] = [
                (uid, slot) for slot in range(max_decisions)
            ]
    elif method == "reference_task_factor":
        for uid in expected_uids:
            record = by_uid[uid]
            cell = (record["task"], record["factor"])
            quota_cells[cell].extend(
                (uid, slot) for slot in range(max_decisions)
            )
    else:
        unreliable_slots = 0
        for uid in expected_uids:
            record = by_uid[uid]
            for slot, (phase, reliable) in enumerate(
                zip(record["phases"], record["phase_reliable"])
            ):
                phase_cell = phase
                if not reliable or phase in {"", "unknown", "None"}:
                    # Preserve uncertainty as an explicit cell rather than
                    # guessing a post-hoc phase or dropping budget.
                    phase_cell = "UNKNOWN"
                    unreliable_slots += 1
                quota_cells[(record["task"], record["factor"], phase_cell)].append(
                    (uid, slot)
                )

    quota_records = []
    for cell in sorted(quota_cells):
        slots = sorted(quota_cells[cell])
        quota = sum(int(by_uid[uid]["modes"][slot]) for uid, slot in slots)
        selected = _ranked_quota_schedule(
            slots,
            quota=quota,
            seed=int(seed),
            method=method,
            cell=cell,
        )
        for uid, slot in selected:
            schedules[uid][slot] = 1
        allocated = sum(schedules[uid][slot] for uid, slot in slots)
        if allocated != quota:
            raise RuntimeError(
                f"quota conservation failed for {method} cell {cell}: "
                f"{allocated} vs {quota}"
            )
        quota_records.append(
            {
                "cell": list(cell),
                "slot_count": len(slots),
                "reference_idm_calls": quota,
                "reserved_idm_calls": allocated,
            }
        )

    total_reference = sum(sum(by_uid[uid]["modes"]) for uid in expected_uids)
    total_reserved = sum(sum(schedules[uid]) for uid in expected_uids)
    if total_reference != total_reserved:
        raise RuntimeError(
            f"global reference quota was not conserved: {total_reserved} vs "
            f"{total_reference}"
        )
    phase_matching = method == "reference_phase"
    return {
        "version": 1,
        "max_decisions": max_decisions,
        "method": method,
        "provenance": {
            "method": method,
            "seed": int(seed),
            "max_decisions": max_decisions,
            "checkpoint_sha256": checkpoint_sha256,
            "episode_manifest_sha256": episode_manifest_sha256,
            "reference_trace_sha256": reference_trace_sha256,
            "reference_gate_checkpoint_sha256": next(
                iter(reference_gate_hashes)
            ),
            "reference_phase_matching": phase_matching,
            "reference_phase_semantics": (
                "learned_reference_trajectory_not_strict_post_branch_matching;"
                "unreliable_labels_preserved_as_UNKNOWN"
                if phase_matching
                else None
            ),
            "reference_phase_unreliable_slots": (
                unreliable_slots if phase_matching else None
            ),
            "determinism": "sha256_order_independent_v1",
            "total_reference_idm_calls": total_reference,
            "total_reserved_idm_calls": total_reserved,
            "quota_conservation": quota_records,
        },
        "episodes": {
            uid: {
                "reserved_modes": schedules[uid],
                "reference_idm_calls": int(sum(by_uid[uid]["modes"])),
                "reserved_idm_calls": int(sum(schedules[uid])),
                "task": by_uid[uid]["task"],
                "factor": by_uid[uid]["factor"],
            }
            for uid in expected_uids
        },
    }


def write_json_atomic(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def checkpoint_sha256_from_config(value: object) -> str | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    path = Path(str(value)).expanduser()
    return sha256_file(path) if path.is_file() else None
