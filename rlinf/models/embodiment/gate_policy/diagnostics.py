# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Training-signal and target-budget diagnostics for the adaptive gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


GATE_DIAGNOSTICS_SCHEMA_VERSION = 3


def _evidence_run_id(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA256 identifier")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        result = float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _count(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0 or not math.isclose(result, round(result), abs_tol=1e-5):
        raise ValueError(f"{name} must be a non-negative count, got {result}")
    return float(round(result))


def _fraction(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {result}")
    return result


def new_gate_diagnostics_state(
    *,
    seed: int,
    target_idm_usage: float | None,
    evidence_run_id: str | None = None,
) -> dict[str, Any]:
    """Create the exact run-level accumulator persisted beside Gate checkpoints."""
    target = None
    if target_idm_usage is not None:
        target = _fraction(target_idm_usage, name="target_idm_usage")
    return {
        "schema_version": GATE_DIAGNOSTICS_SCHEMA_VERSION,
        "evidence_run_id": _evidence_run_id(
            evidence_run_id, name="evidence_run_id"
        ),
        "seed": int(seed),
        "step": 0,
        "nonzero_return_variance_fraction": None,
        "group_return_variance": None,
        "zero_advantage_group_fraction": None,
        "effective_group_count": None,
        "group_count": None,
        "effective_sample_count": None,
        "effective_sample_fraction": None,
        "target_idm_usage": target,
        "idm_usage": None,
        "target_usage_error": None,
        "collapsed": False,
        "ever_collapsed": False,
        "collapse_consecutive": 0,
        "diagnostic_updates": 0,
        "diagnostic_rank_batches": 0,
        "diagnostic_eval_updates": 0,
        "cumulative_group_count": 0.0,
        "cumulative_nonzero_return_variance_group_count": 0.0,
        "cumulative_zero_advantage_group_count": 0.0,
        "cumulative_effective_group_count": 0.0,
        "cumulative_group_return_variance_sum": 0.0,
        "cumulative_sample_count": 0.0,
        "cumulative_effective_sample_count": 0.0,
    }


def _refresh_gate_diagnostic_means(payload: dict[str, Any]) -> None:
    groups = float(payload["cumulative_group_count"])
    samples = float(payload["cumulative_sample_count"])
    if groups > 0.0:
        payload["nonzero_return_variance_fraction"] = float(
            payload["cumulative_nonzero_return_variance_group_count"] / groups
        )
        payload["zero_advantage_group_fraction"] = float(
            payload["cumulative_zero_advantage_group_count"] / groups
        )
        payload["group_return_variance"] = float(
            payload["cumulative_group_return_variance_sum"] / groups
        )
        payload["group_count"] = groups
        payload["effective_group_count"] = float(
            payload["cumulative_effective_group_count"]
        )
    if samples > 0.0:
        payload["effective_sample_count"] = float(
            payload["cumulative_effective_sample_count"]
        )
        payload["effective_sample_fraction"] = float(
            payload["cumulative_effective_sample_count"] / samples
        )


def accumulate_grpo_gate_diagnostics(
    payload: dict[str, Any],
    *,
    step: int,
    rank_metrics: list[Mapping[str, Any]],
    group_size: int,
) -> None:
    """Accumulate one optimizer update of disjoint actor-rank GRPO groups.

    Rank-local means are weighted by their exact group counts. The update clock
    advances once, not once per rank, so the persisted state has an unambiguous
    resume contract.
    """
    if int(group_size) <= 1:
        raise ValueError("cumulative Gate GRPO diagnostics require group_size > 1")
    if not rank_metrics:
        raise ValueError("Gate GRPO diagnostics received no actor-rank metrics")
    expected_step = int(payload.get("step", -1)) + 1
    if int(step) != expected_step:
        raise ValueError(
            f"Gate diagnostic step must advance exactly once: expected "
            f"{expected_step}, got {step}"
        )

    for rank, metrics in enumerate(rank_metrics):
        if not metrics:
            raise ValueError(f"Gate GRPO diagnostics are empty for actor rank {rank}")
        prefix = f"actor rank {rank}"
        group_count = _count(
            metrics.get("gate/group_count"), name=f"{prefix} group_count"
        )
        if group_count <= 0.0:
            raise ValueError(f"{prefix} group_count must be positive")
        nonzero_fraction = _fraction(
            metrics.get("gate/nonzero_return_variance_fraction"),
            name=f"{prefix} nonzero_return_variance_fraction",
        )
        zero_advantage_fraction = _fraction(
            metrics.get("gate/zero_advantage_group_fraction"),
            name=f"{prefix} zero_advantage_group_fraction",
        )
        nonzero_count = _count(
            metrics.get("gate/nonzero_return_variance_group_count"),
            name=f"{prefix} nonzero_return_variance_group_count",
        )
        zero_advantage_count = _count(
            metrics.get("gate/zero_advantage_group_count"),
            name=f"{prefix} zero_advantage_group_count",
        )
        return_variance_sum = _finite_float(
            metrics.get("gate/group_return_variance_sum"),
            name=f"{prefix} group_return_variance_sum",
        )
        if return_variance_sum < 0.0:
            raise ValueError(
                f"{prefix} group_return_variance_sum must be non-negative"
            )
        effective_groups = _count(
            metrics.get("gate/effective_group_count"),
            name=f"{prefix} effective_group_count",
        )
        effective_samples = _count(
            metrics.get("gate/effective_sample_count"),
            name=f"{prefix} effective_sample_count",
        )
        sample_count = group_count * int(group_size)
        if effective_groups > group_count or effective_samples > sample_count:
            raise ValueError(f"{prefix} effective counts exceed sampled counts")
        if nonzero_count > group_count or not math.isclose(
            nonzero_count / group_count, nonzero_fraction, abs_tol=1e-6
        ):
            raise ValueError(
                f"{prefix} nonzero return-variance count/fraction disagree"
            )
        if zero_advantage_count > group_count or not math.isclose(
            zero_advantage_count / group_count,
            zero_advantage_fraction,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{prefix} zero-advantage count/fraction disagree"
            )
        if effective_groups + zero_advantage_count != group_count:
            raise ValueError(
                f"{prefix} effective_group_count disagrees with "
                "zero_advantage_group_count"
            )
        if "gate/group_return_variance" in metrics:
            reported_variance = _finite_float(
                metrics["gate/group_return_variance"],
                name=f"{prefix} group_return_variance",
            )
            if not math.isclose(
                reported_variance,
                return_variance_sum / group_count,
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"{prefix} return-variance sum/mean disagree"
                )
        if "gate/effective_sample_fraction" in metrics:
            reported = _fraction(
                metrics["gate/effective_sample_fraction"],
                name=f"{prefix} effective_sample_fraction",
            )
            expected = effective_samples / sample_count
            if not math.isclose(reported, expected, abs_tol=1e-6):
                raise ValueError(
                    f"{prefix} effective sample count/fraction disagree"
                )
        if "gate/zero_return_variance_fraction" in metrics:
            zero_return_fraction = _fraction(
                metrics["gate/zero_return_variance_fraction"],
                name=f"{prefix} zero_return_variance_fraction",
            )
            if not math.isclose(
                nonzero_fraction + zero_return_fraction, 1.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"{prefix} return-variance fractions do not sum to one"
                )

        payload["cumulative_group_count"] += group_count
        payload["cumulative_nonzero_return_variance_group_count"] += nonzero_count
        payload["cumulative_zero_advantage_group_count"] += zero_advantage_count
        payload["cumulative_effective_group_count"] += effective_groups
        payload["cumulative_group_return_variance_sum"] += return_variance_sum
        payload["cumulative_sample_count"] += sample_count
        payload["cumulative_effective_sample_count"] += effective_samples

    payload["step"] = int(step)
    payload["diagnostic_updates"] += 1
    payload["diagnostic_rank_batches"] += len(rank_metrics)
    _refresh_gate_diagnostic_means(payload)


def update_gate_eval_diagnostics(
    payload: dict[str, Any],
    *,
    idm_usage: Any,
    tracker: "BudgetCollapseTracker | None" = None,
) -> dict[str, float]:
    """Record one evaluation and, when configured, advance collapse state."""
    usage = _fraction(idm_usage, name="eval IDM usage")
    target = payload.get("target_idm_usage")
    if tracker is not None:
        metrics = tracker.update(usage)
        if target != tracker.target_idm_usage:
            raise ValueError("Gate diagnostics and collapse tracker targets disagree")
        payload["collapsed"] = bool(tracker.collapsed)
        payload["ever_collapsed"] = bool(tracker.ever_collapsed)
        payload["collapse_consecutive"] = int(tracker.consecutive)
        payload["target_usage_error"] = float(
            metrics["gate/target_budget_error"]
        )
    else:
        metrics = {}
        payload["target_usage_error"] = (
            None if target is None else abs(usage - float(target))
        )
    payload["idm_usage"] = usage
    payload["diagnostic_eval_updates"] += 1
    return metrics


def validate_gate_diagnostics_state(
    raw_payload: Mapping[str, Any],
    *,
    seed: int,
    target_idm_usage: float | None,
    evidence_run_id: str | None = None,
    step: int,
    group_size: int,
    tracker: "BudgetCollapseTracker | None" = None,
) -> dict[str, Any]:
    """Validate and canonicalize a checkpointed accumulator before actor load."""
    if not isinstance(raw_payload, Mapping):
        raise ValueError("Gate diagnostics resume payload must be a JSON object")
    payload = dict(raw_payload)
    schema_version = _count(
        payload.get("schema_version"), name="schema_version"
    )
    if schema_version != GATE_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("unsupported Gate diagnostics resume schema")
    if _count(payload.get("seed"), name="seed") != _count(seed, name="seed"):
        raise ValueError("Gate diagnostics resume seed does not match config")
    expected_evidence_run_id = _evidence_run_id(
        evidence_run_id, name="configured evidence_run_id"
    )
    actual_evidence_run_id = _evidence_run_id(
        payload.get("evidence_run_id"), name="evidence_run_id"
    )
    if actual_evidence_run_id != expected_evidence_run_id:
        raise ValueError("Gate diagnostics evidence run ID does not match config")
    payload["evidence_run_id"] = actual_evidence_run_id
    expected_target = (
        None
        if target_idm_usage is None
        else _fraction(target_idm_usage, name="configured target_idm_usage")
    )
    actual_target = payload.get("target_idm_usage")
    if actual_target is not None:
        actual_target = _fraction(actual_target, name="target_idm_usage")
    if actual_target != expected_target:
        raise ValueError("Gate diagnostics target budget does not match resume config")
    payload["target_idm_usage"] = actual_target
    checkpoint_step = _count(step, name="checkpoint step")
    if _count(payload.get("step"), name="step") != checkpoint_step:
        raise ValueError("Gate diagnostics step does not match checkpoint directory")
    payload["schema_version"] = GATE_DIAGNOSTICS_SCHEMA_VERSION
    payload["seed"] = int(seed)
    payload["step"] = int(checkpoint_step)

    count_fields = (
        "diagnostic_updates",
        "diagnostic_rank_batches",
        "diagnostic_eval_updates",
        "cumulative_group_count",
        "cumulative_nonzero_return_variance_group_count",
        "cumulative_zero_advantage_group_count",
        "cumulative_effective_group_count",
        "cumulative_sample_count",
        "cumulative_effective_sample_count",
    )
    for field in count_fields:
        payload[field] = _count(payload.get(field), name=field)
    if payload["diagnostic_updates"] != checkpoint_step:
        raise ValueError(
            "Gate diagnostics must contain exactly one cumulative update per "
            "optimizer step"
        )
    variance_sum = _finite_float(
        payload.get("cumulative_group_return_variance_sum"),
        name="cumulative_group_return_variance_sum",
    )
    if variance_sum < 0.0:
        raise ValueError("cumulative group return variance must be non-negative")
    payload["cumulative_group_return_variance_sum"] = variance_sum

    groups = payload["cumulative_group_count"]
    if checkpoint_step > 0 and groups <= 0.0:
        raise ValueError("non-empty Gate checkpoint has no cumulative groups")
    if payload["diagnostic_rank_batches"] < payload["diagnostic_updates"]:
        raise ValueError("Gate diagnostics have fewer rank batches than updates")
    if payload["diagnostic_eval_updates"] > payload["diagnostic_updates"]:
        raise ValueError("Gate diagnostics have more evals than optimizer updates")
    if not math.isclose(
        payload["cumulative_sample_count"], groups * int(group_size), abs_tol=1e-5
    ):
        raise ValueError("Gate cumulative sample/group counts disagree")
    nonzero = payload["cumulative_nonzero_return_variance_group_count"]
    zero_adv = payload["cumulative_zero_advantage_group_count"]
    effective_groups = payload["cumulative_effective_group_count"]
    effective_samples = payload["cumulative_effective_sample_count"]
    if nonzero > groups or zero_adv > groups:
        raise ValueError("Gate cumulative group subsets exceed group count")
    if not math.isclose(effective_groups + zero_adv, groups, abs_tol=1e-5):
        raise ValueError("Gate cumulative effective/zero-advantage groups disagree")
    if effective_samples > payload["cumulative_sample_count"]:
        raise ValueError("Gate cumulative effective samples exceed sample count")

    previous_derived = {
        key: payload.get(key)
        for key in (
            "nonzero_return_variance_fraction",
            "group_return_variance",
            "zero_advantage_group_fraction",
            "effective_group_count",
            "group_count",
            "effective_sample_count",
            "effective_sample_fraction",
        )
    }
    _refresh_gate_diagnostic_means(payload)
    for key, expected in previous_derived.items():
        actual = payload.get(key)
        if expected is None and actual is None:
            continue
        expected_float = _finite_float(expected, name=key)
        if not math.isclose(expected_float, float(actual), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"Gate diagnostics derived field {key} is inconsistent")

    usage = payload.get("idm_usage")
    if usage is not None:
        payload["idm_usage"] = _fraction(usage, name="idm_usage")
    eval_updates = int(payload["diagnostic_eval_updates"])
    if (eval_updates == 0) != (payload.get("idm_usage") is None):
        raise ValueError("Gate diagnostics eval clock and IDM usage disagree")
    expected_error = (
        None
        if expected_target is None or payload.get("idm_usage") is None
        else abs(float(payload["idm_usage"]) - expected_target)
    )
    actual_error = payload.get("target_usage_error")
    if expected_error is None:
        if actual_error is not None:
            raise ValueError("Gate diagnostics have a target error without a target")
    elif not math.isclose(
        _finite_float(actual_error, name="target_usage_error"),
        expected_error,
        abs_tol=1e-7,
    ):
        raise ValueError("Gate diagnostics target usage error is inconsistent")
    if not isinstance(payload.get("collapsed"), bool) or not isinstance(
        payload.get("ever_collapsed"), bool
    ):
        raise ValueError("Gate diagnostics collapse flags must be JSON booleans")
    payload["collapse_consecutive"] = int(
        _count(payload.get("collapse_consecutive"), name="collapse_consecutive")
    )
    if tracker is not None:
        if tracker.eval_count != int(payload["diagnostic_eval_updates"]):
            raise ValueError(
                "collapse tracker and Gate diagnostic eval clocks disagree"
            )
        if bool(payload.get("collapsed")) != tracker.collapsed:
            raise ValueError("collapse tracker and Gate diagnostics disagree")
        if bool(payload.get("ever_collapsed")) != tracker.ever_collapsed:
            raise ValueError("collapse history and Gate diagnostics disagree")
        if int(payload.get("collapse_consecutive", -1)) != tracker.consecutive:
            raise ValueError("collapse streak and Gate diagnostics disagree")
        if tracker.last_idm_usage != payload.get("idm_usage"):
            raise ValueError("collapse tracker and Gate diagnostics IDM usage disagree")
    elif (
        bool(payload.get("collapsed"))
        or bool(payload.get("ever_collapsed"))
        or int(payload.get("collapse_consecutive", -1)) != 0
    ):
        raise ValueError(
            "collapse state is present while collapse tracking is disabled"
        )
    return payload


def _episode_scores_like_grpo(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    reward_type: str,
) -> torch.Tensor:
    """Reproduce embodied GRPO's undiscounted score construction."""
    if reward_type == "chunk_level":
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True).values
    num_chunks, batch_size, chunk_size = rewards.shape
    n_steps = num_chunks * chunk_size
    rewards = rewards.transpose(1, 2).reshape(n_steps, batch_size)
    flattened_dones = dones.transpose(1, 2).reshape(
        (num_chunks + 1) * chunk_size, batch_size
    )
    dones = flattened_dones[-(n_steps + 1) :]
    scores = torch.zeros(
        batch_size, device=rewards.device, dtype=rewards.dtype
    )
    for step in reversed(range(n_steps)):
        scores = scores * ~dones[step + 1]
        scores = scores + rewards[step]
    return scores


def compute_grpo_group_diagnostics(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    advantages: torch.Tensor,
    group_size: int,
    reward_type: str = "chunk_level",
    loss_mask: torch.Tensor | None = None,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Measure whether GRPO groups contain any usable allocation signal.

    Metrics are trajectory/group weighted, rather than decision weighted, so
    early termination cannot make a collapsed group look informative merely by
    changing its episode length.
    """
    group_size = int(group_size)
    if group_size <= 1:
        raise ValueError("GRPO gate diagnostics require group_size > 1")
    scores = _episode_scores_like_grpo(
        rewards, dones, reward_type=reward_type
    ).float()
    if scores.numel() % group_size != 0:
        raise ValueError(
            f"trajectory count {scores.numel()} is not divisible by group_size={group_size}"
        )
    grouped_scores = scores.reshape(-1, group_size)
    variances = grouped_scores.var(dim=-1, unbiased=False)

    if advantages.ndim < 2:
        raise ValueError("advantages must contain time and trajectory dimensions")
    advantage_values = advantages.float()
    if advantage_values.ndim == 2:
        advantage_values = advantage_values.unsqueeze(-1)
    if loss_mask is None:
        valid = torch.ones_like(advantage_values, dtype=torch.bool)
    else:
        valid = loss_mask.to(device=advantage_values.device, dtype=torch.bool)
        if valid.shape != advantage_values.shape:
            valid = torch.broadcast_to(valid, advantage_values.shape)
    trajectory_effective = (
        (advantage_values.abs() > float(tolerance)) & valid
    ).any(dim=0)
    while trajectory_effective.ndim > 1:
        trajectory_effective = trajectory_effective.any(dim=-1)
    if trajectory_effective.numel() != scores.numel():
        raise ValueError(
            "advantage trajectory dimension does not match reward trajectories"
        )
    grouped_effective = trajectory_effective.reshape(-1, group_size)
    zero_advantage_groups = ~grouped_effective.any(dim=-1)
    zero_return_variance = variances <= float(tolerance)
    effective_count = int(trajectory_effective.sum().item())
    trajectory_count = int(trajectory_effective.numel())
    return {
        "gate/group_return_variance": float(variances.mean().item()),
        "gate/group_return_variance_sum": float(variances.sum().item()),
        "gate/nonzero_return_variance_fraction": float(
            (~zero_return_variance).float().mean().item()
        ),
        "gate/nonzero_return_variance_group_count": float(
            (~zero_return_variance).sum().item()
        ),
        "gate/zero_return_variance_fraction": float(
            zero_return_variance.float().mean().item()
        ),
        "gate/zero_advantage_group_fraction": float(
            zero_advantage_groups.float().mean().item()
        ),
        "gate/zero_advantage_group_count": float(
            zero_advantage_groups.sum().item()
        ),
        "gate/effective_sample_fraction": float(
            effective_count / max(trajectory_count, 1)
        ),
        "gate/effective_sample_count": float(effective_count),
        "gate/effective_group_count": float(
            (~zero_advantage_groups).sum().item()
        ),
        "gate/group_count": float(grouped_scores.shape[0]),
    }


class BudgetCollapseTracker:
    """Detect persistent endpoint collapse for an interior IDM-usage target."""

    def __init__(
        self,
        *,
        target_idm_usage: float,
        patience: int = 3,
        low_threshold: float = 0.05,
        high_threshold: float = 0.95,
        budget_error_threshold: float = 0.15,
    ):
        self.target_idm_usage = float(target_idm_usage)
        self.patience = int(patience)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.budget_error_threshold = float(budget_error_threshold)
        if not 0.0 < self.target_idm_usage < 1.0:
            raise ValueError("collapse tracking requires an interior target in (0,1)")
        if self.patience <= 0:
            raise ValueError("collapse patience must be positive")
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError("collapse thresholds must satisfy 0 <= low < high <= 1")
        if self.budget_error_threshold < 0.0:
            raise ValueError("budget_error_threshold must be non-negative")
        self.consecutive = 0
        self.collapsed = False
        self.ever_collapsed = False
        self.eval_count = 0
        self.last_idm_usage: float | None = None

    def update(self, idm_usage: float) -> dict[str, float]:
        usage = float(idm_usage)
        if not 0.0 <= usage <= 1.0 or not torch.isfinite(torch.tensor(usage)):
            raise ValueError(f"IDM usage must be finite and in [0,1], got {usage}")
        error = abs(usage - self.target_idm_usage)
        extreme = usage < self.low_threshold or usage > self.high_threshold
        qualifies = extreme and error > self.budget_error_threshold
        self.consecutive = self.consecutive + 1 if qualifies else 0
        self.collapsed = self.consecutive >= self.patience
        self.ever_collapsed = self.ever_collapsed or self.collapsed
        self.eval_count += 1
        self.last_idm_usage = usage
        return {
            "gate/target_budget_error": error,
            "gate/collapse_consecutive": float(self.consecutive),
            "gate/collapsed": float(self.collapsed),
            "gate/ever_collapsed": float(self.ever_collapsed),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": {
                "target_idm_usage": self.target_idm_usage,
                "patience": self.patience,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "budget_error_threshold": self.budget_error_threshold,
            },
            "state": {
                "consecutive": self.consecutive,
                "collapsed": self.collapsed,
                "ever_collapsed": self.ever_collapsed,
                "eval_count": self.eval_count,
                "last_idm_usage": self.last_idm_usage,
            },
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError("collapse tracker resume payload must be a JSON object")
        if _count(payload.get("version"), name="collapse state version") != 1:
            raise ValueError("unsupported budget collapse tracker state version")
        expected = self.state_dict()["config"]
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("collapse tracker config must be a JSON object")
        actual = dict(config)
        if actual != expected:
            raise ValueError(
                "budget collapse tracker config does not match resume state: "
                f"expected={expected}, actual={actual}"
            )
        state = payload.get("state", {})
        if not isinstance(state, Mapping):
            raise ValueError("collapse tracker state must be a JSON object")
        consecutive = int(
            _count(state.get("consecutive"), name="collapse consecutive")
        )
        eval_count = int(_count(state.get("eval_count"), name="collapse eval_count"))
        if not isinstance(state.get("collapsed"), bool) or not isinstance(
            state.get("ever_collapsed"), bool
        ):
            raise ValueError("collapse flags must be JSON booleans")
        collapsed = state["collapsed"]
        ever_collapsed = state["ever_collapsed"]
        if consecutive < 0 or eval_count < 0 or consecutive > eval_count:
            raise ValueError("invalid collapse tracker counters in resume state")
        if collapsed != (consecutive >= self.patience):
            raise ValueError("collapse flag disagrees with the persisted streak")
        if collapsed and not ever_collapsed:
            raise ValueError("collapsed resume state must preserve collapse history")
        last = state.get("last_idm_usage", None)
        if (eval_count == 0) != (last is None):
            raise ValueError("collapse eval clock and last IDM usage disagree")
        self.consecutive = consecutive
        self.collapsed = collapsed
        self.ever_collapsed = ever_collapsed
        self.eval_count = eval_count
        self.last_idm_usage = (
            None if last is None else _fraction(last, name="last_idm_usage")
        )


def write_collapse_state(path: str | Path, tracker: BudgetCollapseTracker) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(tracker.state_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_collapse_state(path: str | Path, tracker: BudgetCollapseTracker) -> None:
    tracker.load_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_gate_diagnostics(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish the latest machine-readable gate health record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
