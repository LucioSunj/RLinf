# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalized rollout-sample schema for snapshot-grouped WAM training.

One sample is one executed action chunk from one rollout member of one
snapshot group, carrying everything an FPO-style update needs to replay the
old-policy CFM loss: the flow-matching draws ``(tau, eps)`` and the cached
``cfm_loss_old``.

PROVISIONAL SHAPES - pending alignment with the FastWAM sampler interface
(work order W8).  Until W8 lands, this module fixes:

- ``eps``: ``[action_horizon, action_dim]`` float32 - matching the training
  draw ``torch.randn_like(action)`` in FastWAM ``training_loss``.
- ``tau``: scalar (``[]``) float32 - matching ``sample_training_t``, which
  draws ONE timestep per sample, not one per action step.  This deviates from
  the work order's literal "[T, action_dim]" for tau because no FastWAM code
  path produces a per-step tau; the deviation is recorded here and in the PR.

``deserialize_samples`` refuses any payload whose schema string differs from
:data:`WAM_ROLLOUT_SCHEMA`, so a shape change at W8 alignment time bumps the
version and old buffers fail closed instead of being silently misread.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

WAM_ROLLOUT_SCHEMA = "wam-rollout-buffer-v1"


class WamBufferError(ValueError):
    """Raised when a rollout sample or buffer payload violates the schema."""


@dataclass(frozen=True)
class WamBufferConfig:
    """Shape contract shared by every sample in one buffer."""

    action_horizon: int
    action_dim: int
    group_size: int

    def __post_init__(self):
        # Type before magnitude: int() coercion would silently accept floats
        # from a JSON round-trip and let bools (isinstance(True, int)) through.
        for name in ("action_horizon", "action_dim", "group_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise WamBufferError(
                    f"{name} must be an int, got {value!r} ({type(value).__name__})"
                )
        for name in ("action_horizon", "action_dim"):
            if getattr(self, name) < 1:
                raise WamBufferError(f"{name} must be >= 1, got {getattr(self, name)}")
        # GRPO baselines need within-group contrast; a singleton group has none.
        if self.group_size < 2:
            raise WamBufferError(f"group_size must be >= 2, got {self.group_size}")


@dataclass(frozen=True)
class WamRolloutSample:
    """One executed action chunk from one member of one snapshot group."""

    snapshot_id: str
    group_id: int
    group_member_index: int
    episode_id: str
    decision_index: int
    tau: torch.Tensor
    eps: torch.Tensor
    cfm_loss_old: float
    reward: float
    advantage: Optional[float] = None


def _require_tensor(
    value: Any, *, name: str, shape: tuple[int, ...], context: str
) -> None:
    if not isinstance(value, torch.Tensor):
        raise WamBufferError(
            f"{context}: {name} must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.dtype != torch.float32:
        raise WamBufferError(f"{context}: {name} must be float32, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise WamBufferError(
            f"{context}: {name} must have shape {shape}, got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise WamBufferError(f"{context}: {name} contains non-finite values")


def _require_finite_float(value: Any, *, name: str, context: str) -> None:
    if not isinstance(value, float) or not math.isfinite(value):
        raise WamBufferError(f"{context}: {name} must be a finite float, got {value!r}")


def validate_sample(
    sample: WamRolloutSample, config: WamBufferConfig, *, context: str
) -> None:
    """Fail-closed validation of one sample against the buffer contract."""
    for name in ("snapshot_id", "episode_id"):
        value = getattr(sample, name)
        if not isinstance(value, str) or not value:
            raise WamBufferError(
                f"{context}: {name} must be a non-empty string, got {value!r}"
            )
    for name in ("group_id", "decision_index"):
        value = getattr(sample, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise WamBufferError(
                f"{context}: {name} must be a non-negative int, got {value!r}"
            )
    member = sample.group_member_index
    if (
        not isinstance(member, int)
        or isinstance(member, bool)
        or not 0 <= member < config.group_size
    ):
        raise WamBufferError(
            f"{context}: group_member_index must be in [0, {config.group_size}), "
            f"got {member!r}"
        )
    _require_tensor(sample.tau, name="tau", shape=(), context=context)
    _require_tensor(
        sample.eps,
        name="eps",
        shape=(config.action_horizon, config.action_dim),
        context=context,
    )
    _require_finite_float(sample.cfm_loss_old, name="cfm_loss_old", context=context)
    if sample.cfm_loss_old < 0.0:
        raise WamBufferError(
            f"{context}: cfm_loss_old must be >= 0, got {sample.cfm_loss_old}"
        )
    _require_finite_float(sample.reward, name="reward", context=context)
    if sample.advantage is not None:
        _require_finite_float(sample.advantage, name="advantage", context=context)


def serialize_samples(
    samples: Sequence[WamRolloutSample],
    config: WamBufferConfig,
    *,
    context: str = "serialize_samples",
) -> dict[str, Any]:
    """Validate and pack samples into a torch-saveable payload.

    Beyond per-sample validation this enforces buffer-level invariants: one row
    per executed action chunk, so ``(group_id, group_member_index,
    decision_index)`` triples are unique and each member's decision indices are
    contiguous ``0..n-1`` (members may differ in n - early termination); every
    group that appears is complete - exactly ``group_size`` members with
    indices ``0..group_size-1``.  A partially collected group cannot produce a
    valid GRPO baseline, so it is refused here rather than downstream.

    ``group_id`` is PLAN-SCOPED: every plan numbers groups ``0..N-1``, so one
    buffer holds rows from one plan.  Accumulating rollout rounds requires the
    caller to renumber (offset) group ids per round before serializing.
    """
    if not samples:
        raise WamBufferError(f"{context}: refusing to serialize an empty buffer")
    seen: set[tuple[int, int, int]] = set()
    members_by_group: dict[int, set[int]] = {}
    decisions_by_member: dict[tuple[int, int], set[int]] = {}
    snapshot_by_group: dict[int, str] = {}
    for index, sample in enumerate(samples):
        sample_context = f"{context}[{index}]"
        validate_sample(sample, config, context=sample_context)
        key = (sample.group_id, sample.group_member_index, sample.decision_index)
        if key in seen:
            raise WamBufferError(
                f"{sample_context}: duplicate "
                f"(group_id, group_member_index, decision_index) {key}"
            )
        seen.add(key)
        members_by_group.setdefault(sample.group_id, set()).add(
            sample.group_member_index
        )
        decisions_by_member.setdefault(
            (sample.group_id, sample.group_member_index), set()
        ).add(sample.decision_index)
        expected_snapshot = snapshot_by_group.setdefault(
            sample.group_id, sample.snapshot_id
        )
        if sample.snapshot_id != expected_snapshot:
            raise WamBufferError(
                f"{sample_context}: group {sample.group_id} mixes snapshots "
                f"{expected_snapshot!r} and {sample.snapshot_id!r}; a group is "
                "K rollouts from ONE restored snapshot"
            )
    for group_id, members in sorted(members_by_group.items()):
        if members != set(range(config.group_size)):
            raise WamBufferError(
                f"{context}: group {group_id} is incomplete or malformed: "
                f"members {sorted(members)}, expected 0..{config.group_size - 1}"
            )
    for (group_id, member), decisions in sorted(decisions_by_member.items()):
        if decisions != set(range(len(decisions))):
            raise WamBufferError(
                f"{context}: member {member} of group {group_id} has "
                f"non-contiguous decision_index values {sorted(decisions)}; "
                "expected 0..n-1, one row per executed chunk"
            )
    return {
        "schema": WAM_ROLLOUT_SCHEMA,
        "config": asdict(config),
        "snapshot_id": [s.snapshot_id for s in samples],
        "group_id": [s.group_id for s in samples],
        "group_member_index": [s.group_member_index for s in samples],
        "episode_id": [s.episode_id for s in samples],
        "decision_index": [s.decision_index for s in samples],
        "tau": torch.stack([s.tau for s in samples]),
        "eps": torch.stack([s.eps for s in samples]),
        "cfm_loss_old": [s.cfm_loss_old for s in samples],
        "reward": [s.reward for s in samples],
        "advantage": [s.advantage for s in samples],
    }


_PAYLOAD_KEYS = (
    "schema",
    "config",
    "snapshot_id",
    "group_id",
    "group_member_index",
    "episode_id",
    "decision_index",
    "tau",
    "eps",
    "cfm_loss_old",
    "reward",
    "advantage",
)


def deserialize_samples(
    payload: Mapping[str, Any], *, context: str = "deserialize_samples"
) -> tuple[WamBufferConfig, list[WamRolloutSample]]:
    """Unpack and re-validate a payload produced by :func:`serialize_samples`."""
    if not isinstance(payload, Mapping):
        raise WamBufferError(
            f"{context}: payload must be a mapping, got {type(payload).__name__}"
        )
    missing = [key for key in _PAYLOAD_KEYS if key not in payload]
    unexpected = sorted(set(payload) - set(_PAYLOAD_KEYS))
    if missing or unexpected:
        raise WamBufferError(
            f"{context}: payload keys mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if payload["schema"] != WAM_ROLLOUT_SCHEMA:
        raise WamBufferError(
            f"{context}: schema must be {WAM_ROLLOUT_SCHEMA!r}, "
            f"got {payload['schema']!r}"
        )
    try:
        config = WamBufferConfig(**dict(payload["config"]))
    except TypeError as exc:
        raise WamBufferError(f"{context}: malformed config: {exc}") from exc
    count = len(payload["snapshot_id"])
    list_keys = (
        "snapshot_id",
        "group_id",
        "group_member_index",
        "episode_id",
        "decision_index",
        "cfm_loss_old",
        "reward",
        "advantage",
    )
    for key in list_keys:
        if len(payload[key]) != count:
            raise WamBufferError(
                f"{context}: field {key!r} has {len(payload[key])} entries, "
                f"expected {count}"
            )
    for key in ("tau", "eps"):
        value = payload[key]
        if not isinstance(value, torch.Tensor) or value.shape[0] != count:
            raise WamBufferError(
                f"{context}: field {key!r} must be a stacked tensor with {count} rows"
            )
    samples = [
        WamRolloutSample(
            snapshot_id=payload["snapshot_id"][i],
            group_id=payload["group_id"][i],
            group_member_index=payload["group_member_index"][i],
            episode_id=payload["episode_id"][i],
            decision_index=payload["decision_index"][i],
            tau=payload["tau"][i],
            eps=payload["eps"][i],
            cfm_loss_old=payload["cfm_loss_old"][i],
            reward=payload["reward"][i],
            advantage=payload["advantage"][i],
        )
        for i in range(count)
    ]
    # Re-run the full buffer-level validation so a hand-edited payload cannot
    # smuggle in what serialize_samples would have refused.
    serialize_samples(samples, config, context=f"{context}.revalidate")
    return config, samples
