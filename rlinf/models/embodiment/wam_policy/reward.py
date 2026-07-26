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

"""Reward and advantage assembly for snapshot-grouped WAM RL (Route A).

Route A trains the UNCOND branch with three deliberately separated
ingredients:

- a SPARSE TERMINAL RETURN per rollout member - did this member reach task
  success within its remaining horizon after the snapshot restore
  (:func:`terminal_success_return`);
- a GROUP-NORMALIZED ADVANTAGE over the K members of one snapshot group
  (:func:`group_normalized_advantage`) - GRPO-style, critic-free, because all
  K members start from the identical restored simulator state;
- a demo flow-matching ANCHOR that lives at the LOSS level, never in the env
  reward.  This module owns ONLY its schedule (:func:`demo_anchor_weight`);
  the anchor term itself is the W8 ``cfm_loss`` on demo data, composed by the
  WS5 loss as ``total = fpo_term + lambda_bc(step) * cfm_loss_demo``.

:func:`fill_reward_fields` closes the loop: it broadcasts the per-trajectory
return/advantage onto every buffer row of that trajectory and re-runs the
full ``buffer_schema`` validation, so a filled batch is exactly as trustworthy
as a freshly serialized one.

Boundary with ``rlinf.algorithms.advantages.compute_grpo_advantages``: the
dense ``[-1, group_size]`` GRPO path is NOT reimplemented or modified here.
This module operates on explicit ``group_id`` labels straight from the
rollout buffer because the WS5 loss additionally needs (a) EXACT-ZERO
advantages for zero-variance groups - they carry no learning signal, and eps
noise must not be turned into gradients - and (b) the zero-variance /
effective-sample diagnostics that the GRPO health checks track.  Everything
is parameterized; no K, horizon, or seed count is hard-coded (OQ7 ruling).

All functions are pure: no hidden state, fail-closed validation, mirroring
``gate_policy.reward``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Union

import torch

from .buffer_schema import (
    WamBufferConfig,
    WamRolloutSample,
    serialize_samples,
)

#: One trajectory = one rollout member = one ``(group_id, group_member_index)``.
TrajectoryKey = tuple[int, int]


class WamRewardError(ValueError):
    """Raised when reward/advantage assembly violates its contract."""


def _require_real(value: Any, *, name: str, context: str) -> float:
    """Accept int/float (not bool), require finite; return as float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WamRewardError(
            f"{context}: {name} must be a real number, got {value!r} "
            f"({type(value).__name__})"
        )
    if not math.isfinite(value):
        raise WamRewardError(f"{context}: {name} must be finite, got {value!r}")
    return float(value)


def _require_bool_matrix(value: Any, *, name: str, context: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise WamRewardError(
            f"{context}: {name} must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.dtype != torch.bool:
        raise WamRewardError(
            f"{context}: {name} must be a bool tensor, got {value.dtype}"
        )
    if value.ndim != 2:
        raise WamRewardError(
            f"{context}: {name} must be 2-D [num_rollouts, num_steps], "
            f"got shape {tuple(value.shape)}"
        )
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise WamRewardError(
            f"{context}: {name} must be non-empty, got shape {tuple(value.shape)}"
        )


def _first_true_index(flags: torch.Tensor) -> torch.Tensor:
    """Per row: index of the first True, or num_steps if the row has none.

    ``(~flags).cumprod`` counts leading Falses, which IS the first-True index;
    this avoids relying on argmax tie-breaking semantics.
    """
    return (~flags).to(torch.int64).cumprod(dim=1).sum(dim=1)


def terminal_success_return(
    successes: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Sparse terminal return per rollout member.

    Args:
        successes: Bool ``[num_rollouts, num_steps]`` - success flag observed
            at each executed decision step (one column per action chunk).
            Padded columns after termination must stay False.
        dones: Bool ``[num_rollouts, num_steps]`` - episode-termination flag
            per decision step, same shape as ``successes``.  Sticky/padded
            True values after the first done are permitted.
        gamma: Discount in ``(0, 1]``.  With the default ``gamma=1.0`` the
            return is the plain indicator: 1.0 if the member reached success
            within its window, else 0.0.  With ``gamma < 1`` the return is
            ``gamma ** k`` where ``k`` is the 0-based index of the FIRST
            success step (success on the very first chunk is undiscounted:
            ``gamma ** 0 == 1.0``), rewarding faster completion.

    Truncation semantics: a member that runs out its window without a done
    flag simply scores on any success observed inside the window (none => 0);
    the horizon itself is a parameter of the data, never of this function.

    Returns:
        Float32 ``[num_rollouts]`` returns.

    Raises:
        WamRewardError: On non-bool dtypes, shape mismatch, empty input,
            ``gamma`` outside ``(0, 1]``, or a success flagged STRICTLY after
            the first done step (success-after-done inconsistency - the
            rollout stream is lying about one of the two).
    """
    context = "terminal_success_return"
    _require_bool_matrix(successes, name="successes", context=context)
    _require_bool_matrix(dones, name="dones", context=context)
    if successes.shape != dones.shape:
        raise WamRewardError(
            f"{context}: successes shape {tuple(successes.shape)} != "
            f"dones shape {tuple(dones.shape)}"
        )
    gamma_f = _require_real(gamma, name="gamma", context=context)
    if not 0.0 < gamma_f <= 1.0:
        raise WamRewardError(f"{context}: gamma must be in (0, 1], got {gamma_f}")

    num_steps = successes.shape[1]
    first_success = _first_true_index(successes)
    first_done = _first_true_index(dones)
    columns = torch.arange(num_steps, device=successes.device)
    late = successes & (columns.unsqueeze(0) > first_done.unsqueeze(1))
    if bool(late.any()):
        rows = sorted({int(r) for r in torch.nonzero(late)[:, 0]})
        raise WamRewardError(
            f"{context}: rows {rows} flag success strictly after the episode "
            "terminated (success-after-done); refusing an inconsistent "
            "rollout stream"
        )
    succeeded = first_success < num_steps
    if gamma_f == 1.0:
        return succeeded.to(torch.float32)
    discounted = (gamma_f ** first_success.to(torch.float32)).to(torch.float32)
    return torch.where(succeeded, discounted, torch.zeros_like(discounted))


def group_normalized_advantage(
    returns: torch.Tensor,
    group_ids: Union[Sequence[int], torch.Tensor],
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-group normalized advantage ``A = (R - mean_g) / (std_g + eps)``.

    ``std_g`` is the POPULATION std (``correction=0``); the tests pin this
    convention.  A group whose returns are all identical (the common case
    under sparse 0/1 returns: all-success or all-failure) has no within-group
    contrast, so its advantages are EXACTLY zero - not ``0/eps`` float noise -
    and the group is counted in the diagnostics: those samples contribute no
    learning signal and the GRPO health checks must see how many there are.

    Args:
        returns: Floating 1-D ``[num_trajectories]`` per-member returns.
        group_ids: ``[num_trajectories]`` non-negative ints (sequence or int
            tensor), aligned with ``returns``.  Groups need not be contiguous
            or uniformly sized here (buffer-level completeness is enforced at
            serialization), but every group needs >= 2 members.
        eps: Positive denominator guard.

    Returns:
        ``(advantages, diagnostics)`` - advantages as float32
        ``[num_trajectories]`` aligned with the input order (permutation of
        input rows permutes the output identically), and diagnostics with
        ``num_samples``, ``num_groups``, ``group_mean``, ``group_std``,
        ``zero_variance_group_ids``, ``zero_variance_group_count``,
        ``effective_group_count``, ``effective_sample_count`` (samples in
        groups that DO carry signal).

    Raises:
        WamRewardError: On empty/non-1-D/non-floating/non-finite returns,
            length mismatch, bool or negative group ids, a singleton group,
            or ``eps <= 0``.
    """
    context = "group_normalized_advantage"
    if not isinstance(returns, torch.Tensor):
        raise WamRewardError(
            f"{context}: returns must be a torch.Tensor, got {type(returns).__name__}"
        )
    if returns.ndim != 1:
        raise WamRewardError(
            f"{context}: returns must be 1-D, got shape {tuple(returns.shape)}"
        )
    if returns.shape[0] < 1:
        raise WamRewardError(f"{context}: refusing an empty returns tensor")
    if not torch.is_floating_point(returns):
        raise WamRewardError(
            f"{context}: returns must be floating point, got {returns.dtype}"
        )
    if not bool(torch.isfinite(returns).all()):
        raise WamRewardError(f"{context}: returns contains non-finite values")
    eps_f = _require_real(eps, name="eps", context=context)
    if eps_f <= 0.0:
        raise WamRewardError(f"{context}: eps must be > 0, got {eps_f}")

    if isinstance(group_ids, torch.Tensor):
        if group_ids.ndim != 1:
            raise WamRewardError(
                f"{context}: group_ids must be 1-D, got shape {tuple(group_ids.shape)}"
            )
        if group_ids.dtype not in (torch.int32, torch.int64):
            raise WamRewardError(
                f"{context}: group_ids tensor must be int32/int64, "
                f"got {group_ids.dtype}"
            )
        ids = [int(g) for g in group_ids.tolist()]
    else:
        ids = list(group_ids)
        for index, gid in enumerate(ids):
            if not isinstance(gid, int) or isinstance(gid, bool):
                raise WamRewardError(
                    f"{context}: group_ids[{index}] must be an int, got {gid!r}"
                )
    if len(ids) != int(returns.shape[0]):
        raise WamRewardError(
            f"{context}: got {len(ids)} group ids for {int(returns.shape[0])} returns"
        )
    negative = sorted({g for g in ids if g < 0})
    if negative:
        raise WamRewardError(f"{context}: group ids must be >= 0, got {negative}")

    values = returns.to(torch.float32)
    members_by_group: dict[int, list[int]] = {}
    for index, gid in enumerate(ids):
        members_by_group.setdefault(gid, []).append(index)

    advantages = torch.zeros_like(values)
    group_mean: dict[int, float] = {}
    group_std: dict[int, float] = {}
    zero_variance_ids: list[int] = []
    effective_sample_count = 0
    for gid in sorted(members_by_group):
        member_indices = members_by_group[gid]
        if len(member_indices) < 2:
            raise WamRewardError(
                f"{context}: group {gid} has {len(member_indices)} member(s); "
                "a group-normalized baseline needs >= 2 rollouts per group"
            )
        group_values = values[member_indices]
        mean = group_values.mean()
        std = group_values.std(correction=0)
        group_mean[gid] = float(mean)
        group_std[gid] = float(std)
        if bool((group_values == group_values[0]).all()):
            # All-equal detection rather than std == 0: fp summation can make
            # the std of identical values a tiny nonzero, which would leak
            # eps-scaled noise into the advantages.  Advantages stay the
            # exact zeros they were initialized to.
            zero_variance_ids.append(gid)
            continue
        effective_sample_count += len(member_indices)
        advantages[member_indices] = (group_values - mean) / (std + eps_f)

    diagnostics: dict[str, Any] = {
        "num_samples": int(values.shape[0]),
        "num_groups": len(members_by_group),
        "group_mean": group_mean,
        "group_std": group_std,
        "zero_variance_group_ids": tuple(zero_variance_ids),
        "zero_variance_group_count": len(zero_variance_ids),
        "effective_group_count": len(members_by_group) - len(zero_variance_ids),
        "effective_sample_count": effective_sample_count,
    }
    return advantages, diagnostics


@dataclass(frozen=True)
class AnchorSchedule:
    """Explicit ``lambda_bc`` schedule for the demo flow-matching anchor.

    The WS5 loss composes ``total = fpo_term + lambda_bc(step) *
    cfm_loss_demo``; this schedule owns ONLY ``lambda_bc(step)``.  Two kinds:

    - ``constant``: ``lambda_bc(step) == start`` forever; ``end`` and
      ``decay_steps`` must stay at their defaults so a mistyped decay cannot
      silently become a constant.
    - ``linear_decay``: linear from ``start`` at step 0 to ``end`` at
      ``decay_steps``, held at ``end`` afterwards.  ``end <= start`` is
      required - a rising "decay" is a config bug, not a schedule.

    All weights must be non-negative and finite (fail-closed).
    """

    kind: str
    start: float
    end: float = 0.0
    decay_steps: int = 0

    def __post_init__(self):
        context = "AnchorSchedule"
        if self.kind not in ("constant", "linear_decay"):
            raise WamRewardError(
                f"{context}: kind must be 'constant' or 'linear_decay', "
                f"got {self.kind!r}"
            )
        for name in ("start", "end"):
            value = _require_real(getattr(self, name), name=name, context=context)
            if value < 0.0:
                raise WamRewardError(
                    f"{context}: {name} must be >= 0 (anchor weights are "
                    f"non-negative), got {value}"
                )
            object.__setattr__(self, name, value)
        if not isinstance(self.decay_steps, int) or isinstance(self.decay_steps, bool):
            raise WamRewardError(
                f"{context}: decay_steps must be an int, got {self.decay_steps!r}"
            )
        if self.kind == "constant":
            if self.end != 0.0 or self.decay_steps != 0:
                raise WamRewardError(
                    f"{context}: constant schedule must leave end/decay_steps "
                    f"at defaults, got end={self.end}, "
                    f"decay_steps={self.decay_steps}"
                )
        else:
            if self.decay_steps < 1:
                raise WamRewardError(
                    f"{context}: linear_decay needs decay_steps >= 1, "
                    f"got {self.decay_steps}"
                )
            if self.end > self.start:
                raise WamRewardError(
                    f"{context}: linear_decay must not increase: "
                    f"end ({self.end}) > start ({self.start})"
                )

    @classmethod
    def constant(cls, value: float) -> "AnchorSchedule":
        return cls(kind="constant", start=value)

    @classmethod
    def linear_decay(
        cls, *, start: float, end: float, decay_steps: int
    ) -> "AnchorSchedule":
        return cls(kind="linear_decay", start=start, end=end, decay_steps=decay_steps)


def demo_anchor_weight(step: int, *, schedule: AnchorSchedule) -> float:
    """Evaluate the demo-anchor coefficient ``lambda_bc`` at ``step``.

    Pure and stateless: the caller passes the global optimizer step; nothing
    is cached between calls.

    Raises:
        WamRewardError: If ``schedule`` is not an :class:`AnchorSchedule` or
            ``step`` is not a non-negative int.
    """
    context = "demo_anchor_weight"
    if not isinstance(schedule, AnchorSchedule):
        raise WamRewardError(
            f"{context}: schedule must be an AnchorSchedule, "
            f"got {type(schedule).__name__}"
        )
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise WamRewardError(
            f"{context}: step must be a non-negative int, got {step!r}"
        )
    if schedule.kind == "constant":
        return schedule.start
    fraction = min(step / float(schedule.decay_steps), 1.0)
    return schedule.start + (schedule.end - schedule.start) * fraction


def trajectory_value_map(
    keys: Sequence[TrajectoryKey],
    values: torch.Tensor,
    *,
    context: str = "trajectory_value_map",
) -> dict[TrajectoryKey, float]:
    """Zip aligned trajectory keys with a 1-D value tensor, fail-closed.

    Bridges the tensor outputs of :func:`terminal_success_return` /
    :func:`group_normalized_advantage` (aligned with the rollout-record
    order) into the keyed mappings :func:`fill_reward_fields` consumes.
    """
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise WamRewardError(f"{context}: values must be a 1-D torch.Tensor")
    if not torch.is_floating_point(values):
        raise WamRewardError(
            f"{context}: values must be floating point, got {values.dtype}"
        )
    if not bool(torch.isfinite(values).all()):
        raise WamRewardError(f"{context}: values contains non-finite entries")
    if len(keys) != int(values.shape[0]):
        raise WamRewardError(
            f"{context}: got {len(keys)} keys for {int(values.shape[0])} values"
        )
    result: dict[TrajectoryKey, float] = {}
    for index, key in enumerate(keys):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in key)
        ):
            raise WamRewardError(
                f"{context}: keys[{index}] must be a (group_id, "
                f"group_member_index) tuple of non-negative ints, got {key!r}"
            )
        if key in result:
            raise WamRewardError(f"{context}: duplicate trajectory key {key}")
        result[key] = float(values[index])
    return result


def fill_reward_fields(
    samples: Sequence[WamRolloutSample],
    returns: Mapping[TrajectoryKey, float],
    advantages: Mapping[TrajectoryKey, float],
    *,
    config: WamBufferConfig,
    context: str = "fill_reward_fields",
) -> list[WamRolloutSample]:
    """Broadcast per-trajectory return/advantage onto buffer rows, validated.

    One trajectory is one rollout member, keyed ``(group_id,
    group_member_index)``.  Route A's return is terminal-sparse - a
    trajectory-level scalar - so EVERY chunk row of a member carries the same
    ``reward`` (its return; any per-chunk env reward previously stored is
    superseded) and the same ``advantage``.  The key sets of ``returns`` and
    ``advantages`` must equal the samples' trajectory set EXACTLY in both
    directions: an injected key that no sample carries (e.g. a foreign
    ``group_id``) is refused, as is a trajectory left without values.

    Validation is delegated to ``buffer_schema.serialize_samples`` - the full
    per-sample and buffer-level contract (group completeness, snapshot
    consistency, contiguous decisions) runs on the filled batch, so this
    function cannot hand the WS5 loss anything the buffer itself would refuse.

    Returns:
        New validated :class:`WamRolloutSample` list (inputs are never
        mutated - samples are frozen dataclasses).

    Raises:
        WamRewardError: On empty/foreign inputs, key-set mismatches, or
            non-finite values.
        WamBufferError: If the filled batch violates the buffer contract.
    """
    if not samples:
        raise WamRewardError(f"{context}: refusing to fill an empty sample list")
    for index, sample in enumerate(samples):
        if not isinstance(sample, WamRolloutSample):
            raise WamRewardError(
                f"{context}: samples[{index}] must be a WamRolloutSample, "
                f"got {type(sample).__name__}"
            )
    sample_keys = {(s.group_id, s.group_member_index) for s in samples}
    for name, mapping in (("returns", returns), ("advantages", advantages)):
        if not isinstance(mapping, Mapping):
            raise WamRewardError(
                f"{context}: {name} must be a mapping keyed by "
                f"(group_id, group_member_index), got {type(mapping).__name__}"
            )
        keys = set(mapping.keys())
        unknown = sorted(keys - sample_keys, key=repr)
        if unknown:
            raise WamRewardError(
                f"{context}: {name} contains trajectories absent from the "
                f"samples: {unknown}; refusing injected group ids"
            )
        missing = sorted(sample_keys - keys, key=repr)
        if missing:
            raise WamRewardError(
                f"{context}: {name} is missing trajectories present in the "
                f"samples: {missing}"
            )
        for key, value in mapping.items():
            _require_real(value, name=f"{name}[{key}]", context=context)
    filled = [
        replace(
            sample,
            reward=float(returns[(sample.group_id, sample.group_member_index)]),
            advantage=float(
                advantages[(sample.group_id, sample.group_member_index)]
            ),
        )
        for sample in samples
    ]
    # Reuse the buffer schema's own validators end-to-end; the payload is
    # discarded, only the fail-closed check matters here.
    serialize_samples(filled, config, context=f"{context}.validate")
    return filled
