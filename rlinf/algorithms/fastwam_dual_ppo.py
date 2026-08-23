# Copyright 2026 The RLinf Authors.
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

from dataclasses import dataclass

import torch

from rlinf.models.embodiment.wam_policy.contracts import WAMRoute


@dataclass(frozen=True, slots=True, kw_only=True)
class EpsilonMixtureBernoulli:
    """Exact epsilon-mixture Bernoulli behavior-policy quantities."""

    base_probability: torch.Tensor
    behavior_probability: torch.Tensor
    route: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class BranchCostReward:
    """Environment reward after applying one fixed cost per executed chunk."""

    rewards: torch.Tensor
    costs: torch.Tensor


def _as_float_tensor(
    value: float | torch.Tensor,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=reference.device)


def epsilon_mixture_bernoulli(
    logits: torch.Tensor,
    *,
    epsilon: float | torch.Tensor,
    temperature: float | torch.Tensor,
    route: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> EpsilonMixtureBernoulli:
    """Build or sample the Gate's exact epsilon-mixture behavior distribution.

    The Bernoulli event is ``WAMRoute.IDM``. When ``route`` is supplied, this
    function only evaluates that route. Otherwise it samples exactly once from
    the behavior probability.

    Args:
        logits: Unnormalized Gate logits.
        epsilon: Uniform-exploration mixture weight in ``[0, 1]``.
        temperature: Positive logit temperature.
        route: Optional route tensor containing only ``UNCOND``/``IDM`` ids.
        generator: Optional generator used when sampling a route.

    Returns:
        Base and behavior probabilities, selected route, exact log-probability,
        and behavior-distribution entropy. Floating outputs are float32.
    """

    if not logits.is_floating_point():
        raise TypeError(f"logits must use a floating dtype, got {logits.dtype}.")
    logits_float = logits.float()
    epsilon_tensor, temperature_tensor, logits_float = torch.broadcast_tensors(
        _as_float_tensor(epsilon, reference=logits),
        _as_float_tensor(temperature, reference=logits),
        logits_float,
    )
    if bool(
        (
            (~torch.isfinite(epsilon_tensor))
            | (epsilon_tensor < 0)
            | (epsilon_tensor > 1)
        )
        .any()
        .item()
    ):
        raise ValueError("epsilon must be finite and in [0, 1].")
    if bool(
        ((~torch.isfinite(temperature_tensor)) | (temperature_tensor <= 0)).any().item()
    ):
        raise ValueError("temperature must be finite and greater than zero.")

    base_probability = torch.sigmoid(logits_float / temperature_tensor)
    behavior_probability = (
        1.0 - epsilon_tensor
    ) * base_probability + 0.5 * epsilon_tensor

    if route is None:
        selected_route = torch.bernoulli(behavior_probability, generator=generator).to(
            torch.int64
        )
    else:
        selected_route = torch.broadcast_to(route, behavior_probability.shape)
        if selected_route.dtype == torch.bool:
            selected_route = selected_route.to(torch.int64)
        elif selected_route.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                f"route must use an integer or bool dtype, got {selected_route.dtype}."
            )
        invalid_route = (selected_route != int(WAMRoute.UNCOND)) & (
            selected_route != int(WAMRoute.IDM)
        )
        if bool(invalid_route.any().item()):
            raise ValueError("route contains a value outside WAMRoute.")
        selected_route = selected_route.to(torch.int64)

    selected_idm = selected_route == int(WAMRoute.IDM)
    logprob = torch.where(
        selected_idm,
        torch.log(behavior_probability),
        torch.log1p(-behavior_probability),
    )
    entropy = -(
        torch.xlogy(behavior_probability, behavior_probability)
        + torch.xlogy(1.0 - behavior_probability, 1.0 - behavior_probability)
    )
    return EpsilonMixtureBernoulli(
        base_probability=base_probability,
        behavior_probability=behavior_probability,
        route=selected_route,
        logprob=logprob,
        entropy=entropy,
    )


def _expand_trailing_dimensions(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    try:
        return torch.broadcast_to(value, target.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} cannot broadcast to "
            f"log-probability shape {tuple(target.shape)}."
        ) from error


def _squeeze_trailing_singletons(
    value: torch.Tensor,
    *,
    target_ndim: int,
    name: str,
) -> torch.Tensor:
    while value.ndim > target_ndim and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim > target_ndim:
        raise ValueError(
            f"{name} has non-singleton trailing dimensions incompatible with "
            f"a {target_ndim}-D target: {tuple(value.shape)}."
        )
    return value


def _zero_ppo_result(
    logprobs: torch.Tensor,
    *,
    prefix: str,
    selected_loss_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = logprobs.sum() * 0.0
    metric_zero = zero.detach()
    return zero, {
        f"{prefix}/policy_loss": metric_zero,
        f"{prefix}/total_loss": metric_zero,
        f"{prefix}/sample_count": metric_zero,
        f"{prefix}/ratio": metric_zero,
        f"{prefix}/ratio_abs": metric_zero,
        f"{prefix}/log_ratio_max_abs": metric_zero,
        f"{prefix}/approx_kl": metric_zero,
        f"{prefix}/clip_fraction": metric_zero,
        f"{prefix}/entropy": metric_zero,
        f"{prefix}/selected_loss_scale": (
            metric_zero if selected_loss_scale is None else selected_loss_scale.detach()
        ),
    }


_WEIGHTED_METRIC_GROUPS = {
    "gate": (
        "policy_loss",
        "total_loss",
        "ratio",
        "ratio_abs",
        "approx_kl",
        "clip_fraction",
        "entropy",
    ),
    "uncond_flow": (
        "policy_loss",
        "total_loss",
        "ratio",
        "ratio_abs",
        "approx_kl",
        "clip_fraction",
        "entropy",
    ),
    "base_uncond_kl": ("loss",),
}
_WEIGHTED_SUM_SUFFIX = "::weighted_sum"
_MAX_METRIC_KEYS = (
    "gate/log_ratio_max_abs",
    "uncond_flow/log_ratio_max_abs",
    "gate/preupdate_log_ratio_max_abs",
    "uncond_flow/preupdate_log_ratio_max_abs",
    "base_uncond_kl/max",
)


def pop_fastwam_weighted_metric_sums(
    metrics: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert microbatch means into additive numerators and denominators."""

    sums: dict[str, float] = {}
    maxima: dict[str, float] = {}
    for prefix, suffixes in _WEIGHTED_METRIC_GROUPS.items():
        count_key = f"{prefix}/sample_count"
        counts = [float(value) for value in metrics.pop(count_key, [])]
        if not counts:
            continue
        sums[count_key] = sum(counts)
        for suffix in suffixes:
            key = f"{prefix}/{suffix}"
            values = [float(value) for value in metrics.pop(key, [])]
            if len(values) != len(counts):
                raise ValueError(
                    f"Metric {key!r} has {len(values)} values for "
                    f"{len(counts)} sample counts."
                )
            sums[f"{key}{_WEIGHTED_SUM_SUFFIX}"] = sum(
                value * count for value, count in zip(values, counts)
            )
    for key in _MAX_METRIC_KEYS:
        max_values = metrics.pop(key, [])
        if max_values:
            maxima[key] = max(float(value) for value in max_values)
    return sums, maxima


def finalize_fastwam_weighted_metrics(
    reduced_sums: dict[str, float],
) -> dict[str, float]:
    """Recover globally weighted means after SUM-reducing the payload."""

    finalized: dict[str, float] = {}
    for prefix, suffixes in _WEIGHTED_METRIC_GROUPS.items():
        count_key = f"{prefix}/sample_count"
        if count_key not in reduced_sums:
            continue
        count = float(reduced_sums[count_key])
        finalized[count_key] = count
        for suffix in suffixes:
            key = f"{prefix}/{suffix}"
            numerator = float(reduced_sums[f"{key}{_WEIGHTED_SUM_SUFFIX}"])
            finalized[key] = numerator / count if count > 0 else 0.0
    return finalized


def _compute_masked_clipped_ppo_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    prefix: str,
    entropy: torch.Tensor | None = None,
    entropy_coefficient: float = 0.0,
    selected_loss_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logprobs.dtype != torch.float32:
        raise TypeError("logprobs must use torch.float32 for PPO stability.")
    if old_logprobs.dtype != torch.float32:
        raise TypeError("old_logprobs must use torch.float32 for PPO stability.")
    if advantages.dtype != torch.float32:
        raise TypeError("advantages must use torch.float32 for PPO stability.")
    if logprobs.shape != old_logprobs.shape:
        raise ValueError(
            "logprobs and old_logprobs must have the same shape, got "
            f"{tuple(logprobs.shape)} and {tuple(old_logprobs.shape)}."
        )
    if loss_mask.dtype != torch.bool:
        raise TypeError("loss_mask must use torch.bool.")
    if clip_ratio_low < 0 or clip_ratio_high < 0:
        raise ValueError("PPO clip ratios must be non-negative.")
    if entropy_coefficient < 0:
        raise ValueError("entropy_coefficient must be non-negative.")
    if selected_loss_scale is not None:
        selected_loss_scale = torch.as_tensor(
            selected_loss_scale,
            dtype=torch.float32,
            device=logprobs.device,
        )
        if selected_loss_scale.numel() != 1 or not bool(
            torch.isfinite(selected_loss_scale).item()
        ):
            raise ValueError("selected_loss_scale must be one finite scalar.")
        if bool((selected_loss_scale < 0).item()):
            raise ValueError("selected_loss_scale must be non-negative.")

    expanded_advantages = _expand_trailing_dimensions(
        advantages, logprobs, name="advantages"
    )
    expanded_mask = _expand_trailing_dimensions(loss_mask, logprobs, name="loss_mask")
    if not bool(expanded_mask.any().item()):
        return _zero_ppo_result(
            logprobs,
            prefix=prefix,
            selected_loss_scale=selected_loss_scale,
        )

    selected_logprobs = logprobs[expanded_mask]
    selected_old_logprobs = old_logprobs[expanded_mask]
    selected_advantages = expanded_advantages[expanded_mask]
    log_ratio = selected_logprobs - selected_old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    unclipped_objective = -selected_advantages * ratio
    clipped_objective = -selected_advantages * clipped_ratio
    selected_objective = torch.maximum(unclipped_objective, clipped_objective)
    policy_loss = (
        selected_objective.mean()
        if selected_loss_scale is None
        else selected_objective.sum() * selected_loss_scale
    )

    metric_policy_loss = selected_objective.detach().mean()
    entropy_mean = policy_loss.detach() * 0.0
    metric_entropy = metric_policy_loss * 0.0
    if entropy is not None:
        if not entropy.is_floating_point():
            raise TypeError("entropy must use a floating dtype.")
        expanded_entropy = _expand_trailing_dimensions(
            entropy, logprobs, name="entropy"
        )
        selected_entropy = expanded_entropy[expanded_mask].float()
        entropy_mean = (
            selected_entropy.mean()
            if selected_loss_scale is None
            else selected_entropy.sum() * selected_loss_scale
        )
        metric_entropy = selected_entropy.detach().mean()
    total_loss = policy_loss - entropy_coefficient * entropy_mean
    metric_total_loss = metric_policy_loss - entropy_coefficient * metric_entropy

    with torch.no_grad():
        clip_fraction = (
            ((ratio < 1.0 - clip_ratio_low) | (ratio > 1.0 + clip_ratio_high))
            .float()
            .mean()
        )
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        sample_count = torch.tensor(
            float(selected_logprobs.numel()),
            dtype=torch.float32,
            device=logprobs.device,
        )
    metrics = {
        f"{prefix}/policy_loss": metric_policy_loss,
        f"{prefix}/total_loss": metric_total_loss,
        f"{prefix}/sample_count": sample_count,
        f"{prefix}/ratio": ratio.detach().mean(),
        f"{prefix}/ratio_abs": (ratio.detach() - 1.0).abs().mean(),
        f"{prefix}/log_ratio_max_abs": log_ratio.detach().abs().max(),
        f"{prefix}/approx_kl": approx_kl.detach(),
        f"{prefix}/clip_fraction": clip_fraction.detach(),
        f"{prefix}/entropy": metric_entropy,
        f"{prefix}/selected_loss_scale": (
            torch.tensor(
                1.0 / float(selected_logprobs.numel()),
                dtype=torch.float32,
                device=logprobs.device,
            )
            if selected_loss_scale is None
            else selected_loss_scale.detach()
        ),
    }
    return total_loss, metrics


def compute_gate_ppo_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    behavior_probabilities: torch.Tensor | None = None,
    entropy_coefficient: float = 0.0,
    selected_loss_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the separately clipped delayed-Gate PPO loss."""

    entropy = None
    if behavior_probabilities is not None:
        probabilities = behavior_probabilities.float()
        entropy = -(
            torch.xlogy(probabilities, probabilities)
            + torch.xlogy(1.0 - probabilities, 1.0 - probabilities)
        )
    return _compute_masked_clipped_ppo_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        loss_mask=valid_mask,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        prefix="gate",
        entropy=entropy,
        entropy_coefficient=entropy_coefficient,
        selected_loss_scale=selected_loss_scale,
    )


def compute_uncond_flow_ppo_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    route_used: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    valid_mask: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    entropy_coefficient: float = 0.0,
    selected_loss_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute joint chunk-level Flow-SDE PPO only for executed UNCOND."""

    if route_used.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("route_used must use an integer dtype.")
    invalid_route = (route_used != int(WAMRoute.UNCOND)) & (
        route_used != int(WAMRoute.IDM)
    )
    if bool(invalid_route.any().item()):
        raise ValueError("route_used contains a value outside WAMRoute.")
    if logprobs.shape[: route_used.ndim] != route_used.shape:
        raise ValueError(
            "Flow log-probability leading dimensions must match route_used."
        )
    reduction_dims = tuple(range(route_used.ndim, logprobs.ndim))
    if reduction_dims:
        logprobs = logprobs.sum(dim=reduction_dims)
        old_logprobs = old_logprobs.sum(dim=reduction_dims)
    if entropy is not None and entropy.ndim > route_used.ndim:
        entropy = entropy.sum(dim=tuple(range(route_used.ndim, entropy.ndim)))
    advantages = _squeeze_trailing_singletons(
        advantages,
        target_ndim=route_used.ndim,
        name="flow advantages",
    )
    if valid_mask is not None:
        valid_mask = _squeeze_trailing_singletons(
            valid_mask,
            target_ndim=route_used.ndim,
            name="flow valid_mask",
        )
    route_mask = route_used == int(WAMRoute.UNCOND)
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool.")
        try:
            route_mask, valid_mask = torch.broadcast_tensors(route_mask, valid_mask)
        except RuntimeError as error:
            raise ValueError(
                "route_used and valid_mask must have broadcast-compatible shapes."
            ) from error
        route_mask = route_mask & valid_mask
    return _compute_masked_clipped_ppo_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        loss_mask=route_mask,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        prefix="uncond_flow",
        entropy=entropy,
        entropy_coefficient=entropy_coefficient,
        selected_loss_scale=selected_loss_scale,
    )


def compute_fastwam_dual_ppo_loss(
    *,
    gate_logprobs: torch.Tensor,
    gate_old_logprobs: torch.Tensor,
    gate_advantages: torch.Tensor,
    gate_valid_mask: torch.Tensor,
    gate_clip_ratio_low: float,
    gate_clip_ratio_high: float,
    flow_logprobs: torch.Tensor,
    flow_old_logprobs: torch.Tensor,
    flow_advantages: torch.Tensor,
    route_used: torch.Tensor,
    flow_clip_ratio_low: float,
    flow_clip_ratio_high: float,
    flow_valid_mask: torch.Tensor | None = None,
    gate_behavior_probabilities: torch.Tensor | None = None,
    gate_entropy_coefficient: float = 0.0,
    flow_entropy: torch.Tensor | None = None,
    flow_entropy_coefficient: float = 0.0,
    gate_loss_coefficient: float = 1.0,
    flow_loss_coefficient: float = 1.0,
    gate_selected_loss_scale: float | torch.Tensor | None = None,
    flow_selected_loss_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assemble independently clipped Gate and UNCOND Flow-SDE PPO losses."""

    if gate_loss_coefficient < 0 or flow_loss_coefficient < 0:
        raise ValueError("Policy-loss coefficients must be non-negative.")
    gate_loss, gate_metrics = compute_gate_ppo_loss(
        logprobs=gate_logprobs,
        old_logprobs=gate_old_logprobs,
        advantages=gate_advantages,
        valid_mask=gate_valid_mask,
        clip_ratio_low=gate_clip_ratio_low,
        clip_ratio_high=gate_clip_ratio_high,
        behavior_probabilities=gate_behavior_probabilities,
        entropy_coefficient=gate_entropy_coefficient,
        selected_loss_scale=gate_selected_loss_scale,
    )
    flow_loss, flow_metrics = compute_uncond_flow_ppo_loss(
        logprobs=flow_logprobs,
        old_logprobs=flow_old_logprobs,
        advantages=flow_advantages,
        route_used=route_used,
        clip_ratio_low=flow_clip_ratio_low,
        clip_ratio_high=flow_clip_ratio_high,
        valid_mask=flow_valid_mask,
        entropy=flow_entropy,
        entropy_coefficient=flow_entropy_coefficient,
        selected_loss_scale=flow_selected_loss_scale,
    )
    total_loss = gate_loss_coefficient * gate_loss + flow_loss_coefficient * flow_loss
    metrics = {
        **gate_metrics,
        **flow_metrics,
        "fastwam_dual/total_policy_loss": total_loss.detach(),
    }
    return total_loss, metrics


def compute_base_uncond_kl_loss(
    *,
    kl_values: torch.Tensor,
    route_used: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    selected_loss_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce exact equal-variance Gaussian KL over valid UNCOND chunks."""

    if kl_values.dtype != torch.float32:
        raise TypeError("UNCOND transition KL values must use torch.float32.")
    if bool((kl_values < 0).any().item()):
        raise ValueError("UNCOND transition KL values must be non-negative.")
    invalid_route = (route_used != int(WAMRoute.UNCOND)) & (
        route_used != int(WAMRoute.IDM)
    )
    if bool(invalid_route.any().item()):
        raise ValueError("route_used contains a value outside WAMRoute.")
    if kl_values.shape[: route_used.ndim] != route_used.shape:
        raise ValueError("Transition KL leading dimensions must match route_used.")
    reduction_dims = tuple(range(route_used.ndim, kl_values.ndim))
    if reduction_dims:
        kl_values = kl_values.sum(dim=reduction_dims)
    route_mask = route_used == int(WAMRoute.UNCOND)
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool.")
        try:
            route_mask, valid_mask = torch.broadcast_tensors(route_mask, valid_mask)
        except RuntimeError as error:
            raise ValueError(
                "route_used and valid_mask must have broadcast-compatible shapes."
            ) from error
        route_mask = route_mask & valid_mask
    expanded_mask = _expand_trailing_dimensions(
        route_mask, kl_values, name="base_uncond_kl_mask"
    )
    if not bool(expanded_mask.any().item()):
        zero = kl_values.sum() * 0.0
        return zero, {
            "base_uncond_kl/loss": zero.detach(),
            "base_uncond_kl/sample_count": zero.detach(),
            "base_uncond_kl/max": zero.detach(),
        }

    selected_kl = kl_values[expanded_mask]
    if selected_loss_scale is None:
        loss = selected_kl.mean()
    else:
        scale = torch.as_tensor(
            selected_loss_scale,
            dtype=torch.float32,
            device=kl_values.device,
        )
        if scale.numel() != 1 or not bool(torch.isfinite(scale).item()):
            raise ValueError("selected_loss_scale must be one finite scalar.")
        if bool((scale < 0).item()):
            raise ValueError("selected_loss_scale must be non-negative.")
        loss = selected_kl.sum() * scale
    return loss, {
        "base_uncond_kl/loss": selected_kl.detach().mean(),
        "base_uncond_kl/sample_count": torch.tensor(
            float(selected_kl.numel()),
            dtype=torch.float32,
            device=kl_values.device,
        ),
        "base_uncond_kl/max": selected_kl.detach().max(),
    }


def compute_gate_collapse_penalty(
    *,
    base_idm_probabilities: torch.Tensor,
    episode_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_calls: float,
    scope: str = "episode",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the optional exponential expected-call anti-collapse surrogate."""

    if tau_calls <= 0:
        raise ValueError("Collapse-penalty `tau_calls` must be positive.")
    if scope not in {"episode", "microbatch"}:
        raise ValueError("Collapse-penalty scope must be `episode` or `microbatch`.")
    if not base_idm_probabilities.is_floating_point():
        raise TypeError("Gate base probabilities must use a floating dtype.")
    if episode_ids.shape != base_idm_probabilities.shape:
        raise ValueError("episode_ids must match Gate base probabilities.")
    if (
        valid_mask.shape != base_idm_probabilities.shape
        or valid_mask.dtype != torch.bool
    ):
        raise ValueError("valid_mask must be bool and match Gate base probabilities.")
    probabilities = base_idm_probabilities.float()
    if bool(((probabilities < 0) | (probabilities > 1)).any().item()):
        raise ValueError("Gate base probabilities must lie in [0, 1].")
    if not bool(valid_mask.any().item()):
        zero = probabilities.sum() * 0.0
        return zero, {
            "collapse/loss": zero.detach(),
            "collapse/group_count": zero.detach(),
            "collapse/expected_idm_calls": zero.detach(),
            "collapse/expected_uncond_calls": zero.detach(),
        }

    selected_probabilities = probabilities[valid_mask]
    selected_episodes = episode_ids[valid_mask]
    groups = (
        [torch.ones_like(selected_episodes, dtype=torch.bool)]
        if scope == "microbatch"
        else [
            selected_episodes == episode for episode in torch.unique(selected_episodes)
        ]
    )
    idm_calls = torch.stack([selected_probabilities[group].sum() for group in groups])
    uncond_calls = torch.stack(
        [(1.0 - selected_probabilities[group]).sum() for group in groups]
    )
    penalty = (
        torch.exp(-idm_calls / tau_calls) + torch.exp(-uncond_calls / tau_calls)
    ).mean()
    return penalty, {
        "collapse/loss": penalty.detach(),
        "collapse/group_count": torch.tensor(
            float(len(groups)), dtype=torch.float32, device=probabilities.device
        ),
        "collapse/expected_idm_calls": idm_calls.detach().mean(),
        "collapse/expected_uncond_calls": uncond_calls.detach().mean(),
    }


def apply_fixed_branch_cost(
    *,
    environment_rewards: torch.Tensor,
    route_used: torch.Tensor,
    idm_cost: float,
    uncond_cost: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> BranchCostReward:
    """Subtract one fixed branch cost from each valid chunk reward.

    Rewards must already be aggregated to chunk level. A single trailing
    singleton dimension is accepted; primitive-step reward vectors are rejected
    so the branch cost cannot accidentally be charged more than once.
    """

    if not environment_rewards.is_floating_point():
        raise TypeError("environment_rewards must use a floating dtype.")
    if route_used.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("route_used must use an integer dtype.")
    if idm_cost < 0 or uncond_cost < 0:
        raise ValueError("Branch costs must be non-negative.")
    invalid_route = (route_used != int(WAMRoute.UNCOND)) & (
        route_used != int(WAMRoute.IDM)
    )
    if bool(invalid_route.any().item()):
        raise ValueError("route_used contains a value outside WAMRoute.")

    if environment_rewards.shape == route_used.shape:
        expanded_route = route_used
    elif environment_rewards.shape == (*route_used.shape, 1):
        expanded_route = route_used.unsqueeze(-1)
    else:
        raise ValueError(
            "environment_rewards must be chunk-level with shape route_used.shape "
            "or route_used.shape + (1,); got "
            f"{tuple(environment_rewards.shape)} and {tuple(route_used.shape)}."
        )

    costs = torch.where(
        expanded_route == int(WAMRoute.IDM),
        torch.as_tensor(
            idm_cost,
            dtype=environment_rewards.dtype,
            device=environment_rewards.device,
        ),
        torch.as_tensor(
            uncond_cost,
            dtype=environment_rewards.dtype,
            device=environment_rewards.device,
        ),
    )
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool.")
        if valid_mask.shape == route_used.shape and costs.ndim > valid_mask.ndim:
            valid_mask = valid_mask.unsqueeze(-1)
        if valid_mask.shape != costs.shape:
            raise ValueError(
                "valid_mask must match route_used (with an optional trailing "
                "singleton reward dimension)."
            )
        costs = torch.where(valid_mask, costs, torch.zeros_like(costs))
    return BranchCostReward(
        rewards=environment_rewards - costs,
        costs=costs,
    )


__all__ = [
    "BranchCostReward",
    "EpsilonMixtureBernoulli",
    "apply_fixed_branch_cost",
    "compute_base_uncond_kl_loss",
    "compute_fastwam_dual_ppo_loss",
    "compute_gate_collapse_penalty",
    "compute_gate_ppo_loss",
    "compute_uncond_flow_ppo_loss",
    "epsilon_mixture_bernoulli",
]
