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

"""Q-gradient velocity guidance for the frozen UNCOND sampler (route 1.3 / B-guidance).

Stage-2 W18 mechanism layer: bend the sampling ODE of the FROZEN action expert
with the gradient of a critic, touching no weights.  This module never imports
fastwam — it mirrors the merged W8 sampler contract exactly:

* ``FastWAM.infer_action`` accepts ``velocity_hook(x_t, v, timestep,
  step_index) -> delta_v`` and applies ``pred_action = v + delta_v`` BEFORE
  the scheduler step.  ``x_t`` is the pre-step action latent, ``v`` the
  model's raw velocity prediction, ``timestep`` the exact element of the
  inference timestep grid, ``step_index`` counts from 0 and the hook fires
  once per solver step (``num_inference_steps`` times total).
* The scheduler is continuous-time rectified flow
  (``FastWAM .../schedulers/scheduler_continuous.py:63-88``): noising is
  ``x_sigma = (1-sigma)*x0 + sigma*eps`` with regression target
  ``v = eps - x0``; timesteps are parameterized ``t = sigma *
  num_train_timesteps`` (1000 by default, NOT unit-interval sigma); the
  inference grid is ``sigma_k = phi(u_k, shift)`` over ``u = linspace(1, 0)``
  so the per-step deltas ``delta_k = sigma_{k+1} - sigma_k`` are NEGATIVE and
  the Euler update is ``x <- x + v * delta``.
* The call site runs under ``@torch.no_grad()``; a hook that needs gradients
  for its own critic must open a local ``torch.enable_grad()`` block.  The
  caller detaches the returned delta and fail-closes on any
  shape/dtype/device mismatch with ``v``.

Sign derivation (the line everything hinges on).  The guided Euler update is::

    x_{k+1} = x_k + (v_k + delta_v_k) * delta_sigma_k,    delta_sigma_k < 0.

With ``delta_v = -lambda * grad_{x0_hat} Q`` the guidance contribution to the
new latent is ``(-lambda * grad Q) * delta_sigma = +lambda * |delta_sigma| *
grad Q`` — a gradient ASCENT step on Q, precisely BECAUSE ``delta_sigma`` is
negative (sigma runs 1 -> 0).  A ``+lambda`` sign would descend Q.  The
``LinearProbeCritic`` sign pre-flight in ``test_flow_guidance.py`` pins this
against a synthetic Euler loop replicating the scheduler math (ROUTES section 4:
verify the sign with ``Q = <a, u>`` before ever mounting a learned critic).

Gradient discipline (ROUTES section 4, both decisive implementation choices):

* ``x0_hat = x_t - sigma * v`` is formed from the hook arguments alone (no
  extra model forward) and DETACHED into a fresh leaf
  (``detach().requires_grad_(True)``).  The critic is differentiated w.r.t.
  that leaf ONLY — never through ``v_theta`` or the caller's graph, which
  would cost a full action-expert backward per step and eat the UNCOND cost
  advantage the project exists to exploit.  Note ``grad_{x_sigma} Q(x0_hat)
  = grad_{x0_hat} Q`` at fixed ``v`` since ``d x0_hat / d x_sigma = I``, so
  differentiating at the leaf is exactly the ROUTES formula.
* Guidance runs only in the last ``guide_last_k`` solver steps: ``x0_hat`` is
  only accurate at small sigma, and that is also when it actually determines
  the final action.

Trust region: the ROUTES trust-region term ``lambda_tr * ||x0_hat -
x0_hat_base||`` needs the UNGUIDED clean prediction ``x0_hat_base``, which a
single per-step hook cannot see (it would require a parallel unguided solve).
``max_delta_norm`` is the implementable proxy: a per-sample L2 clamp on the
velocity correction, bounding how far one step can push the trajectory off the
frozen flow.

Per OQ7 nothing is hard-coded: window size, solver length, schedule, clamp,
and the timestep parameterization are all :class:`GuidanceConfig` fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

__all__ = [
    "GuidanceConfig",
    "LinearProbeCritic",
    "constant_lambda_after_threshold",
    "make_q_guidance_hook",
    "predicted_clean_action",
]

#: ``lambda(sigma) -> float`` guidance-strength schedule over unit-interval sigma.
LambdaSchedule = Callable[[float], float]

#: Per-sample critic: ``critic(a_chunk [B, T, D] float32) -> [B]`` (or ``[B, 1]``).
#: State conditioning is closed over by the caller — the hook only ever hands
#: the critic a candidate clean action chunk.
Critic = Callable[[torch.Tensor], torch.Tensor]

_NORM_FLOOR = 1e-12


def predicted_clean_action(
    x_t: torch.Tensor, v: torch.Tensor, sigma: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Closed-form clean-action estimate ``x0_hat = x_t - sigma * v``.

    Derivation: rectified flow noises ``x_sigma = (1-sigma)*x0 + sigma*eps``
    and regresses the velocity ``v = eps - x0``.  Substituting
    ``eps = v + x0`` gives ``x_sigma = x0 + sigma*v``, hence
    ``x0 = x_sigma - sigma*v`` exactly — no extra model forward is needed to
    read the current clean-action estimate out of a solver step.

    Args:
        x_t: Noisy action latents ``[B, T, D]`` (any shape; must match ``v``).
        v: Predicted velocity, same shape as ``x_t``.
        sigma: UNIT-INTERVAL noise level in [0, 1]; python float or tensor
            with 1 or batch elements.  Scheduler timesteps are parameterized
            ``t = sigma * num_train_timesteps`` — divide by
            ``num_train_timesteps`` first; passing a raw timestep (e.g. 800.0)
            here is a unit error and fails closed.
    """
    for name, value in (("x_t", x_t), ("v", v)):
        if not torch.is_tensor(value):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value).__name__}")
    if x_t.shape != v.shape:
        raise ValueError(
            f"`x_t` and `v` shapes must match, got {tuple(x_t.shape)} vs {tuple(v.shape)}"
        )
    if isinstance(sigma, (int, float)) and not isinstance(sigma, bool):
        sigma = torch.tensor(float(sigma), dtype=torch.float32)
    if not torch.is_tensor(sigma):
        raise TypeError(
            f"`sigma` must be a float or torch.Tensor, got {type(sigma).__name__}"
        )
    if sigma.numel() not in (1, x_t.shape[0] if x_t.ndim > 0 else 1):
        raise ValueError(
            f"`sigma` must have 1 or batch({x_t.shape[0]}) elements, got {sigma.numel()}"
        )
    sigma_f32 = sigma.detach().to(torch.float32)
    if bool((sigma_f32 < 0.0).any()) or bool((sigma_f32 > 1.0).any()):
        raise ValueError(
            "`sigma` must lie in [0, 1], got range "
            f"[{float(sigma_f32.min())}, {float(sigma_f32.max())}]. Scheduler "
            "timesteps are parameterized t = sigma * num_train_timesteps — divide "
            "by num_train_timesteps before calling predicted_clean_action()."
        )
    sigma = sigma.to(device=x_t.device, dtype=x_t.dtype)
    if sigma.numel() == 1:
        return x_t - sigma.reshape(()) * v
    return x_t - sigma.view(-1, *([1] * (x_t.ndim - 1))) * v


def constant_lambda_after_threshold(value: float, sigma_threshold: float) -> LambdaSchedule:
    """Default (tunable) schedule: ``lambda(sigma) = value`` once ``sigma <=
    sigma_threshold``, else 0.

    Sigma DECREASES over solver steps, so "after the threshold" means the late,
    small-sigma steps where ``x0_hat`` is accurate.  Both knobs are explicit —
    there is no baked-in strength or threshold (OQ7).
    """
    value = float(value)
    sigma_threshold = float(sigma_threshold)
    if not math.isfinite(value):
        raise ValueError(f"`value` must be finite, got {value}")
    if not (0.0 <= sigma_threshold <= 1.0):
        raise ValueError(f"`sigma_threshold` must lie in [0, 1], got {sigma_threshold}")

    def schedule(sigma: float) -> float:
        return value if sigma <= sigma_threshold else 0.0

    return schedule


@dataclass(frozen=True)
class GuidanceConfig:
    """Everything the hook needs, with no hard-coded window/solver constants.

    Attributes:
        lambda_schedule: ``sigma -> float`` guidance strength (unit-interval
            sigma).  Build the default constant-after-threshold shape with
            :func:`constant_lambda_after_threshold`.
        guide_last_k: Guide only the final ``k`` of ``num_inference_steps``
            solver steps.  ``0`` disables guidance entirely (the hook becomes
            an exact-zero passthrough).
        num_inference_steps: The solver length the hook will be mounted on;
            the guided window is ``step_index >= num_inference_steps -
            guide_last_k``.  A ``step_index`` outside ``[0,
            num_inference_steps)`` fails closed — it means the config and the
            actual solver disagree.
        max_delta_norm: Optional per-sample L2 clamp on the returned
            ``delta_v`` — the trust-region proxy (see module docstring: the
            ROUTES ``||x0_hat - x0_hat_base||`` term needs the unguided
            trajectory, unavailable inside a single hook).  ``None`` disables
            the clamp.
        num_train_timesteps: The scheduler's timestep parameterization
            constant (``t = sigma * num_train_timesteps``); FastWAM default
            1000.
    """

    lambda_schedule: LambdaSchedule
    guide_last_k: int
    num_inference_steps: int
    max_delta_norm: Optional[float]
    num_train_timesteps: int = 1000

    def __post_init__(self) -> None:
        if not callable(self.lambda_schedule):
            raise TypeError(
                "`lambda_schedule` must be callable (sigma -> float), got "
                f"{type(self.lambda_schedule).__name__}"
            )
        for name in ("guide_last_k", "num_inference_steps", "num_train_timesteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an int, got {type(value).__name__}")
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` must be positive, got {self.num_inference_steps}"
            )
        if not (0 <= self.guide_last_k <= self.num_inference_steps):
            raise ValueError(
                f"`guide_last_k` must lie in [0, num_inference_steps="
                f"{self.num_inference_steps}], got {self.guide_last_k}"
            )
        if self.num_train_timesteps <= 0:
            raise ValueError(
                f"`num_train_timesteps` must be positive, got {self.num_train_timesteps}"
            )
        if self.max_delta_norm is not None:
            if isinstance(self.max_delta_norm, bool) or not isinstance(
                self.max_delta_norm, (int, float)
            ):
                raise TypeError(
                    "`max_delta_norm` must be a float or None, got "
                    f"{type(self.max_delta_norm).__name__}"
                )
            if not math.isfinite(self.max_delta_norm) or self.max_delta_norm <= 0.0:
                raise ValueError(
                    "`max_delta_norm` must be a finite positive float or None, got "
                    f"{self.max_delta_norm}"
                )


class LinearProbeCritic(torch.nn.Module):
    """Trivial linear critic ``Q(a) = <a, u>`` for sign calibration.

    ``grad_a Q = u`` everywhere, so a correctly signed guidance hook must push
    the final action measurably toward ``+u``.  ROUTES section 4 mandates
    running this probe BEFORE mounting a learned critic: a flipped sign
    produces plausible-looking but exactly inverted behaviour.  ``u`` is a
    parameter so tests can also assert the hook never leaks gradient into
    critic weights.
    """

    def __init__(self, u: torch.Tensor):
        super().__init__()
        if not torch.is_tensor(u) or not torch.is_floating_point(u):
            raise TypeError("`u` must be a floating torch.Tensor")
        self.u = torch.nn.Parameter(u.detach().clone().to(torch.float32))

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(action_chunk):
            raise TypeError(
                f"`action_chunk` must be a torch.Tensor, got {type(action_chunk).__name__}"
            )
        if action_chunk.ndim != self.u.ndim + 1 or action_chunk.shape[1:] != self.u.shape:
            raise ValueError(
                f"`action_chunk` must be [B, *{tuple(self.u.shape)}], got "
                f"{tuple(action_chunk.shape)}"
            )
        batch = action_chunk.shape[0]
        return (action_chunk.reshape(batch, -1) * self.u.reshape(1, -1)).sum(dim=1)


def _validate_hook_inputs(
    x_t: torch.Tensor, v: torch.Tensor, timestep: torch.Tensor, step_index: int, config: GuidanceConfig
) -> None:
    for name, value in (("x_t", x_t), ("v", v)):
        if not torch.is_tensor(value):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value).__name__}")
        if not torch.is_floating_point(value):
            raise ValueError(f"`{name}` must be a floating tensor, got dtype {value.dtype}")
    if x_t.shape != v.shape:
        raise ValueError(
            f"`x_t` shape {tuple(x_t.shape)} does not match `v` shape {tuple(v.shape)}"
        )
    if x_t.ndim < 2:
        raise ValueError(
            "`x_t`/`v` must be batched [B, ...] tensors (FastWAM passes [1, T, D]), "
            f"got ndim={x_t.ndim}"
        )
    if x_t.dtype != v.dtype or x_t.device != v.device:
        raise ValueError(
            f"`x_t` dtype/device ({x_t.dtype}, {x_t.device}) must match `v` "
            f"({v.dtype}, {v.device})"
        )
    if not torch.is_tensor(timestep):
        raise TypeError(f"`timestep` must be a torch.Tensor, got {type(timestep).__name__}")
    if timestep.numel() != 1:
        raise ValueError(
            "`timestep` must be one element of the inference timestep grid, got "
            f"{timestep.numel()} elements"
        )
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise TypeError(f"`step_index` must be an int, got {type(step_index).__name__}")
    if not (0 <= step_index < config.num_inference_steps):
        raise ValueError(
            f"`step_index` {step_index} outside [0, num_inference_steps="
            f"{config.num_inference_steps}): the mounted solver disagrees with "
            "GuidanceConfig.num_inference_steps"
        )


def _sigma_from_timestep(timestep: torch.Tensor, num_train_timesteps: int) -> float:
    sigma = float(timestep.detach().reshape(()).to(torch.float32)) / float(num_train_timesteps)
    if not (0.0 <= sigma <= 1.0):
        raise ValueError(
            f"timestep {float(timestep.detach().reshape(()))} implies sigma={sigma} "
            f"outside [0, 1] under num_train_timesteps={num_train_timesteps}; the "
            "scheduler parameterization is t = sigma * num_train_timesteps — fix "
            "GuidanceConfig.num_train_timesteps"
        )
    return sigma


def _critic_grad_wrt_clean_action(
    critic: Critic, x_t: torch.Tensor, v: torch.Tensor, sigma: float
) -> torch.Tensor:
    """``grad_{x0_hat} sum(Q)`` at a detached float32 leaf; per-sample rows."""
    with torch.enable_grad():
        x0_hat = predicted_clean_action(
            x_t.detach().to(torch.float32), v.detach().to(torch.float32), sigma
        )
        x0_leaf = x0_hat.detach().requires_grad_(True)
        q = critic(x0_leaf)
        if not torch.is_tensor(q) or not torch.is_floating_point(q):
            raise TypeError(
                "critic must return a floating torch.Tensor, got "
                f"{type(q).__name__ if not torch.is_tensor(q) else q.dtype}"
            )
        batch = x0_leaf.shape[0]
        if q.ndim not in (1, 2) or q.shape[0] != batch or q.numel() != batch:
            raise ValueError(
                f"critic must return one scalar per sample [B={batch}] (or [B, 1]), "
                f"got shape {tuple(q.shape)}"
            )
        if not q.requires_grad:
            # A critic that detaches its input (or returns constants) produces
            # a graph-free output; autograd.grad would raise an opaque
            # RuntimeError, so fail closed with the actual diagnosis.
            grad = None
        else:
            # Per-sample critic => sum() is additive across samples, so the
            # grad of the sum recovers each sample's own gradient.
            # torch.autograd.grad w.r.t. the leaf ONLY: no .grad accumulation
            # on critic parameters and no path back into v or the caller's
            # graph.
            (grad,) = torch.autograd.grad(q.sum(), x0_leaf, allow_unused=True)
    if grad is None:
        raise ValueError(
            "critic output does not depend on its action input (gradient is None); "
            "a critic that detaches its input cannot steer the sampler"
        )
    return grad.detach()


def _clamp_per_sample_norm(delta: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Scale each sample's delta so its L2 norm is at most ``max_norm``."""
    batch = delta.shape[0]
    flat = delta.reshape(batch, -1)
    norms = flat.norm(dim=1, keepdim=True)
    scale = (max_norm / norms.clamp_min(_NORM_FLOOR)).clamp(max=1.0)
    return (flat * scale).reshape(delta.shape)


def make_q_guidance_hook(critic: Critic, config: GuidanceConfig):
    """Build a W8 velocity hook applying ``delta_v = -lambda(sigma) * grad Q``.

    ``critic`` maps a float32 clean-action chunk ``[B, T, D]`` to one scalar
    per sample; any state conditioning is closed over by the caller.  The
    returned hook matches ``hook(x_t, v, timestep, step_index) -> delta_v``
    exactly and is total: outside the guided window (or at ``lambda == 0``) it
    returns an EXACT zero tensor shaped/typed like ``v`` — never ``None`` —
    and does not evaluate the critic at all (the route's cost claim is
    ``guide_last_k`` critic forward/backward passes, nothing more).  It never
    mutates ``x_t``/``v``, opens its own local ``torch.enable_grad()`` (the
    FastWAM call site is ``@torch.no_grad()``), and returns a detached tensor
    on ``v``'s dtype/device.
    """
    if not callable(critic):
        raise TypeError(f"`critic` must be callable, got {type(critic).__name__}")
    if not isinstance(config, GuidanceConfig):
        raise TypeError(f"`config` must be a GuidanceConfig, got {type(config).__name__}")

    first_guided_step = config.num_inference_steps - config.guide_last_k

    def hook(
        x_t: torch.Tensor, v: torch.Tensor, timestep: torch.Tensor, step_index: int
    ) -> torch.Tensor:
        _validate_hook_inputs(x_t, v, timestep, step_index, config)
        if step_index < first_guided_step:
            return torch.zeros_like(v)
        sigma = _sigma_from_timestep(timestep, config.num_train_timesteps)
        lam = float(config.lambda_schedule(sigma))
        if not math.isfinite(lam):
            raise ValueError(f"lambda_schedule({sigma}) returned non-finite {lam}")
        if lam == 0.0:
            return torch.zeros_like(v)
        grad = _critic_grad_wrt_clean_action(critic, x_t, v, sigma)
        # Sign: the Euler step is x <- x + (v + delta_v) * delta_sigma with
        # delta_sigma < 0, so -lambda*grad moves the latent by
        # +lambda*|delta_sigma|*grad — ASCENT on Q (module docstring).
        delta = (-lam) * grad
        if config.max_delta_norm is not None:
            delta = _clamp_per_sample_norm(delta, float(config.max_delta_norm))
        return delta.detach().to(dtype=v.dtype, device=v.device)

    return hook
