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

"""Contract tests for the W18 Q-gradient guidance hook (route 1.3 / B-guidance).

Deliberately data-free and fastwam-free: the FastWAM sampler contract is
replicated by a tiny local Euler loop whose schedule math is copied from
``FastWAM .../schedulers/scheduler_continuous.py:63-88`` (shift map ``phi``,
``u = linspace(1, 0)``, NEGATIVE deltas ``sigma[1:] - sigma[:-1]``, update
``x <- x + v * delta``) and whose hook handling mirrors
``FastWAM.infer_action``: called once per solver step under ``torch.no_grad()``
with ``(x_t, v, timestep, step_index)``, delta validated against ``v``'s
shape/dtype/device, detached, and added BEFORE the scheduler step.

The decisive test is the sign pre-flight with ``LinearProbeCritic`` (Q = <a, u>):
guided minus unguided final actions must project POSITIVELY onto u, and the
projection must grow with lambda.
"""

import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _namespace(name: str, path: Path) -> None:
    """Fabricate an empty namespace package so heavy __init__ files never run.

    Same mechanism as test_snapshot_group_sampler: rlinf/__init__ and
    rlinf/models/__init__ pull omegaconf/torch-heavy registration at import
    time, which these unit tests do not need.
    """
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules.setdefault(name, module)


_namespace("rlinf", ROOT / "rlinf")
_namespace("rlinf.models", ROOT / "rlinf" / "models")
_namespace("rlinf.models.embodiment", ROOT / "rlinf" / "models" / "embodiment")
_namespace(
    "rlinf.models.embodiment.wam_policy",
    ROOT / "rlinf" / "models" / "embodiment" / "wam_policy",
)

flow_guidance = importlib.import_module(
    "rlinf.models.embodiment.wam_policy.steering.flow_guidance"
)
steering_pkg = importlib.import_module("rlinf.models.embodiment.wam_policy.steering")

GuidanceConfig = flow_guidance.GuidanceConfig
LinearProbeCritic = flow_guidance.LinearProbeCritic
constant_lambda_after_threshold = flow_guidance.constant_lambda_after_threshold
make_q_guidance_hook = flow_guidance.make_q_guidance_hook
predicted_clean_action = flow_guidance.predicted_clean_action

BATCH, HORIZON, ACT_DIM = 3, 4, 2
N_TRAIN = 1000
SHIFT = 5.0
STEPS = 8


# --------------------------------------------------------------------------- #
# Synthetic sampler: scheduler math copied from scheduler_continuous.py:63-88
# --------------------------------------------------------------------------- #
def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * u / (1.0 + (shift - 1.0) * u)


def _build_schedule(num_steps: int, shift: float = SHIFT, num_train: int = N_TRAIN):
    u_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    sigma_steps = _phi(u_steps, shift)
    timesteps = sigma_steps[:-1] * float(num_train)
    deltas = sigma_steps[1:] - sigma_steps[:-1]
    return timesteps, deltas


def _base_velocity(x: torch.Tensor, sigma: float, x_star: torch.Tensor) -> torch.Tensor:
    """Smooth pull toward a fixed clean target; well-behaved at every sigma."""
    return (x - x_star) / (sigma + 0.25)


def _euler_rollout(
    x_init: torch.Tensor,
    x_star: torch.Tensor,
    hook,
    num_steps: int = STEPS,
    shift: float = SHIFT,
    record: list | None = None,
) -> torch.Tensor:
    """Replicates the FastWAM.infer_action solver semantics around the hook.

    Runs under torch.no_grad() (the real call site is @torch.no_grad()),
    validates the delta against v like fastwam._validate_velocity_delta, and
    adds it BEFORE the scheduler step ``x <- x + v * delta_sigma``.
    """
    timesteps, deltas = _build_schedule(num_steps, shift=shift)
    with torch.no_grad():
        x = x_init.clone()
        for step_index, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            sigma = float(step_t) / N_TRAIN
            v = _base_velocity(x, sigma, x_star)
            if hook is not None:
                delta_v = hook(x, v, step_t, step_index)
                assert torch.is_tensor(delta_v), "hook must return a tensor, never None"
                assert delta_v.shape == v.shape
                assert delta_v.dtype == v.dtype and delta_v.device == v.device
                v = v + delta_v.detach()
            x = x + v * step_delta
            if record is not None:
                record.append(x.clone())
    return x


def _probe_u(seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn((HORIZON, ACT_DIM), generator=gen)
    return u / u.norm()


def _config(**overrides) -> GuidanceConfig:
    base = dict(
        lambda_schedule=constant_lambda_after_threshold(0.5, 1.0),
        guide_last_k=4,
        num_inference_steps=STEPS,
        max_delta_norm=None,
        num_train_timesteps=N_TRAIN,
    )
    base.update(overrides)
    return GuidanceConfig(**base)


def _inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x_init = torch.randn((BATCH, HORIZON, ACT_DIM), generator=gen)
    x_star = torch.randn((BATCH, HORIZON, ACT_DIM), generator=gen)
    return x_init, x_star


def _projection_onto_u(diff: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (diff.reshape(diff.shape[0], -1) * u.reshape(1, -1)).sum(dim=1)


class _CountingCritic(LinearProbeCritic):
    def __init__(self, u: torch.Tensor):
        super().__init__(u)
        self.calls = 0

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(action_chunk)


# --------------------------------------------------------------------------- #
# 1. Sign sanity — THE decisive test (ROUTES section 4 pre-flight)
# --------------------------------------------------------------------------- #
class TestSignSanity:
    @pytest.mark.parametrize("seed", range(5))
    def test_guided_final_moves_toward_plus_u(self, seed):
        u = _probe_u()
        hook = make_q_guidance_hook(LinearProbeCritic(u), _config())
        x_init, x_star = _inputs(seed)
        unguided = _euler_rollout(x_init, x_star, None)
        guided = _euler_rollout(x_init, x_star, hook)
        proj = _projection_onto_u(guided - unguided, u)
        assert bool((proj > 0).all()), f"seed {seed}: projections {proj.tolist()}"

    @pytest.mark.parametrize("shift", (3.0, 5.0))
    def test_sign_holds_across_schedule_shifts(self, shift):
        u = _probe_u()
        hook = make_q_guidance_hook(LinearProbeCritic(u), _config())
        x_init, x_star = _inputs(11)
        unguided = _euler_rollout(x_init, x_star, None, shift=shift)
        guided = _euler_rollout(x_init, x_star, hook, shift=shift)
        proj = _projection_onto_u(guided - unguided, u)
        assert bool((proj > 0).all()), proj.tolist()

    def test_projection_grows_with_lambda(self):
        u = _probe_u()
        x_init, x_star = _inputs(3)
        unguided = _euler_rollout(x_init, x_star, None)
        means = []
        for lam in (0.25, 1.0, 4.0):
            hook = make_q_guidance_hook(
                LinearProbeCritic(u),
                _config(lambda_schedule=constant_lambda_after_threshold(lam, 1.0)),
            )
            guided = _euler_rollout(x_init, x_star, hook)
            means.append(float(_projection_onto_u(guided - unguided, u).mean()))
        assert means[0] > 0.0
        assert means[0] < means[1] < means[2], means

    def test_linear_probe_delta_is_minus_lambda_u(self):
        # For Q = <a, u>, grad_a Q = u exactly, so the raw (unclamped) hook
        # output must be -lambda * u broadcast over the batch.
        u = _probe_u()
        lam = 0.7
        hook = make_q_guidance_hook(
            LinearProbeCritic(u),
            _config(lambda_schedule=constant_lambda_after_threshold(lam, 1.0)),
        )
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        delta = hook(x, v, timesteps[STEPS - 1], STEPS - 1)
        expected = (-lam) * u.expand(BATCH, HORIZON, ACT_DIM)
        torch.testing.assert_close(delta, expected, atol=1e-6, rtol=0.0)


# --------------------------------------------------------------------------- #
# 2. Window semantics
# --------------------------------------------------------------------------- #
class TestWindow:
    def test_called_every_step_zero_outside_last_k(self):
        critic = _CountingCritic(_probe_u())
        inner = make_q_guidance_hook(critic, _config(guide_last_k=3))
        seen = []

        def spy(x_t, v, timestep, step_index):
            delta = inner(x_t, v, timestep, step_index)
            seen.append((step_index, delta.clone(), v.clone()))
            return delta

        x_init, x_star = _inputs(1)
        _euler_rollout(x_init, x_star, spy)
        assert len(seen) == STEPS
        assert [s[0] for s in seen] == list(range(STEPS))
        for step_index, delta, v in seen:
            if step_index < STEPS - 3:
                assert torch.equal(delta, torch.zeros_like(v)), step_index
            else:
                assert bool(delta.abs().sum() > 0), step_index
        # Cost contract: the critic runs ONLY in the guided window.
        assert critic.calls == 3

    def test_guide_last_k_equal_to_steps_guides_every_step(self):
        critic = _CountingCritic(_probe_u())
        hook = make_q_guidance_hook(critic, _config(guide_last_k=STEPS))
        x_init, x_star = _inputs(2)
        _euler_rollout(x_init, x_star, hook)
        assert critic.calls == STEPS

    def test_out_of_window_result_is_a_real_zero_tensor(self):
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config(guide_last_k=1))
        timesteps, _ = _build_schedule(STEPS)
        v = torch.randn(BATCH, HORIZON, ACT_DIM, dtype=torch.float64)
        out = hook(torch.randn_like(v), v, timesteps[0], 0)
        assert torch.is_tensor(out)
        assert out.dtype == v.dtype and out.device == v.device
        assert torch.equal(out, torch.zeros_like(v))


# --------------------------------------------------------------------------- #
# 3. Graph hygiene
# --------------------------------------------------------------------------- #
class TestGraphHygiene:
    def test_runs_entirely_under_no_grad_and_output_is_detached(self):
        # _euler_rollout already wraps everything in torch.no_grad(); assert
        # explicitly the loop completes and nothing carries grad state.
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config())
        x_init, x_star = _inputs(4)
        final = _euler_rollout(x_init, x_star, hook)
        assert final.requires_grad is False
        assert final.grad_fn is None

    def test_delta_is_detached_and_inputs_keep_requires_grad_false(self):
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        with torch.no_grad():
            delta = hook(x, v, timesteps[STEPS - 1], STEPS - 1)
        assert delta.requires_grad is False and delta.grad_fn is None
        assert x.requires_grad is False and v.requires_grad is False

    def test_no_grad_leaks_into_v_or_x_t(self):
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM).requires_grad_(True)
        v = torch.randn(BATCH, HORIZON, ACT_DIM).requires_grad_(True)
        delta = hook(x, v, timesteps[STEPS - 1], STEPS - 1)
        assert delta.requires_grad is False and delta.grad_fn is None
        assert x.grad is None and v.grad is None

    def test_critic_parameters_receive_no_grad(self):
        critic = LinearProbeCritic(_probe_u())
        hook = make_q_guidance_hook(critic, _config())
        x_init, x_star = _inputs(5)
        _euler_rollout(x_init, x_star, hook)
        assert all(p.grad is None for p in critic.parameters())

    def test_hook_never_mutates_x_t_or_v(self):
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        x_ref, v_ref = x.clone(), v.clone()
        hook(x, v, timesteps[STEPS - 1], STEPS - 1)
        assert torch.equal(x, x_ref) and torch.equal(v, v_ref)


# --------------------------------------------------------------------------- #
# 4. Trust-region clamp and zero-config transparency
# --------------------------------------------------------------------------- #
class TestClampAndTransparency:
    def test_per_sample_norm_never_exceeds_max_delta_norm(self):
        max_norm = 0.05
        # Huge u so the raw delta (lambda * ||u||) far exceeds the clamp.
        critic = LinearProbeCritic(_probe_u() * 100.0)
        inner = make_q_guidance_hook(critic, _config(max_delta_norm=max_norm))
        norms = []

        def spy(x_t, v, timestep, step_index):
            delta = inner(x_t, v, timestep, step_index)
            norms.extend(delta.reshape(delta.shape[0], -1).norm(dim=1).tolist())
            return delta

        x_init, x_star = _inputs(6)
        _euler_rollout(x_init, x_star, spy)
        assert norms
        assert all(n <= max_norm * (1.0 + 1e-5) for n in norms), norms
        assert any(n > 0 for n in norms)

    def test_clamp_preserves_direction(self):
        u = _probe_u()
        hook = make_q_guidance_hook(
            LinearProbeCritic(u * 100.0), _config(max_delta_norm=0.05)
        )
        timesteps, _ = _build_schedule(STEPS)
        delta = hook(
            torch.randn(BATCH, HORIZON, ACT_DIM),
            torch.randn(BATCH, HORIZON, ACT_DIM),
            timesteps[STEPS - 1],
            STEPS - 1,
        )
        flat = delta.reshape(BATCH, -1)
        cos = torch.nn.functional.cosine_similarity(
            flat, (-u).reshape(1, -1).expand_as(flat), dim=1
        )
        torch.testing.assert_close(cos, torch.ones(BATCH), atol=1e-5, rtol=0.0)

    def test_loose_clamp_is_inactive(self):
        critic = LinearProbeCritic(_probe_u())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        free = make_q_guidance_hook(critic, _config(max_delta_norm=None))
        clamped = make_q_guidance_hook(critic, _config(max_delta_norm=1e6))
        a = free(x, v, timesteps[STEPS - 1], STEPS - 1)
        b = clamped(x, v, timesteps[STEPS - 1], STEPS - 1)
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        "overrides",
        (
            {"lambda_schedule": constant_lambda_after_threshold(0.0, 1.0)},
            {"guide_last_k": 0},
        ),
        ids=("lambda_zero", "window_zero"),
    )
    def test_zero_config_reproduces_unguided_trajectory_bitwise(self, overrides):
        critic = _CountingCritic(_probe_u())
        hook = make_q_guidance_hook(critic, _config(**overrides))
        x_init, x_star = _inputs(8)
        ref_states: list = []
        hook_states: list = []
        ref = _euler_rollout(x_init, x_star, None, record=ref_states)
        out = _euler_rollout(x_init, x_star, hook, record=hook_states)
        assert torch.equal(out, ref)
        assert len(ref_states) == len(hook_states) == STEPS
        for k, (a, b) in enumerate(zip(ref_states, hook_states)):
            assert torch.equal(a, b), f"trajectories diverge at step {k}"
        assert critic.calls == 0


# --------------------------------------------------------------------------- #
# 5. predicted_clean_action analytics
# --------------------------------------------------------------------------- #
class TestPredictedCleanAction:
    def test_recovers_x0_from_the_flow_identity(self):
        gen = torch.Generator().manual_seed(9)
        x0 = torch.randn((BATCH, HORIZON, ACT_DIM), generator=gen)
        eps = torch.randn((BATCH, HORIZON, ACT_DIM), generator=gen)
        for sigma in (0.05, 0.4, 0.97):
            x_sigma = (1.0 - sigma) * x0 + sigma * eps
            v = eps - x0
            torch.testing.assert_close(
                predicted_clean_action(x_sigma, v, sigma), x0, atol=1e-6, rtol=0.0
            )

    def test_accepts_tensor_sigma_scalar_and_per_sample(self):
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        scalar = predicted_clean_action(x, v, torch.tensor(0.5))
        torch.testing.assert_close(scalar, x - 0.5 * v)
        per_sample = predicted_clean_action(x, v, torch.tensor([0.1, 0.5, 0.9]))
        for i, s in enumerate((0.1, 0.5, 0.9)):
            torch.testing.assert_close(per_sample[i], x[i] - s * v[i])

    def test_sigma_boundaries_are_exact(self):
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        assert torch.equal(predicted_clean_action(x, v, 0.0), x)
        torch.testing.assert_close(predicted_clean_action(x, v, 1.0), x - v)

    @pytest.mark.parametrize(
        "sigma, match",
        ((-0.1, r"\[0, 1\]"), (1.5, r"\[0, 1\]"), (800.0, "num_train_timesteps")),
    )
    def test_raw_timestep_or_out_of_range_sigma_fails_closed(self, sigma, match):
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match=match):
            predicted_clean_action(x, torch.randn_like(x), sigma)

    def test_shape_mismatch_and_non_tensor_fail_closed(self):
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match="shape"):
            predicted_clean_action(x, torch.randn(BATCH, HORIZON, ACT_DIM + 1), 0.5)
        with pytest.raises(TypeError, match="torch.Tensor"):
            predicted_clean_action("not a tensor", x, 0.5)
        with pytest.raises(ValueError, match="elements"):
            predicted_clean_action(x, torch.randn_like(x), torch.tensor([0.1, 0.2]))


# --------------------------------------------------------------------------- #
# 6. Fail-closed validation of config, hook inputs, and the critic contract
# --------------------------------------------------------------------------- #
class TestValidation:
    @pytest.mark.parametrize(
        "overrides, err, match",
        (
            ({"lambda_schedule": 0.5}, TypeError, "callable"),
            ({"guide_last_k": -1}, ValueError, "guide_last_k"),
            ({"guide_last_k": STEPS + 1}, ValueError, "guide_last_k"),
            ({"guide_last_k": 2.0}, TypeError, "int"),
            ({"num_inference_steps": 0}, ValueError, "num_inference_steps"),
            ({"num_train_timesteps": 0}, ValueError, "num_train_timesteps"),
            ({"max_delta_norm": 0.0}, ValueError, "max_delta_norm"),
            ({"max_delta_norm": -1.0}, ValueError, "max_delta_norm"),
            ({"max_delta_norm": float("nan")}, ValueError, "max_delta_norm"),
            ({"max_delta_norm": "big"}, TypeError, "max_delta_norm"),
        ),
    )
    def test_config_fails_closed(self, overrides, err, match):
        with pytest.raises(err, match=match):
            _config(**overrides)

    def test_hook_factory_validates_its_arguments(self):
        with pytest.raises(TypeError, match="callable"):
            make_q_guidance_hook("not a critic", _config())
        with pytest.raises(TypeError, match="GuidanceConfig"):
            make_q_guidance_hook(LinearProbeCritic(_probe_u()), {"guide_last_k": 3})

    def test_hook_input_mismatches_raise_clearly(self):
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config())
        timesteps, _ = _build_schedule(STEPS)
        t_last = timesteps[STEPS - 1]
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match="shape"):
            hook(x, torch.randn(BATCH, HORIZON, ACT_DIM + 1), t_last, STEPS - 1)
        with pytest.raises(TypeError, match="torch.Tensor"):
            hook("not a tensor", v, t_last, STEPS - 1)
        with pytest.raises(ValueError, match="dtype/device"):
            hook(x.to(torch.float64), v, t_last, STEPS - 1)
        with pytest.raises(ValueError, match="floating"):
            hook(
                torch.zeros(BATCH, HORIZON, ACT_DIM, dtype=torch.int64),
                torch.zeros(BATCH, HORIZON, ACT_DIM, dtype=torch.int64),
                t_last,
                STEPS - 1,
            )
        with pytest.raises(ValueError, match="batched"):
            hook(torch.randn(HORIZON), torch.randn(HORIZON), t_last, STEPS - 1)
        with pytest.raises(TypeError, match="timestep"):
            hook(x, v, 800.0, STEPS - 1)
        with pytest.raises(ValueError, match="element"):
            hook(x, v, timesteps, STEPS - 1)
        with pytest.raises(ValueError, match="step_index"):
            hook(x, v, t_last, STEPS)
        with pytest.raises(ValueError, match="step_index"):
            hook(x, v, t_last, -1)
        with pytest.raises(TypeError, match="step_index"):
            hook(x, v, t_last, 1.0)

    def test_timestep_grid_inconsistent_with_num_train_timesteps_fails(self):
        # A t=800 grid element under num_train_timesteps=100 implies sigma=8.
        hook = make_q_guidance_hook(
            LinearProbeCritic(_probe_u()), _config(num_train_timesteps=100)
        )
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match="num_train_timesteps"):
            hook(x, torch.randn_like(x), torch.tensor(800.0), STEPS - 1)

    @pytest.mark.parametrize(
        "bad_output, err, match",
        (
            (lambda a: a.sum(), ValueError, "per sample"),
            (lambda a: a.sum(dim=-1), ValueError, "per sample"),
            (lambda a: (a.sum(dim=(1, 2)) > 0), TypeError, "floating"),
            (lambda a: "score", TypeError, "torch.Tensor"),
            (lambda a: a.detach().sum(dim=(1, 2)), ValueError, "does not depend"),
        ),
        ids=("scalar", "wrong_shape", "bool_tensor", "non_tensor", "detached_input"),
    )
    def test_bad_critics_fail_closed(self, bad_output, err, match):
        hook = make_q_guidance_hook(bad_output, _config())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        with pytest.raises(err, match=match):
            hook(x, torch.randn_like(x), timesteps[STEPS - 1], STEPS - 1)

    def test_critic_returning_b_by_1_is_accepted(self):
        hook = make_q_guidance_hook(
            lambda a: a.sum(dim=(1, 2), keepdim=False).unsqueeze(1), _config()
        )
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        delta = hook(x, torch.randn_like(x), timesteps[STEPS - 1], STEPS - 1)
        assert delta.shape == x.shape

    def test_non_finite_lambda_fails_closed(self):
        hook = make_q_guidance_hook(
            LinearProbeCritic(_probe_u()),
            _config(lambda_schedule=lambda sigma: float("inf")),
        )
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match="non-finite"):
            hook(x, torch.randn_like(x), timesteps[STEPS - 1], STEPS - 1)

    def test_constant_after_threshold_shape_and_validation(self):
        schedule = constant_lambda_after_threshold(0.3, 0.25)
        assert schedule(0.1) == 0.3
        assert schedule(0.25) == 0.3
        assert schedule(0.26) == 0.0
        assert schedule(0.9) == 0.0
        with pytest.raises(ValueError, match="sigma_threshold"):
            constant_lambda_after_threshold(0.3, 1.5)
        with pytest.raises(ValueError, match="finite"):
            constant_lambda_after_threshold(float("nan"), 0.5)

    def test_linear_probe_critic_validates(self):
        with pytest.raises(TypeError, match="floating"):
            LinearProbeCritic(torch.zeros(2, 2, dtype=torch.int64))
        critic = LinearProbeCritic(_probe_u())
        with pytest.raises(ValueError, match="B"):
            critic(torch.randn(HORIZON, ACT_DIM))
        with pytest.raises(TypeError, match="torch.Tensor"):
            critic("chunk")


# --------------------------------------------------------------------------- #
# 7. Dtype/device fidelity and package surface
# --------------------------------------------------------------------------- #
class TestDtypeAndSurface:
    @pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
    def test_delta_matches_v_dtype_and_device(self, dtype):
        hook = make_q_guidance_hook(LinearProbeCritic(_probe_u()), _config())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM).to(dtype)
        v = torch.randn(BATCH, HORIZON, ACT_DIM).to(dtype)
        delta = hook(x, v, timesteps[STEPS - 1], STEPS - 1)
        assert delta.dtype == dtype
        assert delta.device == v.device

    def test_internal_math_is_float32_regardless_of_input_dtype(self):
        # Same inputs through float32 and float64 must agree to float32
        # precision: the critic grad path is computed in float32 by contract.
        u = _probe_u()
        hook = make_q_guidance_hook(LinearProbeCritic(u), _config())
        timesteps, _ = _build_schedule(STEPS)
        x = torch.randn(BATCH, HORIZON, ACT_DIM)
        v = torch.randn(BATCH, HORIZON, ACT_DIM)
        d32 = hook(x, v, timesteps[STEPS - 1], STEPS - 1)
        d64 = hook(
            x.to(torch.float64), v.to(torch.float64), timesteps[STEPS - 1], STEPS - 1
        )
        torch.testing.assert_close(d64.to(torch.float32), d32, atol=1e-6, rtol=0.0)

    def test_steering_package_reexports_the_public_surface(self):
        assert steering_pkg.make_q_guidance_hook is flow_guidance.make_q_guidance_hook
        assert steering_pkg.GuidanceConfig is flow_guidance.GuidanceConfig
        assert steering_pkg.LinearProbeCritic is flow_guidance.LinearProbeCritic
        assert steering_pkg.predicted_clean_action is flow_guidance.predicted_clean_action
        assert (
            steering_pkg.constant_lambda_after_threshold
            is flow_guidance.constant_lambda_after_threshold
        )
        assert set(steering_pkg.__all__) == set(flow_guidance.__all__)

    def test_schedule_deltas_are_negative_and_sum_to_minus_one(self):
        # Guard on the copied scheduler math itself: sigma runs 1 -> 0, so all
        # Euler deltas are negative and telescope to -1. If this ever fails the
        # fixture no longer replicates scheduler_continuous.py and every sign
        # conclusion above is void.
        for shift in (3.0, 5.0):
            timesteps, deltas = _build_schedule(STEPS, shift=shift)
            assert bool((deltas < 0).all())
            assert math.isclose(float(deltas.sum()), -1.0, abs_tol=1e-6)
            assert timesteps.shape == (STEPS,) and float(timesteps[0]) == N_TRAIN
