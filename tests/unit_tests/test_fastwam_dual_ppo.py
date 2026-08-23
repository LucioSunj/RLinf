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

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from rlinf.algorithms.losses import compute_ppo_critic_loss


def _load_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, repo_root / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_module(
    "rlinf.models.embodiment.wam_policy.contracts",
    "rlinf/models/embodiment/wam_policy/contracts.py",
)
dual_ppo = _load_module(
    "fastwam_dual_ppo_under_test",
    "rlinf/algorithms/fastwam_dual_ppo.py",
)
WAMRoute = contracts.WAMRoute


def test_shared_critic_clip_metric_counts_unclipped_value_deltas():
    _loss, metrics = compute_ppo_critic_loss(
        values=torch.tensor([0.0, 0.1, 0.3, -0.4]),
        returns=torch.zeros(4),
        prev_values=torch.zeros(4),
        value_clip=0.2,
        huber_delta=1.0,
    )

    assert metrics["critic/value_clip_ratio"].item() == pytest.approx(0.5)


def test_epsilon_mixture_bernoulli_probability_and_logprob_are_exact():
    logits = torch.tensor([0.0, math.log(3.0)])
    selected_route = torch.tensor([WAMRoute.UNCOND, WAMRoute.IDM])
    result = dual_ppo.epsilon_mixture_bernoulli(
        logits,
        epsilon=0.2,
        temperature=1.0,
        route=selected_route,
    )

    assert torch.allclose(result.base_probability, torch.tensor([0.5, 0.75]))
    assert torch.allclose(result.behavior_probability, torch.tensor([0.5, 0.7]))
    assert torch.allclose(
        result.logprob,
        torch.tensor([math.log(0.5), math.log(0.7)]),
    )
    expected_entropy = -torch.tensor([0.5, 0.7]) * torch.log(
        torch.tensor([0.5, 0.7])
    ) - torch.tensor([0.5, 0.3]) * torch.log(torch.tensor([0.5, 0.3]))
    assert torch.allclose(result.entropy, expected_entropy)


def test_epsilon_one_is_uniform_independent_of_logits():
    result = dual_ppo.epsilon_mixture_bernoulli(
        torch.tensor([-20.0, 20.0]),
        epsilon=1.0,
        temperature=0.25,
        route=torch.tensor([WAMRoute.UNCOND, WAMRoute.IDM]),
    )
    assert torch.equal(result.behavior_probability, torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.logprob, torch.full((2,), math.log(0.5)))


def test_gate_ppo_clips_independently_and_reports_selected_count():
    old_logprobs = torch.zeros(2, dtype=torch.float32)
    logprobs = torch.log(torch.tensor([1.5, 0.5], dtype=torch.float32))
    loss, metrics = dual_ppo.compute_gate_ppo_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=torch.ones(2, dtype=torch.float32),
        valid_mask=torch.ones(2, dtype=torch.bool),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    assert torch.allclose(loss, torch.tensor(-0.85))
    assert metrics["gate/sample_count"].item() == 2
    assert metrics["gate/clip_fraction"].item() == 1.0


def test_dual_ppo_uses_gate_mask_and_uncond_route_mask_separately():
    gate_logprobs = torch.zeros(3, requires_grad=True)
    flow_logprobs = torch.zeros(3, requires_grad=True)
    loss, metrics = dual_ppo.compute_fastwam_dual_ppo_loss(
        gate_logprobs=gate_logprobs,
        gate_old_logprobs=torch.zeros(3),
        gate_advantages=torch.ones(3),
        gate_valid_mask=torch.tensor([False, True, False]),
        gate_clip_ratio_low=0.2,
        gate_clip_ratio_high=0.2,
        flow_logprobs=flow_logprobs,
        flow_old_logprobs=torch.zeros(3),
        flow_advantages=torch.ones(3),
        route_used=torch.tensor([WAMRoute.IDM, WAMRoute.UNCOND, WAMRoute.IDM]),
        flow_clip_ratio_low=0.1,
        flow_clip_ratio_high=0.1,
    )
    loss.backward()

    assert metrics["gate/sample_count"].item() == 1
    assert metrics["uncond_flow/sample_count"].item() == 1
    assert torch.equal(gate_logprobs.grad, torch.tensor([0.0, -1.0, 0.0]))
    assert torch.equal(flow_logprobs.grad, torch.tensor([0.0, -1.0, 0.0]))


def test_uncond_flow_ppo_sums_joint_chunk_logprob_before_clipping():
    logprobs = torch.full(
        (1, 1, 2),
        math.log(1.1),
        dtype=torch.float32,
        requires_grad=True,
    )
    loss, metrics = dual_ppo.compute_uncond_flow_ppo_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=torch.ones(1, 1),
        route_used=torch.tensor([WAMRoute.UNCOND]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    assert loss.item() == pytest.approx(-1.2)
    assert metrics["uncond_flow/ratio"].item() == pytest.approx(1.21)
    assert metrics["uncond_flow/sample_count"].item() == 1


def test_selected_loss_scale_matches_global_mean_across_uneven_microbatches():
    logprobs = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])
    routes = torch.full((4,), WAMRoute.UNCOND, dtype=torch.long)
    # Two microbatches, one and three selected samples. The worker later divides
    # every microbatch loss by gradient_accumulation=2.
    scale = 2.0 / 4.0
    first, _ = dual_ppo.compute_uncond_flow_ppo_loss(
        logprobs=logprobs[:1],
        old_logprobs=torch.zeros(1),
        advantages=advantages[:1],
        route_used=routes[:1],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        selected_loss_scale=scale,
    )
    second, _ = dual_ppo.compute_uncond_flow_ppo_loss(
        logprobs=logprobs[1:],
        old_logprobs=torch.zeros(3),
        advantages=advantages[1:],
        route_used=routes[1:],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        selected_loss_scale=scale,
    )
    accumulated = (first + second) / 2.0
    accumulated.backward()

    assert accumulated.item() == pytest.approx(-advantages.mean().item())
    assert torch.allclose(logprobs.grad, -advantages / 4.0)


def test_selected_loss_scale_handles_an_empty_local_microbatch():
    logprobs = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    scale = 2.0
    empty, _ = dual_ppo.compute_uncond_flow_ppo_loss(
        logprobs=logprobs[:1],
        old_logprobs=torch.zeros(1),
        advantages=torch.ones(1),
        route_used=torch.tensor([WAMRoute.IDM]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        selected_loss_scale=scale,
    )
    selected, _ = dual_ppo.compute_uncond_flow_ppo_loss(
        logprobs=logprobs[1:],
        old_logprobs=torch.zeros(1),
        advantages=torch.ones(1),
        route_used=torch.tensor([WAMRoute.UNCOND]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        selected_loss_scale=scale,
    )
    accumulated = (empty + selected) / 2.0
    accumulated.backward()

    assert accumulated.item() == pytest.approx(-1.0)
    assert torch.equal(logprobs.grad, torch.tensor([0.0, -1.0]))


def test_empty_policy_masks_return_finite_differentiable_zero():
    gate_logprobs = torch.randn(3, requires_grad=True)
    gate_loss, gate_metrics = dual_ppo.compute_gate_ppo_loss(
        logprobs=gate_logprobs,
        old_logprobs=torch.zeros(3),
        advantages=torch.ones(3),
        valid_mask=torch.zeros(3, dtype=torch.bool),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    flow_logprobs = torch.randn(3, requires_grad=True)
    flow_loss, flow_metrics = dual_ppo.compute_uncond_flow_ppo_loss(
        logprobs=flow_logprobs,
        old_logprobs=torch.zeros(3),
        advantages=torch.ones(3),
        route_used=torch.full((3,), WAMRoute.IDM, dtype=torch.int64),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    (gate_loss + flow_loss).backward()

    assert torch.isfinite(gate_loss) and gate_loss.item() == 0.0
    assert torch.isfinite(flow_loss) and flow_loss.item() == 0.0
    assert gate_metrics["gate/sample_count"].item() == 0.0
    assert flow_metrics["uncond_flow/sample_count"].item() == 0.0
    assert torch.equal(gate_logprobs.grad, torch.zeros_like(gate_logprobs))
    assert torch.equal(flow_logprobs.grad, torch.zeros_like(flow_logprobs))


def test_fixed_branch_cost_is_charged_once_per_valid_idm_chunk():
    result = dual_ppo.apply_fixed_branch_cost(
        environment_rewards=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        route_used=torch.tensor(
            [
                [WAMRoute.UNCOND, WAMRoute.IDM],
                [WAMRoute.IDM, WAMRoute.UNCOND],
            ]
        ),
        idm_cost=0.25,
        valid_mask=torch.tensor([[True, True], [False, True]]),
    )

    assert torch.equal(result.costs, torch.tensor([[0.0, 0.25], [0.0, 0.0]]))
    assert torch.equal(result.rewards, torch.tensor([[1.0, 1.75], [3.0, 4.0]]))


def test_fixed_branch_cost_rejects_primitive_step_rewards():
    with pytest.raises(ValueError, match="chunk-level"):
        dual_ppo.apply_fixed_branch_cost(
            environment_rewards=torch.zeros(2, 3, 5),
            route_used=torch.zeros(2, 3, dtype=torch.int64),
            idm_cost=1.0,
        )


def test_base_uncond_kl_sums_joint_chunk_and_masks_idm():
    kl_values = torch.tensor(
        [[[0.25, 0.75]], [[3.0, 4.0]]],
        requires_grad=True,
    )
    loss, metrics = dual_ppo.compute_base_uncond_kl_loss(
        kl_values=kl_values,
        route_used=torch.tensor([WAMRoute.UNCOND, WAMRoute.IDM]),
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert metrics["base_uncond_kl/sample_count"].item() == 1
    assert torch.equal(
        kl_values.grad,
        torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]]),
    )


def test_reported_policy_metrics_are_weighted_by_selected_sample_count():
    metrics = {
        "gate/sample_count": [1.0, 3.0],
        "gate/policy_loss": [2.0, 6.0],
        "gate/total_loss": [2.0, 6.0],
        "gate/ratio": [1.0, 3.0],
        "gate/ratio_abs": [0.0, 2.0],
        "gate/approx_kl": [0.0, 4.0],
        "gate/clip_fraction": [0.0, 1.0],
        "gate/entropy": [0.5, 0.25],
    }

    sums, maxima = dual_ppo.pop_fastwam_weighted_metric_sums(metrics)
    reduced = dual_ppo.finalize_fastwam_weighted_metrics(sums)

    assert maxima == {}
    assert metrics == {}
    assert reduced["gate/sample_count"] == 4.0
    assert reduced["gate/policy_loss"] == pytest.approx(5.0)
    assert reduced["gate/ratio"] == pytest.approx(2.5)
    assert reduced["gate/clip_fraction"] == pytest.approx(0.75)


def test_collapse_penalty_uses_differentiable_expected_calls_per_episode():
    probabilities = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    penalty, metrics = dual_ppo.compute_gate_collapse_penalty(
        base_idm_probabilities=probabilities,
        episode_ids=torch.tensor([4, 4, 9]),
        valid_mask=torch.tensor([True, True, True]),
        tau_calls=1.0,
        scope="episode",
    )
    penalty.backward()

    expected = (math.exp(-0.5) + math.exp(-1.5) + math.exp(-1.0) + math.exp(0.0)) / 2
    assert penalty.item() == pytest.approx(expected)
    assert metrics["collapse/group_count"].item() == 2
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


def test_collapse_penalty_empty_mask_is_differentiable_zero():
    probabilities = torch.tensor([0.2, 0.8], requires_grad=True)
    penalty, _ = dual_ppo.compute_gate_collapse_penalty(
        base_idm_probabilities=probabilities,
        episode_ids=torch.tensor([1, 1]),
        valid_mask=torch.zeros(2, dtype=torch.bool),
        tau_calls=1.0,
    )
    penalty.backward()
    assert penalty.item() == 0.0
    assert torch.equal(probabilities.grad, torch.zeros_like(probabilities))
