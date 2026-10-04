# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check pure PPO gradients, masks, Adam state and pre-compaction metrics on CPU."""

import copy
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.fastwam_dual_ppo import (
    finalize_fastwam_weighted_metrics,
    pop_fastwam_weighted_metric_sums,
)
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
from rlinf.models.embodiment.wam_policy.uncond_rl_actor import (
    UncondRLFSDPActor,
    compact_uncond_global_batch,
    finalize_uncond_compaction_metrics,
)
from rlinf.models.embodiment.wam_policy.uncond_rl_helpers import (
    HelperLossContext,
    TrainableParameterLayout,
    backward_helper_microbatch,
    map_prepared_tensors,
    microbatch_assignments,
)


@dataclass(frozen=True)
class _Route:
    route_used: torch.Tensor


class _PurePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_lora = nn.Parameter(torch.tensor([0.03, -0.02]))
        self.video_lora = nn.Parameter(torch.tensor([0.01, 0.04]))
        self.value_head = nn.Linear(2, 1)
        with torch.no_grad():
            self.value_head.weight.copy_(torch.tensor([[0.05, 0.1]]))
            self.value_head.bias.fill_(0.02)

    def forward(self, forward_inputs, **kwargs):
        x = forward_inputs["x"]
        logprobs = (x * self.action_lora * (1 + self.video_lora)).sum(-1, keepdim=True)
        return {
            "flow_logprobs": logprobs,
            "flow_entropy": logprobs * 0 + 0.5,
            "values": self.value_head(x.detach()),
        }


def _config():
    return OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 392,
                "micro_batch_size": 4,
                "model": {"uncond_rl": {"critic_warmup_updates": 10}},
            },
            "algorithm": {
                "adv_type": "gae",
                "uncond_flow_ppo": {
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.2,
                    "entropy_coefficient": 0.01,
                    "loss_weight": 1.0,
                },
                "critic_loss": {
                    "value_clip": 0.2,
                    "huber_delta": 10.0,
                    "loss_weight": 1.0,
                },
            },
            "env": {"train": {"max_episode_steps": 700}},
        }
    )


def _batch(kind):
    generator = torch.Generator().manual_seed(311)
    flow = torch.zeros(392, dtype=torch.bool)
    critic = torch.zeros(392, 1, dtype=torch.bool)
    if kind == "sparse":
        flow[torch.tensor([2, 13, 29, 61, 109, 156, 193])] = True
        critic[torch.tensor([2, 13, 29, 61, 109, 156, 193, 23, 33, 166, 191])] = True
    elif kind == "full":
        flow[:] = True
        critic[:] = True
    elif kind != "empty":
        raise ValueError(kind)
    return {
        "flow_valid_mask": flow,
        "loss_mask": critic,
        "loss_mask_sum": torch.randint(4, 701, (392, 1), generator=generator),
        "route_info": _Route(torch.full((392,), int(WAMRoute.UNCOND))),
        "forward_inputs": {"x": torch.randn(392, 2, generator=generator)},
        "flow_advantages": torch.randn(392, 1, generator=generator),
        "prev_logprobs": torch.randn(392, 1, generator=generator) * 0.05,
        "returns": torch.randn(392, 1, generator=generator) * 0.1,
        "prev_values": torch.randn(392, 1, generator=generator) * 0.2,
    }


def _microbatches(batch):
    return [
        map_prepared_tensors(batch, lambda value: value[start : start + 4])
        for start in range(0, len(batch["flow_valid_mask"]), 4)
    ]


def _backward(model, batches, context, scale):
    metrics = {}
    for batch in batches:
        values = backward_helper_microbatch(
            model,
            context,
            batch,
            selected_loss_scales={"flow": scale},
            normalization_statistics={"floor_hit_fraction": 0.0},
        )
        for name, value in values.items():
            metrics.setdefault(name, []).append(float(value))
    return metrics


def test_uncond_actor_releases_previous_replay_before_next_receive():
    """A new N84 receive cannot retain the preceding flattened replay."""

    released = []
    old_batch = {"forward_inputs": {"large_replay": torch.ones(2, 3)}}
    worker = object.__new__(UncondRLFSDPActor)
    worker._rank = 0
    worker.version = 5
    worker.rollout_batch = old_batch
    worker._release_replay_host_memory = lambda **kwargs: released.append(kwargs)
    assert UncondRLFSDPActor._consume_rollout_batch_during_train_preparation(worker)
    UncondRLFSDPActor._after_rollout_batch_train_preparation(worker)
    UncondRLFSDPActor._release_consumed_rollout_batch_before_receive(worker)
    assert worker.rollout_batch is None
    assert [entry["phase"] for entry in released] == [
        "post_train_preparation",
        "pre_trajectory_receive",
    ]
    UncondRLFSDPActor._release_consumed_rollout_batch_before_receive(worker)
    assert len(released) == 2


def test_uncond_actor_releases_replay_after_training_frame_exits(monkeypatch):
    """No completed replay storage may overlap the following rollout."""

    import rlinf.models.embodiment.wam_policy.uncond_rl_actor as actor_module

    worker = object.__new__(UncondRLFSDPActor)
    worker._rank = 0
    worker.version = 5
    worker.rollout_batch = {"replay": torch.ones(2, 3)}
    replay = weakref.ref(worker.rollout_batch["replay"])
    events, logs = [], []
    worker.log_info = logs.append
    metrics = {"actor/total_loss": 0.25}
    request, response = object(), object()

    def train(self, kv_request_channel, kv_response_channel):
        assert (kv_request_channel, kv_response_channel) == (request, response)
        last_microbatch = self.rollout_batch["replay"][:1]
        assert last_microbatch.shape == (1, 3) and replay() is not None
        events.append("trained")
        return metrics

    def release(**kwargs):
        assert replay() is None and worker.rollout_batch is None
        events.append(kwargs["phase"])
        return kwargs

    monkeypatch.setattr(actor_module.EmbodiedFSDPActor, "run_training", train)
    monkeypatch.setattr(actor_module, "release_pad_host_memory", release)
    monkeypatch.setattr(
        torch.cuda.memory,
        "host_memory_stats",
        lambda: {"allocated_bytes.current": 0, "reserved_bytes.current": 1024},
    )
    assert worker.run_training(request, response) is metrics
    assert events == ["trained", "post_training"]
    assert '"reserved_bytes.current": 1024' in logs[0]


@pytest.mark.parametrize("kind", ["sparse", "full", "empty"])
@pytest.mark.parametrize("actor_version", [9, 10])
def test_compaction_and_p7_preserve_pure_loss_gradients_adam_and_metrics(
    kind, actor_version
):
    cfg = _config()
    context = HelperLossContext(cfg, actor_version=actor_version)
    assert context.gradient_accumulation == 98
    original = _batch(kind)
    compacted, preparation = compact_uncond_global_batch(original, micro_batch_size=4)
    assert preparation["perf/actor_rows_original"] == 392
    if kind == "empty":
        assert len(compacted["flow_valid_mask"]) == 4
    elif kind == "sparse":
        assert len(compacted["flow_valid_mask"]) == 12
        assert compacted["flow_valid_mask"].sum() == 7
        assert compacted["loss_mask"].sum() == 11
    scale = 98 / int(original["flow_valid_mask"].sum()) if kind != "empty" else 0.0
    reference = _PurePolicy()
    owner = copy.deepcopy(reference)
    participants = [owner, *(copy.deepcopy(reference) for _ in range(6))]
    original_metrics = _backward(reference, _microbatches(original), context, scale)
    compacted_batches = _microbatches(compacted)
    gathered = {}
    for model, assigned in zip(
        participants,
        microbatch_assignments(len(compacted_batches), tuple(range(6))),
        strict=True,
    ):
        partial = _backward(
            model, [compacted_batches[i] for i in assigned], context, scale
        )
        for name, values in partial.items():
            gathered.setdefault(name, []).extend(values)
    layout = TrainableParameterLayout(owner, bucket_bytes=8)
    for model in participants[1:]:
        helper = TrainableParameterLayout(model, bucket_bytes=8)
        for bucket in layout.buckets:
            layout.add_gradient_bucket(
                bucket, helper.pack_bucket(bucket, gradients=True), helper.presence()
            )
    for expected, actual in zip(
        reference.parameters(), owner.parameters(), strict=True
    ):
        assert (expected.grad is None) == (actual.grad is None)
        if expected.grad is not None:
            torch.testing.assert_close(expected.grad, actual.grad, rtol=2e-6, atol=2e-8)
    if actor_version < 10:
        assert owner.action_lora.grad is None
        assert owner.video_lora.grad is None
    else:
        assert owner.action_lora.grad is not None
        assert owner.video_lora.grad is not None
        if kind == "empty":
            assert torch.equal(owner.action_lora.grad, torch.zeros(2))
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3) for model in (reference, owner)
    ]
    for optimizer in optimizers:
        optimizer.step()
    for expected, actual in zip(
        reference.parameters(), owner.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-8)
        assert (
            optimizers[0].state[expected].keys() == optimizers[1].state[actual].keys()
        )
        for key in optimizers[0].state[expected]:
            torch.testing.assert_close(
                optimizers[0].state[expected][key],
                optimizers[1].state[actual][key],
                rtol=2e-6,
                atol=2e-8,
            )
    gathered.update({name: [value] for name, value in preparation.items()})
    gathered["_uncond_compaction/full_value_clip_ratio"] = [
        sum(original_metrics["critic/value_clip_ratio"]) / 98
    ]
    finalize_uncond_compaction_metrics(gathered)
    for name in (
        "critic/value_loss",
        "critic/value_clip_ratio",
        "fastwam/total_loss",
        "actor/total_loss",
        "uncond_flow/selected_loss_scale",
    ):
        assert gathered[name][0] == pytest.approx(
            sum(original_metrics[name]) / 98, rel=2e-6, abs=2e-8
        )
    expected_sums, expected_maxima = pop_fastwam_weighted_metric_sums(original_metrics)
    actual_sums, actual_maxima = pop_fastwam_weighted_metric_sums(gathered)
    for name, expected in finalize_fastwam_weighted_metrics(expected_sums).items():
        assert finalize_fastwam_weighted_metrics(actual_sums)[name] == pytest.approx(
            expected, rel=2e-6, abs=2e-8
        )
    assert actual_maxima == pytest.approx(expected_maxima)


def test_full_critic_diagnostic_preserves_fsdp_gradients_and_next_forward(tmp_path):
    """A direct head diagnostic after backward must not corrupt nested FSDP."""
    from contextlib import nullcontext

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from rlinf.models.embodiment.wam_policy.uncond_rl_actor import UncondRLFSDPActor

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.value_head = FSDP(
                nn.Linear(2, 1), device_id=torch.device("cpu"), use_orig_params=True
            )

        def value_from_features(self, features):
            return self.value_head(features)[:, 0]

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.critic = Critic()

        def _require_critic(self):
            return self.critic

        def forward(self, features):
            return self.critic.value_from_features(features)

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1
    )
    try:
        policy = Policy()
        model = FSDP(policy, device_id=torch.device("cpu"), use_orig_params=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
        model(inputs[:4]).sum().backward()
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        with torch.no_grad():
            expected = (model(inputs).reshape(-1, 1).abs() > 0.2).float().mean().item()
        worker = SimpleNamespace(
            cfg=_config(),
            device="cpu",
            amp_context=nullcontext(),
            _fastwam_policy_module=lambda: policy,
            _uncond_original_critic_inputs=(inputs, torch.zeros(12, 1)),
        )
        assert UncondRLFSDPActor._original_critic_clip_ratio(worker) == expected
        for parameter, expected_gradient in zip(
            model.parameters(), gradients, strict=True
        ):
            torch.testing.assert_close(
                parameter.grad, expected_gradient, rtol=0, atol=0
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(inputs[4:8]).sum().backward()
        assert policy.critic.value_head._is_root is False
        assert all(parameter.grad is not None for parameter in model.parameters())
    finally:
        torch.distributed.destroy_process_group()
