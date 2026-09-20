# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.fastwam_dual_ppo import (
    compute_gate_ppo_loss,
    compute_uncond_flow_ppo_loss,
    finalize_fastwam_weighted_metrics,
    pop_fastwam_weighted_metric_sums,
)
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    LiberoFastWAMRuntime,
    _load_cached_text_contexts,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.task_metrics import (
    accumulate_task_losses,
    finalize_task_losses,
)
from rlinf.utils.metric_utils import materialize_scalar_metrics
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def test_microbatch_normalization_preserves_detached_loss_metrics():
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def compute_loss(**kwargs):
        loss = parameter.square()
        return loss, {"fastwam/total_loss": loss.detach()}

    actor = SimpleNamespace(
        device=torch.device("cpu"),
        before_micro_batch=lambda *args, **kwargs: nullcontext(),
        cfg=OmegaConf.create(
            {
                "actor": {"model": {"model_type": "fastwam_adaptive"}},
                "algorithm": {"adv_type": "gae"},
            }
        ),
        model=lambda **kwargs: {},
        _allows_absent_action_logprobs=lambda: False,
        _compute_fastwam_loss=compute_loss,
        amp_context=nullcontext(),
        enable_sft_co_train=False,
        gradient_accumulation=4,
        grad_scaler=torch.amp.GradScaler("cuda", enabled=False),
        _restore_fastwam_fsdp_parameter_views_after_backward=lambda: 0,
    )
    batch = {
        "advantages": torch.ones(1),
        "prev_logprobs": torch.zeros(1),
        "route_info": object(),
        "emitted_gate": object(),
        "forward_inputs": {},
    }
    metrics = {}
    EmbodiedFSDPActor.train_micro_batch(actor, batch, metrics, is_last=True)
    materialize_scalar_metrics(metrics)
    assert metrics["fastwam/total_loss"] == [4.0]
    assert metrics["actor/total_loss"] == [1.0]
    assert parameter.grad.item() == 1.0


@pytest.mark.parametrize("chunk_sizes", [[7], [4, 3], [1, 2, 4]])
@pytest.mark.parametrize("empty", [False, True])
def test_task_statistics_match_original_ppo_with_uneven_microbatches(
    monkeypatch, chunk_sizes, empty
):
    tasks = torch.tensor([0, 3, 0, 8, 3, 8, 9])
    routes = torch.tensor([0, 1, 0, 0, 0, 1, 0])
    gate_valid = torch.tensor([True, True, False, True, True, False, True])
    flow_valid = torch.tensor([True, True, False, True, False, True, True])
    if empty:
        gate_valid.zero_()
        flow_valid.zero_()
    old_gate = torch.full((7,), -0.7)
    gate_logprob = (old_gate + torch.linspace(-0.9, 0.9, 7)).requires_grad_()
    old_flow = torch.zeros(7, 2, 3)
    flow_logprob = torch.linspace(-0.15, 0.15, 42).reshape(7, 2, 3).requires_grad_()
    advantage = torch.tensor([2.0, -1.0, 9.0, -3.0, 0.5, -0.8, 4.0])
    bc = torch.arange(7).float().requires_grad_()
    clip = SimpleNamespace(clip_ratio_low=0.15, clip_ratio_high=0.25)
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(gate_ppo=clip, uncond_flow_ppo=clip)
    )
    totals = {}
    offset = 0
    with monkeypatch.context() as scoped:

        def no_cpu(*args, **kwargs):
            raise AssertionError("Task accounting must not copy a microbatch to CPU.")

        scoped.setattr(torch.Tensor, "cpu", no_cpu)
        for count in chunk_sizes:
            part = slice(offset, offset + count)
            batch = {
                "forward_inputs": {"multitask_task_id": tasks[part]},
                "route_info": SimpleNamespace(route_used=routes[part]),
                "emitted_gate": SimpleNamespace(old_logprob=old_gate[part]),
                "gate_valid_mask": gate_valid[part],
                "flow_valid_mask": flow_valid[part, None],
                "gate_advantages": advantage[part, None],
                "flow_advantages": advantage[part, None],
                "prev_logprobs": old_flow[part],
            }
            output = {
                "gate_logprobs": gate_logprob[part],
                "flow_logprobs": flow_logprob[part],
                "online_idm_bc_per_sample_loss": bc[part],
            }
            accumulate_task_losses(totals, batch, output, cfg)
            offset += count
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for tensor in totals.values()
    )
    actual = finalize_task_losses(totals)
    for task in range(10):
        selected = tasks == task
        _, gate = compute_gate_ppo_loss(
            logprobs=gate_logprob.detach(),
            old_logprobs=old_gate,
            advantages=advantage,
            valid_mask=gate_valid & selected,
            clip_ratio_low=0.15,
            clip_ratio_high=0.25,
        )
        _, flow = compute_uncond_flow_ppo_loss(
            logprobs=flow_logprob.detach(),
            old_logprobs=old_flow,
            advantages=advantage[:, None],
            route_used=routes,
            valid_mask=flow_valid & selected,
            clip_ratio_low=0.15,
            clip_ratio_high=0.25,
        )
        for owner, reference, original in (
            ("gate", gate, "gate"),
            ("flow", flow, "uncond_flow"),
        ):
            assert (
                actual[f"task/{task}/{owner}_sample_count"]
                == reference[f"{original}/sample_count"]
            )
            assert actual[f"task/{task}/{owner}_raw_loss"] == pytest.approx(
                float(reference[f"{original}/policy_loss"]), rel=2e-6, abs=1e-7
            )
        bc_mask = selected & flow_valid & (routes == 0)
        assert actual[f"task/{task}/bc_sample_count"] == int(bc_mask.sum())
        assert actual[f"task/{task}/bc_raw_loss"] == pytest.approx(
            float(bc.detach()[bc_mask].sum()) / max(int(bc_mask.sum()), 1)
        )


def test_bulk_materialization_preserves_weighted_counts_maxima_and_order():
    histories = {}
    for valid, offset in (
        (torch.tensor([True, False, False]), 0.1),
        (torch.ones(3, dtype=torch.bool), 0.3),
    ):
        _, metrics = compute_gate_ppo_loss(
            logprobs=torch.full((3,), offset),
            old_logprobs=torch.zeros(3),
            advantages=torch.tensor([1.0, -2.0, 3.0]),
            valid_mask=valid,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
        for key, value in metrics.items():
            histories.setdefault(key, []).append(value.detach())
    histories["ordinary"] = [0.3, torch.tensor(0.7, dtype=torch.float64)]
    expected = {
        key: [float(value) for value in values] for key, values in histories.items()
    }
    materialize_scalar_metrics(histories)
    assert histories == expected
    actual_sums, actual_max = pop_fastwam_weighted_metric_sums(histories)
    expected_sums, expected_max = pop_fastwam_weighted_metric_sums(expected)
    assert actual_sums == expected_sums
    assert actual_max == expected_max
    assert finalize_fastwam_weighted_metrics(
        actual_sums
    ) == finalize_fastwam_weighted_metrics(expected_sums)
    assert actual_sums["gate/sample_count"] == 4


def _save_context(directory, prompt, value):
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    torch.save(
        {
            "context": torch.full((3, 2), value, dtype=torch.bfloat16),
            "mask": torch.tensor([True, True, False]),
        },
        directory / f"{digest}.t5_len3.wan22ti2v5b.pt",
    )


def test_worker_text_cache_loads_ten_tasks_once_and_keeps_state_fresh(
    tmp_path, monkeypatch
):
    class Actor(torch.nn.Module):
        text_dim = 2

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        def _append_proprio_to_context(self, *, context, context_mask, proprio):
            return torch.cat((context, proprio[:, None]), dim=1), torch.cat(
                (context_mask, torch.ones(len(proprio), 1, dtype=torch.bool)), dim=1
            )

    runtime = LiberoFastWAMRuntime(
        actor=Actor(),
        lora_adapter=None,
        text_embedding_cache_dir=str(tmp_path),
        text_embedding_context_len=3,
        prompt_template="{task}",
    )
    runtime._model_images = lambda obs: obs["main_images"]
    runtime._normalized_proprio = lambda states: states.to(torch.bfloat16)
    prompts = [f"task {task}" for task in range(10)]
    for task, prompt in enumerate(prompts):
        _save_context(tmp_path, prompt, task + 1)
    calls = []
    original = torch.load

    def counted_load(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    for step, order in enumerate((prompts, list(reversed(prompts)), [prompts[2]] * 4)):
        states = torch.full((len(order), 2), step + 20.0)
        images = torch.full((len(order), 3, 4, 4), step)
        observed_images, context, mask = runtime._encode_condition(
            {
                "task_descriptions": order,
                "states": states,
                "main_images": images,
            }
        )
        assert torch.equal(observed_images, images)
        assert torch.equal(context[:, -1], states.to(torch.bfloat16))
        assert not context[:, 2].any()
        assert mask.all()
        assert context[:, 0, 0].tolist() == [
            prompts.index(prompt) + 1 for prompt in order
        ]
        context.zero_()  # Consumers cannot mutate the cached raw language tensor.
    assert len(calls) == 10
    assert len(runtime._text_context_cache) == 10


def test_text_cache_is_bounded_for_supported_large_prompt_suites(tmp_path):
    prompts = [f"variant {index}" for index in range(34)]
    for prompt in prompts:
        _save_context(tmp_path, prompt, 1)
    cache = OrderedDict()
    _load_cached_text_contexts(
        prompts,
        cache_dir=tmp_path,
        context_len=3,
        expected_dim=2,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        memory_cache=cache,
    )
    assert len(cache) == 32
    assert list(cache) == prompts[2:]
