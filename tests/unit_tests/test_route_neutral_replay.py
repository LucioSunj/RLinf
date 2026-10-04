# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exercise compact replay through collection, transport, shuffle and replay."""

import gc
import pickle
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    convert_trajectories_to_batch,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
    RouteNeutralOnlineIDMBCFSDPActor,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.replay import (
    REPLAY_ROW,
    TEXT_ID,
    RouteNeutralReplayStore,
    RouteNeutralRolloutResult,
)
from rlinf.scheduler.cluster.utils import (
    extract_dataclass_tensor_fields,
    unflatten_dataclass_tensor_fields,
)
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_train


def _step(rank, step, *, dones):
    text_ids = torch.tensor([0, 1, 0, 1])
    text = (text_ids[:, None, None] + rank * 10).expand(4, 8, 16).float()
    state = (torch.arange(4) + step * 10 + rank * 100).float()
    context = torch.cat((text, state[:, None, None].expand(4, 1, 16)), dim=1)
    return ChunkStepResult(
        forward_inputs={
            TEXT_ID: text_ids,
            "fastwam_context": context,
            "fastwam_context_mask": torch.ones(4, 9, dtype=torch.bool),
            "fastwam_first_frame_latents": state[:, None].expand(4, 12).clone(),
            "gate_condition_visual_layer_0_key": state[:, None].expand(4, 32).clone(),
            "gate_condition_visual_layer_0_mask": torch.ones(4, 2, dtype=torch.bool),
            "critic_pooled": state[:, None].expand(4, 3).clone(),
            "denoise_indices": torch.tensor([0, -1, 2, -1]),
            "online_idm_bc_teacher_present": torch.tensor([True, False, True, False]),
            "online_idm_bc_teacher_seconds": state,
        },
        dones=dones,
        prev_logprobs=state[:, None],
        prev_values=(state + 1)[:, None],
        rewards=state[:, None],
    )


def _collect(rank, *, mask_after_terminal=True):
    compact = RouteNeutralRolloutResult(mask_after_terminal=mask_after_terminal)
    original = EmbodiedRolloutResult()
    for step in range(4):
        dones = torch.zeros(4, 2, dtype=torch.bool)
        if step == 1:
            dones[1, 0] = True
        if step == 2:
            dones[2, 1] = True
        if step == 3:
            dones[:] = True
        result = _step(rank, step, dones=dones)
        compact.append_step_result(result)
        original.append_step_result(result)
    bootstrap = ChunkStepResult(
        dones=torch.ones(4, 2, dtype=torch.bool),
        prev_values=torch.zeros(4, 1),
    )
    compact.append_step_result(bootstrap)
    original.append_step_result(bootstrap)
    return compact, original


def _channel_roundtrip(payload):
    fields, tensors, metadata = extract_dataclass_tensor_fields(payload)
    assert set(fields) == {"replay_tensors"}
    assert len(tensors) == len(payload.replay_tensors)
    assert all(tensor.is_contiguous() for tensor in tensors)
    skeleton = replace(payload, **dict.fromkeys(fields))
    # Exercise the actual Channel split between tensor transfer and pickle.
    assert len(pickle.dumps(skeleton)) < 25_000
    return replace(
        pickle.loads(pickle.dumps(skeleton)),
        **unflatten_dataclass_tensor_fields(
            metadata, [value.clone() for value in tensors]
        ),
    )


@pytest.mark.parametrize("mask_after_terminal", [True, False])
def test_packed_rows_preserve_time_masks_shuffle_and_rank_text_namespaces(
    mask_after_terminal,
):
    store = RouteNeutralReplayStore()
    packed_trajectories, original_trajectories = [], []
    for rank in range(2):
        compact, original = _collect(rank, mask_after_terminal=mask_after_terminal)
        payload = compact.to_splited_trajectories_by_sizes([4], consume=True)[0]
        assert (
            len([key for key in payload.replay_tensors if key.startswith("text/")]) == 2
        )
        assert sum(payload.block_sizes) == (9 if mask_after_terminal else 16)
        assert compact.replay_tensors == {}
        compact.clear()
        packed_trajectories.append(store.add(_channel_roundtrip(payload)))
        original_trajectories.append(original.to_trajectory())
    packed = convert_trajectories_to_batch(packed_trajectories, consume=True)
    original = convert_trajectories_to_batch(original_trajectories, consume=True)
    for name in ("dones", "prev_values", "prev_logprobs", "rewards"):
        assert torch.equal(packed[name], original[name])
    mask, counts = compute_loss_mask(original["dones"])
    if mask_after_terminal:
        assert torch.equal(packed["forward_inputs"][REPLAY_ROW] >= 0, mask.any(-1))
    for batch in (packed, original):
        batch["loss_mask"] = mask.any(-1, keepdim=True)
        batch["loss_mask_sum"] = counts[..., -1:]
        batch.pop("dones")
    shuffle = torch.randperm(32, generator=torch.Generator().manual_seed(42))
    packed = process_nested_dict_for_train(packed, shuffle, consume=True)
    original = process_nested_dict_for_train(original, shuffle, consume=True)
    # Keep the same optimizer partitions, including an all-masked microbatch.
    for compact_batch, old_batch in zip(
        split_dict_to_chunk(packed, 8), split_dict_to_chunk(original, 8), strict=True
    ):
        restored = store.materialize(compact_batch["forward_inputs"])
        valid = compact_batch["loss_mask"].reshape(-1)
        assert torch.equal(compact_batch["loss_mask"], old_batch["loss_mask"])
        assert torch.equal(compact_batch["loss_mask_sum"], old_batch["loss_mask_sum"])
        for name, expected in old_batch["forward_inputs"].items():
            if mask_after_terminal:
                assert torch.equal(restored[name][valid], expected[valid]), name
            else:
                assert torch.equal(restored[name], expected), name
        assert torch.isfinite(restored["fastwam_context"]).all()
        assert torch.equal(
            restored["online_idm_bc_teacher_present"],
            old_batch["forward_inputs"]["online_idm_bc_teacher_present"],
        )


def test_packed_collector_does_not_retain_source_storage_or_terminal_rows():
    result = RouteNeutralRolloutResult(mask_after_terminal=True)
    first = _step(0, 0, dones=torch.zeros(4, 2, dtype=torch.bool))
    source = weakref.ref(first.forward_inputs["fastwam_context"])
    result.append_step_result(first)
    first.forward_inputs["fastwam_context"].fill_(-999)
    del first
    gc.collect()
    assert source() is None
    retained_bytes = sum(
        t.numel() * t.element_size() for t in result.replay_tensors.values()
    )
    result.append_step_result(_step(0, 1, dones=torch.ones(4, 2, dtype=torch.bool)))
    assert (
        sum(t.numel() * t.element_size() for t in result.replay_tensors.values())
        == retained_bytes
    )
    assert result.forward_inputs[-1][REPLAY_ROW].tolist() == [-1] * 4
    store = RouteNeutralReplayStore()
    payload = result.to_splited_trajectories_by_sizes([4], consume=True)[0]
    trajectory = store.add(payload)
    restored = store.materialize(
        {key: value[0] for key, value in trajectory.forward_inputs.items()}
    )
    assert restored["fastwam_context"][0, 0, 0] == 0
    # Text occupies one bank per unique instruction, regardless of time count.
    assert len([key for key in payload.replay_tensors if key.startswith("text/")]) == 2


def test_bootstrap_resets_terminal_tracking_for_the_next_rollout_epoch():
    result = RouteNeutralRolloutResult(mask_after_terminal=True)
    result.append_step_result(_step(0, 0, dones=torch.zeros(4, 2, dtype=torch.bool)))
    result.append_step_result(_step(0, 1, dones=torch.ones(4, 2, dtype=torch.bool)))
    result.append_step_result(ChunkStepResult(dones=torch.ones(4, 2, dtype=torch.bool)))
    result.append_step_result(_step(0, 2, dones=torch.zeros(4, 2, dtype=torch.bool)))
    assert result.forward_inputs[-1][REPLAY_ROW].tolist() == [4, 5, 6, 7]


def test_actor_hydrates_only_current_microbatch_and_releases_bank(monkeypatch):
    from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import (
        OnlineIDMBCFSDPActor,
    )

    actor = object.__new__(RouteNeutralOnlineIDMBCFSDPActor)
    actor.cfg = SimpleNamespace(
        route_neutral_online_implementation=SimpleNamespace(
            release_host_memory_after_trajectory_receive=True,
        )
    )
    collector, _ = _collect(0)
    received = collector.to_splited_trajectories_by_sizes([4], consume=True)
    trajectories = actor._prepare_received_trajectories(received)
    bank = weakref.ref(actor._route_neutral_replay)
    forward = {
        key: value[0, :1] for key, value in trajectories[0].forward_inputs.items()
    }
    observed = []
    monkeypatch.setattr(
        OnlineIDMBCFSDPActor,
        "train_micro_batch",
        lambda self, batch, *args, **kwargs: observed.append(batch),
    )
    actor.train_micro_batch({"forward_inputs": forward}, {}, is_last=True)
    assert observed[0]["forward_inputs"]["fastwam_context"].shape[0] == 1
    assert "fastwam_context" not in forward
    actor._release_consumed_rollout_batch(phase="test")
    assert bank() is None
