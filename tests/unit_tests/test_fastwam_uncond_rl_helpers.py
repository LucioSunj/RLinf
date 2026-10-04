# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exercise synchronous helper messages, buckets, and optimizer boundaries."""

import copy
import queue
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.wam_policy import uncond_rl_helpers as h


@dataclass(frozen=True, slots=True)
class _ReplayRecord:
    tensor: torch.Tensor
    name: str

    def __post_init__(self):
        if self.name != "already-valid":
            raise ValueError("Invalid replay fixture.")


def test_message_roundtrip_keeps_all_tensors_in_channel_tensor_field(monkeypatch):
    source = torch.arange(12).reshape(3, 4).t()
    original = {
        "record": _ReplayRecord(source, "already-valid"),
        "nested": ([], {"x": source[1]}),
    }

    def repeat_validation(self):
        raise AssertionError("Replay was validated before transport.")

    monkeypatch.setattr(_ReplayRecord, "__post_init__", repeat_validation)
    message = h.pack_helper_message(
        "microbatch", original, version=3, optimizer_step=4, index=7
    )
    assert len(message.tensors) == 2
    assert all(
        t.device.type == "cpu" and t.is_contiguous() for t in message.tensors.values()
    )
    restored = h.unpack_helper_message(message)
    assert isinstance(restored["record"], _ReplayRecord)
    assert isinstance(restored["nested"], tuple)
    assert torch.equal(restored["record"].tensor, source)
    moved = h.map_prepared_tensors(restored, lambda tensor: tensor + 1)
    assert torch.equal(moved["record"].tensor, source + 1)
    assert torch.equal(original["record"].tensor, source)


def test_cpu_prefetch_prepares_once_and_retains_original_order():
    calls = []

    def prepare(batch):
        calls.append(batch["index"])
        return {"index": batch["index"], "ready": True}

    result = list(
        h.cpu_to_device_prefetch(
            ({"index": i} for i in range(5)), prepare=prepare, device="cpu"
        )
    )
    assert calls == list(range(5))
    assert result == [(i, {"index": i, "ready": True}) for i in range(5)]


def test_two_transfer_slots_wait_for_backward_and_reuse_capacity(monkeypatch):
    events = []
    created = []

    class Event:
        def __init__(self):
            self.name = ("copy" if len(created) % 2 == 0 else "compute") + str(
                len(created) // 2
            )
            created.append(self)

        def record(self, stream):
            events.append(("record", self.name))

        def synchronize(self):
            events.append(("synchronize", self.name))

    class Stream:
        def wait_event(self, event):
            events.append(("wait", event.name))

    allocations = []
    original_empty = torch.empty

    def allocate(*args, **kwargs):
        allocations.append(
            (args, kwargs.get("device"), kwargs.get("pin_memory", False))
        )
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda target: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    batches = [
        {"value": torch.full((size,), float(index))}
        for index, size in enumerate((4, 4, 3, 4, 6))
    ]
    addresses = []
    for index, batch in h.cpu_to_device_prefetch(
        batches, prepare=lambda x: x, device="cuda:0"
    ):
        assert torch.equal(batch["value"], batches[index]["value"])
        addresses.append(batch["value"].data_ptr())
        events.append(("backward", index))
    assert len(created) == 4
    assert len(allocations) == 6  # two banks, plus one growth of slot zero
    assert addresses[0] == addresses[2]
    assert addresses[1] == addresses[3]
    for index in range(5):
        backward = events.index(("backward", index))
        assert events[backward + 1] == ("record", f"compute{index % 2}")
        if index < 3:
            assert events[backward + 2] == ("synchronize", f"copy{index % 2}")
            assert events[backward + 3] == ("wait", f"compute{index % 2}")
    assert events[-4:] == [
        ("synchronize", "copy0"),
        ("synchronize", "compute0"),
        ("synchronize", "copy1"),
        ("synchronize", "compute1"),
    ]


class _ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25, -0.3, 0.7]))
        self.conditional = nn.Parameter(torch.tensor([0.2, -0.4]))
        self.real_zero = nn.Parameter(torch.tensor([0.8]))
        self.never_used = nn.Parameter(torch.tensor([0.9]))
        self.alias = self.weight

    def rollout_runtime_state_dict(self):
        return {"episode": torch.tensor([17, 21]), "history": {"route": [0, 1]}}


@dataclass(frozen=True)
class _Route:
    route_used: torch.Tensor


def _backward(model, context, batch, *, selected_loss_scales, normalization_statistics):
    prediction = (model.weight * batch["x"]).sum()
    if batch["selected"]:
        prediction = prediction + (model.conditional * batch["x"][:2]).sum()
    # Exercise genuine zero gradients, which Adam must not mistake for None.
    if batch["zero_used"]:
        prediction = prediction + model.real_zero.sum() * 0
    loss = (prediction - batch["target"]).square() / context.gradient_accumulation
    loss.backward()
    return {"loss": loss.detach(), "index": batch["index"]}


class _FSDPNameWrapper(nn.Module):
    """Reproduce FSDP's registered child names without a device or process group."""

    def __init__(self, model):
        super().__init__()
        self._fsdp_wrapped_module = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)


@pytest.mark.parametrize("nested", [False, True])
def test_helper_manifest_matches_fsdp_owner_without_replacing_parameters(nested):
    helper = nn.Sequential(_ToyPolicy(), nn.Linear(3, 2))
    owner = copy.deepcopy(helper)
    original_parameters = tuple(owner.parameters())
    if nested:
        owner[0] = _FSDPNameWrapper(owner[0])
    owner = _FSDPNameWrapper(owner)
    owner_layout = h.TrainableParameterLayout(owner, bucket_bytes=8)
    helper_layout = h.TrainableParameterLayout(helper, bucket_bytes=8)

    assert owner_layout.manifest == helper_layout.manifest
    assert owner_layout.buckets == helper_layout.buckets
    assert len(owner_layout.parameters) == len(original_parameters)
    assert all(
        actual is expected
        for actual, expected in zip(
            owner_layout.parameters, original_parameters, strict=True
        )
    )
    assert any(
        "_fsdp_wrapped_module." in name for name, _ in owner_layout.named_parameters
    )
    # Canonical names must still retain the actual policy structure and shape.
    different_shape = nn.Sequential(_ToyPolicy(), nn.Linear(3, 1))
    assert owner_layout.manifest != h.TrainableParameterLayout(different_shape).manifest


def test_buckets_preserve_aliases_none_gradients_and_real_zeros():
    model = _ToyPolicy()
    layout = h.TrainableParameterLayout(model, bucket_bytes=8)
    assert len(layout.parameters) == 4
    assert (
        max(sum(part.end - part.begin for part in bucket) for bucket in layout.buckets)
        <= 2
    )
    model.weight.grad = torch.tensor([1.0, 2.0, 3.0])
    model.real_zero.grad = torch.zeros_like(model.real_zero)
    destination = copy.deepcopy(model)
    target = h.TrainableParameterLayout(destination, bucket_bytes=8)
    for bucket in layout.buckets:
        target.add_gradient_bucket(
            bucket, layout.pack_bucket(bucket, gradients=True), layout.presence()
        )
    assert torch.equal(destination.weight.grad, model.weight.grad)
    assert destination.real_zero.grad is not None
    assert destination.conditional.grad is None
    assert destination.never_used.grad is None
    weight_identity = id(destination.weight)
    for bucket in layout.buckets:
        target.copy_weight_bucket(
            bucket, layout.pack_bucket(bucket, gradients=False) + 1
        )
    assert id(destination.weight) == weight_identity
    assert destination.alias is destination.weight
    assert torch.equal(destination.weight, model.weight + 1)


class _Channel:
    def __init__(self, helpers):
        self.queues = {rank: queue.Queue(maxsize=2) for rank in helpers}

    def put(self, item, *, key):
        self.queues[key].put(item, timeout=10)

    def get(self, *, key):
        return self.queues[key].get(timeout=10)


class _TensorTransport:
    def __init__(self, helpers):
        self.gradients = {rank: queue.Queue(maxsize=1) for rank in helpers}
        self.weights = {rank: queue.Queue(maxsize=1) for rank in helpers}
        self.helpers = helpers
        self.broadcasts = dict.fromkeys((-1, *helpers), 0)

    def broadcast(self, rank, value):
        self.broadcasts[rank] += 1
        if rank == -1:
            for helper in self.helpers:
                self.weights[helper].put(value.clone(), timeout=10)
            return value
        return self.weights[rank].get(timeout=10)


class _Owner(h.UncondTrainingOwnerMixin):
    def __init__(self, model, cfg, transport):
        self.model = model
        self.cfg = cfg
        self.transport = transport
        self._world_size = 1
        self.device = "cpu"
        self.version = 3
        self.optimizer_steps = 0
        self._sync_weight_comm_options = None
        self.gradient_accumulation = (
            cfg.actor.global_batch_size // cfg.actor.micro_batch_size
        )
        self.grad_scaler = SimpleNamespace(is_enabled=lambda: False)

    def _fastwam_policy_module(self):
        return self.model

    def prepare_cpu_microbatch(self, batch):
        return batch

    def train_micro_batch(self, *, micro_batch, metrics, is_last, selected_loss_scales):
        values = _backward(
            self.model,
            self,
            micro_batch,
            selected_loss_scales=selected_loss_scales,
            normalization_statistics=None,
        )
        metrics.update({name: [value] for name, value in values.items()})

    def recv_tensor(self, destination, *, src_rank, **kwargs):
        destination.copy_(self.transport.gradients[src_rank].get(timeout=10))

    def broadcast(self, *, object, **kwargs):
        return self.transport.broadcast(-1, object)


@pytest.mark.parametrize("participant_count", [2, 7])
def test_helper_service_matches_single_owner_across_fifteen_adam_steps(
    participant_count, monkeypatch
):
    helpers = tuple(range(participant_count - 1))
    commands, results = _Channel(helpers), _Channel(helpers)
    transport = _TensorTransport(helpers)
    reference = _ToyPolicy()
    owner = _Owner(
        copy.deepcopy(reference),
        SimpleNamespace(
            actor=SimpleNamespace(
                group_name="actor",
                global_batch_size=392,
                micro_batch_size=4,
                fsdp_config=SimpleNamespace(
                    amp_autocast=SimpleNamespace(enabled=False)
                ),
            ),
            rollout=SimpleNamespace(group_name="rollout"),
        ),
        transport,
    )
    owner.configure_training_helpers(
        helper_ranks=helpers, command_channel=commands, result_channel=results
    )
    # Small buckets force parameters to span several transport messages.
    owner._training_helper_layout = h.TrainableParameterLayout(
        owner.model, bucket_bytes=8
    )
    real_layout = h.TrainableParameterLayout
    monkeypatch.setattr(
        h, "TrainableParameterLayout", lambda model: real_layout(model, bucket_bytes=8)
    )
    monkeypatch.setattr(h, "backward_helper_microbatch", _backward)
    reports, failures, workers, threads = {}, [], [], []

    def run(worker):
        try:
            reports[worker._rank] = h.serve_actor_training(
                worker,
                helper_ranks=helpers,
                command_channel=commands,
                result_channel=results,
                version=3,
            )
        except Exception as exc:
            failures.append(exc)

    for rank in helpers:
        worker = SimpleNamespace(
            hf_model=copy.deepcopy(reference),
            _rank=rank,
            cfg=owner.cfg,
            version=3,
            device="cpu",
            _sync_weight_comm_options=None,
        )
        worker.send_tensor = lambda tensor, rank=rank, **kwargs: transport.gradients[
            rank
        ].put(tensor.clone(), timeout=10)
        worker.broadcast = lambda object, rank=rank, **kwargs: transport.broadcast(
            rank, object
        )
        workers.append(worker)
        thread = threading.Thread(target=run, args=(worker,), daemon=True)
        thread.start()
        threads.append(thread)

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
        for model in (reference, owner.model)
    ]
    context = h.HelperLossContext(owner.cfg, actor_version=3)
    assert context.gradient_accumulation == owner.gradient_accumulation == 98
    generator = torch.Generator().manual_seed(73)
    for step in range(15):
        # Include uneven splits and opportunities with zero assigned helper work.
        count = (98, 31, 7, 1)[step % 4]
        batches = [
            {
                "x": torch.randn(3, generator=generator),
                "target": torch.randn((), generator=generator),
                "selected": index % 3 == 0 and step % 3 != 1,
                "zero_used": index % 2 == 0 and step % 3 != 1,
                "index": index,
            }
            for index in range(count)
        ]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            _backward(
                reference,
                context,
                batch,
                selected_loss_scales=None,
                normalization_statistics=None,
            )
        metrics = {}
        owner._execute_training_microbatches(
            list(batches), metrics=metrics, selected_loss_scales={"flow": 0.5}
        )
        assert metrics["index"] == list(range(count))
        for left, right in zip(
            reference.parameters(), owner.model.parameters(), strict=True
        ):
            assert (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)
        for model, optimizer in zip((reference, owner.model), optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
        owner.optimizer_steps += 1
        owner._after_training_optimizer_step()
        for left, right in zip(
            reference.parameters(), owner.model.parameters(), strict=True
        ):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7)
            for name in optimizers[0].state[left]:
                torch.testing.assert_close(
                    optimizers[0].state[left][name],
                    optimizers[1].state[right][name],
                    rtol=2e-5,
                    atol=2e-7,
                )
    for rank in helpers:
        commands.put(
            h.pack_helper_message(
                "finish_update",
                {"optimizer_opportunities": 15},
                version=3,
                optimizer_step=-1,
            ),
            key=rank,
        )
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), (
            "Helper did not leave its final broadcast boundary."
        )
    assert not failures
    assert set(reports) == set(helpers)
    for worker in workers:
        assert worker.version == 3
        assert not worker.hf_model.training
        assert reports[worker._rank]["optimizer_opportunities"] == 15
        assert all(parameter.grad is None for parameter in worker.hf_model.parameters())
        for left, right in zip(
            owner.model.parameters(), worker.hf_model.parameters(), strict=True
        ):
            assert torch.equal(left, right)
    assert set(transport.broadcasts.values()) == {
        15 * len(owner._training_helper_layout.buckets)
    }


def test_helper_service_restores_rng_on_command_failure(monkeypatch):
    """A failed service cannot consume the rollout's saved random stream."""
    import random

    import numpy as np

    from rlinf.utils.utils import get_rng_state

    cfg = SimpleNamespace()
    worker = SimpleNamespace(hf_model=_ToyPolicy(), _rank=0, version=10, cfg=cfg)
    expected = get_rng_state()

    class FailingChannel:
        def get(self, *, key):
            random.random()
            np.random.random()
            torch.rand(3)
            raise RuntimeError("test command failure")

    with pytest.raises(RuntimeError, match="test command failure"):
        h.serve_actor_training(
            worker,
            helper_ranks=(0,),
            command_channel=FailingChannel(),
            result_channel=None,
            version=10,
        )
    assert h._same_runtime(expected, get_rng_state())
    assert not worker.hf_model.training
    assert all(parameter.grad is None for parameter in worker.hf_model.parameters())
