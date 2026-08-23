# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from rlinf.hybrid_engines.fsdp.strategy.fsdp import mixed_precision_from_config


def test_mixed_precision_carries_the_configured_root_cast():
    config = OmegaConf.create(
        {
            "param_dtype": "fp32",
            "reduce_dtype": "fp32",
            "buffer_dtype": "bf16",
            "cast_root_forward_inputs": False,
        }
    )

    policy = mixed_precision_from_config(config)

    assert policy.param_dtype is torch.float32
    assert policy.reduce_dtype is torch.float32
    assert policy.buffer_dtype is torch.bfloat16
    assert policy.cast_root_forward_inputs is False


def test_mixed_precision_keeps_the_upstream_root_cast_default():
    config = OmegaConf.create(
        {"param_dtype": "bf16", "reduce_dtype": "bf16", "buffer_dtype": "bf16"}
    )

    assert mixed_precision_from_config(config).cast_root_forward_inputs is True


class _FrozenParent(nn.Module):
    """Stands in for the frozen BF16 VAE and DiT kept outside FSDP."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)


class _Composite(nn.Module):
    """A frozen BF16 parent plus one trainable FP32 master, as FastWAM builds."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen = _FrozenParent().to(dtype=torch.bfloat16)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)
        self.trainable = nn.Linear(4, 2)

    def forward(self, image: torch.Tensor):
        observed_dtype = image.dtype
        latents = self.frozen(image)
        return observed_dtype, self.trainable(latents.float().mean(dim=(2, 3, 4)))


def _wrap(cast_root_forward_inputs: bool) -> FSDP:
    model = _Composite().cuda()
    return FSDP(
        module=model,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_root_forward_inputs=cast_root_forward_inputs,
        ),
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
        ignored_states=tuple(
            parameter for parameter in model.parameters() if not parameter.requires_grad
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_root_cast_decides_whether_frozen_parents_receive_their_own_dtype():
    """An FP32 param_dtype must not upcast inputs bound for frozen BF16 parents.

    With the root cast left on, the FP32 master-weight fix reaches the frozen
    VAE as `RuntimeError: Input type (float) and bias type (c10::BFloat16)
    should be the same`. See `docs/BF16_PARAMETER_UPDATE_LOSS.md`.
    """

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29583")
        dist.init_process_group("nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)
    try:
        image = torch.zeros(1, 3, 1, 4, 4, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(RuntimeError, match="should be the same"):
            _wrap(cast_root_forward_inputs=True)(image)

        observed_dtype, values = _wrap(cast_root_forward_inputs=False)(image)
        assert observed_dtype is torch.bfloat16
        assert values.dtype is torch.float32
    finally:
        if owns_process_group:
            dist.destroy_process_group()
