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

from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.wam_policy.critic import (
    CriticKind,
    FastWAMCurrentFrameFeatureConfig,
    FastWAMCurrentFrameValueCritic,
    critic_parent_checkpoint_sha256,
    pool_current_frame_video_values,
)


def _condition(values: torch.Tensor, *, layer_index: int = 14):
    caches = [
        {"v": torch.zeros(values.shape[0], values.shape[1], values.shape[2])}
        for _ in range(30)
    ]
    caches[layer_index] = {"v": values}
    return SimpleNamespace(
        video_kv_cache=caches,
        current_frame_video_tokens=2,
    )


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("mean_token", [[2.0, 3.0], [10.0, 11.0]]),
        ("first_token", [[1.0, 2.0], [9.0, 10.0]]),
        ("last_token", [[3.0, 4.0], [11.0, 12.0]]),
    ],
)
def test_current_frame_video_value_pooling_is_detached_fp32(
    pooling: str,
    expected: list[list[float]],
) -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [10_000.0, 10_000.0]],
            [[9.0, 10.0], [11.0, 12.0], [-10_000.0, -10_000.0]],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    config = FastWAMCurrentFrameFeatureConfig(
        input_dim=2,
        layer_index=14,
        pooling=pooling,
    )

    pooled = pool_current_frame_video_values(_condition(values), config)

    assert pooled.dtype is torch.float32
    assert pooled.shape == (2, 2)
    assert pooled.is_contiguous()
    assert not pooled.requires_grad
    assert torch.equal(pooled, torch.tensor(expected))


def test_current_frame_feature_config_rejects_invalid_source_layer_and_width() -> None:
    with pytest.raises(ValueError, match="source"):
        FastWAMCurrentFrameFeatureConfig(input_dim=2, source="video_key")
    with pytest.raises(ValueError, match="layer_index"):
        FastWAMCurrentFrameFeatureConfig(input_dim=2, layer_index=-1)
    with pytest.raises(ValueError, match="layer_index"):
        FastWAMCurrentFrameFeatureConfig(input_dim=2, layer_index=14.5)
    with pytest.raises(ValueError, match="input_dim"):
        FastWAMCurrentFrameFeatureConfig(input_dim=2.5)

    values = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match="input width"):
        pool_current_frame_video_values(
            _condition(values),
            FastWAMCurrentFrameFeatureConfig(input_dim=3),
        )
    with pytest.raises(ValueError, match="outside"):
        pool_current_frame_video_values(
            _condition(values),
            FastWAMCurrentFrameFeatureConfig(input_dim=4, layer_index=30),
        )


def test_future_video_values_do_not_change_current_frame_feature() -> None:
    current = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    first = torch.cat((current, torch.full((1, 3, 2), -100.0)), dim=1)
    second = torch.cat((current, torch.full((1, 3, 2), 100.0)), dim=1)
    config = FastWAMCurrentFrameFeatureConfig(input_dim=2)

    first_feature = pool_current_frame_video_values(_condition(first), config)
    second_feature = pool_current_frame_video_values(_condition(second), config)

    torch.testing.assert_close(first_feature, second_feature)


def test_current_frame_critic_updates_only_its_value_head() -> None:
    critic = FastWAMCurrentFrameValueCritic(
        input_dim=3,
        hidden_sizes=(4, 2),
        activation="tanh",
        bias_last=True,
    )
    features = torch.randn(5, 3, requires_grad=True)

    values = critic.value_from_features(features)
    values.sum().backward()

    assert values.shape == (5,)
    assert features.grad is None
    assert critic.trainable_parameters() == list(critic.value_head.parameters())
    assert all(parameter.grad is not None for parameter in critic.parameters())
    assert list(critic.named_modules())[0][0] == ""
    assert not hasattr(critic, "actor")


def test_current_frame_critic_supports_gelu_and_scalar_output() -> None:
    critic = FastWAMCurrentFrameValueCritic(
        input_dim=3,
        hidden_sizes=(4,),
        activation="gelu",
        bias_last=False,
    )

    assert critic.value_from_features(torch.ones(2, 3)).shape == (2,)
    assert critic.value_head.mlp[-1].bias is None


def test_critic_parent_identity_depends_on_backend() -> None:
    digest = "a" * 64
    assert (
        critic_parent_checkpoint_sha256(
            {
                "kind": CriticKind.PI0_5_VALUE_AFTER_VLM.value,
                "backbone_checkpoint_sha256": digest,
            }
        )
        == digest
    )
    assert (
        critic_parent_checkpoint_sha256(
            {
                "kind": CriticKind.FASTWAM_CURRENT_FRAME_VALUE.value,
                "backbone_checkpoint_sha256": None,
            }
        )
        is None
    )


def test_unknown_critic_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported FastWAM critic kind"):
        CriticKind.parse("custom_target")
