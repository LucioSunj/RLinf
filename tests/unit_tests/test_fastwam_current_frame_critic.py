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

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.condition_kv import ConditionLayerKV
from fastwam.models.wan22.kv_tap import (
    GateKVSnapshot,
    GateLayerKV,
    KeyValueBank,
    KVSource,
)

from rlinf.models.embodiment.wam_policy import libero_runtime as _runtime_module
from rlinf.models.embodiment.wam_policy.critic import (
    CriticKind,
    FastWAMCurrentFrameValueCritic,
    FastWAMValueFeatures,
    FastWAMValueTransformer,
    FastWAMValueTransformerConfig,
    critic_parent_checkpoint_sha256,
    extract_fastwam_value_features,
)


def _config(
    *,
    pooling: str = "mean_token",
    layer_indices: tuple[int, ...] = (1,),
    sources: tuple[str, ...] = ("current_frame_video", "text_state_context"),
) -> FastWAMValueTransformerConfig:
    return FastWAMValueTransformerConfig(
        num_mot_layers=3,
        source_num_heads=2,
        source_head_dim=2,
        layer_indices=layer_indices,
        sources=sources,
        hidden_dim=6,
        num_query_tokens=3,
        ffn_multiplier=2,
        pooling=pooling,
    )


def _bank(
    source: KVSource,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> KeyValueBank:
    return KeyValueBank(
        source=source,
        key=value + 0.25,
        value=value,
        valid_mask=(
            torch.ones(value.shape[:2], dtype=torch.bool) if mask is None else mask
        ),
    )


def _features(
    *,
    video_offset: float = 0.0,
    context_offset: float = 0.0,
    requires_grad: bool = False,
    layer_indices: tuple[int, ...] = (1,),
) -> FastWAMValueFeatures:
    generator = torch.Generator().manual_seed(9)
    video = (torch.randn(2, 4, 4, generator=generator) + video_offset).requires_grad_(
        requires_grad
    )
    context = (
        torch.randn(2, 5, 4, generator=generator) + context_offset
    ).requires_grad_(requires_grad)
    context_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    return FastWAMValueFeatures(
        tuple(
            ConditionLayerKV(
                layer_index=index,
                current_frame_video=_bank(KVSource.CURRENT_FRAME_VIDEO, video),
                context=_bank(
                    KVSource.TEXT_STATE_CONTEXT,
                    context,
                    mask=context_mask,
                ),
            )
            for index in layer_indices
        )
    )


def test_idm_rollout_replay_and_bootstrap_use_identical_critic_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require three-path parity measured at max-abs 0.0 in this synthetic case."""

    class _Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.action_expert = SimpleNamespace(action_dim=3)
            self.infer_action_scheduler = SimpleNamespace(num_train_timesteps=1000)

    runtime = object.__new__(_runtime_module.LiberoFastWAMRuntime)
    runtime.actor = _Actor()
    runtime.critic_feature_config = _config()
    runtime.flow_sde_noise_level = 0.5
    runtime.flow_sde_ignore_last_transition = True
    runtime.seeded_noise_device = "cpu"
    runtime.gate_denoise_last_n = 1
    runtime.gate_replay_backend = _runtime_module.GateKVReplayBackend.STORED
    runtime.action_protocol = _runtime_module.LiberoActionProtocol(
        generation_horizon=2,
        execution_horizon=1,
        prediction_video_frames=3,
        reset_wait_steps=0,
        max_episode_steps=2,
    )
    images = torch.zeros(1, 3, 4, 4)
    context = torch.zeros(1, 5, 2)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    fixed_observation = {"sample_id": torch.tensor([17])}

    def encode_condition(env_obs):
        assert env_obs is fixed_observation
        return images, context, context_mask

    runtime._encode_condition = encode_condition
    runtime._action_schedule = lambda: (
        torch.tensor([1.0]),
        torch.tensor([-1.0]),
    )

    def prepare_condition(**kwargs):
        regime = kwargs["regime"]
        marker = 1.0 if regime is PolicyRegime.UNCOND else 9.0
        return SimpleNamespace(regime=regime, marker=marker), None

    runtime._prepare_action_condition = prepare_condition

    def critic_features(condition):
        video = torch.full((1, 2, 4), condition.marker)
        text_state = torch.full((1, 3, 4), condition.marker + 0.5)
        return FastWAMValueFeatures(
            (
                ConditionLayerKV(
                    layer_index=1,
                    current_frame_video=_bank(
                        KVSource.CURRENT_FRAME_VIDEO,
                        video,
                    ),
                    context=_bank(
                        KVSource.TEXT_STATE_CONTEXT,
                        text_state,
                    ),
                ),
            )
        )

    runtime._critic_features_from_condition = critic_features
    runtime._velocity = lambda condition, **_kwargs: condition
    runtime._denormalize_action_stages = lambda actions, env_obs: (actions, None)

    def sample_flow(initial_noise, *, velocity_fn, **_kwargs):
        gate_value = torch.zeros(1, 1, 4)
        snapshot = GateKVSnapshot(
            (
                GateLayerKV(
                    layer_index=0,
                    denoise_timestep=torch.zeros(1),
                    current_mode=(velocity_fn.regime,),
                    current_frame_video=_bank(
                        KVSource.CURRENT_FRAME_VIDEO,
                        gate_value,
                    ),
                    action=_bank(KVSource.ACTION, gate_value),
                    context=_bank(KVSource.TEXT_STATE_CONTEXT, gate_value),
                    actor_version=7,
                ),
            )
        )
        return SimpleNamespace(
            actions=torch.zeros_like(initial_noise),
            old_log_probs=torch.zeros_like(initial_noise),
            chains=torch.zeros(1, 2, *initial_noise.shape[1:]),
            denoise_indices=torch.zeros(1, dtype=torch.long),
            gate_taps=(snapshot,),
        )

    monkeypatch.setattr(_runtime_module, "sample_action_flow_sde", sample_flow)
    route = int(_runtime_module.WAMRoute.IDM)
    rollout = runtime.sample_action_batch(
        env_obs=fixed_observation,
        routes=torch.tensor([route]),
        mode="train",
        actor_version=7,
        collect_replay=False,
    )
    route_info = _runtime_module.ChunkRouteRecord(
        route_used=torch.tensor([route]),
        route_was_forced=torch.tensor([True]),
        chunk_ids=torch.tensor([0]),
        episode_ids=torch.tensor([17]),
        route_source_chunk_ids=torch.tensor([-1]),
        actor_versions=torch.tensor([7]),
    )
    replay = runtime.replay_action_batch(
        forward_inputs={
            "flow_chains": rollout.flow_chains,
            "denoise_indices": rollout.denoise_indices,
            "fastwam_images": images,
            "fastwam_context": context,
            "fastwam_context_mask": context_mask,
        },
        route_info=route_info,
    )
    bootstrap = runtime.critic_features(env_obs=fixed_observation)

    def max_abs_diff(
        left: FastWAMValueFeatures,
        right: FastWAMValueFeatures,
    ) -> float:
        assert left.layer_indices == right.layer_indices
        differences = []
        for layer_index in left.layer_indices:
            left_layer = left.layer(layer_index)
            right_layer = right.layer(layer_index)
            for source_name in ("current_frame_video", "context"):
                left_bank = getattr(left_layer, source_name)
                right_bank = getattr(right_layer, source_name)
                assert torch.equal(left_bank.valid_mask, right_bank.valid_mask)
                differences.extend(
                    (left_bank.key - right_bank.key).abs().reshape(-1).tolist()
                )
                differences.extend(
                    (left_bank.value - right_bank.value).abs().reshape(-1).tolist()
                )
        return max(differences, default=0.0)

    rollout_replay_max_abs = max_abs_diff(
        rollout.critic_features,
        replay["critic_features"],
    )
    rollout_bootstrap_max_abs = max_abs_diff(
        rollout.critic_features,
        bootstrap,
    )
    parity_atol = 0.0
    assert rollout_replay_max_abs <= parity_atol
    assert rollout_bootstrap_max_abs <= parity_atol


def test_value_feature_batching_preserves_sources_masks_and_detaches() -> None:
    first = _features(requires_grad=True)
    second = _features(video_offset=3.0, context_offset=-2.0, requires_grad=True)

    batched = FastWAMValueFeatures.cat((first, second))

    assert batched.batch_size == 4
    assert batched.feature_dim == 4
    assert batched.layer_indices == (1,)
    assert batched.layer(1).current_frame_video.source is KVSource.CURRENT_FRAME_VIDEO
    assert batched.layer(1).context.source is KVSource.TEXT_STATE_CONTEXT
    assert batched.layer(1).context.valid_mask.shape == (4, 5)
    assert not batched.layer(1).current_frame_video.key.requires_grad
    assert not batched.layer(1).context.value.requires_grad


def test_feature_extraction_uses_frozen_text_projection_and_selected_layers() -> None:
    events = []
    action_expert = nn.Module()
    action_expert.text_embedding = nn.Linear(3, 4, bias=False)

    class _MoT:
        def read_condition_layer_kv(self, **kwargs):
            events.append(("layer", kwargs["layer_index"]))
            context = kwargs["context"]
            video = torch.full(
                (context.shape[0], kwargs["current_frame_video_tokens"], 4),
                float(kwargs["layer_index"] + 1),
            )
            return ConditionLayerKV(
                layer_index=kwargs["layer_index"],
                current_frame_video=_bank(KVSource.CURRENT_FRAME_VIDEO, video),
                context=_bank(
                    KVSource.TEXT_STATE_CONTEXT,
                    context,
                    mask=kwargs["context_mask"],
                ),
            )

    class _RegimeContext:
        @contextmanager
        def use(self, regime):
            events.append(("enter", regime))
            yield
            events.append(("exit", regime))

    condition = SimpleNamespace(
        context=torch.randn(2, 5, 3, requires_grad=True),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        video_kv_cache=object(),
        current_frame_video_tokens=2,
    )
    features = extract_fastwam_value_features(
        condition,
        mot=_MoT(),
        action_expert=action_expert,
        config=_config(layer_indices=(0, 2)),
        regime_context=_RegimeContext(),
    )

    assert features.layer_indices == (0, 2)
    assert events == [
        ("enter", PolicyRegime.IDM),
        ("layer", 0),
        ("layer", 2),
        ("exit", PolicyRegime.IDM),
    ]
    assert not features.layer(0).context.key.requires_grad


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("mean_token", [[4.0, 5.0], [10.0, 11.0]]),
        ("first_token", [[1.0, 2.0], [7.0, 8.0]]),
        ("last_token", [[7.0, 8.0], [13.0, 14.0]]),
    ],
)
def test_learned_query_pooling_modes(
    pooling: str,
    expected: list[list[float]],
) -> None:
    transformer = FastWAMValueTransformer(_config(pooling=pooling))
    transformer.output_norm = nn.Identity()
    queries = torch.tensor(
        [
            [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]],
            [[7.0, 8.0], [10.0, 11.0], [13.0, 14.0]],
        ]
    )

    assert torch.equal(transformer._pool_queries(queries), torch.tensor(expected))


def test_value_transformer_config_rejects_bad_layers_sources_and_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        FastWAMValueTransformerConfig(
            num_mot_layers=3,
            source_num_heads=True,
            source_head_dim=2,
        )
    with pytest.raises(ValueError, match="unique and increasing"):
        _config(layer_indices=(1, 1))
    with pytest.raises(ValueError, match="outside"):
        _config(layer_indices=(3,))
    with pytest.raises(ValueError, match="sources"):
        _config(sources=("action",))
    with pytest.raises(ValueError, match="pooling"):
        _config(pooling="max_token")


def test_video_and_prompt_state_sources_each_affect_value() -> None:
    torch.manual_seed(13)
    critic = FastWAMCurrentFrameValueCritic(config=_config())
    baseline = critic.value_from_features(_features())
    changed_video = critic.value_from_features(_features(video_offset=5.0))
    changed_context = critic.value_from_features(_features(context_offset=5.0))

    assert not torch.allclose(baseline, changed_video)
    assert not torch.allclose(baseline, changed_context)


def test_current_frame_critic_updates_only_value_head_and_detaches_sources() -> None:
    critic = FastWAMCurrentFrameValueCritic(
        config=_config(),
        hidden_sizes=(4,),
        activation="tanh",
        bias_last=True,
    )
    features = _features(requires_grad=True)
    source_video = features.layer(1).current_frame_video.value
    source_context = features.layer(1).context.value

    values = critic.value_from_features(features)
    values.sum().backward()

    assert values.shape == (2,)
    assert source_video.grad is None
    assert source_context.grad is None
    assert critic.trainable_parameters() == list(critic.value_head.parameters())
    assert all(parameter.grad is not None for parameter in critic.parameters())
    assert not hasattr(critic, "actor")


def test_current_frame_critic_supports_source_and_layer_ablation() -> None:
    critic = FastWAMCurrentFrameValueCritic(
        config=_config(
            layer_indices=(0, 2),
            sources=("text_state_context",),
        ),
        hidden_sizes=(),
        activation="gelu",
        bias_last=False,
    )

    values = critic.value_from_features(_features(layer_indices=(0, 2)))

    assert values.shape == (2,)
    assert len(critic.value_head.transformer.blocks) == 2
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
