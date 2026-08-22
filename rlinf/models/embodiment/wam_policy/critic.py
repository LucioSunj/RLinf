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

"""Configurable critic contracts for the adaptive FastWAM policy."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.condition_kv import ConditionLayerKV
from fastwam.models.wan22.gate_transformer import DirectKVAttention
from fastwam.models.wan22.kv_tap import (
    KeyValueBank,
    KVSource,
)

from rlinf.models.embodiment.modules.value_head import ValueHead


class CriticKind(str, Enum):
    """Built-in critic feature backends supported by adaptive FastWAM."""

    PI0_5_VALUE_AFTER_VLM = "pi0_5_value_after_vlm"
    FASTWAM_CURRENT_FRAME_VALUE = "fastwam_current_frame_value"

    @classmethod
    def parse(cls, value: CriticKind | str) -> CriticKind:
        """Normalize one configured critic kind."""

        try:
            return value if isinstance(value, cls) else cls(str(value))
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported FastWAM critic kind {value!r}; expected one of "
                f"{supported}."
            ) from exc


class CurrentFramePooling(str, Enum):
    """Learned-query pooling modes for the FastWAM value transformer."""

    MEAN_TOKEN = "mean_token"
    FIRST_TOKEN = "first_token"
    LAST_TOKEN = "last_token"

    @classmethod
    def parse(cls, value: CurrentFramePooling | str) -> CurrentFramePooling:
        """Normalize one configured current-frame pooling mode."""

        try:
            return value if isinstance(value, cls) else cls(str(value))
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported FastWAM critic pooling {value!r}; expected one of "
                f"{supported}."
            ) from exc


@dataclass(frozen=True)
class FastWAMValueTransformerConfig:
    """Architecture and source selection for the FastWAM value sidecar."""

    num_mot_layers: int
    source_num_heads: int
    source_head_dim: int
    layer_indices: tuple[int, ...] = (14,)
    sources: tuple[str, ...] = (
        KVSource.CURRENT_FRAME_VIDEO.value,
        KVSource.TEXT_STATE_CONTEXT.value,
    )
    hidden_dim: int = 256
    num_query_tokens: int = 4
    ffn_multiplier: int = 4
    share_blocks: bool = False
    layer_index_embedding: bool = True
    pooling: CurrentFramePooling | str = CurrentFramePooling.MEAN_TOKEN

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_indices", tuple(self.layer_indices))
        object.__setattr__(
            self, "sources", tuple(str(source) for source in self.sources)
        )
        integer_fields = {
            "num_mot_layers": self.num_mot_layers,
            "source_num_heads": self.source_num_heads,
            "source_head_dim": self.source_head_dim,
            "hidden_dim": self.hidden_dim,
            "num_query_tokens": self.num_query_tokens,
            "ffn_multiplier": self.ffn_multiplier,
        }
        invalid_integers = {
            name: value
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int) or value < 1
        }
        if invalid_integers:
            raise ValueError(
                "FastWAM value-transformer dimensions/counts must be positive "
                f"integers, got {invalid_integers}."
            )
        if not self.layer_indices:
            raise ValueError("FastWAM critic `layer_indices` cannot be empty.")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.layer_indices
        ):
            raise TypeError("FastWAM critic layer indices must be integers.")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError(
                "FastWAM critic `layer_indices` must be unique and increasing."
            )
        invalid_layers = [
            index
            for index in self.layer_indices
            if index < 0 or index >= self.num_mot_layers
        ]
        if invalid_layers:
            raise ValueError(
                "FastWAM critic layer indices are outside the configured MoT: "
                f"{invalid_layers} for {self.num_mot_layers} layers."
            )
        if not self.sources:
            raise ValueError("FastWAM critic `sources` cannot be empty.")
        supported_sources = {
            KVSource.CURRENT_FRAME_VIDEO.value,
            KVSource.TEXT_STATE_CONTEXT.value,
        }
        unknown_sources = [
            source for source in self.sources if source not in supported_sources
        ]
        if unknown_sources:
            raise ValueError(
                "FastWAM critic sources must be current_frame_video and/or "
                f"text_state_context, got {unknown_sources}."
            )
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("FastWAM critic `sources` must be unique.")
        if not isinstance(self.share_blocks, bool):
            raise TypeError("FastWAM critic `share_blocks` must be a boolean.")
        if not isinstance(self.layer_index_embedding, bool):
            raise TypeError("FastWAM critic `layer_index_embedding` must be a boolean.")
        object.__setattr__(self, "pooling", CurrentFramePooling.parse(self.pooling))

    @property
    def source_dim(self) -> int:
        """Return the flattened projected K/V width."""

        return self.source_num_heads * self.source_head_dim


def _cat_value_bank(banks: Sequence[KeyValueBank]) -> KeyValueBank:
    """Batch compatible read-only source banks."""

    if not banks:
        raise ValueError("Cannot concatenate an empty K/V-bank sequence.")
    first = banks[0]
    for bank in banks[1:]:
        if (
            bank.source is not first.source
            or bank.sequence_length != first.sequence_length
            or bank.feature_dim != first.feature_dim
            or bank.key.dtype != first.key.dtype
            or bank.contains_generated_future_video
            != first.contains_generated_future_video
        ):
            raise ValueError("Cannot batch incompatible FastWAM value K/V banks.")
    return KeyValueBank(
        source=first.source,
        key=torch.cat([bank.key for bank in banks], dim=0),
        value=torch.cat([bank.value for bank in banks], dim=0),
        valid_mask=torch.cat([bank.valid_mask for bank in banks], dim=0),
        contains_generated_future_video=first.contains_generated_future_video,
    )


@dataclass(frozen=True)
class FastWAMValueFeatures:
    """Detached multi-layer condition K/V consumed by the value sidecar."""

    layers: tuple[ConditionLayerKV, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("FastWAM value features require at least one MoT layer.")
        indices = self.layer_indices
        if tuple(sorted(set(indices))) != indices:
            raise ValueError(
                "FastWAM value-feature layers must be unique and increasing."
            )
        first = self.layers[0]
        for layer in self.layers[1:]:
            if layer.batch_size != first.batch_size:
                raise ValueError(
                    "FastWAM value-feature batches must match across layers."
                )
            if layer.feature_dim != first.feature_dim:
                raise ValueError(
                    "FastWAM value-feature widths must match across layers."
                )
            if (
                layer.current_frame_video.key.device
                != first.current_frame_video.key.device
            ):
                raise ValueError("FastWAM value-feature layers must share one device.")

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def feature_dim(self) -> int:
        return self.layers[0].feature_dim

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers)

    def layer(self, layer_index: int) -> ConditionLayerKV:
        for layer in self.layers:
            if layer.layer_index == layer_index:
                return layer
        raise KeyError(
            f"Layer {layer_index} is absent from FastWAM value features "
            f"{self.layer_indices}."
        )

    def detached(self) -> FastWAMValueFeatures:
        return FastWAMValueFeatures(tuple(layer.detached() for layer in self.layers))

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> FastWAMValueFeatures:
        """Move value features to their query-transformer compute placement."""

        return FastWAMValueFeatures(
            tuple(
                layer.to(device=device, dtype=dtype, non_blocking=non_blocking)
                for layer in self.layers
            )
        )

    @classmethod
    def cat(cls, features: Sequence[FastWAMValueFeatures]) -> FastWAMValueFeatures:
        """Batch per-sample condition features without serializing them."""

        if not features:
            raise ValueError("Cannot concatenate an empty value-feature sequence.")
        first = features[0]
        if any(item.layer_indices != first.layer_indices for item in features[1:]):
            raise ValueError("Cannot batch value features with different layer taps.")
        layers = []
        for layer_index in first.layer_indices:
            source_layers = [item.layer(layer_index) for item in features]
            layers.append(
                ConditionLayerKV(
                    layer_index=layer_index,
                    current_frame_video=_cat_value_bank(
                        [layer.current_frame_video for layer in source_layers]
                    ),
                    context=_cat_value_bank([layer.context for layer in source_layers]),
                )
            )
        return cls(tuple(layers)).detached()


def extract_fastwam_value_features(
    condition: Any,
    *,
    mot: nn.Module,
    action_expert: nn.Module,
    config: FastWAMValueTransformerConfig,
    regime_context: Any | None = None,
) -> FastWAMValueFeatures:
    """Extract base-regime current-video and prompt/state K/V without actions."""

    if not hasattr(action_expert, "text_embedding"):
        raise TypeError(
            "FastWAM action expert must expose `text_embedding` for value features."
        )
    regime_scope = (
        nullcontext()
        if regime_context is None
        else regime_context.use(PolicyRegime.IDM)
    )
    with torch.no_grad(), regime_scope:
        context = action_expert.text_embedding(condition.context)
        context_mask = condition.context_mask
        layers = tuple(
            mot.read_condition_layer_kv(
                layer_index=layer_index,
                video_kv_cache=condition.video_kv_cache,
                current_frame_video_tokens=condition.current_frame_video_tokens,
                context=context,
                context_mask=context_mask,
            )
            for layer_index in config.layer_indices
        )
    features = FastWAMValueFeatures(layers).detached()
    if features.feature_dim != config.source_dim:
        raise ValueError(
            "FastWAM value source width does not match the configured attention: "
            f"observed={features.feature_dim}, configured={config.source_dim}."
        )
    return features


class FastWAMValueTransformerBlock(nn.Module):
    """One Gate-style read-only attention block for one configured MoT layer."""

    def __init__(self, config: FastWAMValueTransformerConfig) -> None:
        super().__init__()
        self.sources = config.sources
        self.source_norms = nn.ModuleDict(
            {source: nn.LayerNorm(config.hidden_dim) for source in self.sources}
        )
        self.source_attentions = nn.ModuleDict(
            {
                source: DirectKVAttention(
                    config.hidden_dim,
                    config.source_num_heads,
                    config.source_head_dim,
                )
                for source in self.sources
            }
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_multiplier * config.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_multiplier * config.hidden_dim, config.hidden_dim),
        )

    @staticmethod
    def _source_bank(layer: ConditionLayerKV, source: str) -> KeyValueBank:
        if source == KVSource.CURRENT_FRAME_VIDEO.value:
            return layer.current_frame_video
        if source == KVSource.TEXT_STATE_CONTEXT.value:
            return layer.context
        raise RuntimeError(f"Unvalidated FastWAM value source {source!r}.")

    def forward(
        self,
        query: torch.Tensor,
        layer: ConditionLayerKV,
    ) -> torch.Tensor:
        layer = layer.detached()
        for source in self.sources:
            query = query + self.source_attentions[source](
                self.source_norms[source](query),
                self._source_bank(layer, source),
            )
        return query + self.ffn(self.ffn_norm(query))


class FastWAMValueTransformer(nn.Module):
    """Learned value queries over frozen FastWAM condition K/V."""

    def __init__(self, config: FastWAMValueTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.num_query_tokens, config.hidden_dim)
        )
        nn.init.normal_(self.query_tokens, std=config.hidden_dim**-0.5)
        block_count = 1 if config.share_blocks else len(config.layer_indices)
        self.blocks = nn.ModuleList(
            FastWAMValueTransformerBlock(config) for _ in range(block_count)
        )
        self.layer_embedding = (
            nn.Embedding(config.num_mot_layers, config.hidden_dim)
            if config.layer_index_embedding
            else None
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def _pool_queries(self, query: torch.Tensor) -> torch.Tensor:
        query = self.output_norm(query)
        if self.config.pooling is CurrentFramePooling.MEAN_TOKEN:
            return query.mean(dim=1)
        if self.config.pooling is CurrentFramePooling.FIRST_TOKEN:
            return query[:, 0]
        return query[:, -1]

    def forward(self, features: FastWAMValueFeatures) -> torch.Tensor:
        features = features.detached()
        missing = [
            index
            for index in self.config.layer_indices
            if index not in features.layer_indices
        ]
        if missing:
            raise ValueError(
                f"FastWAM value features are missing configured layers {missing}; "
                f"available={features.layer_indices}."
            )
        if features.feature_dim != self.config.source_dim:
            raise ValueError(
                f"FastWAM value transformer expected source dim "
                f"{self.config.source_dim}, got {features.feature_dim}."
            )
        first = features.layer(self.config.layer_indices[0])
        if first.current_frame_video.key.device != self.query_tokens.device:
            raise ValueError(
                "FastWAM value K/V and value-head parameters must share a device."
            )
        query = self.query_tokens.expand(features.batch_size, -1, -1)
        for offset, layer_index in enumerate(self.config.layer_indices):
            if self.layer_embedding is not None:
                layer_ids = torch.full(
                    (features.batch_size,),
                    layer_index,
                    dtype=torch.long,
                    device=query.device,
                )
                query = query + self.layer_embedding(layer_ids).unsqueeze(1)
            block = self.blocks[0] if self.config.share_blocks else self.blocks[offset]
            query = block(query, features.layer(layer_index))
        return self._pool_queries(query)


class FastWAMValueHead(ValueHead):
    """FSDP-compatible query transformer plus configurable scalar MLP."""

    def __init__(
        self,
        *,
        config: FastWAMValueTransformerConfig,
        hidden_sizes: tuple[int, ...],
        activation: str,
        bias_last: bool,
    ) -> None:
        super().__init__(
            input_dim=config.hidden_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            activation=activation,
            bias_last=bias_last,
        )
        self.transformer = FastWAMValueTransformer(config)

    def forward(self, features: FastWAMValueFeatures) -> torch.Tensor:
        return self.mlp(self.transformer(features))


class FastWAMCurrentFrameValueCritic(nn.Module):
    """A fresh Gate-style value head over frozen FastWAM condition K/V."""

    kind = CriticKind.FASTWAM_CURRENT_FRAME_VALUE

    def __init__(
        self,
        *,
        config: FastWAMValueTransformerConfig,
        hidden_sizes: tuple[int, ...] = (),
        activation: str = "relu",
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_dim
        self.value_head = FastWAMValueHead(
            config=config,
            hidden_sizes=hidden_sizes,
            activation=activation,
            bias_last=bias_last,
        )

    def value_from_features(self, features: FastWAMValueFeatures) -> torch.Tensor:
        """Predict scalar values without propagating into FastWAM features."""

        if not isinstance(features, FastWAMValueFeatures):
            raise TypeError("FastWAM critic requires `FastWAMValueFeatures`.")
        parameter = next(self.value_head.parameters())
        features = features.detached().to(device=parameter.device)
        return self.value_head(features)[:, 0]

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the value-head parameters and assert exclusive ownership."""

        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        expected = list(self.value_head.parameters())
        if {id(parameter) for parameter in trainable} != {
            id(parameter) for parameter in expected
        }:
            raise RuntimeError(
                "FastWAM current-frame critic has trainable parameters outside "
                "the fresh value head."
            )
        return trainable


def critic_parent_checkpoint_sha256(critic_cfg: Any) -> str | None:
    """Return the external critic parent identity required by one backend."""

    def get(name: str, default: Any = None) -> Any:
        if hasattr(critic_cfg, "get"):
            return critic_cfg.get(name, default)
        return getattr(critic_cfg, name, default)

    kind = CriticKind.parse(get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM))
    if kind is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
        return None
    value = str(get("backbone_checkpoint_sha256", "")).strip().lower()
    return value


__all__ = [
    "CriticKind",
    "CurrentFramePooling",
    "FastWAMCurrentFrameValueCritic",
    "FastWAMValueFeatures",
    "FastWAMValueHead",
    "FastWAMValueTransformer",
    "FastWAMValueTransformerBlock",
    "FastWAMValueTransformerConfig",
    "critic_parent_checkpoint_sha256",
    "extract_fastwam_value_features",
]
