# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Current-only Gate and replay codec owned by PAD-RV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from fastwam.models.wan22.condition_kv import ConditionLayerKV
from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource

from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMValueFeatures,
    FastWAMValueTransformer,
    FastWAMValueTransformerConfig,
)


def serialize_condition_features(
    features: FastWAMValueFeatures,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Serialize detached condition K/V without action-denoising metadata."""

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("PAD condition prefix must be alphanumeric text.")
    result: dict[str, torch.Tensor] = {}
    for layer in features.layers:
        stem = f"{prefix}_layer_{layer.layer_index}"
        result.update(
            {
                f"{stem}_current_key": layer.current_frame_video.key.detach(),
                f"{stem}_current_value": layer.current_frame_video.value.detach(),
                f"{stem}_current_mask": layer.current_frame_video.valid_mask.detach(),
                f"{stem}_context_key": layer.context.key.detach(),
                f"{stem}_context_value": layer.context.value.detach(),
                f"{stem}_context_mask": layer.context.valid_mask.detach(),
            }
        )
    return result


def deserialize_condition_features(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    prefix: str,
    layer_indices: Sequence[int],
) -> FastWAMValueFeatures:
    """Restore PAD condition K/V using the existing value-feature contract."""

    layers: list[ConditionLayerKV] = []
    for raw_index in layer_indices:
        layer_index = int(raw_index)
        stem = f"{prefix}_layer_{layer_index}"
        names = {
            name: f"{stem}_{suffix}"
            for name, suffix in (
                ("current_key", "current_key"),
                ("current_value", "current_value"),
                ("current_mask", "current_mask"),
                ("context_key", "context_key"),
                ("context_value", "context_value"),
                ("context_mask", "context_mask"),
            )
        }
        missing = [name for name in names.values() if name not in forward_inputs]
        if missing:
            raise KeyError(f"PAD condition replay is missing tensors: {missing}.")
        layers.append(
            ConditionLayerKV(
                layer_index=layer_index,
                current_frame_video=KeyValueBank(
                    source=KVSource.CURRENT_FRAME_VIDEO,
                    key=forward_inputs[names["current_key"]],
                    value=forward_inputs[names["current_value"]],
                    valid_mask=forward_inputs[names["current_mask"]],
                ),
                context=KeyValueBank(
                    source=KVSource.TEXT_STATE_CONTEXT,
                    key=forward_inputs[names["context_key"]],
                    value=forward_inputs[names["context_value"]],
                    valid_mask=forward_inputs[names["context_mask"]],
                ),
            )
        )
    return FastWAMValueFeatures(tuple(layers)).detached()


class PadCurrentStepGate(nn.Module):
    """Route policy over current-frame and text/state condition K/V only."""

    def __init__(self, config: FastWAMValueTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = FastWAMValueTransformer(config)
        self.output = nn.Linear(config.hidden_dim, 1)

    def forward(self, features: FastWAMValueFeatures) -> torch.Tensor:
        if not isinstance(features, FastWAMValueFeatures):
            raise TypeError("PAD current-step Gate requires FastWAMValueFeatures.")
        return self.output(self.transformer(features)).squeeze(-1)
