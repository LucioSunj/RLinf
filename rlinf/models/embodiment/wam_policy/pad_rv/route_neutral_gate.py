# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Route-neutral current-step Gate features, history, replay, and policy head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource

from rlinf.models.embodiment.wam_policy.critic import (
    FastWAMValueFeatures,
    FastWAMValueTransformer,
    FastWAMValueTransformerConfig,
)

from .route_neutral_contracts import RouteNeutralGateInputContract


@dataclass(frozen=True)
class RouteNeutralVisualLayer:
    """One current-frame-only K/V layer; context and action banks do not exist."""

    layer_index: int
    current_frame_video: KeyValueBank

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("Route-neutral layer index must be non-negative.")
        if self.current_frame_video.source is not KVSource.CURRENT_FRAME_VIDEO:
            raise ValueError("Route-neutral visual layer has the wrong K/V source.")
        if self.current_frame_video.contains_generated_future_video:
            raise ValueError(
                "Route-neutral visual K/V cannot contain generated future."
            )

    @property
    def batch_size(self) -> int:
        return self.current_frame_video.batch_size

    @property
    def feature_dim(self) -> int:
        return self.current_frame_video.feature_dim

    def detached(self) -> "RouteNeutralVisualLayer":
        return RouteNeutralVisualLayer(
            layer_index=self.layer_index,
            current_frame_video=self.current_frame_video.detached(),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "RouteNeutralVisualLayer":
        return RouteNeutralVisualLayer(
            layer_index=self.layer_index,
            current_frame_video=self.current_frame_video.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
        )


@dataclass(frozen=True)
class RouteNeutralVisualFeatures:
    """Duck-typed FastWAM value features narrowed to current-frame K/V."""

    layers: tuple[RouteNeutralVisualLayer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError(
                "Route-neutral visual features require at least one layer."
            )
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError(
                "Route-neutral visual layers must be unique and increasing."
            )
        first = self.layers[0]
        for layer in self.layers[1:]:
            if layer.batch_size != first.batch_size:
                raise ValueError("Route-neutral visual batches differ across layers.")
            if layer.feature_dim != first.feature_dim:
                raise ValueError("Route-neutral visual widths differ across layers.")

    @classmethod
    def from_value_features(
        cls,
        features: FastWAMValueFeatures,
    ) -> "RouteNeutralVisualFeatures":
        """Discard projected context immediately after the canonical producer."""

        if not isinstance(features, FastWAMValueFeatures):
            raise TypeError(
                "Route-neutral visual narrowing requires FastWAMValueFeatures."
            )
        return cls(
            tuple(
                RouteNeutralVisualLayer(
                    layer_index=layer.layer_index,
                    current_frame_video=layer.current_frame_video.detached(),
                )
                for layer in features.layers
            )
        )

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def feature_dim(self) -> int:
        return self.layers[0].feature_dim

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers)

    def layer(self, layer_index: int) -> RouteNeutralVisualLayer:
        for layer in self.layers:
            if layer.layer_index == layer_index:
                return layer
        raise KeyError(f"Route-neutral visual layer {layer_index} is absent.")

    def detached(self) -> "RouteNeutralVisualFeatures":
        return RouteNeutralVisualFeatures(
            tuple(layer.detached() for layer in self.layers)
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "RouteNeutralVisualFeatures":
        return RouteNeutralVisualFeatures(
            tuple(
                layer.to(device=device, dtype=dtype, non_blocking=non_blocking)
                for layer in self.layers
            )
        )


@dataclass(frozen=True)
class RouteNeutralGateFeatures:
    """The complete actor-facing Gate input, with no route-derived fields."""

    visual: RouteNeutralVisualFeatures
    language: torch.Tensor
    language_mask: torch.Tensor
    state: torch.Tensor
    physical_history: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = self.visual.batch_size
        if self.language.ndim != 3 or self.language.shape[0] != batch_size:
            raise ValueError("Route-neutral language must have shape [B,L,D].")
        if self.language_mask.shape != self.language.shape[:2]:
            raise ValueError("Route-neutral language mask must match [B,L].")
        if self.language_mask.dtype is not torch.bool:
            raise TypeError("Route-neutral language mask must be boolean.")
        if not bool(self.language_mask.any(dim=1).all().item()):
            raise ValueError("Each route-neutral language row needs a valid token.")
        if self.state.ndim != 2 or self.state.shape[0] != batch_size:
            raise ValueError("Route-neutral state must have shape [B,D].")
        if (
            self.physical_history.ndim != 3
            or self.physical_history.shape[0] != batch_size
            or self.physical_history.shape[2] != self.state.shape[1]
        ):
            raise ValueError("Route-neutral physical history must have shape [B,H,D].")
        tensors = (
            self.language,
            self.language_mask,
            self.state,
            self.physical_history,
        )
        visual_device = self.visual.layers[0].current_frame_video.key.device
        if any(tensor.device != visual_device for tensor in tensors):
            raise ValueError("Route-neutral Gate features must share one device.")

    @property
    def batch_size(self) -> int:
        return self.visual.batch_size

    def detached(self) -> "RouteNeutralGateFeatures":
        return RouteNeutralGateFeatures(
            visual=self.visual.detached(),
            language=self.language.detach(),
            language_mask=self.language_mask.detach(),
            state=self.state.detach(),
            physical_history=self.physical_history.detach(),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "RouteNeutralGateFeatures":
        return RouteNeutralGateFeatures(
            visual=self.visual.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            language=self.language.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            language_mask=self.language_mask.to(
                device=device, non_blocking=non_blocking
            ),
            state=self.state.to(device=device, dtype=dtype, non_blocking=non_blocking),
            physical_history=self.physical_history.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
        )


class PhysicalStateHistoryTracker:
    """Fixed-width causal proprio history with no route or clock inputs."""

    _SCHEMA = "pad-physical-state-history-v1"

    def __init__(self, contract: RouteNeutralGateInputContract) -> None:
        self.contract = contract
        self._states: dict[int, list[torch.Tensor]] = {}

    def features_and_append(
        self,
        *,
        env_ids: torch.Tensor,
        reset_mask: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        if env_ids.ndim != 1 or env_ids.shape != reset_mask.shape:
            raise ValueError("Physical-history IDs and reset mask must have shape [B].")
        if reset_mask.dtype is not torch.bool:
            raise TypeError("Physical-history reset mask must be boolean.")
        if current_state.shape != (env_ids.numel(), self.contract.state_dim):
            raise ValueError("Physical-history current state has the wrong shape.")
        ids = [int(value) for value in env_ids.detach().cpu().tolist()]
        if len(ids) != len(set(ids)):
            raise ValueError("Physical-history environment IDs must be batch-unique.")

        rows: list[torch.Tensor] = []
        length = self.contract.history_length_chunks
        current_cpu = current_state.detach().to(device="cpu", dtype=torch.float32)
        for offset, (env_id, reset) in enumerate(
            zip(ids, reset_mask.detach().cpu().tolist(), strict=True)
        ):
            if bool(reset):
                self._states.pop(env_id, None)
            prior = self._states.get(env_id, [])
            if prior:
                selected = prior[-length:]
                padded = [selected[0]] * (length - len(selected)) + selected
            else:
                padded = [current_cpu[offset]] * length
            rows.append(torch.stack([value.clone() for value in padded]))
            retained = prior[-(length - 1) :] if length > 1 else []
            self._states[env_id] = [
                *retained,
                current_cpu[offset].clone(),
            ]
        return torch.stack(rows).to(
            device=current_state.device, dtype=current_state.dtype
        )

    def state_dict(self) -> dict:
        return {
            "schema": self._SCHEMA,
            "history_length_chunks": self.contract.history_length_chunks,
            "state_dim": self.contract.state_dim,
            "states": {
                env_id: tuple(value.clone() for value in values)
                for env_id, values in self._states.items()
            },
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        expected = {"schema", "history_length_chunks", "state_dim", "states"}
        if set(payload) != expected or payload.get("schema") != self._SCHEMA:
            raise ValueError("Unsupported physical-history checkpoint state.")
        if (
            int(payload["history_length_chunks"]) != self.contract.history_length_chunks
            or int(payload["state_dim"]) != self.contract.state_dim
        ):
            raise ValueError("Physical-history checkpoint contract changed.")
        raw_states = payload["states"]
        if not isinstance(raw_states, Mapping):
            raise TypeError("Physical-history checkpoint states must be a mapping.")
        restored: dict[int, list[torch.Tensor]] = {}
        for raw_env_id, raw_values in raw_states.items():
            env_id = int(raw_env_id)
            if env_id < 0 or not isinstance(raw_values, (tuple, list)):
                raise ValueError("Physical-history checkpoint entry is malformed.")
            values = []
            for raw_value in raw_values:
                if not isinstance(raw_value, torch.Tensor):
                    raise TypeError(
                        "Physical-history checkpoint values must be tensors."
                    )
                value = raw_value.detach().to(device="cpu", dtype=torch.float32)
                if value.shape != (self.contract.state_dim,) or not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise ValueError("Physical-history checkpoint tensor is invalid.")
                values.append(value.clone())
            if not 1 <= len(values) <= self.contract.history_length_chunks:
                raise ValueError("Physical-history checkpoint length is invalid.")
            restored[env_id] = values
        self._states = restored


@dataclass(frozen=True)
class PadRouteNeutralGateConfig:
    visual: FastWAMValueTransformerConfig
    language_dim: int
    state_dim: int
    history_length_chunks: int

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self.visual.layer_indices

    @property
    def hidden_dim(self) -> int:
        return self.visual.hidden_dim


class PadRouteNeutralCurrentStepGate(nn.Module):
    """Fuse canonical current-frame K/V with raw language and physical state."""

    def __init__(self, config: PadRouteNeutralGateConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.visual = FastWAMValueTransformer(config.visual)
        self.language_norm = nn.LayerNorm(config.language_dim)
        self.language_projection = nn.Linear(config.language_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(config.state_dim)
        self.state_projection = nn.Linear(config.state_dim, hidden_dim)
        self.history_projection = nn.Linear(config.state_dim, hidden_dim)
        self.history_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: RouteNeutralGateFeatures) -> torch.Tensor:
        if not isinstance(features, RouteNeutralGateFeatures):
            raise TypeError("Route-neutral Gate requires RouteNeutralGateFeatures.")
        if features.visual.layer_indices != self.config.layer_indices:
            raise ValueError("Route-neutral Gate visual layer contract changed.")
        if features.state.shape[1] != self.config.state_dim:
            raise ValueError("Route-neutral Gate state width changed.")
        if features.physical_history.shape[1:] != (
            self.config.history_length_chunks,
            self.config.state_dim,
        ):
            raise ValueError("Route-neutral Gate history shape changed.")

        parameter = next(self.parameters())
        features = features.detached().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        visual = self.visual(features.visual)
        mask = features.language_mask.unsqueeze(-1)
        language_tokens = self.language_norm(features.language)
        language = (language_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        language = self.language_projection(language)
        state = self.state_projection(self.state_norm(features.state))
        history_input = self.history_projection(features.physical_history)
        _history_sequence, history_hidden = self.history_encoder(history_input)
        history = history_hidden[-1]
        return self.fusion(torch.cat([visual, language, state, history], dim=-1))[:, 0]


def serialize_route_neutral_features(
    features: RouteNeutralGateFeatures,
    *,
    prefix: str = "gate_condition",
) -> dict[str, torch.Tensor]:
    """Serialize only the declared route-neutral actor-facing tensors."""

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("Route-neutral replay prefix must be alphanumeric text.")
    result: dict[str, torch.Tensor] = {
        f"{prefix}_language": features.language.detach(),
        f"{prefix}_language_mask": features.language_mask.detach(),
        f"{prefix}_state": features.state.detach(),
        f"{prefix}_physical_history": features.physical_history.detach(),
    }
    for layer in features.visual.layers:
        stem = f"{prefix}_visual_layer_{layer.layer_index}"
        result.update(
            {
                f"{stem}_key": layer.current_frame_video.key.detach(),
                f"{stem}_value": layer.current_frame_video.value.detach(),
                f"{stem}_mask": layer.current_frame_video.valid_mask.detach(),
            }
        )
    return result


def deserialize_route_neutral_features(
    forward_inputs: Mapping[str, torch.Tensor],
    *,
    layer_indices: Sequence[int],
    prefix: str = "gate_condition",
) -> RouteNeutralGateFeatures:
    """Restore the narrowed Gate contract without creating context/action fields."""

    scalar_names = {
        name: f"{prefix}_{suffix}"
        for name, suffix in (
            ("language", "language"),
            ("language_mask", "language_mask"),
            ("state", "state"),
            ("physical_history", "physical_history"),
        )
    }
    missing = [name for name in scalar_names.values() if name not in forward_inputs]
    layers: list[RouteNeutralVisualLayer] = []
    for raw_index in layer_indices:
        layer_index = int(raw_index)
        stem = f"{prefix}_visual_layer_{layer_index}"
        names = {kind: f"{stem}_{kind}" for kind in ("key", "value", "mask")}
        missing.extend(name for name in names.values() if name not in forward_inputs)
        if not any(name not in forward_inputs for name in names.values()):
            layers.append(
                RouteNeutralVisualLayer(
                    layer_index=layer_index,
                    current_frame_video=KeyValueBank(
                        source=KVSource.CURRENT_FRAME_VIDEO,
                        key=forward_inputs[names["key"]],
                        value=forward_inputs[names["value"]],
                        valid_mask=forward_inputs[names["mask"]],
                        contains_generated_future_video=False,
                    ),
                )
            )
    if missing:
        raise KeyError(f"Route-neutral Gate replay is missing tensors: {missing}.")
    return RouteNeutralGateFeatures(
        visual=RouteNeutralVisualFeatures(tuple(layers)),
        language=forward_inputs[scalar_names["language"]],
        language_mask=forward_inputs[scalar_names["language_mask"]],
        state=forward_inputs[scalar_names["state"]],
        physical_history=forward_inputs[scalar_names["physical_history"]],
    ).detached()


__all__ = [
    "PadRouteNeutralCurrentStepGate",
    "PadRouteNeutralGateConfig",
    "PhysicalStateHistoryTracker",
    "RouteNeutralGateFeatures",
    "RouteNeutralVisualFeatures",
    "RouteNeutralVisualLayer",
    "deserialize_route_neutral_features",
    "serialize_route_neutral_features",
]
