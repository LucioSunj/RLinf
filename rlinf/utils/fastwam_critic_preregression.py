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

"""Offline value-only preregression for a native FastWAM step-zero checkpoint.

The input dataset is a ``torch.save`` payload with this exact structure::

    {
        "schema": "fastwam-critic-preregression-dataset-v1",
        "feature_kind": "pi05_prefix_mean_token",  # or "value_head_input"
        "features": FloatTensor[num_samples, num_tokens, input_dim],
        "returns": FloatTensor[num_samples],
        "validation_mask": BoolTensor[num_samples],
        "valid_mask": BoolTensor[num_samples],  # optional
    }

The tool changes only ``policy.value_head`` in a copy of a native step-zero
actor checkpoint. It leaves Gate, LoRA, route/RNG state, optimizer, scheduler,
and every checkpoint contract field untouched. Normal training can therefore
load the output actor directory through its existing ``runner.ckpt_path``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.metric_utils import (
    compute_critic_explained_variance_from_stats,
    compute_critic_explained_variance_stats,
)

DATASET_SCHEMA = "fastwam-critic-preregression-dataset-v1"
MANIFEST_SCHEMA = "fastwam-critic-preregression-v1"
_LINEAR_WEIGHT = re.compile(r"^mlp\.(\d+)\.weight$")


@dataclass(frozen=True, slots=True)
class CriticPreregressionDataset:
    """Fixed train/held-out value-head inputs and return targets."""

    inputs: torch.Tensor
    returns: torch.Tensor
    validation_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class CriticPreregressionResult:
    """Best held-out result and strictly loadable value-head state."""

    value_head_state: dict[str, torch.Tensor]
    initial_heldout_explained_variance: float
    heldout_explained_variance: float
    heldout_mse: float
    best_epoch: int
    train_sample_count: int
    heldout_sample_count: int


def _tensor_mapping(value: Any, *, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{name} must be a non-empty tensor mapping.")
    result = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must contain only string/tensor entries.")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} tensor {key!r} must be finite and floating.")
        result[key] = tensor.detach().cpu().contiguous().clone()
    return result


def build_value_head_from_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    activation: str,
) -> ValueHead:
    """Infer the configured MLP shape and strictly restore its native state."""

    state = _tensor_mapping(state, name="FastWAM value-head state")
    layers = []
    for key, tensor in state.items():
        match = _LINEAR_WEIGHT.fullmatch(key)
        if match is not None:
            if tensor.ndim != 2:
                raise ValueError(f"Value-head weight {key!r} must be rank two.")
            layers.append((int(match.group(1)), tensor))
    layers.sort(key=lambda item: item[0])
    if not layers or layers[-1][1].shape[0] != 1:
        raise ValueError("FastWAM value head must end in one scalar output.")
    for previous, current in zip(layers, layers[1:], strict=False):
        if previous[1].shape[0] != current[1].shape[1]:
            raise ValueError("FastWAM value-head linear dimensions do not chain.")
    input_dim = int(layers[0][1].shape[1])
    hidden_sizes = tuple(int(tensor.shape[0]) for _index, tensor in layers[:-1])
    final_index = layers[-1][0]
    value_head = ValueHead(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=1,
        activation=activation,
        bias_last=f"mlp.{final_index}.bias" in state,
    )
    value_head.load_state_dict(state, strict=True)
    return value_head.float()


def load_preregression_dataset(path: str | Path) -> CriticPreregressionDataset:
    """Load and validate one explicitly split fixed-rollout feature dataset."""

    payload = torch.load(
        Path(path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or payload.get("schema") != DATASET_SCHEMA:
        raise ValueError("Unsupported FastWAM critic-preregression dataset schema.")
    feature_kind = str(payload.get("feature_kind", "")).strip()
    features = payload.get("features")
    returns = payload.get("returns")
    validation_mask = payload.get("validation_mask")
    valid_mask = payload.get("valid_mask")
    if not isinstance(features, torch.Tensor) or not features.is_floating_point():
        raise TypeError("Critic preregression features must be a floating tensor.")
    if feature_kind == "pi05_prefix_mean_token":
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError(
                "pi0.5 prefix features must have shape [sample, token, dim]."
            )
        inputs = features.float().mean(dim=1)
    elif feature_kind == "value_head_input":
        if features.ndim != 2:
            raise ValueError("Value-head inputs must have shape [sample, dim].")
        inputs = features.float()
    else:
        raise ValueError(f"Unsupported critic feature_kind {feature_kind!r}.")
    if not isinstance(returns, torch.Tensor) or not returns.is_floating_point():
        raise TypeError("Critic preregression returns must be a floating tensor.")
    returns = returns.float().reshape(-1)
    if not isinstance(validation_mask, torch.Tensor) or (
        validation_mask.dtype != torch.bool
    ):
        raise TypeError("validation_mask must be a boolean tensor.")
    validation_mask = validation_mask.reshape(-1)
    sample_count = int(inputs.shape[0])
    if returns.shape != (sample_count,) or validation_mask.shape != (sample_count,):
        raise ValueError("Features, returns, and validation_mask must align.")
    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean tensor when present.")
        valid_mask = valid_mask.reshape(-1)
        if valid_mask.shape != (sample_count,):
            raise ValueError("valid_mask must align with critic features.")
        inputs = inputs[valid_mask]
        returns = returns[valid_mask]
        validation_mask = validation_mask[valid_mask]
    if not bool(torch.isfinite(inputs).all()) or not bool(
        torch.isfinite(returns).all()
    ):
        raise ValueError("Critic preregression data must be finite.")
    if (
        inputs.shape[0] < 2
        or not bool(validation_mask.any())
        or not bool((~validation_mask).any())
    ):
        raise ValueError(
            "Critic preregression requires non-empty train and held-out sets."
        )
    heldout_returns = returns[validation_mask]
    if float(heldout_returns.var(unbiased=False).item()) <= 0.0:
        raise ValueError("Held-out critic returns must have nonzero variance.")
    return CriticPreregressionDataset(
        inputs=inputs.contiguous(),
        returns=returns.contiguous(),
        validation_mask=validation_mask.contiguous(),
    )


def explained_variance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute the production RLinf explained variance on one held-out split."""

    predictions = predictions.float().reshape(-1)
    targets = targets.float().reshape(-1)
    if predictions.shape != targets.shape or predictions.numel() < 1:
        raise ValueError("Explained-variance predictions and targets must align.")
    result = float(
        compute_critic_explained_variance_from_stats(
            compute_critic_explained_variance_stats(
                returns=targets,
                values=predictions,
            )
        ).item()
    )
    if not math.isfinite(result):
        raise ValueError(
            "Critic explained variance is undefined or non-finite on held-out data."
        )
    return result


@torch.no_grad()
def _heldout_metrics(
    value_head: ValueHead,
    inputs: torch.Tensor,
    returns: torch.Tensor,
) -> tuple[float, float]:
    value_head.eval()
    predictions = value_head(inputs)[:, 0]
    return (
        explained_variance(predictions, returns),
        float(F.mse_loss(predictions, returns).item()),
    )


def fit_value_head(
    *,
    initial_state: Mapping[str, torch.Tensor],
    dataset: CriticPreregressionDataset,
    activation: str = "gelu",
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
) -> CriticPreregressionResult:
    """Fit only the value head and select by fixed held-out explained variance."""

    if epochs < 1 or batch_size < 1:
        raise ValueError("Critic preregression epochs and batch_size must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Critic preregression learning_rate must be positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("Critic preregression weight_decay must be non-negative.")
    torch.manual_seed(int(seed))
    value_head = build_value_head_from_state_dict(
        initial_state,
        activation=activation,
    )
    if value_head.mlp[0].in_features != dataset.inputs.shape[1]:
        raise ValueError(
            "Critic feature dimension does not match the native value head: "
            f"{dataset.inputs.shape[1]} != {value_head.mlp[0].in_features}."
        )
    train_mask = ~dataset.validation_mask
    train_inputs = dataset.inputs[train_mask]
    train_returns = dataset.returns[train_mask]
    heldout_inputs = dataset.inputs[dataset.validation_mask]
    heldout_returns = dataset.returns[dataset.validation_mask]
    initial_ev, _initial_mse = _heldout_metrics(
        value_head,
        heldout_inputs,
        heldout_returns,
    )
    best_ev = initial_ev
    best_mse = _initial_mse
    best_epoch = 0
    best_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in value_head.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        value_head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for epoch in range(1, int(epochs) + 1):
        value_head.train()
        order = torch.randperm(train_inputs.shape[0], generator=generator)
        for start in range(0, train_inputs.shape[0], int(batch_size)):
            indices = order[start : start + int(batch_size)]
            predictions = value_head(train_inputs[indices])[:, 0]
            loss = F.mse_loss(predictions, train_returns[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        heldout_ev, heldout_mse = _heldout_metrics(
            value_head,
            heldout_inputs,
            heldout_returns,
        )
        if heldout_ev > best_ev:
            best_ev = heldout_ev
            best_mse = heldout_mse
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().contiguous().clone()
                for key, tensor in value_head.state_dict().items()
            }
    strict_reload = build_value_head_from_state_dict(
        best_state,
        activation=activation,
    )
    reloaded_ev, reloaded_mse = _heldout_metrics(
        strict_reload,
        heldout_inputs,
        heldout_returns,
    )
    if reloaded_ev != best_ev or reloaded_mse != best_mse:
        raise RuntimeError("Strict value-head reload changed held-out metrics.")
    return CriticPreregressionResult(
        value_head_state=best_state,
        initial_heldout_explained_variance=initial_ev,
        heldout_explained_variance=best_ev,
        heldout_mse=best_mse,
        best_epoch=best_epoch,
        train_sample_count=int(train_inputs.shape[0]),
        heldout_sample_count=int(heldout_inputs.shape[0]),
    )


def _validate_native_step_zero(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("FastWAM native step-zero checkpoint must be a mapping.")
    if payload.get("schema") != "fastwam-adaptive-rl-checkpoint-v1":
        raise ValueError("Critic preregression requires a native FastWAM checkpoint.")
    if isinstance(payload.get("step"), bool) or int(payload.get("step", -1)) != 0:
        raise ValueError("Critic preregression requires checkpoint step zero.")
    if int(payload.get("optimizer_steps", -1)) != 0:
        raise ValueError("Critic preregression requires zero optimizer steps.")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping) or policy.get("schema") != (
        "fastwam-adaptive-policy-v1"
    ):
        raise ValueError("Native FastWAM checkpoint has no adaptive policy payload.")
    if int(policy.get("actor_version", -1)) != 0:
        raise ValueError("Critic preregression requires actor version zero.")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("state"):
        raise ValueError("Native step-zero optimizer state must be empty.")
    return _tensor_mapping(
        policy.get("value_head"),
        name="Native FastWAM value-head state",
    )


def write_preregressed_step_zero(
    *,
    input_actor_dir: str | Path,
    output_actor_dir: str | Path,
    value_head_state: Mapping[str, torch.Tensor],
    activation: str,
) -> int:
    """Patch every native actor rank and strictly reload the written weights."""

    input_actor_dir = Path(input_actor_dir).expanduser().resolve()
    output_actor_dir = Path(output_actor_dir).expanduser().resolve()
    if not input_actor_dir.is_dir():
        raise FileNotFoundError(
            f"Input actor directory does not exist: {input_actor_dir}"
        )
    rank_paths = sorted(input_actor_dir.glob("rank_*.pt"))
    if not rank_paths:
        raise FileNotFoundError("Input actor directory contains no rank checkpoints.")
    if output_actor_dir.exists():
        raise FileExistsError(
            f"Critic preregression output already exists: {output_actor_dir}"
        )
    value_head_state = _tensor_mapping(
        value_head_state,
        name="Preregressed value-head state",
    )
    build_value_head_from_state_dict(value_head_state, activation=activation)
    payloads = []
    baseline_state = None
    for rank_path in rank_paths:
        payload = torch.load(rank_path, map_location="cpu", weights_only=False)
        existing_state = _validate_native_step_zero(payload)
        if set(existing_state) != set(value_head_state):
            raise ValueError("Preregressed value-head keys differ from step zero.")
        if any(
            existing_state[key].shape != value_head_state[key].shape
            or existing_state[key].dtype != value_head_state[key].dtype
            for key in existing_state
        ):
            raise ValueError("Preregressed value-head tensor metadata changed.")
        if baseline_state is None:
            baseline_state = existing_state
        elif any(
            not torch.equal(existing_state[key], baseline_state[key])
            for key in baseline_state
        ):
            raise ValueError("Native actor ranks disagree on initial value-head state.")
        updated = copy.deepcopy(payload)
        updated["policy"]["value_head"] = {
            key: tensor.detach().cpu().contiguous().clone()
            for key, tensor in value_head_state.items()
        }
        payloads.append((rank_path.name, updated))
    output_actor_dir.mkdir(parents=True)
    for filename, payload in payloads:
        target = output_actor_dir / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
    for output_path in sorted(output_actor_dir.glob("rank_*.pt")):
        written = torch.load(output_path, map_location="cpu", weights_only=False)
        written_state = _validate_native_step_zero(written)
        strict_head = build_value_head_from_state_dict(
            written_state,
            activation=activation,
        )
        strict_head.load_state_dict(value_head_state, strict=True)
        if any(
            not torch.equal(written_state[key], value_head_state[key])
            for key in value_head_state
        ):
            raise RuntimeError("Written preregression weights changed on reload.")
    return len(payloads)


def run_critic_preregression(
    *,
    dataset_path: str | Path,
    input_actor_dir: str | Path,
    output_actor_dir: str | Path,
    activation: str = "gelu",
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    minimum_heldout_explained_variance: float = 0.35,
) -> dict[str, Any]:
    """Fit, enforce the held-out EV gate, and emit a native training bootstrap."""

    if not math.isfinite(minimum_heldout_explained_variance):
        raise ValueError("Minimum held-out explained variance must be finite.")
    input_actor_dir = Path(input_actor_dir).expanduser().resolve()
    rank_paths = sorted(input_actor_dir.glob("rank_*.pt"))
    if not rank_paths:
        raise FileNotFoundError("Input actor directory contains no rank checkpoints.")
    first_payload = torch.load(rank_paths[0], map_location="cpu", weights_only=False)
    initial_state = _validate_native_step_zero(first_payload)
    dataset = load_preregression_dataset(dataset_path)
    result = fit_value_head(
        initial_state=initial_state,
        dataset=dataset,
        activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    if result.heldout_explained_variance < minimum_heldout_explained_variance:
        raise RuntimeError(
            "Critic preregression held-out explained variance missed its gate: "
            f"{result.heldout_explained_variance} < "
            f"{minimum_heldout_explained_variance}."
        )
    rank_count = write_preregressed_step_zero(
        input_actor_dir=input_actor_dir,
        output_actor_dir=output_actor_dir,
        value_head_state=result.value_head_state,
        activation=activation,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "input_actor_dir": str(input_actor_dir),
        "output_actor_dir": str(Path(output_actor_dir).expanduser().resolve()),
        "rank_count": rank_count,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "activation": activation,
        "train_sample_count": result.train_sample_count,
        "heldout_sample_count": result.heldout_sample_count,
        "initial_heldout_explained_variance": (
            result.initial_heldout_explained_variance
        ),
        "heldout_explained_variance": result.heldout_explained_variance,
        "heldout_mse": result.heldout_mse,
        "minimum_heldout_explained_variance": float(minimum_heldout_explained_variance),
        "best_epoch": result.best_epoch,
        "strict_checkpoint_reload": True,
    }
    manifest_path = Path(output_actor_dir) / "critic_preregression_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-actor-dir", required=True)
    parser.add_argument("--output-actor-dir", required=True)
    parser.add_argument(
        "--activation", default="gelu", choices=("relu", "gelu", "tanh")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-heldout-ev", type=float, default=0.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_critic_preregression(
        dataset_path=args.dataset,
        input_actor_dir=args.input_actor_dir,
        output_actor_dir=args.output_actor_dir,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        minimum_heldout_explained_variance=args.minimum_heldout_ev,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
