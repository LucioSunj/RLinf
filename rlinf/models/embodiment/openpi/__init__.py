# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import os
import pathlib
from collections.abc import Mapping

import torch
from omegaconf import DictConfig

_VLM_STATE_PREFIX = "paligemma_with_expert.paligemma."


def _is_legacy_value_head_key(key: str) -> bool:
    return "value_head" in key.split(".")


def _merge_vlm_checkpoint_aliases(
    aliases: dict[str, str],
    metadata: Mapping[str, str] | None,
    *,
    source: str,
) -> None:
    """Merge safetensors-declared VLM aliases without accepting conflicts."""

    for alias, canonical in (metadata or {}).items():
        alias_is_vlm = alias.startswith(_VLM_STATE_PREFIX)
        canonical_is_vlm = canonical.startswith(_VLM_STATE_PREFIX)
        if alias_is_vlm != canonical_is_vlm:
            raise ValueError(
                "pi0.5 safetensors alias metadata crosses the VLM boundary: "
                f"{alias!r} -> {canonical!r} in {source}."
            )
        if not alias_is_vlm:
            continue
        previous = aliases.get(alias)
        if previous is not None and previous != canonical:
            raise ValueError(
                "Conflicting pi0.5 safetensors alias metadata for "
                f"{alias!r}: {previous!r} versus {canonical!r} in {source}."
            )
        aliases[alias] = canonical


def _is_verified_tied_vlm_alias(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    missing_key: str,
    checkpoint_aliases: Mapping[str, str],
) -> bool:
    """Return whether a missing VLM key is a verified safetensors tied alias."""

    canonical_key = checkpoint_aliases.get(missing_key)
    if canonical_key is None or not canonical_key.startswith(_VLM_STATE_PREFIX):
        return False
    checkpoint_tensor = state_dict.get(canonical_key)
    if checkpoint_tensor is None:
        return False
    try:
        alias_parameter = model.get_parameter(missing_key)
        canonical_parameter = model.get_parameter(canonical_key)
    except AttributeError:
        return False
    return (
        alias_parameter is canonical_parameter
        and checkpoint_tensor.shape == alias_parameter.shape
        and checkpoint_tensor.shape == canonical_parameter.shape
        and (
            checkpoint_tensor.dtype == alias_parameter.dtype
            or (
                checkpoint_tensor.is_floating_point()
                and alias_parameter.is_floating_point()
            )
        )
    )


def _load_openpi_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict_vlm_checkpoint: bool,
    checkpoint_aliases: Mapping[str, str] | None = None,
) -> None:
    """Load an OpenPi parent while fail-closing on a partial VLM restore."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    if not strict_vlm_checkpoint:
        return

    expected_vlm = {
        key for key in model.state_dict() if key.startswith(_VLM_STATE_PREFIX)
    }
    if not expected_vlm:
        raise ValueError(
            "Strict pi0.5 loading found no PaliGemma VLM parameters in the model."
        )
    aliases = checkpoint_aliases or {}
    missing_vlm = sorted(
        key
        for key in incompatible.missing_keys
        if key.startswith(_VLM_STATE_PREFIX)
        and not _is_verified_tied_vlm_alias(model, state_dict, key, aliases)
    )
    unexpected = sorted(
        key
        for key in incompatible.unexpected_keys
        if not _is_legacy_value_head_key(key)
    )
    if missing_vlm or unexpected:
        raise ValueError(
            "Strict pi0.5 VLM checkpoint mismatch: "
            f"missing_vlm={missing_vlm[:8]}, unexpected={unexpected[:8]}."
        )


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    strict_vlm_checkpoint = bool(getattr(cfg, "strict_vlm_checkpoint", False))
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        _load_openpi_state_dict(
            model,
            model_state_dict,
            strict_vlm_checkpoint=strict_vlm_checkpoint,
        )
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        _load_openpi_state_dict(
            model,
            model_state_dict,
            strict_vlm_checkpoint=strict_vlm_checkpoint,
        )
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        checkpoint_aliases: dict[str, str] = {}
        for weight_path in weight_paths:
            with safetensors.safe_open(
                weight_path, framework="pt", device="cpu"
            ) as handle:
                _merge_vlm_checkpoint_aliases(
                    checkpoint_aliases,
                    handle.metadata(),
                    source=weight_path,
                )
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            duplicate_keys = sorted(all_state_dict.keys() & state_dict.keys())
            if duplicate_keys:
                raise ValueError(
                    "Duplicate tensors across pi0.5 safetensors shards: "
                    f"{duplicate_keys[:8]}."
                )
            all_state_dict.update(state_dict)
        _load_openpi_state_dict(
            model,
            all_state_dict,
            strict_vlm_checkpoint=strict_vlm_checkpoint,
            checkpoint_aliases=checkpoint_aliases,
        )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
