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

"""LIBERO observation and action runtime for the composite FastWAM policy."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Literal

import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
)
from fastwam.models.wan22.adaptive_sampler import (
    replay_action_flow_sde_transition,
    sample_action_flow_sde,
)
from fastwam.models.wan22.kv_tap import (
    GateKVSnapshot,
    GateLayerKV,
    KeyValueBank,
)

from rlinf.envs.action_contract import (
    DENORMALIZED_ACTION_STAGE,
    GRIPPER_CONVERTED_ACTION_STAGE,
    NORMALIZED_ACTION_STAGE,
    ActionExecutionTrace,
    ActionStageStatistics,
)
from rlinf.envs.libero.action_protocol import (
    LiberoActionProtocol,
    select_executed_action_prefix,
    select_executed_flow_statistics,
)
from rlinf.envs.libero.image_preprocessing import (
    OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
    prepare_libero_camera_batch,
)

from .adaptive_policy import FastWAMChunkSample
from .contracts import ChunkRouteRecord, WAMRoute
from .critic import (
    FastWAMCurrentFrameFeatureConfig,
    pool_current_frame_video_values,
)
from .kv_replay import GateKVReplayBackend

DEFAULT_FASTWAM_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _format_fastwam_prompts(
    task_descriptions: str | list[str] | tuple[str, ...],
    *,
    prompt_template: str,
) -> list[str]:
    if "{task}" not in prompt_template:
        raise ValueError("FastWAM `prompt_template` must contain `{task}`.")
    descriptions = (
        [task_descriptions]
        if isinstance(task_descriptions, str)
        else list(task_descriptions)
    )
    return [prompt_template.format(task=str(task)) for task in descriptions]


def _load_cached_text_contexts(
    prompts: list[str],
    *,
    cache_dir: Path,
    context_len: int,
    expected_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load deterministic precomputed Wan2.2 contexts without loading UMT5."""

    if context_len < 1 or expected_dim < 1:
        raise ValueError("Cached text context length and dimension must be positive.")
    contexts = []
    masks = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing FastWAM evaluation text context for prompt hash {digest}: "
                f"{path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"context", "mask"}:
            raise ValueError(f"Cached text context has an invalid schema: {path}")
        context = payload["context"]
        mask = payload["mask"]
        if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError(f"Cached text context values must be tensors: {path}")
        if context.shape != torch.Size([context_len, expected_dim]):
            raise ValueError(
                "Cached text context shape mismatch: "
                f"expected {(context_len, expected_dim)}, got {tuple(context.shape)} "
                f"in {path}."
            )
        if mask.shape != torch.Size([context_len]):
            raise ValueError(
                f"Cached text mask shape mismatch in {path}: {tuple(mask.shape)}."
            )
        if context.dtype is not torch.bfloat16 or mask.dtype is not torch.bool:
            raise TypeError(
                "Cached text context must use BF16 context and bool mask: "
                f"context={context.dtype}, mask={mask.dtype}, path={path}."
            )
        if not torch.isfinite(context.float()).all():
            raise ValueError(f"Cached text context contains non-finite values: {path}")
        # Match both FastWAM training and ``FastWAM.encode_prompt``: the UMT5
        # padding embeddings remain present in the raw cache, but consumers
        # must zero them and then expose an all-valid mask because Wan2.2's
        # original cross-attention behavior sees the zero padding tokens.
        context = context.clone()
        context[~mask] = 0
        mask = torch.ones_like(mask)
        masks.append(mask)
        contexts.append(context)
    return (
        torch.stack(contexts).to(device=device, dtype=dtype),
        torch.stack(masks).to(device=device, dtype=torch.bool),
    )


def _cat_bank(banks: list[KeyValueBank]) -> KeyValueBank:
    first = banks[0]
    if any(
        bank.source is not first.source
        or bank.contains_generated_future_video != first.contains_generated_future_video
        for bank in banks[1:]
    ):
        raise ValueError("Cannot batch Gate K/V banks with different sources.")
    return KeyValueBank(
        source=first.source,
        key=torch.cat([bank.key for bank in banks], dim=0),
        value=torch.cat([bank.value for bank in banks], dim=0),
        valid_mask=torch.cat([bank.valid_mask for bank in banks], dim=0),
        contains_generated_future_video=first.contains_generated_future_video,
    )


def _cat_snapshots(snapshots: list[GateKVSnapshot]) -> GateKVSnapshot:
    first = snapshots[0]
    if any(snapshot.layer_indices != first.layer_indices for snapshot in snapshots[1:]):
        raise ValueError("Cannot batch Gate snapshots with different layer taps.")
    actor_versions = {snapshot.layers[0].actor_version for snapshot in snapshots}
    if len(actor_versions) != 1:
        raise ValueError("Cannot batch Gate snapshots from different actor versions.")
    layers = []
    for layer_index in first.layer_indices:
        source_layers = [snapshot.layer(layer_index) for snapshot in snapshots]
        layers.append(
            GateLayerKV(
                layer_index=layer_index,
                denoise_timestep=torch.cat(
                    [layer.denoise_timestep for layer in source_layers],
                    dim=0,
                ),
                current_mode=tuple(
                    mode for layer in source_layers for mode in layer.current_mode
                ),
                current_frame_video=_cat_bank(
                    [layer.current_frame_video for layer in source_layers]
                ),
                action=_cat_bank([layer.action for layer in source_layers]),
                context=_cat_bank([layer.context for layer in source_layers]),
                actor_version=source_layers[0].actor_version,
            )
        )
    return GateKVSnapshot(tuple(layers))


def _validate_flow_sde_sampling(
    *,
    mode: Literal["train", "eval"],
    routes: torch.Tensor,
    noise_level: float,
) -> None:
    if mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported FastWAM runtime mode {mode!r}.")
    if not math.isfinite(noise_level) or noise_level < 0:
        raise ValueError("`flow_sde_noise_level` must be finite and non-negative.")
    has_uncond = bool((routes == int(WAMRoute.UNCOND)).any().item())
    if mode == "train" and has_uncond and noise_level <= 0:
        raise ValueError(
            "Training an UNCOND Flow-SDE transition requires "
            "`flow_sde_noise_level > 0`; zero is valid only for deterministic eval."
        )


def _validate_noise_seeds(
    value: Any,
    *,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    """Validate compact per-sample seeds without touching global RNG state."""

    seeds = torch.as_tensor(value)
    if seeds.ndim != 1:
        raise ValueError(f"FastWAM {name} seeds must be one-dimensional.")
    if seeds.shape[0] != batch_size:
        raise ValueError(
            f"FastWAM {name} seed batch must match routes: "
            f"{seeds.shape[0]} != {batch_size}."
        )
    if seeds.dtype == torch.bool or seeds.dtype.is_floating_point:
        raise TypeError(f"FastWAM {name} seeds must use an integer dtype.")
    seeds = seeds.to(device="cpu", dtype=torch.long)
    if bool((seeds < 0).any()):
        raise ValueError(f"FastWAM {name} seeds must be non-negative.")
    return seeds


def _seeded_randn(
    seed: int,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    rand_device: str = "cpu",
) -> torch.Tensor:
    """Generate one noise tensor from a local generator."""

    seed = int(seed)
    if seed < 0:
        raise ValueError("FastWAM local noise seed must be non-negative.")
    if not shape or any(int(dimension) < 1 for dimension in shape):
        raise ValueError("FastWAM seeded noise shape must be non-empty and positive.")
    if rand_device not in {"cpu", "model"}:
        raise ValueError("FastWAM seeded noise device must be either 'cpu' or 'model'.")
    generation_device = device if rand_device == "model" else torch.device("cpu")
    generator = torch.Generator(device=generation_device)
    generator.manual_seed(seed)
    return torch.randn(
        shape,
        device=generation_device,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)


def _domain_separated_noise_seed(seed: int, *, domain: str) -> int:
    """Derive a local substream without advancing any global RNG state."""

    payload = b"\0".join(
        (
            b"fastwam-formal-training-noise-v1",
            str(int(seed)).encode("ascii"),
            str(domain).encode("utf-8"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _align_linear_normalizer(normalizer: Any, reference: torch.Tensor) -> None:
    """Move FastWAM's plain-tensor normalizer state beside its input."""

    for name in ("scale", "offset"):
        value = getattr(normalizer, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"FastWAM normalizer is missing tensor `{name}`.")
        if value.device != reference.device or value.dtype != reference.dtype:
            setattr(
                normalizer,
                name,
                value.to(device=reference.device, dtype=reference.dtype),
            )


def _convert_fastwam_gripper_to_libero(
    actions: torch.Tensor,
    *,
    binarize: bool,
) -> torch.Tensor:
    """Match FastWAM's official LIBERO action postprocessing."""

    if actions.shape[-1] < 1:
        raise ValueError("FastWAM actions must contain a gripper dimension.")
    result = actions.clone()
    # Dataset convention: 0=close, 1=open. LIBERO convention after the
    # official FastWAM sign inversion: +1=close, -1=open.
    result[..., -1] = 1.0 - 2.0 * result[..., -1]
    if binarize:
        result[..., -1] = torch.sign(result[..., -1])
    return result


def _action_contract_metadata(
    env_obs: dict[str, Any],
    *,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, str] | None:
    """Validate compact exact-spec metadata supplied by the eval collector."""

    keys = (
        "_fastwam_action_contract_low",
        "_fastwam_action_contract_high",
        "_fastwam_action_gripper_indices",
        "_fastwam_action_contract_sha256",
    )
    present = [key in env_obs for key in keys]
    if not any(present):
        return None
    if not all(present):
        missing = [key for key, exists in zip(keys, present) if not exists]
        raise ValueError(f"Incomplete FastWAM Action contract metadata: {missing}.")
    batch_size = int(actions.shape[0])
    action_dim = int(actions.shape[-1])
    low = torch.as_tensor(env_obs[keys[0]], device=actions.device, dtype=torch.float32)
    high = torch.as_tensor(env_obs[keys[1]], device=actions.device, dtype=torch.float32)
    if low.shape != (batch_size, action_dim) or high.shape != low.shape:
        raise ValueError("FastWAM Action contract bounds must have shape [B, D].")
    gripper_indices = torch.as_tensor(env_obs[keys[2]], dtype=torch.long)
    if gripper_indices.shape != (batch_size,):
        raise ValueError("FastWAM gripper indices must have shape [B].")
    unique_gripper = torch.unique(gripper_indices)
    if unique_gripper.numel() != 1:
        raise ValueError("One FastWAM Action batch cannot mix gripper indices.")
    gripper_index = int(unique_gripper.item())
    if not 0 <= gripper_index < action_dim:
        raise ValueError("FastWAM gripper index is outside the Action dimension.")
    hashes = env_obs[keys[3]]
    if isinstance(hashes, str):
        hashes = [hashes] * batch_size
    if not isinstance(hashes, (list, tuple)) or len(hashes) != batch_size:
        raise ValueError("FastWAM Action contract hashes must align with the batch.")
    hashes = [str(item) for item in hashes]
    if len(set(hashes)) != 1:
        raise ValueError("One FastWAM Action batch cannot mix contract hashes.")
    contract_hash = hashes[0]
    if len(contract_hash) != 64 or any(
        character not in "0123456789abcdef" for character in contract_hash
    ):
        raise ValueError("FastWAM Action contract SHA256 is invalid.")
    return low, high, gripper_index, contract_hash


class LiberoFastWAMRuntime:
    """Correctness-first mixed-route runtime using the checked-out FastWAM."""

    def __init__(
        self,
        *,
        actor,
        lora_adapter,
        processor: Any | None = None,
        processor_stats_path: str | None = None,
        generation_horizon: int = 32,
        execution_horizon: int = 10,
        num_video_frames: int = 9,
        reset_wait_steps: int = 30,
        max_episode_steps: int = 700,
        num_inference_steps: int = 10,
        seeded_noise_device: str = "cpu",
        sigma_shift: float | None = None,
        flow_sde_noise_level: float = 0.5,
        flow_sde_ignore_last_transition: bool = False,
        gate_layer_indices: tuple[int, ...] | list[int] | None = None,
        gate_denoise_last_n: int = 1,
        gate_replay_backend: GateKVReplayBackend | str = GateKVReplayBackend.STORED,
        critic_feature_config: FastWAMCurrentFrameFeatureConfig | None = None,
        camera_height: int = 224,
        camera_width: int = 224,
        camera_concat: str = "horizontal",
        tiled_vae: bool = False,
        camera_resize_mode: str = OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
        binarize_gripper: bool = False,
        prompt_template: str = DEFAULT_FASTWAM_PROMPT_TEMPLATE,
        text_embedding_cache_dir: str | None = None,
        text_embedding_context_len: int = 128,
    ) -> None:
        self.actor = actor
        self.lora_adapter = lora_adapter
        self.processor = processor
        self.processor_stats_path = processor_stats_path
        self.num_video_frames = int(num_video_frames)
        self.num_inference_steps = int(num_inference_steps)
        self.seeded_noise_device = str(seeded_noise_device)
        self.action_protocol = LiberoActionProtocol(
            generation_horizon=generation_horizon,
            execution_horizon=execution_horizon,
            prediction_video_frames=num_video_frames,
            reset_wait_steps=reset_wait_steps,
            max_episode_steps=max_episode_steps,
        )
        self.sigma_shift = sigma_shift
        self.flow_sde_noise_level = float(flow_sde_noise_level)
        self.flow_sde_ignore_last_transition = bool(flow_sde_ignore_last_transition)
        self.gate_layer_indices = (
            None
            if gate_layer_indices is None
            else tuple(int(index) for index in gate_layer_indices)
        )
        self.gate_denoise_last_n = int(gate_denoise_last_n)
        self.gate_replay_backend = GateKVReplayBackend(gate_replay_backend)
        self.critic_feature_config = critic_feature_config
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.camera_concat = str(camera_concat)
        self.tiled_vae = bool(tiled_vae)
        self.camera_resize_mode = str(camera_resize_mode)
        self.binarize_gripper = bool(binarize_gripper)
        self.prompt_template = str(prompt_template)
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser().resolve()
        )
        self.text_embedding_context_len = int(text_embedding_context_len)
        if self.num_inference_steps < 1:
            raise ValueError("Inference steps must be positive.")
        if self.flow_sde_ignore_last_transition and self.num_inference_steps < 2:
            raise ValueError(
                "Ignoring the final Flow-SDE transition requires at least two "
                "inference steps."
            )
        if self.seeded_noise_device not in {"cpu", "model"}:
            raise ValueError("seeded_noise_device must be either 'cpu' or 'model'.")
        if self.num_video_frames % 4 != 1:
            raise ValueError("`num_video_frames` must satisfy T % 4 == 1.")
        if (
            not math.isfinite(self.flow_sde_noise_level)
            or self.flow_sde_noise_level < 0
        ):
            raise ValueError("Flow-SDE noise level must be finite and non-negative.")
        if self.gate_denoise_last_n < 1:
            raise ValueError("`gate_denoise_last_n` must be positive.")
        if self.camera_concat not in {"horizontal", "vertical", "main_only"}:
            raise ValueError("Unsupported camera concatenation.")
        if self.camera_resize_mode != OFFICIAL_LIBERO_CAMERA_RESIZE_MODE:
            raise ValueError(
                f"Unsupported LIBERO camera resize mode: {self.camera_resize_mode!r}."
            )
        if (self.processor is None) != (self.processor_stats_path is None):
            raise ValueError(
                "FastWAM processor and `processor_stats_path` must be configured together."
            )
        if self.text_embedding_context_len < 1:
            raise ValueError("`text_embedding_context_len` must be positive.")
        if self.text_embedding_cache_dir is not None and not (
            self.text_embedding_cache_dir.is_dir()
        ):
            raise FileNotFoundError(self.text_embedding_cache_dir)
        _format_fastwam_prompts("validation", prompt_template=self.prompt_template)
        if self.processor is not None:
            from fastwam.datasets.lerobot.utils.normalizer import (
                load_dataset_stats_from_json,
            )

            stats = load_dataset_stats_from_json(str(self.processor_stats_path))
            self.processor.eval()
            self.processor.set_normalizer_from_stats(stats)

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.actor.parameters()).dtype

    def _camera_tensor(self, image: torch.Tensor) -> torch.Tensor:
        image = prepare_libero_camera_batch(
            image,
            height=self.camera_height,
            width=self.camera_width,
            resize_mode=self.camera_resize_mode,
        )
        return image.to(device=self.device, dtype=torch.float32)

    def _model_images(self, env_obs: dict[str, Any]) -> torch.Tensor:
        main = self._camera_tensor(env_obs["main_images"])
        if self.camera_concat == "main_only":
            image = main
        else:
            wrist = self._camera_tensor(env_obs["wrist_images"])
            dim = 3 if self.camera_concat == "horizontal" else 2
            image = torch.cat([main, wrist], dim=dim)
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("Combined FastWAM input image must be divisible by 16.")
        return image.to(dtype=self.dtype) * (2.0 / 255.0) - 1.0

    def _normalized_proprio(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(device=self.device, dtype=torch.float32)
        if self.processor is None:
            return states.to(dtype=self.dtype)
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("FastWAM LIBERO runtime expects one merged state key.")
        state_key = state_meta[0]["key"]
        batch = {"state": {state_key: states}}
        batch = self.processor.action_state_transform(batch)
        _align_linear_normalizer(
            self.processor.normalizer.normalizers["state"][state_key],
            batch["state"][state_key],
        )
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][state_key].to(device=self.device, dtype=self.dtype)

    def _denormalize_action_stages(
        self,
        actions: torch.Tensor,
        *,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, ActionExecutionTrace | None]:
        metadata = _action_contract_metadata(env_obs, actions=actions)
        if self.processor is None:
            denormalized = actions.float()
        else:
            action_meta = self.processor.shape_meta["action"]
            if len(action_meta) != 1:
                raise ValueError(
                    "FastWAM LIBERO runtime expects one merged action key."
                )
            action_key = action_meta[0]["key"]
            normalizer = self.processor.normalizer.normalizers["action"][action_key]
            float_actions = actions.float()
            _align_linear_normalizer(normalizer, float_actions)
            denormalized = normalizer.backward(float_actions).to(device=actions.device)
        converted = _convert_fastwam_gripper_to_libero(
            denormalized,
            binarize=self.binarize_gripper,
        )
        if metadata is None:
            return converted, None
        low, high, gripper_index, contract_hash = metadata
        trace = ActionExecutionTrace(
            stages=tuple(
                ActionStageStatistics.from_values(
                    stage=stage,
                    values=values,
                    low=low,
                    high=high,
                    gripper_dimension_index=gripper_index,
                    action_contract_sha256=contract_hash,
                )
                for stage, values in (
                    (NORMALIZED_ACTION_STAGE, actions),
                    (DENORMALIZED_ACTION_STAGE, denormalized),
                    (GRIPPER_CONVERTED_ACTION_STAGE, converted),
                )
            )
        )
        return converted, trace

    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Preserve the production conversion API outside instrumented eval."""

        converted, trace = self._denormalize_action_stages(
            actions,
            env_obs={},
        )
        if trace is not None:
            raise AssertionError("Uninstrumented conversion unexpectedly made a trace.")
        return converted

    def _encode_condition(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = self._model_images(env_obs)
        prompts = _format_fastwam_prompts(
            env_obs["task_descriptions"],
            prompt_template=self.prompt_template,
        )
        if self.text_embedding_cache_dir is None:
            context, context_mask = self.actor.encode_prompt(prompts)
        else:
            context, context_mask = _load_cached_text_contexts(
                prompts,
                cache_dir=self.text_embedding_cache_dir,
                context_len=self.text_embedding_context_len,
                expected_dim=int(self.actor.text_dim),
                device=self.device,
                dtype=self.dtype,
            )
        proprio = self._normalized_proprio(env_obs["states"])
        context, context_mask = self.actor._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        return images, context, context_mask

    @torch.no_grad()
    def _prepare_action_condition(
        self,
        *,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        regime: PolicyRegime,
        idm_initial_latents: torch.Tensor | None = None,
        idm_noise_seed: int | None = None,
    ) -> tuple[CachedActionCondition, torch.Tensor | None]:
        first_frame = self.actor._encode_input_image_latents_tensor(
            image,
            tiled=self.tiled_vae,
        )
        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        video_latents = first_frame
        replay_initial_latents = None
        if regime is PolicyRegime.IDM:
            latent_t = (
                self.num_video_frames - 1
            ) // self.actor.vae.temporal_downsample_factor + 1
            latent_h = image.shape[-2] // self.actor.vae.upsampling_factor
            latent_w = image.shape[-1] // self.actor.vae.upsampling_factor
            expected_shape = (
                1,
                self.actor.vae.model.z_dim,
                latent_t,
                latent_h,
                latent_w,
            )
            if idm_initial_latents is not None and idm_noise_seed is not None:
                raise ValueError(
                    "Specify either injected IDM latents or an IDM seed, not both."
                )
            if idm_initial_latents is None:
                video_latents = (
                    torch.randn(
                        expected_shape,
                        device=self.device,
                        dtype=torch.float32,
                    ).to(dtype=self.dtype)
                    if idm_noise_seed is None
                    else _seeded_randn(
                        idm_noise_seed,
                        expected_shape,
                        device=self.device,
                        dtype=self.dtype,
                        rand_device=self.seeded_noise_device,
                    )
                )
            else:
                if tuple(idm_initial_latents.shape) != expected_shape:
                    raise ValueError(
                        "Recomputed IDM initial latents have shape "
                        f"{tuple(idm_initial_latents.shape)}, expected {expected_shape}."
                    )
                video_latents = idm_initial_latents.to(
                    device=self.device,
                    dtype=self.dtype,
                ).clone()
            replay_initial_latents = video_latents.detach().clone()
            video_latents[:, :, :1] = first_frame
            video_timesteps, video_deltas = (
                self.actor.infer_video_scheduler.build_inference_schedule(
                    num_inference_steps=self.num_inference_steps,
                    device=self.device,
                    dtype=self.dtype,
                    shift_override=self.sigma_shift,
                )
            )
            for timestep, delta in zip(video_timesteps, video_deltas):
                timestep_batch = timestep.expand(1).to(dtype=self.dtype)
                velocity = self.actor._video_denoise_step_compiled(
                    latents_video=video_latents,
                    timestep_video=timestep_batch,
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                )
                video_latents = self.actor.infer_video_scheduler.step(
                    velocity,
                    delta,
                    video_latents,
                )
                video_latents[:, :, :1] = first_frame

        video_pre = self.actor.video_expert.pre_dit(
            x=video_latents,
            timestep=torch.zeros(1, device=self.device, dtype=self.dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = self.actor._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.action_protocol.generation_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        video_cache = self.actor.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            gate_current_frame_video_tokens=tokens_per_frame,
        )
        return (
            CachedActionCondition(
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                current_frame_video_tokens=tokens_per_frame,
            ),
            replay_initial_latents,
        )

    def _action_schedule(self):
        # Keep schedule arithmetic in FP32. The sampler casts model timesteps
        # and ODE deltas to the action dtype only at their point of use; building
        # both arrays in BF16 can otherwise round the final next-time below zero.
        return self.actor.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            dtype=torch.float32,
            shift_override=self.sigma_shift,
        )

    def _velocity(
        self,
        condition: CachedActionCondition,
        *,
        regime: PolicyRegime,
        capture_gate_kv: bool,
        actor_version: int,
    ) -> CachedActionVelocity:
        return CachedActionVelocity(
            action_expert=self.actor.action_expert,
            mot=self.actor.mot,
            condition=condition,
            regime=regime,
            regime_context=self.lora_adapter.regime_context,
            gate_layer_indices=self.gate_layer_indices,
            capture_gate_kv=capture_gate_kv,
            actor_version=actor_version,
        )

    def sample_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        mode: Literal["train", "eval"],
        actor_version: int,
        collect_replay: bool = True,
    ) -> FastWAMChunkSample:
        _validate_flow_sde_sampling(
            mode=mode,
            routes=routes,
            noise_level=self.flow_sde_noise_level,
        )
        images, context, context_mask = self._encode_condition(env_obs)
        timesteps, deltas = self._action_schedule()
        action_noise_override = env_obs.get("_fastwam_action_initial_noise")
        if action_noise_override is not None:
            action_noise_override = torch.as_tensor(
                action_noise_override,
                device=self.device,
                dtype=self.dtype,
            )
            expected = (
                routes.shape[0],
                self.action_protocol.generation_horizon,
                self.actor.action_expert.action_dim,
            )
            if tuple(action_noise_override.shape) != expected:
                raise ValueError(
                    "Injected FastWAM action noise has shape "
                    f"{tuple(action_noise_override.shape)}, expected {expected}."
                )
        action_noise_seeds = env_obs.get("_fastwam_action_noise_seeds")
        if action_noise_seeds is not None:
            if action_noise_override is not None:
                raise ValueError(
                    "Specify either injected action noise or action seeds, not both."
                )
            action_noise_seeds = _validate_noise_seeds(
                action_noise_seeds,
                batch_size=routes.shape[0],
                name="action noise",
            )
        video_noise_override = env_obs.get("_fastwam_idm_initial_latents")
        if video_noise_override is not None:
            video_noise_override = torch.as_tensor(
                video_noise_override,
                device=self.device,
                dtype=self.dtype,
            )
            if video_noise_override.shape[0] != routes.shape[0]:
                raise ValueError(
                    "Injected FastWAM IDM video noise batch must match routes."
                )
        idm_noise_seeds = env_obs.get("_fastwam_idm_noise_seeds")
        if idm_noise_seeds is not None:
            if video_noise_override is not None:
                raise ValueError(
                    "Specify either injected IDM latents or IDM seeds, not both."
                )
            idm_noise_seeds = _validate_noise_seeds(
                idm_noise_seeds,
                batch_size=routes.shape[0],
                name="IDM video noise",
            )
        rollouts = []
        idm_initial_latents = []
        critic_features = []
        for index, route_value in enumerate(routes.tolist()):
            regime = (
                PolicyRegime.IDM
                if int(route_value) == int(WAMRoute.IDM)
                else PolicyRegime.UNCOND
            )
            condition, replay_video_noise = self._prepare_action_condition(
                image=images[index : index + 1],
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                regime=regime,
                idm_initial_latents=(
                    video_noise_override[index : index + 1]
                    if video_noise_override is not None and regime is PolicyRegime.IDM
                    else None
                ),
                idm_noise_seed=(
                    int(idm_noise_seeds[index])
                    if idm_noise_seeds is not None and regime is PolicyRegime.IDM
                    else None
                ),
            )
            if self.critic_feature_config is not None and mode == "train":
                critic_features.append(
                    pool_current_frame_video_values(
                        condition,
                        self.critic_feature_config,
                    )
                )
            if (
                collect_replay
                and self.gate_replay_backend is GateKVReplayBackend.RECOMPUTE
            ):
                if replay_video_noise is None:
                    latent_t = (
                        self.num_video_frames - 1
                    ) // self.actor.vae.temporal_downsample_factor + 1
                    replay_video_noise = torch.zeros(
                        (
                            1,
                            self.actor.vae.model.z_dim,
                            latent_t,
                            images.shape[-2] // self.actor.vae.upsampling_factor,
                            images.shape[-1] // self.actor.vae.upsampling_factor,
                        ),
                        device=self.device,
                        dtype=self.dtype,
                    )
                idm_initial_latents.append(replay_video_noise)
            if action_noise_override is None:
                action_shape = (
                    1,
                    self.action_protocol.generation_horizon,
                    self.actor.action_expert.action_dim,
                )
                initial_noise = (
                    torch.randn(
                        action_shape,
                        device=self.device,
                        dtype=torch.float32,
                    ).to(dtype=self.dtype)
                    if action_noise_seeds is None
                    else _seeded_randn(
                        int(action_noise_seeds[index]),
                        action_shape,
                        device=self.device,
                        dtype=self.dtype,
                        rand_device=self.seeded_noise_device,
                    )
                )
            else:
                initial_noise = action_noise_override[index : index + 1]
            flow_generator = None
            if action_noise_seeds is not None:
                flow_generator = torch.Generator(device=self.device)
                flow_generator.manual_seed(
                    _domain_separated_noise_seed(
                        int(action_noise_seeds[index]),
                        domain="flow-sde",
                    )
                )
            rollout = sample_action_flow_sde(
                initial_noise,
                velocity_fn=self._velocity(
                    condition,
                    regime=regime,
                    capture_gate_kv=True,
                    actor_version=actor_version,
                ),
                timesteps=timesteps,
                scheduler_deltas=deltas,
                num_train_timesteps=(
                    self.actor.infer_action_scheduler.num_train_timesteps
                ),
                noise_level=self.flow_sde_noise_level,
                generator=flow_generator,
                gate_last_n=self.gate_denoise_last_n,
                ignore_last_transition=self.flow_sde_ignore_last_transition,
                stochastic=mode == "train" and regime is PolicyRegime.UNCOND,
                collect_replay=collect_replay,
            )
            rollouts.append(rollout)

        gate_snapshots = tuple(
            _cat_snapshots([rollout.gate_taps[tap_index] for rollout in rollouts])
            for tap_index in range(self.gate_denoise_last_n)
        )
        normalized_actions = torch.cat(
            [rollout.actions for rollout in rollouts],
            dim=0,
        )
        executed_normalized_actions = select_executed_action_prefix(
            normalized_actions,
            protocol=self.action_protocol,
        )
        processed_actions, action_execution_trace = self._denormalize_action_stages(
            executed_normalized_actions,
            env_obs=env_obs,
        )
        replay_inputs = {}
        if collect_replay:
            replay_inputs = {
                "fastwam_images": images.detach(),
                "fastwam_context": context.detach(),
                "fastwam_context_mask": context_mask.detach(),
            }
        if collect_replay and self.gate_replay_backend is GateKVReplayBackend.RECOMPUTE:
            replay_inputs["fastwam_idm_initial_latents"] = torch.cat(
                idm_initial_latents,
                dim=0,
            ).detach()
        return FastWAMChunkSample(
            actions=processed_actions,
            old_flow_logprobs=select_executed_flow_statistics(
                torch.cat(
                    [rollout.old_log_probs for rollout in rollouts],
                    dim=0,
                ),
                protocol=self.action_protocol,
            ),
            flow_chains=torch.cat([rollout.chains for rollout in rollouts], dim=0),
            denoise_indices=torch.cat(
                [rollout.denoise_indices for rollout in rollouts],
                dim=0,
            ),
            gate_snapshots=gate_snapshots,
            forward_inputs=replay_inputs,
            critic_features=(
                None if not critic_features else torch.cat(critic_features, dim=0)
            ),
            action_execution_trace=action_execution_trace,
        )

    @torch.no_grad()
    def critic_features(self, *, env_obs: dict[str, Any]) -> torch.Tensor:
        """Encode current-frame critic features without action or route updates."""

        if self.critic_feature_config is None:
            raise RuntimeError(
                "FastWAM current-frame critic features were not configured."
            )
        images, context, context_mask = self._encode_condition(env_obs)
        features = []
        for index in range(images.shape[0]):
            condition, _ = self._prepare_action_condition(
                image=images[index : index + 1],
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                regime=PolicyRegime.UNCOND,
            )
            features.append(
                pool_current_frame_video_values(
                    condition,
                    self.critic_feature_config,
                )
            )
        return torch.cat(features, dim=0)

    def replay_action_batch(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
        compute_base_logprobs: bool = False,
    ) -> dict[str, torch.Tensor]:
        _validate_flow_sde_sampling(
            mode="train",
            routes=route_info.route_used,
            noise_level=self.flow_sde_noise_level,
        )
        chains = forward_inputs["flow_chains"]
        indices = forward_inputs["denoise_indices"]
        images = forward_inputs["fastwam_images"]
        context = forward_inputs["fastwam_context"]
        context_mask = forward_inputs["fastwam_context_mask"]
        timesteps, deltas = self._action_schedule()
        logprobs = torch.zeros_like(chains[:, 0], dtype=torch.float32)
        entropies = torch.zeros_like(chains[:, 0], dtype=torch.float32)
        base_kl = (
            torch.zeros_like(chains[:, 0], dtype=torch.float32)
            if compute_base_logprobs
            else None
        )
        for index in range(chains.shape[0]):
            if int(route_info.route_used[index]) != int(WAMRoute.UNCOND):
                continue
            condition, _ = self._prepare_action_condition(
                image=images[index : index + 1],
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                regime=PolicyRegime.UNCOND,
            )
            current_replay = replay_action_flow_sde_transition(
                chains[index : index + 1],
                indices[index : index + 1],
                velocity_fn=self._velocity(
                    condition,
                    regime=PolicyRegime.UNCOND,
                    capture_gate_kv=False,
                    actor_version=int(route_info.actor_versions[index]),
                ),
                timesteps=timesteps,
                scheduler_deltas=deltas,
                num_train_timesteps=(
                    self.actor.infer_action_scheduler.num_train_timesteps
                ),
                noise_level=self.flow_sde_noise_level,
            )
            logprobs[index : index + 1] = current_replay.log_prob
            entropies[index : index + 1] = torch.broadcast_to(
                current_replay.std.float().log()
                + 0.5 * math.log(2.0 * math.pi * math.e),
                current_replay.mean.shape,
            )
            if base_kl is not None:
                with torch.no_grad():
                    base_replay = replay_action_flow_sde_transition(
                        chains[index : index + 1],
                        indices[index : index + 1],
                        velocity_fn=self._velocity(
                            condition,
                            # The conditioning remains UNCOND; the IDM regime
                            # only disables the regime-gated LoRA contribution.
                            regime=PolicyRegime.IDM,
                            capture_gate_kv=False,
                            actor_version=int(route_info.actor_versions[index]),
                        ),
                        timesteps=timesteps,
                        scheduler_deltas=deltas,
                        num_train_timesteps=(
                            self.actor.infer_action_scheduler.num_train_timesteps
                        ),
                        noise_level=self.flow_sde_noise_level,
                    )
                base_kl[index : index + 1] = (
                    0.5
                    * (
                        (current_replay.mean.float() - base_replay.mean.float())
                        / current_replay.std.float()
                    ).square()
                )
        result = {
            "flow_logprobs": select_executed_action_prefix(
                logprobs,
                protocol=self.action_protocol,
            ),
            "flow_entropy": select_executed_action_prefix(
                entropies,
                protocol=self.action_protocol,
            ),
        }
        if base_kl is not None:
            result["base_uncond_kl"] = select_executed_action_prefix(
                base_kl,
                protocol=self.action_protocol,
            )
        return result

    @torch.no_grad()
    def recompute_gate_snapshots(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
    ) -> tuple[GateKVSnapshot, ...]:
        """Rebuild the configured final Gate taps from stored chain material."""

        if self.gate_replay_backend is not GateKVReplayBackend.RECOMPUTE:
            raise RuntimeError("Gate K/V recomputation is disabled for this runtime.")
        required = {
            "flow_chains",
            "fastwam_images",
            "fastwam_context",
            "fastwam_context_mask",
            "fastwam_idm_initial_latents",
        }
        missing = sorted(required - set(forward_inputs))
        if missing:
            raise KeyError(f"Gate K/V recomputation is missing inputs: {missing}.")

        chains = forward_inputs["flow_chains"]
        images = forward_inputs["fastwam_images"]
        context = forward_inputs["fastwam_context"]
        context_mask = forward_inputs["fastwam_context_mask"]
        video_noise = forward_inputs["fastwam_idm_initial_latents"]
        timesteps, _ = self._action_schedule()
        if chains.shape[1] != timesteps.numel() + 1:
            raise ValueError(
                "Stored action chain does not match the current scheduler."
            )
        if self.gate_denoise_last_n > timesteps.numel():
            raise ValueError("Gate denoising tap count exceeds the action schedule.")

        actor_versions = torch.unique(route_info.actor_versions)
        if actor_versions.numel() != 1:
            raise ValueError(
                "One Gate K/V recompute batch must contain one actor version."
            )
        replay_actor_version = int(actor_versions.item())
        snapshots: list[GateKVSnapshot] = []
        with self.lora_adapter.use_replay_reference(actor_version=replay_actor_version):
            first_step = timesteps.numel() - self.gate_denoise_last_n
            for step_index in range(first_step, timesteps.numel()):
                per_sample: list[GateKVSnapshot] = []
                for index, route_value in enumerate(route_info.route_used.tolist()):
                    regime = (
                        PolicyRegime.IDM
                        if int(route_value) == int(WAMRoute.IDM)
                        else PolicyRegime.UNCOND
                    )
                    condition, _ = self._prepare_action_condition(
                        image=images[index : index + 1],
                        context=context[index : index + 1],
                        context_mask=context_mask[index : index + 1],
                        regime=regime,
                        idm_initial_latents=(
                            video_noise[index : index + 1]
                            if regime is PolicyRegime.IDM
                            else None
                        ),
                    )
                    timestep = (
                        timesteps[step_index]
                        .to(
                            device=self.device,
                            dtype=self.dtype,
                        )
                        .expand(1)
                    )
                    output = self._velocity(
                        condition,
                        regime=regime,
                        capture_gate_kv=True,
                        actor_version=replay_actor_version,
                    )(
                        chains[index : index + 1, step_index].to(
                            device=self.device,
                            dtype=self.dtype,
                        ),
                        timestep,
                    )
                    if output.gate_tap is None:
                        raise RuntimeError(
                            "FastWAM Gate K/V recomputation produced no tap."
                        )
                    per_sample.append(output.gate_tap)
                snapshots.append(_cat_snapshots(per_sample))
        return tuple(snapshots)

    def critic_observation(
        self,
        *,
        env_obs: dict[str, Any] | None = None,
        forward_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if env_obs is None:
            raise ValueError(
                "Critic replay uses stored pi0.5 prefix features, not raw observations."
            )
        observation = {
            key: value
            for key, value in env_obs.items()
            if not key.startswith("_fastwam_")
        }
        observation.setdefault("wrist_images", None)
        observation.setdefault("extra_view_images", None)
        return observation
