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

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig as OpenVLAOFTConfig,
)
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from transformers.generation import TopKLogitsWarper

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction, BasePolicy):
    def __init__(
        self,
        config: OpenVLAOFTConfig,
        action_dim,
        num_action_chunks,
        add_value_head,
        max_prompt_length,
        add_q_head=False,
    ) -> None:
        super().__init__(config)

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.flat_action_dim = self.action_dim * self.num_action_chunks
        self.hidden_size = self.config.hidden_size
        self.add_q_head = add_q_head or getattr(config, "add_q_head", False)

        self.unnorm_key = config.unnorm_key
        if (
            self.unnorm_key not in self.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.norm_stats, (
            f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        )

        if add_value_head:
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(
                input_dim=self.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        if self.add_q_head:
            self.q_head = MultiQHead(
                hidden_size=self.hidden_size,
                action_feature_dim=self.flat_action_dim,
                hidden_dims=[512, 256],
                num_q_heads=getattr(config, "num_q_heads", 2),
            )

        self.max_prompt_length = max_prompt_length

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        n_patch_tokens = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm label & mask & embedding
        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * self.num_action_chunks :] = (
            True  # [B, L + act + 1], [many x 0; act x 1; 0]
        )

        input_embeddings = self.get_input_embeddings()(input_ids)  # [B, L + act + 1, D]
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        # vision
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, None, use_film=False
        )
        # [B, 256 * num_images, D]
        assert projected_patch_embeddings.shape[1] == n_patch_tokens

        # multimodal embeddings
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        )
        assert (
            multimodal_embeddings.shape[1]
            == input_embeddings.shape[1] + projected_patch_embeddings.shape[1]
        )
        assert (
            multimodal_attention_mask.shape[1]
            == attention_mask.shape[1] + projected_patch_embeddings.shape[1]
        )

        return multimodal_embeddings, multimodal_attention_mask

    def _get_action_stats(self) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], self.action_dim * self.num_action_chunks))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones(
                (
                    attention_mask.shape[0],
                    input_ids.shape[-1] - attention_mask.shape[-1],
                )
            )
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        action_high = action_high.repeat(repeat_factor)
        action_low = action_low.repeat(repeat_factor)
        mask = mask * repeat_factor

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

        return actions

    def _format_env_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        assert images.ndim == 5, f"Expected image tensor with 5 dims, got {images.shape}"
        if images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3)
        return images

    def _build_pixel_values_from_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype
        if "pixel_values" in obs:
            return obs["pixel_values"].to(device=device, dtype=precision)

        main_images = self._format_env_images(obs["main_images"])
        all_images = [main_images]
        if (
            self.vision_backbone.get_num_images_in_input() > 1
            and obs.get("wrist_images", None) is not None
        ):
            wrist_images = self._format_env_images(obs["wrist_images"])
            all_images.extend(
                [wrist_images[:, i : i + 1] for i in range(wrist_images.shape[1])]
            )

        pixel_values = self.input_processor.image_processor(
            all_images[0], return_tensors="pt"
        )["pixel_values"]
        if len(all_images) > 1:
            extra_pixel_values = [
                self.input_processor.image_processor(image, return_tensors="pt")[
                    "pixel_values"
                ]
                for image in all_images[1:]
            ]
            pixel_values = torch.cat([pixel_values] + extra_pixel_values, dim=1)
        return pixel_values.to(device=device, dtype=precision)

    def _prepare_policy_inputs(
        self, obs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if "input_ids" in obs and "attention_mask" in obs:
            input_ids = obs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = obs["attention_mask"].to(device=device)
            pixel_values = self._build_pixel_values_from_obs(obs)
            return input_ids, attention_mask, pixel_values

        task_descriptions = [
            f"In: What action should the robot take to {t.lower()}?\nOut: "
            for t in obs["task_descriptions"]
        ]
        main_images = self._format_env_images(obs["main_images"])
        inputs = self.input_processor(
            text=task_descriptions,
            images={"images": main_images},
            proprio_states=obs["states"],
            padding="max_length",
            max_length=self.max_prompt_length,
        )
        pixel_values = inputs["pixel_values"]
        if (
            self.vision_backbone.get_num_images_in_input() > 1
            and obs.get("wrist_images", None) is not None
        ):
            wrist_images = self._format_env_images(obs["wrist_images"])
            extra_pixel_values = []
            for idx in range(wrist_images.shape[1]):
                wrist_inputs = self.input_processor(
                    text=task_descriptions,
                    images={"images": wrist_images[:, idx : idx + 1]},
                    proprio_states=obs["states"],
                    padding="max_length",
                    max_length=self.max_prompt_length,
                )
                extra_pixel_values.append(wrist_inputs["pixel_values"])
            if extra_pixel_values:
                pixel_values = torch.cat([pixel_values] + extra_pixel_values, dim=1)

        return (
            inputs["input_ids"].to(device=device, dtype=torch.long),
            inputs["attention_mask"].to(device=device, dtype=torch.bool),
            pixel_values.to(device=device, dtype=next(self.parameters()).dtype),
        )

    def _encode_policy_feature(
        self, obs: dict[str, Any], detach_encoder: bool = False
    ) -> torch.Tensor:
        input_ids, attention_mask, pixel_values = self._prepare_policy_inputs(obs)

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask.to(torch.long)
        )
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_states = outputs.hidden_states[-1]
        action_hidden_states = last_hidden_states[:, -self.flat_action_dim - 1 : -1]
        pooled_features = action_hidden_states.mean(dim=1).to(dtype=torch.float32)
        if detach_encoder:
            pooled_features = pooled_features.detach()
        return pooled_features

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_norm_stats = self._get_action_stats()

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        flat_actions = actions.reshape(actions.shape[0], -1).to(dtype=torch.float32)
        repeat_factor = flat_actions.shape[-1] // action_high.shape[0]
        action_high_t = torch.as_tensor(
            np.tile(action_high, repeat_factor),
            device=flat_actions.device,
            dtype=flat_actions.dtype,
        )
        action_low_t = torch.as_tensor(
            np.tile(action_low, repeat_factor),
            device=flat_actions.device,
            dtype=flat_actions.dtype,
        )
        mask_t = torch.as_tensor(
            np.tile(mask, repeat_factor),
            device=flat_actions.device,
            dtype=torch.bool,
        )

        normalized = flat_actions.clone()
        denom = torch.clamp(action_high_t - action_low_t, min=1.0e-8)
        normalized_masked = 2.0 * (flat_actions - action_low_t) / denom - 1.0
        normalized[..., mask_t] = normalized_masked[..., mask_t]
        return normalized.clamp(-1.0, 1.0)

    def _continuous_actions_to_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        normalized_actions = self._normalize_actions(actions)
        bin_centers = torch.as_tensor(
            self.bin_centers,
            device=normalized_actions.device,
            dtype=normalized_actions.dtype,
        )
        distances = torch.abs(
            normalized_actions.unsqueeze(-1) - bin_centers.view(1, 1, -1)
        )
        discretized = distances.argmin(dim=-1)
        action_tokens = self.vocab_size - (discretized + 1)
        return action_tokens.to(dtype=torch.long)

    def _build_forward_inputs_from_obs_and_actions(
        self, obs: dict[str, Any], actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        input_ids, attention_mask, pixel_values = self._prepare_policy_inputs(obs)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action_tokens": self._continuous_actions_to_tokens(actions),
        }

    def iql_actor_forward(self, **kwargs) -> torch.Tensor:
        obs = kwargs.get("observations", kwargs.get("obs"))
        actions = kwargs.get("actions")
        if obs is None or actions is None:
            raise ValueError("IQL actor forward expects observations and actions.")

        forward_inputs = self._build_forward_inputs_from_obs_and_actions(obs, actions)
        outputs = self.default_forward(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_values=False,
            temperature=float(kwargs.get("temperature", 1.0)),
            top_k=int(kwargs.get("top_k", self.config.n_action_bins)),
        )
        return outputs["logprobs"].sum(dim=-1)

    def iql_value_forward(self, **kwargs) -> torch.Tensor:
        if not hasattr(self, "value_head"):
            raise RuntimeError("IQL value forward requires add_value_head=True.")
        obs = kwargs.get("observations", kwargs.get("obs"))
        if obs is None:
            raise ValueError("IQL value forward expects observations.")
        values = self.value_head(
            self._encode_policy_feature(
                obs, detach_encoder=bool(kwargs.get("detach_encoder", False))
            )
        )
        if values.ndim > 1 and values.shape[-1] != 1:
            values = values.mean(dim=-1, keepdim=True)
        return values.squeeze(-1)

    def iql_critic_forward(self, **kwargs) -> torch.Tensor:
        if not hasattr(self, "q_head"):
            raise RuntimeError("IQL critic forward requires add_q_head=True.")
        obs = kwargs.get("observations", kwargs.get("obs"))
        actions = kwargs.get("actions")
        if obs is None or actions is None:
            raise ValueError("IQL critic forward expects observations and actions.")
        detach_encoder = bool(kwargs.get("detach_encoder", True))
        state_features = self._encode_policy_feature(obs, detach_encoder=detach_encoder)
        action_features = actions.reshape(actions.shape[0], -1).to(dtype=torch.float32)
        return self.q_head(state_features, action_features)

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")

        if env_obs is not None:
            task_descriptions = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in env_obs["task_descriptions"]
            ]
            if env_obs["main_images"].ndim == 4:
                env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
            assert env_obs["main_images"].ndim == 5

            all_images = [
                env_obs["main_images"].permute(0, 1, 4, 2, 3)
            ]  # [B, 1, H, W, C] -> [B, 1, C, H, W]
            if self.vision_backbone.get_num_images_in_input() > 1:
                if env_obs["wrist_images"].ndim == 4:
                    env_obs["wrist_images"] = env_obs["wrist_images"].unsqueeze(1)
                assert env_obs["wrist_images"].ndim == 5
                wrist_imgs = env_obs["wrist_images"].permute(
                    0, 1, 4, 2, 3
                )  # [B, N_IMG, H, W, C] -> [B, N_IMG, C, H, W]
                all_images.extend(
                    [wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])]
                )

            max_length = self.max_prompt_length
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype

            primary_image = all_images.pop(0)
            images = {"images": primary_image}
            inputs = self.input_processor(
                text=task_descriptions,
                images=images,
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=max_length,
            )

            if all_images:
                all_wrist_inputs = [
                    self.input_processor(
                        text=task_descriptions,
                        images={"images": wrist_image.unsqueeze(1)},
                        proprio_states=env_obs["states"],
                        padding="max_length",
                        max_length=max_length,
                    )
                    for wrist_image in all_images
                ]

                # Concatenate all images
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [
                    wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
                ]
                inputs["pixel_values"] = torch.cat(
                    [primary_pixel_values] + all_wrist_pixel_values, dim=1
                )

            input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = inputs["attention_mask"].to(
                device=device, dtype=torch.bool
            )
            pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)

            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(B, N * C, H, W)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        # assert first token is 1
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        n_prompt_tokens = input_ids.shape[-1] - 1
        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            attention_mask[:, -1 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]

        # multimodal
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = outputs.hidden_states[-1]  # (B, seq_len, D)
        assert last_hidden_states.shape[1] == mm_embeddings.shape[1]

        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + self.action_dim * self.num_action_chunks,
            :,
        ]  # [B, act, vocab_size + 64]

        last_hidden_states = last_hidden_states[
            :, -self.action_dim * self.num_action_chunks - 1 : -1
        ]

        logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits_tensor = logits_tensor / kwargs["temperature"]
            top_k = min(
                kwargs["top_k"], processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)
            processed_logprob_tensor = F.log_softmax(
                processed_logits_tensor, dim=-1
            )  # [B, act, vocab_size + 64]

            probs_tensor = torch.exp(
                processed_logprob_tensor
            )  # [B, act, vocab_size + 64]
            probs_flat = probs_tensor.view(
                -1, processed_logprob_tensor.shape[-1]
            )  # [B * act, vocab_size + 64]

            sample_flat = torch.multinomial(
                probs_flat, num_samples=1, replacement=True
            )  # [B * act, 1]
            idxs = sample_flat.view(
                processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
            )  # [B, act]
        else:
            processed_logits_tensor = logits_tensor
            idxs = processed_logits_tensor.argmax(dim=-1)  # [B, act]

        # assert torch.all(idxs >= 0) and torch.all(idxs < self.config.n_action_bins)
        # generated_ids = idxs + (self.vocab_size - self.config.n_action_bins)
        assert torch.all(
            idxs >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(idxs < self.vocab_size)

        chunk_action_tokens = idxs.reshape(-1, self.action_dim)
        predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        # normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )  # [B, dim]
        normalized_actions = normalized_actions.reshape(-1, self.action_dim)

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits_tensor
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf

        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * self.num_action_chunks
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = torch.as_tensor(
            actions.reshape(-1, self.num_action_chunks, self.action_dim)
        )
        chunk_action_tokens = idxs.reshape(-1, self.num_action_chunks, self.action_dim)

        forward_inputs["action_tokens"] = chunk_action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }

        return chunk_actions, result

    def preprocess_for_train(self, data):
        # action-token: [bsz, chunk-step, action-dim] -> [bsz, chunk-step x action-dim]
        for key in ["action_tokens"]:
            value = data[key]
            data[key] = value.reshape(
                value.shape[0],
                self.action_dim * self.num_action_chunks,
                *value.shape[3:],
            )
        return data

    def setup_config_and_processor(self, model_config, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0

        self.input_processor = input_processor

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.IQL_ACTOR:
            return self.iql_actor_forward(**kwargs)
        if forward_type == ForwardType.IQL_CRITIC:
            return self.iql_critic_forward(**kwargs)
        if forward_type == ForwardType.IQL_VALUE:
            return self.iql_value_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: bool = False,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if forward_inputs is not None:
            forward_inputs = self.preprocess_for_train(forward_inputs)
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]

            action_tokens = forward_inputs["action_tokens"]

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)
        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            input_ids[:, -self.action_dim * self.num_action_chunks - 2] == 29871
        )
        assert torch.all(
            attention_mask[:, -2 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]

        # multimodal
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        if compute_values:
            output_hidden_states = True

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        if compute_logprobs:
            logits = outputs.logits[
                :, -self.action_dim * self.num_action_chunks - 1 : -1
            ]  # [B, action-dim, vocab-size]

            processed_logits_tensor = logits / kwargs["temperature"]
            top_k = min(
                kwargs["top_k"], processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor
            action_logits[
                ..., : self.vocab_size - self.config.n_action_bins
            ] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(
                logits=action_logits, target=action_tokens
            )

            entropy = None
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[
                :, -self.action_dim * self.num_action_chunks - 1
            ]  # [batch_size, hidden_dim]
            values = self.value_head(hidden_features)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result
