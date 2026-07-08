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

"""Adaptive-prediction GATE policy.

A small 3-way categorical controller over {SKIP, LATENT, FULL} sitting on top of a
FROZEN fast-wam world-action model (wrapped by `fastwam.adaptive_gate.WAMModeAdapter`).

Design (mirrors MLPPolicy's BasePolicy contract so RLinf's FSDP actor + HF rollout
workers drive it unchanged):
- The policy-gradient ACTION is the discrete MODE (one decision per chunk-step).
  Therefore `action_dim = 1`, `num_action_chunks = 1` (set these in the config too,
  so RLinf's logprob/advantage reshaping in algorithms/utils.py stays consistent).
- The ROBOT action chunk that the simulator executes is produced by the frozen WAM
  for the chosen mode and returned as `chunk_actions`; it is NOT trained (the WAM is
  frozen; only the gate trains).
- `world_feat` (the cheap current-context latent) + proprio form the gate input,
  computed in ALL modes BEFORE the mode-specific compute.

Same module is used for joint and idm backbones (decided by the injected adapter's
`backbone_kind`) and identically in training (default_forward) and rollout
(predict_action_batch) — one interface, per decision #2.

The gate trains WITHOUT supervision by default: pure GRPO on success - lambda*cost,
stabilized label-free by the entropy bonus, the lambda warmup, the near-uniform
logits init, and `explore_eps` (uniform-mixture rollout sampling with exact
behavior logprobs). The BC warm-start + KL-to-BC prior (`mode_logits`,
`attach_bc_prior`, `kl_to_prior`) are an OPTIONAL accelerator/ablation on
self-generated oracle labels — everything about them defaults to off, and no
training path requires them.

# TODO(shape-reconcile): `chunk_actions` (robot, [B, Ta, A_robot]) differs from the
#   trained action (mode, [B, 1]). RLinf's buffer/advantage path is validated for
#   policies where these coincide; confirm on-server that the discrete-mode logprob
#   path ([B,1], num_action_chunks=1) flows through the GRPO actor unchanged
#   (STEP-0 open question #8). If not, register a small categorical policy-loss.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init
from rlinf.models.embodiment.modules.value_head import ValueHead


class _GatePriorHead(nn.Module):
    """Frozen (backbone, logits_head) clone used as the KL-to-BC prior."""

    def __init__(self, backbone: nn.Module, logits_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.logits_head = logits_head

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        return self.logits_head(self.backbone(gate_input))


class GatePolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        world_feat_dim: int,
        proprio_dim: int,
        *,
        num_modes: int = 3,
        hidden_sizes: tuple[int, ...] = (256, 256),
        add_value_head: bool = True,
        use_last_action: bool = False,
        activation: str = "tanh",
        explore_eps: float = 0.0,
        wam_adapter=None,
        obs_preprocessor=None,
    ):
        super().__init__()
        self.world_feat_dim = int(world_feat_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_modes = int(num_modes)
        self.use_last_action = bool(use_last_action)
        # Label-free collapse prevention: during TRAIN rollouts sample from the
        # mixture (1-eps)*pi + eps*Uniform(modes) and record the MIXTURE logprob
        # as the behavior logprob, so the PPO/GRPO ratio pi_theta/mu stays exact.
        # Guarantees every GRPO group keeps contrast across modes even if pi
        # momentarily collapses — no supervision/warm-start needed. Anneal via
        # `set_explore_eps` (runner hook) or keep a small constant.
        self.explore_eps = float(explore_eps)
        # Discrete mode is the policy-gradient action (one per chunk-step).
        self.action_dim = 1
        self.num_action_chunks = 1

        # gate input = [world_feat, proprio, (optional) last-mode one-hot]
        # TODO(phase/last-action, decision optional): also allow a phase scalar.
        self.gate_input_dim = self.world_feat_dim + self.proprio_dim + (
            self.num_modes if self.use_last_action else 0
        )

        act = get_act_func(activation)
        layers, d = [], self.gate_input_dim
        for h in hidden_sizes:
            layers += [layer_init(nn.Linear(d, h)), act()]
            d = h
        self.backbone = nn.Sequential(*layers)
        # small init on the logits head -> near-uniform mode prior at start
        # (helps avoid early all-SKIP / all-FULL collapse before entropy/BC kick in).
        self.logits_head = layer_init(nn.Linear(d, self.num_modes), std=0.01)

        self.add_value_head = bool(add_value_head)
        if self.add_value_head:
            self.value_head = ValueHead(
                self.gate_input_dim, hidden_sizes=(256, 256), activation=activation
            )

        # Frozen WAM wrapper (set here or after construction by the factory).
        self.wam_adapter = wam_adapter
        # Hook to turn raw env_obs into fastwam inputs (image/proprio/context);
        # suite-specific, injected by the env wiring. If None, env_obs is expected
        # to already carry {input_image, proprio, context, context_mask}.
        self.obs_preprocessor = obs_preprocessor
        self.cuda_graph_manager = None
        # Optional FROZEN BC prior (M3). Held in a plain list so it is NOT a
        # registered submodule: state_dict()/FSDP wrapping/`runner.ckpt_path`
        # strict loads stay exactly the BC-compatible gate keys.
        self._bc_prior_container: list[nn.Module] = []

    # ----- helpers ------------------------------------------------------- #
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _logits(self, gate_input: torch.Tensor) -> torch.Tensor:
        return self.logits_head(self.backbone(gate_input))

    def _build_gate_input(self, world_feat: torch.Tensor, proprio: torch.Tensor,
                          last_mode_onehot: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [world_feat, proprio]
        if self.use_last_action:
            if last_mode_onehot is None:
                last_mode_onehot = torch.zeros(
                    world_feat.shape[0], self.num_modes, device=world_feat.device, dtype=world_feat.dtype
                )
            parts.append(last_mode_onehot)
        return torch.cat(parts, dim=-1)

    def mode_logits(
        self,
        world_feat: torch.Tensor,
        proprio: torch.Tensor,
        last_mode_onehot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Categorical mode logits from gate inputs [B, num_modes].

        The single logits path shared by BC/SFT training (`gate_policy.bc`) and
        the RL rollout, so warm-start and RL are architecturally identical.
        """
        return self._logits(self._build_gate_input(world_feat, proprio, last_mode_onehot))

    def set_explore_eps(self, eps: float) -> None:
        """Runner/rollout hook to anneal the uniform-mixture exploration rate."""
        eps = float(eps)
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"explore_eps must be in [0, 1], got {eps}")
        self.explore_eps = eps

    # ----- KL-to-BC prior (optional; OFF by default) ----------------------- #
    @property
    def bc_prior(self) -> Optional[nn.Module]:
        return self._bc_prior_container[0] if self._bc_prior_container else None

    def attach_bc_prior(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Attach a FROZEN copy of a BC-trained gate as the KL prior.

        `state_dict` is a GatePolicy state dict (e.g. the artifact written by
        `train_gate_bc.py`); only its `backbone.*`/`logits_head.*` keys are used,
        so it also accepts checkpoints trained with a value head. The prior lives
        OUTSIDE the module tree (see `_bc_prior_container`).
        """
        prior = _GatePriorHead(
            copy.deepcopy(self.backbone), copy.deepcopy(self.logits_head)
        )
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k.startswith(("backbone.", "logits_head."))
        }
        if not filtered:
            raise ValueError(
                "BC prior state_dict has no `backbone.*`/`logits_head.*` keys; "
                "expected a GatePolicy checkpoint from train_gate_bc.py."
            )
        prior.load_state_dict(filtered, strict=True)
        prior.eval()
        prior.requires_grad_(False)
        self._bc_prior_container = [prior]

    def _prior_logits(self, gate_input: torch.Tensor) -> torch.Tensor:
        prior = self.bc_prior
        prior_param = next(prior.parameters())
        if prior_param.device != gate_input.device or prior_param.dtype != gate_input.dtype:
            # not a registered submodule -> move lazily (kept for later calls)
            prior.to(device=gate_input.device, dtype=gate_input.dtype)
        return prior(gate_input)

    def preprocess_env_obs(self, env_obs):
        """Turn raw env_obs into the fastwam inputs the WAM consumes.

        Returns a dict with batched: input_image [B,3,H,W], proprio [B,P],
        context [B,L,D], context_mask [B,L]. Delegates to the injected
        `obs_preprocessor` (suite-specific, wired by the env). If absent, expects
        env_obs to already contain these keys (passthrough; used by unit tests).
        """
        if self.obs_preprocessor is not None:
            return self.obs_preprocessor(env_obs)
        keys = ("input_image", "proprio", "context", "context_mask")
        if all(k in env_obs for k in keys):
            return {k: env_obs[k] for k in keys}
        raise NotImplementedError(
            "GatePolicy needs an `obs_preprocessor` (wired by the env) to map raw "
            "env_obs {main_images, wrist_images, states, task_descriptions} to "
            "{input_image, proprio, context, context_mask}."
        )

    # ----- training forward (replay) ------------------------------------- #
    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        gate_input = forward_inputs["gate_input"]
        action = forward_inputs["action"]  # mode index, [B] or [B,1]
        mode_idx = action.reshape(action.shape[0]).long()

        logits = self._logits(gate_input)
        dist = Categorical(logits=logits)

        output_dict = {}
        if compute_logprobs:
            output_dict["logprobs"] = dist.log_prob(mode_idx).unsqueeze(-1)  # [B,1]
        if compute_entropy:
            output_dict["entropy"] = dist.entropy().unsqueeze(-1)  # [B,1]
        if compute_values:
            if not self.add_value_head:
                raise NotImplementedError("value head disabled; set add_value_head=True for GAE/critic.")
            output_dict["values"] = self.value_head(gate_input)  # [B,1]
        return output_dict

    # ----- rollout forward ----------------------------------------------- #
    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        if self.wam_adapter is None:
            raise RuntimeError("GatePolicy.wam_adapter is not set (inject the frozen WAMModeAdapter).")
        device = self._device()
        inputs = self.preprocess_env_obs(env_obs)
        input_image = inputs["input_image"].to(device)
        proprio = inputs["proprio"].to(device)
        context = inputs["context"].to(device)
        context_mask = inputs["context_mask"].to(device)
        batch = input_image.shape[0]

        # world_feat in ALL modes, before mode-specific compute (per-env; fastwam
        # infer is batch-1). # TODO(batch): batch the WAM forward if it ever supports it.
        world_feat = torch.stack(
            [self.wam_adapter.encode_world_feat(input_image[i : i + 1]) for i in range(batch)],
            dim=0,
        ).to(device)

        gate_input = self._build_gate_input(world_feat, proprio)
        logits = self._logits(gate_input)
        dist = Categorical(logits=logits)
        eps = float(self.explore_eps) if mode == "train" else 0.0
        if mode != "train":
            mode_idx = torch.argmax(logits, dim=-1)
            chunk_logprobs = dist.log_prob(mode_idx).unsqueeze(-1)  # [B,1]
        elif eps > 0.0:
            # behavior policy mu = (1-eps)*pi + eps*Uniform; record mu's logprob
            # (NOT pi's) so the actor-side ratio pi_theta/mu is an exact
            # importance weight that PPO/GRPO clipping handles.
            mix_probs = (1.0 - eps) * dist.probs + eps / float(self.num_modes)
            mix_dist = Categorical(probs=mix_probs)
            mode_idx = mix_dist.sample()
            chunk_logprobs = mix_dist.log_prob(mode_idx).unsqueeze(-1)  # [B,1]
        else:
            mode_idx = dist.sample()
            chunk_logprobs = dist.log_prob(mode_idx).unsqueeze(-1)  # [B,1]
        chunk_values = (
            self.value_head(gate_input)
            if (self.add_value_head and calculate_values)
            else torch.zeros(batch, 1, device=device)
        )

        # Run the frozen WAM in the chosen mode per env -> robot action chunk + cost.
        robot_chunks, costs = [], []
        for i in range(batch):
            out = self.wam_adapter.act(
                input_image=input_image[i : i + 1],
                mode=int(mode_idx[i].item()),
                proprio=proprio[i : i + 1] if proprio.ndim == 2 else None,
                context=context[i : i + 1],
                context_mask=context_mask[i : i + 1],
                world_feat=world_feat[i],
            )
            robot_chunks.append(out["action_chunk"])
            costs.append(out["cost"])
        chunk_actions = torch.stack([c.to(device) for c in robot_chunks], dim=0)  # [B, Ta, A_robot]
        if hasattr(self.obs_preprocessor, "denormalize_actions"):
            chunk_actions = self.obs_preprocessor.denormalize_actions(chunk_actions)
        mode_cost = torch.tensor(costs, device=device, dtype=torch.float32).unsqueeze(-1)  # [B,1]

        forward_inputs = {
            "gate_input": gate_input,
            "action": mode_idx.long().unsqueeze(-1),  # trained action = mode, [B,1]
            # carried in the buffer for the reward hook (-lambda*cost) and logging:
            "mode": mode_idx.long().unsqueeze(-1),    # [B,1]
            "mode_cost": mode_cost,                    # [B,1], cost(FULL)=1
        }
        if return_obs:
            forward_inputs["world_feat"] = world_feat
            forward_inputs["proprio"] = proprio

        # M3: exact per-decision KL(pi || pi_BC), carried in the buffer for the
        # reward hook (-beta * KL keeps early RL near the BC warm-start).
        kl_to_prior = None
        if self.bc_prior is not None:
            prior_dist = Categorical(logits=self._prior_logits(gate_input))
            kl_to_prior = kl_divergence(dist, prior_dist).unsqueeze(-1)  # [B,1]
            forward_inputs["kl_to_prior"] = kl_to_prior

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
            # convenience top-level mirrors for mode-usage logging:
            "mode": mode_idx.long(),
            "mode_cost": mode_cost.squeeze(-1),
            "mode_logits": logits.detach(),
        }
        if kl_to_prior is not None:
            result["kl_to_prior"] = kl_to_prior.squeeze(-1)
        return chunk_actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"GatePolicy does not support forward_type={forward_type}")
