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

"""Adaptive-prediction gate over UNCOND and full IDM inference.

The policy-gradient action is one categorical mode per executed robot chunk.  The
frozen WAM produces the robot action chunk; only this small gate is optimized.

Design (mirrors MLPPolicy's BasePolicy contract so RLinf's FSDP actor + HF rollout
workers drive it unchanged):
- The policy-gradient ACTION is the discrete MODE (one decision per chunk-step).
  Therefore `action_dim = 1`, `num_action_chunks = 1` (set these in the config too,
  so RLinf's logprob/advantage reshaping in algorithms/utils.py stays consistent).
- The ROBOT action chunk that the simulator executes is produced by the frozen WAM
  for the chosen mode and returned as `chunk_actions`; it is NOT trained (the WAM is
  frozen; only the gate trains).
- Spatial world-latent statistics, proprio, and a fixed-size pooled task-text
  feature form the gate input.  The encoded first-frame state is reused by the WAM,
  so routing does not introduce a second VAE encode.

The same module is used for replay training and rollout; the WAM adapter is IDM-only
and exposes UNCOND=0 / IDM=1 through one interface.

An optional frozen BC prior is regularized with a differentiable actor-side
``KL(pi || pi_BC)``.  It is deliberately not injected into the environment reward:
a detached state-wise exact KL is action-independent and therefore is not a valid
score-function reward penalty.
"""
from __future__ import annotations

import copy
import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from rlinf.models.embodiment.gate_policy.control_eval import (
    configured_control_kind,
)
from rlinf.models.embodiment.gate_policy.mode_selectors import (
    ForcedModeSelector,
    ModeSelection,
    build_eval_mode_selector,
    normalize_gate_context,
)
from rlinf.models.embodiment.gate_policy.obs_preprocessor import pool_text_context

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
        text_feat_dim: int = 64,
        num_modes: int = 2,
        hidden_sizes: tuple[int, ...] = (256, 256),
        add_value_head: bool = True,
        activation: str = "tanh",
        explore_eps: float = 0.0,
        force_mode: Optional[int] = None,
        eval_policy=None,
        eval_control=None,
        eval_control_runtime=None,
        allow_legacy_gate_checkpoint: bool = False,
        kl_prior_beta: float = 0.0,
        kl_prior_beta_end: float = 0.0,
        kl_prior_decay_steps: int = 0,
        wam_adapter=None,
        obs_preprocessor=None,
    ):
        super().__init__()
        self.world_feat_dim = int(world_feat_dim)
        self.proprio_dim = int(proprio_dim)
        self.text_feat_dim = int(text_feat_dim)
        self.num_modes = int(num_modes)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.activation = str(activation)
        if self.num_modes != 2:
            raise ValueError(
                "GatePolicy supports exactly two modes: UNCOND=0 and IDM=1; "
                f"got num_modes={self.num_modes}."
            )
        if min(self.world_feat_dim, self.proprio_dim, self.text_feat_dim) < 0:
            raise ValueError("gate feature dimensions must be non-negative")
        # Label-free collapse prevention: during training the optimized policy is
        # mu_theta=(1-eps)*pi_theta+eps*Uniform. Rollout stores eps with every
        # sample, so actor replay recomputes the same mu_theta and the pre-update
        # ratio is exactly one. This also keeps both modes reachable if pi collapses.
        # It does not guarantee that every finite GRPO group contains both modes.
        # Anneal via `set_explore_eps` (runner hook) or keep a small constant.
        self.explore_eps = float(explore_eps)
        if not 0.0 <= self.explore_eps <= 1.0:
            raise ValueError(f"explore_eps must be in [0, 1], got {self.explore_eps}")
        if force_mode is None:
            self.force_mode = None
        elif isinstance(force_mode, str) and force_mode in {"0", "1"}:
            self.force_mode = int(force_mode)
        elif isinstance(force_mode, numbers.Integral) and not isinstance(force_mode, bool):
            self.force_mode = int(force_mode)
        else:
            raise ValueError("force_mode must be null, 0 (UNCOND), or 1 (IDM)")
        if self.force_mode not in (None, 0, 1):
            raise ValueError("force_mode must be null, 0 (UNCOND), or 1 (IDM)")
        self.eval_mode_selector = build_eval_mode_selector(eval_policy)
        self.eval_control_kind = configured_control_kind(eval_control)
        self.eval_control_runtime = eval_control_runtime
        if self.eval_control_kind is not None:
            if self.force_mode is not None:
                raise ValueError(
                    "gate.force_mode cannot be combined with gate.eval_control"
                )
            if self.eval_mode_selector.kind != "learned":
                raise ValueError(
                    "gate.eval_control owns the evaluation intervention and cannot "
                    "be combined with a non-learned gate.eval_policy"
                )
            if (
                self.eval_control_runtime is not None
                and self.eval_control_runtime.kind != self.eval_control_kind
            ):
                raise ValueError(
                    "configured eval-control kind does not match its runtime: "
                    f"{self.eval_control_kind!r} vs "
                    f"{self.eval_control_runtime.kind!r}"
                )
        if self.force_mode is not None:
            if self.eval_mode_selector.kind != "learned":
                configured_mode = getattr(self.eval_mode_selector, "mode", None)
                if not (
                    self.eval_mode_selector.kind == "forced"
                    and configured_mode == self.force_mode
                ):
                    raise ValueError(
                        "legacy gate.force_mode conflicts with gate.eval_policy; "
                        "prefer eval_policy.kind=forced"
                    )
            self.eval_mode_selector = ForcedModeSelector(
                mode=self.force_mode,
                max_decisions=self.eval_mode_selector.max_decisions,
                seed=self.eval_mode_selector.seed,
            )
        self.eval_trace_path: Optional[str] = None
        self.wam_checkpoint_sha256: Optional[str] = None
        self.allow_legacy_gate_checkpoint = bool(allow_legacy_gate_checkpoint)
        # Discrete mode is the policy-gradient action (one per chunk-step).
        self.action_dim = 1
        self.num_action_chunks = 1

        # A last-mode feature is intentionally absent: implementing it correctly
        # requires reset-aware per-environment recurrent state.
        self.gate_input_dim = (
            self.world_feat_dim + self.proprio_dim + self.text_feat_dim
        )

        act = get_act_func(self.activation)
        layers, d = [], self.gate_input_dim
        for h in self.hidden_sizes:
            layers += [layer_init(nn.Linear(d, h)), act()]
            d = h
        self.backbone = nn.Sequential(*layers)
        # small init on the logits head -> near-uniform mode prior at start
        # (helps avoid early all-UNCOND / all-IDM collapse before entropy kicks in).
        self.logits_head = layer_init(nn.Linear(d, self.num_modes), std=0.01)

        self.add_value_head = bool(add_value_head)
        if self.add_value_head:
            self.value_head = ValueHead(
                self.gate_input_dim, hidden_sizes=(256, 256), activation=self.activation
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
        # Populated by the factory from the WAM task/cost profile. BC loading
        # requires it so a same-shaped checkpoint from another task/WAM cannot
        # be silently attached.
        self.bc_expected_provenance: Optional[dict] = None
        self.kl_prior_beta = float(kl_prior_beta)
        self.kl_prior_beta_end = float(kl_prior_beta_end)
        self.kl_prior_decay_steps = int(kl_prior_decay_steps)
        self.global_step = 0
        if self.kl_prior_beta < 0 or self.kl_prior_beta_end < 0:
            raise ValueError("KL-prior beta values must be non-negative")
        if self.kl_prior_beta_end > self.kl_prior_beta:
            raise ValueError("KL-prior beta_end must not exceed beta")
        if self.kl_prior_decay_steps < 0:
            raise ValueError("KL-prior decay_steps must be non-negative")

    # ----- helpers ------------------------------------------------------- #
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _logits(self, gate_input: torch.Tensor) -> torch.Tensor:
        param = next(self.backbone.parameters(), self.logits_head.weight)
        return self.logits_head(
            self.backbone(gate_input.to(device=param.device, dtype=param.dtype))
        )

    def _values(self, gate_input: torch.Tensor) -> torch.Tensor:
        param = next(self.value_head.parameters())
        return self.value_head(
            gate_input.to(device=param.device, dtype=param.dtype)
        )

    def _policy_distributions(
        self,
        logits: torch.Tensor,
        explore_eps: torch.Tensor | float | None = None,
    ) -> tuple[Categorical, Categorical]:
        """Return base ``pi`` and the actual training policy ``mu`` in FP32."""
        base_dist = Categorical(logits=logits.float())
        if explore_eps is None:
            return base_dist, base_dist
        eps = torch.as_tensor(
            explore_eps, device=logits.device, dtype=torch.float32
        )
        if eps.ndim == 0:
            eps = eps.expand(logits.shape[0])
        eps = eps.reshape(-1, 1)
        if eps.shape[0] != logits.shape[0]:
            raise ValueError(
                "explore_eps must be scalar or have one value per gate sample; "
                f"got {tuple(eps.shape)} for batch {logits.shape[0]}."
            )
        if not bool(torch.isfinite(eps).all()) or bool(((eps < 0) | (eps > 1)).any()):
            raise ValueError("explore_eps values must be finite and in [0, 1]")
        mix_probs = (1.0 - eps) * base_dist.probs + eps / float(self.num_modes)
        return base_dist, Categorical(probs=mix_probs)

    def _build_gate_input(
        self,
        world_feat: torch.Tensor,
        proprio: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        features = (
            ("world_feat", world_feat, self.world_feat_dim),
            ("proprio", proprio, self.proprio_dim),
            ("text_feat", text_feat, self.text_feat_dim),
        )
        batch = int(world_feat.shape[0])
        for name, value, expected_dim in features:
            if value.ndim != 2 or value.shape[0] != batch or value.shape[-1] != expected_dim:
                raise ValueError(
                    f"{name} must have shape [B,{expected_dim}], got {tuple(value.shape)}"
                )
        device = world_feat.device
        return torch.cat(
            [value.to(device=device, dtype=torch.float32) for _, value, _ in features],
            dim=-1,
        )

    def mode_logits(
        self,
        world_feat: torch.Tensor,
        proprio: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Categorical mode logits from gate inputs [B, num_modes].

        The single logits path shared by BC/SFT training (`gate_policy.bc`) and
        the RL rollout, so warm-start and RL are architecturally identical.
        """
        return self._logits(self._build_gate_input(world_feat, proprio, text_feat))

    def set_explore_eps(self, eps: float) -> None:
        """Runner/rollout hook to anneal the uniform-mixture exploration rate."""
        eps = float(eps)
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"explore_eps must be in [0, 1], got {eps}")
        self.explore_eps = eps

    def set_global_step(self, global_step: int) -> None:
        """Set the optimizer/global runner step used by the KL decay schedule."""
        self.global_step = max(int(global_step), 0)

    @property
    def effective_eval_method(self) -> str:
        if self.eval_control_kind is not None:
            return f"control:{self.eval_control_kind}"
        return self.eval_mode_selector.kind

    @property
    def effective_eval_max_decisions(self) -> int:
        return int(self.eval_mode_selector.max_decisions)

    def effective_eval_provenance(self) -> dict:
        if self.eval_control_kind is None:
            return self.eval_mode_selector.provenance()
        if self.eval_control_runtime is None:
            raise RuntimeError(
                "evaluation control is configured but its FastWAM runtime was not "
                "loaded; evaluation requires rollout.model.load_wam=true"
            )
        return {
            **self.eval_control_runtime.provenance(),
            "max_decisions": self.effective_eval_max_decisions,
        }

    def attach_eval_control_runtime(self, runtime) -> None:
        if self.eval_control_kind is None:
            raise ValueError(
                "cannot attach an evaluation-control runtime when "
                "gate.eval_control.kind is null"
            )
        if runtime is None or runtime.kind != self.eval_control_kind:
            actual = None if runtime is None else runtime.kind
            raise ValueError(
                "configured eval-control kind does not match its runtime: "
                f"{self.eval_control_kind!r} vs {actual!r}"
            )
        self.eval_control_runtime = runtime

    def _select_eval_control(
        self,
        logits: torch.Tensor,
        gate_context,
    ) -> ModeSelection:
        runtime = self.eval_control_runtime
        if runtime is None:
            raise RuntimeError(
                "evaluation control is configured but its FastWAM runtime was not "
                "loaded; evaluation requires rollout.model.load_wam=true"
            )
        normalized = normalize_gate_context(
            gate_context,
            batch_size=int(logits.shape[0]),
            device=logits.device,
            max_decisions=self.effective_eval_max_decisions,
            # The runtime validates phase only for active shuffled samples. This
            # lets fixed-horizon reference inference continue after absorption.
            require_phase=False,
        )
        branch_mode = int(runtime.branch_mode)
        modes = torch.full(
            (int(logits.shape[0]),),
            branch_mode,
            device=logits.device,
            dtype=torch.long,
        )
        reserved_modes = torch.full(
            (int(logits.shape[0]), self.effective_eval_max_decisions),
            branch_mode,
            device=logits.device,
            dtype=torch.long,
        )
        return ModeSelection(
            modes=modes,
            method=self.effective_eval_method,
            episode_uids=normalized.episode_uids,
            decision_indices=normalized.decision_indices,
            reserved_modes=reserved_modes,
        )

    def current_kl_beta(self) -> float:
        if self.kl_prior_decay_steps <= 0:
            return self.kl_prior_beta
        frac = min(self.global_step / float(self.kl_prior_decay_steps), 1.0)
        return self.kl_prior_beta + (
            self.kl_prior_beta_end - self.kl_prior_beta
        ) * frac

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

    def load_bc_init(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        """Load BC gate weights, allowing only value-head presence differences."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        bad_missing = [key for key in missing if not key.startswith("value_head.")]
        bad_unexpected = [
            key for key in unexpected if not key.startswith("value_head.")
        ]
        if bad_missing or bad_unexpected:
            raise ValueError(
                "BC checkpoint does not match the gate architecture: "
                f"missing={bad_missing}, unexpected={bad_unexpected}."
            )
        return list(missing), list(unexpected)

    def _prior_logits(self, gate_input: torch.Tensor) -> torch.Tensor:
        prior = self.bc_prior
        if prior is None:
            raise RuntimeError("BC prior is not attached")
        prior_param = next(prior.parameters())
        if prior_param.device != gate_input.device or prior_param.dtype != gate_input.dtype:
            # not a registered submodule -> move lazily (kept for later calls)
            prior.to(device=gate_input.device, dtype=gate_input.dtype)
        return prior(gate_input)

    def preprocess_env_obs(self, env_obs):
        """Turn raw env_obs into the fastwam inputs the WAM consumes.

        Returns batched input_image, proprio, context/context_mask and text_feat.
        Delegates to the injected
        `obs_preprocessor` (suite-specific, wired by the env). If absent, expects
        env_obs to already contain these keys (passthrough; used by unit tests).
        """
        if self.obs_preprocessor is not None:
            return self.obs_preprocessor(env_obs)
        keys = ("input_image", "proprio", "context", "context_mask")
        if all(k in env_obs for k in keys):
            result = {k: env_obs[k] for k in keys}
            result["text_feat"] = env_obs.get(
                "text_feat",
                pool_text_context(
                    result["context"], result["context_mask"], self.text_feat_dim
                ),
            )
            return result
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

        # Categorical math is always FP32, even under the actor's BF16 autocast.
        logits = self._logits(gate_input)
        base_dist, train_dist = self._policy_distributions(
            logits, forward_inputs.get("explore_eps")
        )

        output_dict = {}
        if compute_logprobs:
            output_dict["logprobs"] = train_dist.log_prob(mode_idx).unsqueeze(-1).float()
        if compute_entropy:
            output_dict["entropy"] = train_dist.entropy().unsqueeze(-1).float()
        if compute_values:
            if not self.add_value_head:
                raise NotImplementedError(
                    "value head disabled; set add_value_head=True for GAE/critic."
                )
            output_dict["values"] = self._values(gate_input)  # [B,1]
        beta = self.current_kl_beta()
        if self.bc_prior is not None and beta > 0.0:
            prior_dist = Categorical(logits=self._prior_logits(gate_input).float())
            # The prior regularizes the learnable base gate, not the externally
            # imposed uniform exploration floor.
            kl = kl_divergence(base_dist, prior_dist).unsqueeze(-1).float()
            output_dict["kl_to_prior"] = kl
            output_dict["aux_loss"] = kl * beta
            output_dict["kl_beta"] = torch.tensor(
                beta, device=kl.device, dtype=torch.float32
            )
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
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be `train` or `eval`, got {mode!r}")
        if self.eval_control_kind is not None and mode == "train":
            raise RuntimeError(
                "gate.eval_control is evaluation-only and cannot be used in a "
                "training rollout"
            )
        if self.force_mode is not None and mode == "train":
            raise RuntimeError(
                "force_mode is evaluation-only; training with forced modes "
                "would invalidate the stored behavior logprob."
            )
        if self.wam_adapter is None:
            raise RuntimeError(
                "GatePolicy.wam_adapter is not set (inject the frozen WAMModeAdapter)."
            )
        device = self._device()
        inputs = self.preprocess_env_obs(env_obs)
        input_image = inputs["input_image"].to(device)
        proprio = inputs["proprio"].to(device)
        context = inputs["context"].to(device)
        context_mask = inputs["context_mask"].to(device)
        text_feat = inputs["text_feat"].to(device=device, dtype=torch.float32)
        batch = input_image.shape[0]

        # FastWAM inference is currently batch-1. Keep each encoded state so the
        # selected branch reuses its first-frame latents instead of encoding twice.
        encoded_states = [
            self.wam_adapter.encode_world_state(input_image[i : i + 1])
            for i in range(batch)
        ]
        world_feat = torch.stack(
            [state.world_feat for state in encoded_states],
            dim=0,
        ).to(device=device, dtype=torch.float32)

        gate_input = self._build_gate_input(world_feat, proprio, text_feat)
        logits = self._logits(gate_input)
        eps = float(self.explore_eps) if mode == "train" else 0.0
        base_dist, policy_dist = self._policy_distributions(logits, eps)
        selection = None
        if mode == "eval":
            gate_context = (
                env_obs.get("gate_context")
                if isinstance(env_obs, dict)
                else None
            )
            selection = (
                self._select_eval_control(logits, gate_context)
                if self.eval_control_kind is not None
                else self.eval_mode_selector.select(logits, gate_context)
            )
            mode_idx = selection.modes
            chunk_logprobs = base_dist.log_prob(mode_idx).unsqueeze(-1).float()
        else:
            mode_idx = policy_dist.sample()
            chunk_logprobs = policy_dist.log_prob(mode_idx).unsqueeze(-1).float()
        chunk_values = (
            self._values(gate_input)
            if (self.add_value_head and calculate_values)
            else torch.zeros(batch, 1, device=device)
        )

        # Run the frozen WAM in the chosen mode per env -> robot action chunk + cost.
        robot_chunks, costs, control_artifacts = [], [], []
        for i in range(batch):
            common = {
                "input_image": input_image[i : i + 1],
                "proprio": proprio[i : i + 1] if proprio.ndim == 2 else None,
                "context": context[i : i + 1],
                "context_mask": context_mask[i : i + 1],
                "encoded_state": encoded_states[i],
            }
            if self.eval_control_kind is not None:
                out = self.eval_control_runtime.act(
                    self.wam_adapter,
                    **common,
                    gate_context=gate_context,
                    batch_index=i,
                )
            else:
                out = self.wam_adapter.act(
                    mode=int(mode_idx[i].item()),
                    **common,
                )
            robot_chunks.append(out["action_chunk"])
            costs.append(out["cost"])
            control_artifacts.append(
                dict(out.get("aux", {})).get("donor_artifact")
            )
        # FastWAM returns decoded robot chunks on CPU. Keep them there: action
        # denormalization/gripper conversion is CPU work and the rollout channel
        # ultimately sends CPU tensors to the environment. Only gate features and
        # policy logits need to reside on the accelerator.
        chunk_actions = torch.stack(
            [c.detach().to(device="cpu", dtype=torch.float32) for c in robot_chunks],
            dim=0,
        )  # [B, Ta, A_robot]
        if hasattr(self.obs_preprocessor, "process_actions"):
            chunk_actions = self.obs_preprocessor.process_actions(chunk_actions)
        elif hasattr(self.obs_preprocessor, "denormalize_actions"):
            chunk_actions = self.obs_preprocessor.denormalize_actions(chunk_actions)
        mode_cost = torch.tensor(costs, device=device, dtype=torch.float32).unsqueeze(-1)  # [B,1]

        forward_inputs = {
            "gate_input": gate_input,
            "action": mode_idx.long().unsqueeze(-1),  # trained action = mode, [B,1]
            # carried in the buffer for the reward hook (-lambda*cost) and logging:
            "mode": mode_idx.long().unsqueeze(-1),    # [B,1]
            "mode_cost": mode_cost,                    # [B,1], cost(IDM)=1
            "mode_entropy": policy_dist.entropy().unsqueeze(-1).float(),
            # Replay must use the exact policy that collected each state/action.
            # Keep this per sample so schedules can change between rollouts.
            "explore_eps": torch.full(
                (batch, 1), eps, device=device, dtype=torch.float32
            ),
        }

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
            # convenience top-level mirrors for mode-usage logging:
            "mode": mode_idx.long(),
            "mode_cost": mode_cost.squeeze(-1),
            "mode_logits": logits.detach().float(),
        }
        if selection is not None:
            result.update(
                {
                    "eval_policy_method": selection.method,
                    "eval_policy_provenance": self.effective_eval_provenance(),
                    "reserved_modes": (
                        None
                        if selection.reserved_modes is None
                        else selection.reserved_modes.detach().cpu()
                    ),
                    "schedule_decision_index": (
                        None
                        if selection.decision_indices is None
                        else selection.decision_indices.detach().cpu()
                    ),
                }
            )
        if self.eval_control_kind is not None:
            result["control_artifacts"] = control_artifacts
        return chunk_actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"GatePolicy does not support forward_type={forward_type}")
