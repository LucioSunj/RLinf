# Copyright 2026 The RLinf Authors.
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

import os
from typing import Any, Optional

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    compute_observation_stats,
    replay_buffer_collate_fn,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_iql_policy_worker import (
    EmbodiedIQLFSDPPolicy,
    iql_expectile_loss,
)


class FSDPOfflineIQLPolicy(EmbodiedIQLFSDPPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._model_type = str(self.cfg.actor.model.model_type)
        self._trajectory_batch_mode = False
        self._shared_iql_backbone = False
        self.replay_buffer = None
        self.observation_stats = None

    def build_offline_dataloader(self) -> None:
        dataset_cfg = self.cfg.get("data", {})
        dataset_type = str(dataset_cfg.get("dataset_type", "trajectory_replay")).lower()
        if dataset_type == "d4rl":
            self._trajectory_batch_mode = False
            return super().build_offline_dataloader()

        if dataset_type not in {"trajectory_replay", "trajectory", "embodied_replay"}:
            raise AssertionError(
                "offline_rl only supports data.dataset_type in "
                "{'trajectory_replay', 'trajectory', 'embodied_replay', 'd4rl'}; "
                f"got {dataset_type!r}."
            )

        self._trajectory_batch_mode = True
        replay_cfg = self.cfg.algorithm.get("replay_buffer", {})
        dataset_cls = (
            PreloadReplayBufferDataset
            if replay_cfg.get("enable_preload", False)
            else ReplayBufferDataset
        )

        self._offline_dataset = dataset_cls.from_offline_path(
            self.cfg.data.offline_dataset_path,
            batch_size=int(self.batch_size),
            min_replay_buffer_size=0,
            min_demo_buffer_size=0,
            prefetch_size=int(replay_cfg.get("prefetch_size", 10)),
            seed=int(self.cfg.actor.get("seed", 1234)),
            enable_cache=bool(replay_cfg.get("enable_cache", True)),
            cache_size=int(replay_cfg.get("cache_size", 32)),
            sample_window_size=int(replay_cfg.get("sample_window_size", 0)),
            trajectory_format=str(replay_cfg.get("trajectory_format", "pt")),
            reward_scale=float(self.cfg.data.get("reward_scale", 1.0)),
            reward_bias=float(self.cfg.data.get("reward_bias", 0.0)),
        )
        self.replay_buffer = self._offline_dataset.replay_buffer
        if self.cfg.data.get("normalize_observations", False):
            self.observation_stats = compute_observation_stats(self.replay_buffer)
            existing_transform = self._offline_dataset.batch_transform

            def _normalized_batch_transform(batch: dict[str, Any]) -> dict[str, Any]:
                if existing_transform is not None:
                    batch = existing_transform(batch)
                from rlinf.data.embodied_buffer_dataset import (
                    apply_observation_normalizer,
                )

                return apply_observation_normalizer(batch, self.observation_stats)

            self._offline_dataset.batch_transform = _normalized_batch_transform

        self.offline_data_loader = DataLoader(
            self._offline_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.offline_data_iter = iter(self.offline_data_loader)
        self._dataset_size = int(self.replay_buffer.total_samples)
        self._data_epoch = 0
        self._data_iter_offset = 0

        sample_batch = self.replay_buffer.sample(1)
        sample_actions = sample_batch["actions"]
        self._action_dim = int(sample_actions.shape[-1])
        if self._model_type == "mlp_policy":
            sample_states = sample_batch["curr_obs"]["states"]
            self._obs_dim = int(sample_states.reshape(sample_states.shape[0], -1).shape[-1])
        self.log_info(
            "offline_rl: replay-buffer dataloader with "
            f"{self._dataset_size} samples, per_rank_batch_size={self.batch_size}."
        )

    def setup_model_and_optimizer(self, initialize_target: bool = True) -> None:
        if self._model_type == "openvla_oft":
            self._setup_openvla_iql_model_and_optimizer(initialize_target)
            return
        super().setup_model_and_optimizer(initialize_target=initialize_target)

    def _setup_openvla_iql_model_and_optimizer(
        self, initialize_target: bool = True
    ) -> None:
        self._shared_iql_backbone = True
        self.offline_torch_compile = False
        model_cfg = OmegaConf.create(
            OmegaConf.to_container(self.cfg.actor.model, resolve=True)
        )
        with open_dict(model_cfg):
            model_cfg.action_dim = int(self.cfg.actor.model.action_dim)
            model_cfg.num_action_chunks = int(self.cfg.actor.model.num_action_chunks)
            model_cfg.add_value_head = True
            model_cfg.add_q_head = True
        module = get_model(model_cfg)
        target_module = None
        if initialize_target:
            target_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg.actor.model, resolve=True)
            )
            with open_dict(target_cfg):
                target_cfg.action_dim = int(self.cfg.actor.model.action_dim)
                target_cfg.num_action_chunks = int(self.cfg.actor.model.num_action_chunks)
                target_cfg.add_value_head = False
                target_cfg.add_q_head = True
            target_module = get_model(target_cfg)

        use_fsdp_wrap = self.cfg.actor.get("use_fsdp_wrap", True)
        if use_fsdp_wrap:
            self.model = self._strategy.wrap_model(
                model=module, device_mesh=self._device_mesh
            )
            if initialize_target:
                self.target_model = self._strategy.wrap_model(
                    model=target_module, device_mesh=self._device_mesh
                )
        else:
            self.model = module
            if initialize_target:
                self.target_model = target_module
        self._use_fsdp_wrap = use_fsdp_wrap
        self.critic_model = self.model
        self.value_model = self.model

        if initialize_target:
            if self._use_fsdp_wrap:
                online_state = self._strategy.get_model_state_dict(
                    self.model,
                    cpu_offload=False,
                    full_state_dict=True,
                )
                target_state = self._strategy.get_model_state_dict(
                    self.target_model,
                    cpu_offload=False,
                    full_state_dict=True,
                )
            else:
                online_state = self.model.state_dict()
                target_state = self.target_model.state_dict()
            filtered_state = {
                key: value
                for key, value in online_state.items()
                if key in target_state
            }
            if self._use_fsdp_wrap:
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    filtered_state,
                    cpu_offload=False,
                    full_state_dict=True,
                )
            else:
                self.target_model.load_state_dict(filtered_state, strict=False)
            self.target_model.eval()
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        actor_params = []
        q_params = []
        value_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "q_head" in name:
                q_params.append(param)
            elif "value_head" in name:
                value_params.append(param)
            else:
                actor_params.append(param)

        if not actor_params:
            raise RuntimeError(
                "openvla_oft offline IQL found no trainable actor/backbone parameters."
            )
        if not q_params:
            raise RuntimeError(
                "openvla_oft offline IQL requires add_q_head=True and trainable q_head parameters."
            )
        if not value_params:
            raise RuntimeError(
                "openvla_oft offline IQL requires add_value_head=True and trainable value_head parameters."
            )

        actor_optim_cfg = self.cfg.actor.optim
        critic_optim_cfg = self.cfg.actor.critic_optim
        value_optim_cfg = self.cfg.actor.value_optim
        self.optimizer = torch.optim.Adam(
            actor_params,
            lr=float(actor_optim_cfg.lr),
            betas=(
                float(actor_optim_cfg.adam_beta1),
                float(actor_optim_cfg.adam_beta2),
            ),
            eps=float(actor_optim_cfg.adam_eps),
        )
        self.qf_optimizer = torch.optim.Adam(
            q_params,
            lr=float(critic_optim_cfg.lr),
            betas=(
                float(critic_optim_cfg.adam_beta1),
                float(critic_optim_cfg.adam_beta2),
            ),
            eps=float(critic_optim_cfg.adam_eps),
        )
        self.vf_optimizer = torch.optim.Adam(
            value_params,
            lr=float(value_optim_cfg.lr),
            betas=(
                float(value_optim_cfg.adam_beta1),
                float(value_optim_cfg.adam_beta2),
            ),
            eps=float(value_optim_cfg.adam_eps),
        )
        self.build_lr_schedulers()

    @staticmethod
    def _augment_obs_with_forward_inputs(
        obs: dict[str, Any],
        forward_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(obs)
        for key in (
            "input_ids",
            "attention_mask",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "pixel_values",
        ):
            if key in forward_inputs and key not in merged:
                merged[key] = forward_inputs[key]
        return merged

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self._trajectory_batch_mode:
            return super().prepare_batch(batch)

        from rlinf.utils.nested_dict_process import put_tensor_device

        batch = put_tensor_device(batch, self.device)
        curr_obs = self._augment_obs_with_forward_inputs(
            batch["curr_obs"], batch.get("forward_inputs", {})
        )
        next_obs = self._augment_obs_with_forward_inputs(
            batch["next_obs"], batch.get("forward_inputs", {})
        )

        actions = batch["actions"].to(dtype=torch.float32)
        rewards = batch["rewards"].reshape(batch["rewards"].shape[0], -1).sum(dim=-1)
        done_source = batch.get("terminations", None)
        if done_source is None:
            done_source = batch.get("dones", batch.get("truncations", None))
        if done_source is None:
            dones = torch.zeros_like(rewards, dtype=torch.float32)
        else:
            dones = (
                done_source.reshape(done_source.shape[0], -1).any(dim=-1).float()
            )
        masks = 1.0 - dones

        if self._model_type == "mlp_policy":
            observations = curr_obs["states"].reshape(curr_obs["states"].shape[0], -1)
            next_observations = next_obs["states"].reshape(
                next_obs["states"].shape[0], -1
            )
            actions = actions.reshape(actions.shape[0], -1)
        else:
            observations = curr_obs
            next_observations = next_obs

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_observations": next_observations,
        }

    def _pack_train_batch(
        self, prepared: dict[str, Any]
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        return (
            prepared["observations"],
            prepared["actions"],
            prepared["rewards"],
            prepared["masks"],
            prepared["next_observations"],
        )

    def forward_critic_module(
        self,
        critic_module,
        observations,
        actions: torch.Tensor,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_iql_backbone:
            q_values = critic_module(
                forward_type=ForwardType.IQL_CRITIC,
                observations=observations,
                actions=actions,
                detach_encoder=detach_encoder,
            )
            if q_values.ndim == 1:
                return q_values, q_values
            return q_values[..., 0], q_values[..., 1]
        return super().forward_critic_module(critic_module, observations, actions)

    def forward_value(
        self, obs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._shared_iql_backbone:
            return super().forward_value(obs, actions)

        assert self.vf_optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        with torch.no_grad():
            q1_t, q2_t = self.forward_critic_module(
                self.target_model, obs, actions, detach_encoder=True
            )
            q_t = torch.min(q1_t, q2_t)
        v = self.model(
            forward_type=ForwardType.IQL_VALUE,
            observations=obs,
            detach_encoder=True,
        )
        value_loss = iql_expectile_loss(q_t - v, self.expectile).mean()
        self.vf_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.vf_optimizer.step()
        return v, value_loss

    def forward_actor(
        self, obs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._shared_iql_backbone:
            return super().forward_actor(obs, actions)

        assert self.optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        with torch.no_grad():
            new_v = self.model(
                forward_type=ForwardType.IQL_VALUE,
                observations=obs,
                detach_encoder=True,
            )
            q1_t, q2_t = self.forward_critic_module(
                self.target_model, obs, actions, detach_encoder=True
            )
            q_t = torch.min(q1_t, q2_t)
            adv = q_t - new_v
            exp_a = torch.exp(adv * self.temperature).clamp(max=100.0)
        log_probs = self.model(
            forward_type=ForwardType.IQL_ACTOR,
            observations=obs,
            actions=actions,
            temperature=1.0,
            top_k=int(getattr(self.model.config, "n_action_bins", 64)),
        )
        actor_loss = -(exp_a * log_probs).mean()
        self.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return adv, actor_loss

    def forward_critic(
        self,
        obs,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        next_obs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._shared_iql_backbone:
            return super().forward_critic(obs, actions, rewards, masks, next_obs)

        assert self.qf_optimizer is not None, (
            "setup_model_and_optimizer must be called first."
        )
        with torch.no_grad():
            next_v = self.model(
                forward_type=ForwardType.IQL_VALUE,
                observations=next_obs,
                detach_encoder=True,
            )
            target_q = rewards + self.discount * masks * next_v
        q1, q2 = self.forward_critic_module(
            self.model,
            obs,
            actions,
            detach_encoder=True,
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        self.qf_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.qf_optimizer.step()
        return q1, q2, critic_loss

    def update_one_epoch(
        self,
        batch: tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any],
    ) -> dict[str, Any]:
        if not self._shared_iql_backbone:
            return super().update_one_epoch(batch)

        self.model.train()
        obs, actions, rewards, masks, next_obs = batch
        if int(actions.shape[0]) != self.batch_size:
            raise ValueError(
                f"Offline IQL requires static batch size. Got {int(actions.shape[0])}, expected {self.batch_size}."
            )
        flat = self.update_step_forward(obs, actions, rewards, masks, next_obs)
        metric_device = actions.device
        return {
            "critic_loss": flat[0].detach(),
            "q1": flat[1].detach(),
            "q2": flat[2].detach(),
            "value_loss": flat[3].detach(),
            "v": flat[4].detach(),
            "actor_loss": flat[5].detach(),
            "adv_mean": flat[6].detach(),
            "adv_std": flat[7].detach(),
            "lr_actor": torch.tensor(
                float(self.optimizer.param_groups[0]["lr"]),
                device=metric_device,
            ),
            "lr_value": torch.tensor(
                float(self.vf_optimizer.param_groups[0]["lr"]),
                device=metric_device,
            ),
            "lr_critic": torch.tensor(
                float(self.qf_optimizer.param_groups[0]["lr"]),
                device=metric_device,
            ),
            "use_fsdp_wrap": self._use_fsdp_wrap,
        }

    def soft_update_target_model(self):
        if not self._shared_iql_backbone:
            return super().soft_update_target_model()

        assert self.target_model_initialized
        one_minus_tau = 1.0 - float(self.tau)
        with torch.no_grad():
            if self._use_fsdp_wrap:
                target_state = self._strategy.get_model_state_dict(
                    self.target_model,
                    cpu_offload=False,
                    full_state_dict=True,
                )
                online_state = self._strategy.get_model_state_dict(
                    self.model,
                    cpu_offload=False,
                    full_state_dict=True,
                )
                mixed_state: dict[str, Any] = {}
                for name, target_value in target_state.items():
                    online_value = online_state.get(name, None)
                    if isinstance(target_value, torch.Tensor) and isinstance(
                        online_value, torch.Tensor
                    ):
                        mixed_state[name] = (
                            target_value * one_minus_tau
                            + online_value * float(self.tau)
                        )
                    else:
                        mixed_state[name] = target_value
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    mixed_state,
                    cpu_offload=False,
                    full_state_dict=True,
                )
            else:
                online_state = self.model.state_dict()
                for name, target_value in self.target_model.state_dict().items():
                    online_value = online_state.get(name, None)
                    if isinstance(target_value, torch.Tensor) and isinstance(
                        online_value, torch.Tensor
                    ):
                        target_value.mul_(one_minus_tau).add_(
                            online_value, alpha=float(self.tau)
                        )

    def save_checkpoint(self, save_base_path, step):
        if not self._shared_iql_backbone:
            return super().save_checkpoint(save_base_path, step)

        os.makedirs(save_base_path, exist_ok=True)
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[opt for opt in [self.optimizer, self.qf_optimizer, self.vf_optimizer] if opt is not None],
            lr_schedulers=[sched for sched in [self.lr_scheduler] if sched is not None],
            save_path=os.path.join(save_base_path, "actor_policy"),
            checkpoint_format="local_shard",
        )
        components_path = os.path.join(save_base_path, "iql_components")
        os.makedirs(components_path, exist_ok=True)
        if self._use_fsdp_wrap:
            target_state = self._strategy.get_model_state_dict(
                self.target_model, cpu_offload=False, full_state_dict=True
            )
        else:
            target_state = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.target_model.state_dict().items()
            }
        torch.save(target_state, os.path.join(components_path, "target_critic.pt"))
        state_payload = {
            "step": int(step),
            "global_step": int(self._global_step),
            "data_epoch": int(self._data_epoch),
            "data_iter_offset": int(self._data_iter_offset),
        }
        torch.save(state_payload, os.path.join(components_path, "state.pt"))

    def load_checkpoint(self, load_base_path: str):
        if not self._shared_iql_backbone:
            return super().load_checkpoint(load_base_path)

        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[opt for opt in [self.optimizer, self.qf_optimizer, self.vf_optimizer] if opt is not None],
            lr_schedulers=[sched for sched in [self.lr_scheduler] if sched is not None],
            load_path=os.path.join(load_base_path, "actor_policy"),
            checkpoint_format="local_shard",
        )
        components_path = os.path.join(load_base_path, "iql_components")
        target_path = os.path.join(components_path, "target_critic.pt")
        if os.path.exists(target_path):
            target_state = torch.load(
                target_path, map_location=self.device, weights_only=True
            )
            if self._use_fsdp_wrap:
                self._strategy.load_model_with_state_dict(
                    self.target_model,
                    target_state,
                    cpu_offload=False,
                    full_state_dict=True,
                )
            else:
                self.target_model.load_state_dict(target_state, strict=False)
        state_path = os.path.join(components_path, "state.pt")
        if os.path.exists(state_path):
            state_payload = torch.load(state_path, map_location=self.device)
            self._global_step = int(state_payload["global_step"])
            self._load_offline_data_state(state_payload)
