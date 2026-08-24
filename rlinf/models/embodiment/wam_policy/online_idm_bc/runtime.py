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

"""Same-state frozen-IDM teacher and actor-side BC replay for LIBERO."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_sampler import sample_action_flow_sde
from fastwam.uncond_bc import (
    compute_action_flow_matching_bc_loss,
    stateless_validation_flow_inputs,
)

from rlinf.models.embodiment.wam_policy.adaptive_policy import FastWAMChunkSample
from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord, WAMRoute
from rlinf.models.embodiment.wam_policy.libero_runtime import LiberoFastWAMRuntime

from .config import (
    ONLINE_IDM_BC_FLOW_VALID,
    ONLINE_IDM_BC_FORWARD_KEYS,
    ONLINE_IDM_BC_SAMPLE_IDENTITIES,
    ONLINE_IDM_BC_TEACHER_ACTIONS,
    ONLINE_IDM_BC_TEACHER_BYTES,
    ONLINE_IDM_BC_TEACHER_PRESENT,
    ONLINE_IDM_BC_TEACHER_SECONDS,
)


@dataclass(frozen=True, slots=True)
class OnlineIDMBCLossBatch:
    """Differentiable BC numerator plus compact per-microbatch diagnostics."""

    loss_sum: torch.Tensor
    raw_loss: torch.Tensor
    selected_count: torch.Tensor
    expected_count: torch.Tensor
    present_count: torch.Tensor
    mse_per_dimension: torch.Tensor
    mse_pose: torch.Tensor
    mse_gripper: torch.Tensor
    mse_by_timestep_bin: torch.Tensor
    timestep_bin_count: torch.Tensor
    valid_action_count: torch.Tensor
    full_action_mse: torch.Tensor
    executed_prefix_mse: torch.Tensor
    teacher_seconds_sum: torch.Tensor
    teacher_bytes_sum: torch.Tensor

    def as_forward_outputs(self) -> dict[str, torch.Tensor]:
        """Return stable actor-forward keys without detaching the loss."""

        return {
            "online_idm_bc_loss_sum": self.loss_sum,
            "online_idm_bc_raw_loss": self.raw_loss,
            "online_idm_bc_selected_count": self.selected_count,
            "online_idm_bc_expected_count": self.expected_count,
            "online_idm_bc_present_count": self.present_count,
            "online_idm_bc_mse_per_dimension": self.mse_per_dimension,
            "online_idm_bc_mse_pose": self.mse_pose,
            "online_idm_bc_mse_gripper": self.mse_gripper,
            "online_idm_bc_mse_by_timestep_bin": self.mse_by_timestep_bin,
            "online_idm_bc_timestep_bin_count": self.timestep_bin_count,
            "online_idm_bc_valid_action_count": self.valid_action_count,
            "online_idm_bc_full_action_mse": self.full_action_mse,
            "online_idm_bc_executed_prefix_mse": self.executed_prefix_mse,
            "online_idm_bc_teacher_seconds_sum": self.teacher_seconds_sum,
            "online_idm_bc_teacher_bytes_sum": self.teacher_bytes_sum,
        }


def _batch_long_tensor(value: Any, *, batch_size: int, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, device="cpu", dtype=torch.long).reshape(-1)
    if result.shape != (batch_size,):
        raise ValueError(
            f"Online IDM BC {name} must have shape ({batch_size},), got "
            f"{tuple(result.shape)}."
        )
    return result


class OnlineIDMTeacherLiberoRuntime(LiberoFastWAMRuntime):
    """Add an IDM action target for every generated training UNCOND chunk."""

    @torch.no_grad()
    def _sample_idm_teacher_action(
        self,
        *,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        initial_action_noise: torch.Tensor,
        idm_noise_seed: int,
        actor_version: int,
    ) -> torch.Tensor:
        """Run deterministic frozen IDM from the student's initial action noise."""

        condition, _ = self._prepare_action_condition(
            image=image,
            context=context,
            context_mask=context_mask,
            regime=PolicyRegime.IDM,
            idm_noise_seed=int(idm_noise_seed),
        )
        timesteps, deltas = self._action_schedule()
        rollout = sample_action_flow_sde(
            initial_action_noise.to(device=self.device, dtype=self.dtype),
            velocity_fn=self._velocity(
                condition,
                regime=PolicyRegime.IDM,
                capture_gate_kv=False,
                actor_version=int(actor_version),
            ),
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=self.actor.infer_action_scheduler.num_train_timesteps,
            noise_level=self.flow_sde_noise_level,
            gate_last_n=1,
            ignore_last_transition=self.flow_sde_ignore_last_transition,
            stochastic=False,
            collect_replay=False,
        )
        return rollout.actions

    def sample_action_batch(
        self,
        *,
        env_obs: dict[str, Any],
        routes: torch.Tensor,
        mode: Literal["train", "eval"],
        actor_version: int,
        collect_replay: bool = True,
    ) -> FastWAMChunkSample:
        """Run the student, then query frozen IDM only for training UNCOND rows."""

        sample = super().sample_action_batch(
            env_obs=env_obs,
            routes=routes,
            mode=mode,
            actor_version=actor_version,
            collect_replay=collect_replay,
        )
        if mode != "train" or not collect_replay:
            return sample

        batch_size = int(routes.numel())
        action_seeds = _batch_long_tensor(
            env_obs.get("_fastwam_action_noise_seeds"),
            batch_size=batch_size,
            name="action-noise seeds",
        )
        idm_seeds = _batch_long_tensor(
            env_obs.get("_fastwam_idm_noise_seeds"),
            batch_size=batch_size,
            name="IDM-video seeds",
        )
        required_replay = {
            "fastwam_images",
            "fastwam_context",
            "fastwam_context_mask",
        }
        missing_replay = sorted(required_replay - set(sample.forward_inputs))
        if missing_replay:
            raise KeyError(
                f"Online IDM teacher lacks same-state replay inputs: {missing_replay}."
            )
        if sample.flow_chains.ndim != 4 or sample.flow_chains.shape[0] != batch_size:
            raise ValueError("Online IDM teacher requires full [B,S+1,32,7] chains.")

        teacher_actions = torch.zeros_like(
            sample.flow_chains[:, -1],
            dtype=torch.bfloat16,
        )
        teacher_present = torch.zeros(
            batch_size,
            device=routes.device,
            dtype=torch.bool,
        )
        teacher_seconds = torch.zeros(
            batch_size,
            device=routes.device,
            dtype=torch.float32,
        )
        teacher_bytes = torch.zeros(
            batch_size,
            device=routes.device,
            dtype=torch.long,
        )
        images = sample.forward_inputs["fastwam_images"]
        context = sample.forward_inputs["fastwam_context"]
        context_mask = sample.forward_inputs["fastwam_context_mask"]
        bytes_per_target = int(
            teacher_actions[0].numel() * teacher_actions.element_size()
        )

        for index, route in enumerate(routes.detach().cpu().reshape(-1).tolist()):
            if int(route) != int(WAMRoute.UNCOND):
                continue
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            target = self._sample_idm_teacher_action(
                image=images[index : index + 1],
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                initial_action_noise=sample.flow_chains[index : index + 1, 0],
                idm_noise_seed=int(idm_seeds[index].item()),
                actor_version=int(actor_version),
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            if target.shape != teacher_actions[index : index + 1].shape:
                raise ValueError(
                    "IDM teacher action shape changed: "
                    f"{tuple(target.shape)} != "
                    f"{tuple(teacher_actions[index : index + 1].shape)}."
                )
            if not bool(torch.isfinite(target.float()).all().item()):
                raise FloatingPointError("IDM teacher produced a non-finite action.")
            teacher_actions[index].copy_(target[0].to(dtype=torch.bfloat16))
            teacher_present[index] = True
            teacher_seconds[index] = float(elapsed)
            teacher_bytes[index] = bytes_per_target

        forward_inputs = dict(sample.forward_inputs)
        collisions = sorted(
            set(forward_inputs).intersection(ONLINE_IDM_BC_FORWARD_KEYS)
        )
        if collisions:
            raise KeyError(f"Online IDM BC replay fields already exist: {collisions}.")
        forward_inputs.update(
            {
                ONLINE_IDM_BC_TEACHER_ACTIONS: teacher_actions.detach(),
                ONLINE_IDM_BC_TEACHER_PRESENT: teacher_present.detach(),
                ONLINE_IDM_BC_SAMPLE_IDENTITIES: action_seeds.to(routes.device),
                ONLINE_IDM_BC_TEACHER_SECONDS: teacher_seconds.detach(),
                ONLINE_IDM_BC_TEACHER_BYTES: teacher_bytes.detach(),
            }
        )
        return replace(sample, forward_inputs=forward_inputs)

    def compute_online_idm_bc_loss(
        self,
        *,
        forward_inputs: dict[str, torch.Tensor],
        route_info: ChunkRouteRecord,
    ) -> OnlineIDMBCLossBatch:
        """Rebuild live UNCOND velocities against detached online IDM targets."""

        required = set(ONLINE_IDM_BC_FORWARD_KEYS) | {
            ONLINE_IDM_BC_FLOW_VALID,
            "flow_chains",
            "fastwam_images",
            "fastwam_context",
            "fastwam_context_mask",
        }
        missing = sorted(required - set(forward_inputs))
        if missing:
            raise KeyError(f"Online IDM BC actor replay is missing fields: {missing}.")

        routes = route_info.route_used.reshape(-1)
        batch_size = int(routes.numel())
        flow_valid = forward_inputs[ONLINE_IDM_BC_FLOW_VALID].bool().reshape(-1)
        present = forward_inputs[ONLINE_IDM_BC_TEACHER_PRESENT].bool().reshape(-1)
        identities = forward_inputs[ONLINE_IDM_BC_SAMPLE_IDENTITIES].long().reshape(-1)
        teacher_actions = forward_inputs[ONLINE_IDM_BC_TEACHER_ACTIONS]
        teacher_seconds = (
            forward_inputs[ONLINE_IDM_BC_TEACHER_SECONDS].float().reshape(-1)
        )
        teacher_bytes = forward_inputs[ONLINE_IDM_BC_TEACHER_BYTES].long().reshape(-1)
        expected_shape = (batch_size, self.action_protocol.generation_horizon, 7)
        if tuple(teacher_actions.shape) != expected_shape:
            raise ValueError(
                "Online IDM BC teacher actions must have shape "
                f"{expected_shape}, got {tuple(teacher_actions.shape)}."
            )
        for name, value in {
            "flow-valid": flow_valid,
            "teacher-present": present,
            "sample identities": identities,
            "teacher seconds": teacher_seconds,
            "teacher bytes": teacher_bytes,
        }.items():
            if value.shape != (batch_size,):
                raise ValueError(
                    f"Online IDM BC {name} must have shape ({batch_size},)."
                )
        if teacher_actions.dtype is not torch.bfloat16:
            raise TypeError(
                "Online IDM BC teacher actions must be transported as BF16."
            )
        if bool((teacher_seconds < 0).any().item()) or bool(
            (teacher_bytes < 0).any().item()
        ):
            raise ValueError(
                "Online IDM BC teacher time/byte metrics must be nonnegative."
            )

        is_uncond = routes == int(WAMRoute.UNCOND)
        expected = flow_valid & is_uncond
        missing_teacher = is_uncond & ~present
        if bool(missing_teacher.any().item()):
            indices = missing_teacher.nonzero(as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                f"UNCOND rows lack IDM teacher targets at indices {indices}."
            )
        unexpected_teacher = present & ~is_uncond
        if bool(unexpected_teacher.any().item()):
            indices = unexpected_teacher.nonzero(as_tuple=False).reshape(-1).tolist()
            raise RuntimeError(
                f"IDM-routed rows carry teacher targets at indices {indices}."
            )
        selected = expected & present
        selected_indices = selected.nonzero(as_tuple=False).reshape(-1)

        lora_parameter = next(self.lora_adapter.lora_parameters())
        differentiable_zero = lora_parameter.reshape(-1)[0] * 0.0
        action_dim = int(teacher_actions.shape[-1])
        timestep_bins = 10
        dimension_sum = torch.zeros(action_dim, device=self.device, dtype=torch.float32)
        timestep_mse_sum = torch.zeros(
            timestep_bins,
            device=self.device,
            dtype=torch.float32,
        )
        timestep_counts = torch.zeros(
            timestep_bins,
            device=self.device,
            dtype=torch.long,
        )
        losses: list[torch.Tensor] = []
        pose_values: list[torch.Tensor] = []
        gripper_values: list[torch.Tensor] = []
        valid_action_count = torch.zeros((), device=self.device, dtype=torch.long)

        images = forward_inputs["fastwam_images"]
        context = forward_inputs["fastwam_context"]
        context_mask = forward_inputs["fastwam_context_mask"]
        for raw_index in selected_indices.tolist():
            index = int(raw_index)
            action = teacher_actions[index : index + 1].to(
                device=self.device,
                dtype=self.dtype,
            )
            timestep, noise = stateless_validation_flow_inputs(
                sample_identities=[int(identities[index].item())],
                action_shape=tuple(action.shape),
                scheduler=self.actor.train_action_scheduler,
                seed=0,
                device=self.device,
                dtype=self.dtype,
            )
            noisy_action = self.actor.train_action_scheduler.add_noise(
                action,
                noise,
                timestep,
            )
            velocity_target = self.actor.train_action_scheduler.training_target(
                action,
                noise,
                timestep,
            )
            condition, _ = self._prepare_action_condition(
                image=images[index : index + 1],
                context=context[index : index + 1],
                context_mask=context_mask[index : index + 1],
                regime=PolicyRegime.UNCOND,
            )
            prediction = self._velocity(
                condition,
                regime=PolicyRegime.UNCOND,
                capture_gate_kv=False,
                actor_version=int(route_info.actor_versions.reshape(-1)[index].item()),
            )(noisy_action, timestep).velocity
            result = compute_action_flow_matching_bc_loss(
                prediction=prediction,
                target=velocity_target,
                timestep=timestep,
                action_is_pad=None,
                scheduler=self.actor.train_action_scheduler,
                gripper_dimension=6,
                timestep_bins=timestep_bins,
            )
            losses.append(result.loss_action_bc)
            dimension_sum += result.mse_per_dimension.detach()
            pose_values.append(result.mse_pose.detach())
            gripper_values.append(result.mse_gripper.detach())
            timestep_mse_sum += (
                result.mse_by_timestep_bin.detach()
                * result.timestep_bin_count.to(dtype=torch.float32)
            )
            timestep_counts += result.timestep_bin_count.detach()
            valid_action_count += result.valid_action_count.detach()

        if losses:
            loss_stack = torch.stack(losses)
            loss_sum = loss_stack.sum()
            raw_loss = loss_stack.mean().detach()
            divisor = float(len(losses))
            mse_per_dimension = dimension_sum / divisor
            mse_pose = torch.stack(pose_values).mean()
            mse_gripper = torch.stack(gripper_values).mean()
        else:
            loss_sum = differentiable_zero
            raw_loss = differentiable_zero.detach()
            mse_per_dimension = dimension_sum
            mse_pose = differentiable_zero.detach()
            mse_gripper = differentiable_zero.detach()
        mse_by_timestep_bin = timestep_mse_sum / timestep_counts.to(
            dtype=torch.float32
        ).clamp(min=1.0)

        student_actions = forward_inputs["flow_chains"][:, -1].float()
        action_error = (student_actions - teacher_actions.float()).square()
        if selected_indices.numel():
            selected_error = action_error[selected]
            full_action_mse = selected_error.mean()
            executed_prefix_mse = selected_error[
                :, : self.action_protocol.execution_horizon
            ].mean()
        else:
            full_action_mse = differentiable_zero.detach()
            executed_prefix_mse = differentiable_zero.detach()

        selected_count = selected.sum().to(device=self.device, dtype=torch.float32)
        expected_count = expected.sum().to(device=self.device, dtype=torch.float32)
        present_count = present.sum().to(device=self.device, dtype=torch.float32)
        return OnlineIDMBCLossBatch(
            loss_sum=loss_sum.float(),
            raw_loss=raw_loss.float(),
            selected_count=selected_count,
            expected_count=expected_count,
            present_count=present_count,
            mse_per_dimension=mse_per_dimension.float(),
            mse_pose=mse_pose.float(),
            mse_gripper=mse_gripper.float(),
            mse_by_timestep_bin=mse_by_timestep_bin.float(),
            timestep_bin_count=timestep_counts,
            valid_action_count=valid_action_count,
            full_action_mse=full_action_mse.detach().float(),
            executed_prefix_mse=executed_prefix_mse.detach().float(),
            teacher_seconds_sum=teacher_seconds[present].sum().to(self.device),
            teacher_bytes_sum=teacher_bytes[present].sum().to(self.device),
        )
