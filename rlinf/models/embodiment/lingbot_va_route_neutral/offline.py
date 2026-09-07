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

"""Native joint flow-matching SFT and causal-history-only UNCOND BC."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .data import DemonstrationDataset
from .functional import mesh
from .runtime import cpu_noise


def native_joint_inputs(window, video_scheduler, action_scheduler, *, reference, seed):
    """Use the native noisy/clean block-causal training contract and masks."""
    generator = torch.Generator().manual_seed(seed)
    text = (
        window["negative_text"]
        if torch.rand((), generator=generator) < 0.1
        else window["text"]
    )
    result = {
        "chunk_size": int(torch.randint(1, 5, (), generator=generator)),
        "window_size": int(torch.randint(4, 65, (), generator=generator)),
    }
    for action, scheduler, key in (
        (False, video_scheduler, "latents"),
        (True, action_scheduler, "actions"),
    ):
        clean = window[key].to(reference)
        frames, height, width = clean.shape[-3:]
        ids = torch.randint(1000, (frames,), generator=generator)
        times = scheduler.timesteps[ids].to(reference.device)
        sigma = scheduler.sigmas[ids].to(reference).reshape(1, 1, frames, 1, 1)
        noise = cpu_noise(clean.shape, seed + int(action) + 1, reference)
        mask = (
            window["action_mask"].to(reference.device)
            if action
            else torch.ones_like(clean, dtype=torch.bool)
        )
        noisy = ((1 - sigma) * clean + sigma * noise) * mask
        target = (noise - clean) * mask
        condition = clean * mask
        cond_times = torch.zeros_like(times)
        # Match native SFT's independently noised visual history augmentation.
        if not action and torch.rand((), generator=generator) < 0.5:
            cond_ids = torch.randint(500, 1000, (frames,), generator=generator)
            cond_times = scheduler.timesteps[cond_ids].to(reference.device)
            cond_sigma = (
                scheduler.sigmas[cond_ids].to(reference).reshape(1, 1, frames, 1, 1)
            )
            condition = (1 - cond_sigma) * clean + cond_sigma * cpu_noise(
                clean.shape, seed + 99, reference
            )
        grid, _ = mesh(
            frames,
            height if action else height // 2,
            width if action else width // 2,
            start=window["start"],
            action=action,
            device=reference.device,
        )
        result["action_dict" if action else "latent_dict"] = {
            "timesteps": times[None],
            "cond_timesteps": cond_times[None],
            "noisy_latents": noisy,
            "targets": target,
            "latent": condition,
            "grid_id": grid[None],
            "text_emb": text.to(reference),
            "actions_mask": mask,
        }
    return result


def native_joint_loss(model, inputs, video_scheduler, action_scheduler):
    """Both native losses; padded first actions never enter the action loss."""
    from wan_va.utils.utils import data_seq_to_patch

    video, actions = model(inputs, train_mode=True)
    latent = inputs["latent_dict"]
    action = inputs["action_dict"]
    frames, height, width = latent["targets"].shape[-3:]
    video = data_seq_to_patch(
        model.patch_size, video, frames, height, width, batch_size=1
    )
    actions = actions.reshape(1, frames, 4, 30).permute(0, 3, 1, 2)[..., None]
    video_weight = video_scheduler.training_weight(latent["timesteps"][0])[
        None, None, :, None, None
    ]
    action_weight = action_scheduler.training_weight(action["timesteps"][0])[
        None, None, :, None, None
    ]
    video_loss = (
        (video.float() - latent["targets"].float()).square() * video_weight
    ).mean()
    action_loss = (
        (actions.float() - action["targets"].float()).square() * action_weight
    )[action["actions_mask"]].mean()
    return video_loss + action_loss


def _schedulers():
    from wan_va.utils.scheduler import FlowMatchScheduler

    result = [
        FlowMatchScheduler(shift=s, sigma_min=0, extra_one_step=True)
        for s in (5.0, 1.0)
    ]
    for scheduler in result:
        scheduler.set_timesteps(1000, training=True)
    return result


def train_parent(cfg):
    """Four-GPU FP32 master parameters, RLinf FSDP2 and native BF16 forward."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from wan_va.modules.utils import load_transformer

    from rlinf.hybrid_engines.fsdp import MixedPrecisionPolicy, OffloadPolicy
    from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model

    world = int(os.environ.get("WORLD_SIZE", 1))
    if world != 4:
        raise ValueError(
            "The reference parent SFT uses torchrun with exactly four GPUs."
        )
    local_rank, rank = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(42)
    output = Path(cfg.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    parent = load_transformer(
        str(Path(cfg.base_model_path) / "transformer"),
        torch_dtype=torch.float32,
        torch_device="cpu",
        attn_mode="flex",
    )
    parent.requires_grad_(True).train()
    model = apply_fsdp2_to_model(
        parent,
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": [type(parent.blocks[0]).__name__]
            }
        },
        init_device_mesh("cuda", (world,)),
        MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
        OffloadPolicy(),
        True,
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    for i, block in enumerate(model.blocks):
        model.blocks[i] = checkpoint_wrapper(block, preserve_rng_state=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.1
    )
    train, validation = (
        DemonstrationDataset(cfg.prepared, "train"),
        DemonstrationDataset(cfg.prepared, "validation"),
    )
    video_scheduler, action_scheduler = _schedulers()
    reference = torch.empty((), device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    best = float("inf")
    steps = int(cfg.get("steps", 2000))
    if not 100 <= steps <= 2000 or steps % 100:
        raise ValueError(
            "Parent SFT ends on a 100-step checkpoint, at most 2000 steps."
        )
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=reference.device)
        for micro in range(8):
            sample_seed = 42 + (step - 1) * 32 + rank * 8 + micro
            idx = int(
                torch.randint(
                    len(train), (), generator=torch.Generator().manual_seed(sample_seed)
                )
            )
            window = train.sft_window(train[idx], sample_seed)
            inputs = native_joint_inputs(
                window,
                video_scheduler,
                action_scheduler,
                reference=reference,
                seed=sample_seed,
            )
            model.set_requires_gradient_sync(micro == 7)
            loss = native_joint_loss(model, inputs, video_scheduler, action_scheduler)
            (loss / 8).backward()
            train_loss += loss.detach() / 8
        # FSDP DTensor norms reduce across this run's four ranks.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        dist.all_reduce(train_loss)
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"step": step, "train_loss": float(train_loss / world)})
                    + "\n"
                )
        if step % 100:
            continue
        model.eval()
        # Reference validation has 60 episodes, divisible by four. Require equal
        # forward counts because FSDP's parameter gathers are collective.
        if len(validation) % world:
            raise ValueError(
                "The fixed validation episodes must divide across four SFT ranks."
            )
        totals = torch.zeros(2, device=reference.device)
        with torch.no_grad():
            for i in range(rank, len(validation), world):
                window = validation.sft_window(validation[i], 42000 + i)
                inputs = native_joint_inputs(
                    window,
                    video_scheduler,
                    action_scheduler,
                    reference=reference,
                    seed=42000 + i,
                )
                totals[0] += native_joint_loss(
                    model, inputs, video_scheduler, action_scheduler
                )
                totals[1] += 1
        dist.all_reduce(totals)
        val = float(totals[0] / totals[1])
        state = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if rank == 0:
            checkpoint = output / f"step_{step}"
            parent.save_pretrained(
                checkpoint / "transformer",
                state_dict={k: v.to(torch.bfloat16) for k, v in state.items()},
            )
            metadata = {
                "stage": "parent_sft",
                "model_type": "lingbot_va_route_neutral",
                "step": step,
                "validation_loss": val,
                "base_model_path": str(Path(cfg.base_model_path).resolve()),
                "prepared": str(Path(cfg.prepared).resolve()),
                "global_batch": 32,
                "learning_rate": 1e-5,
            }
            (checkpoint / "parent.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            if val < best:
                best = val
                (output / "best.json").write_text(
                    json.dumps(
                        {**metadata, "parent_path": str(checkpoint.resolve())}, indent=2
                    )
                    + "\n"
                )
        del state
        dist.barrier()
        model.train()
    dist.destroy_process_group()


def train_bc(policy, cfg):
    """FP32 adapters, frozen real-history prefill, no predicted video condition."""
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    train, validation = (
        DemonstrationDataset(cfg.prepared, "train"),
        DemonstrationDataset(cfg.prepared, "validation"),
    )
    optimizer = torch.optim.AdamW(
        policy.core.adapters.parameters(), lr=1e-4, weight_decay=0
    )
    steps = int(cfg.get("steps", 2000))
    if not 100 <= steps <= 2000 or steps % 100:
        raise ValueError("BC ends on a 100-step checkpoint, at most 2000 steps.")
    best = float("inf")
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(32):
            seed = 42 + (step - 1) * 32 + micro
            index = int(
                torch.randint(
                    len(train), (), generator=torch.Generator().manual_seed(seed)
                )
            )
            context, target = train.bc_window(train[index], seed)
            loss = policy.runtime.bc_loss(context, target, seed)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BC loss; no checkpoint selected.")
            (loss / 32).backward()
            total += float(loss.detach()) / 32
        torch.nn.utils.clip_grad_norm_(
            policy.core.adapters.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "train_loss": total}) + "\n")
        if step % 100:
            continue
        with torch.no_grad():
            validation_losses = []
            for i in range(len(validation)):
                context, target = validation.bc_window(validation[i], 42000 + i)
                validation_losses.append(
                    float(policy.runtime.bc_loss(context, target, 42000 + i))
                )
        val = sum(validation_losses) / len(validation_losses)
        path = output / f"step_{step}.pt"
        torch.save(
            {
                "stage": "bc",
                "model_type": "lingbot_va_route_neutral",
                "step": step,
                "lora": policy.core.adapters.state_dict(),
                "optimizer": optimizer.state_dict(),
                "validation_loss": val,
                "parent_path": policy.parent_path,
                "routing_config": policy.config.to_dict(),
                "normalizer": policy.runtime.normalizer.to_dict(),
                "prepared": str(Path(cfg.prepared).resolve()),
                "learning_rate": 1e-4,
                "global_batch": 32,
            },
            path,
        )
        if val < best:
            best = val
            (output / "best.json").write_text(
                json.dumps(
                    {
                        "bc_path": str(path.resolve()),
                        "step": step,
                        "validation_loss": val,
                    },
                    indent=2,
                )
                + "\n"
            )
