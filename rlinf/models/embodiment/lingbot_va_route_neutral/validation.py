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

"""Native-weight checks and recorded-observation latency measurements."""

from __future__ import annotations

import numpy as np
import torch

from .contracts import Route
from .functional import mesh
from .runtime import ChunkNoise, clock_sync, cpu_noise


@torch.no_grad()
def native_parity(policy):
    """Compare the actual loaded parent against its cache-free native forward."""
    from wan_va.utils.utils import data_seq_to_patch

    parent, functional = policy.core.parent, policy.core
    reference = policy.runtime.reference
    text = cpu_noise((1, 512, parent.config.text_dim), 100, reference)
    results = {}
    for action in (False, True):
        shape = (
            (1, 30, 4, 4, 1) if action else (1, parent.config.in_channels, 4, 16, 32)
        )
        x = cpu_noise(shape, 42, reference)
        height, width = shape[-2:]
        grid, _ = mesh(
            4,
            height if action else height // 2,
            width if action else width // 2,
            start=0,
            action=action,
            device=reference.device,
        )
        time = torch.full((1, 4), 500.0, device=reference.device)
        native = parent(
            {
                "noisy_latents": x,
                "timesteps": time,
                "grid_id": grid[None],
                "text_emb": text,
            },
            action_mode=action,
        )
        native = (
            native.reshape(1, 4, 4, 30).permute(0, 3, 1, 2)[..., None]
            if action
            else data_seq_to_patch(parent.patch_size, native, 4, 16, 32)
        )
        actual, _ = functional(x, time, text, action=action)
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
        results["action" if action else "video"] = {
            "equal": True,
            "values_compared": actual.numel(),
        }
    return {
        "status": "PASS",
        "scope": "loaded parent video/action forward, LoRA disabled",
        "checks": results,
        "parent_path": policy.parent_path,
        "device": str(reference.device),
    }


def replay_recording(policy, recording, instruction, *, benchmark=False):
    """Use actual execution feedback; no robot commands are issued by this path."""
    with np.load(recording, allow_pickle=False) as handle:
        raw = {k: handle[k] for k in handle.files}
    observations = [
        {
            "images": {key: raw[key][i] for key in ("external", "wrist")},
            "state": raw["states"][i],
            "timestamp": float(raw["observation_timestamps"][i]),
        }
        for i in range(len(raw["states"]))
    ]
    if len(observations) != len(raw["actions"]) + 1:
        raise ValueError(
            "Recorded feedback must contain T+1 observations and T actions."
        )
    results, passes = [], []
    modes = (
        [
            (Route.IDM, True),
            (Route.UNCOND, True),
            (Route.IDM, False),
            (Route.UNCOND, False),
        ]
        if benchmark
        else [(None, True)]
    )
    for mode, gate_framework in modes:
        reference = policy.runtime.reference
        if reference.is_cuda:
            torch.cuda.reset_peak_memory_stats(reference.device)
        started = clock_sync(reference)
        policy.reset_episode(instruction, observations[0])
        reset_s = clock_sync(reference) - started
        policy.rng.manual_seed(42)
        for start in range(0, len(raw["actions"]), 8):
            end = min(start + 8, len(raw["actions"]))
            route = (
                mode
                if mode is not None
                else (Route.IDM if start // 8 % 3 == 1 else Route.UNCOND)
            )
            decision = policy.decide(
                observations[start], route=route, gate_framework=gate_framework
            )
            if route is Route.UNCOND and decision.sample.timings["video_forwards"] != 0:
                raise AssertionError(
                    "UNCOND unexpectedly invoked future video denoising."
                )
            if not benchmark and route is Route.UNCOND:
                # Training replay uses the identical frozen, raw historical prefix.
                sample = policy.runtime.sample_from_context(
                    route,
                    ChunkNoise(start + 1, start + 2, start + 3),
                    decision.sample.context,
                    training=True,
                )
                logprob = policy.replay_uncond_transition(sample.flow).detach().cpu()
                torch.testing.assert_close(
                    logprob, sample.flow.old_log_prob, atol=1e-4, rtol=0
                )
            policy.commit_execution(
                observations[start + 1 : end + 1],
                raw["actions"][start:end],
                terminated=end == len(raw["actions"]),
            )
            results.append(
                {
                    "route": route.value,
                    "chunk": start // 8,
                    "gate_framework": gate_framework,
                    "timings": decision.sample.timings,
                }
            )
        passes.append(
            {
                "route": mode.value if mode is not None else "alternating",
                "gate_framework": gate_framework,
                "episode_reset_s": reset_s,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    reference.device
                )
                if reference.is_cuda
                else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(reference.device)
                if reference.is_cuda
                else None,
            }
        )
    return {
        "status": "PASS",
        "scope": "recorded feedback; no physical execution",
        "benchmark": benchmark,
        "parent_path": policy.parent_path,
        "passes": passes,
        "chunks": results,
    }
