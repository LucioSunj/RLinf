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

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
from omegaconf import OmegaConf

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.calvin.subtask_collection import (
    CalvinSubtaskEnv,
    CollectedChunk,
    build_trajectory,
    ceil_div,
    load_calvin_subtask_segments,
    make_model_weights_id,
    obs_for_storage,
    pad_or_trim_actions,
    set_collection_seed,
)
from rlinf.models import get_model


def _as_plain_list(value) -> list:
    if value is None:
        return []
    converted = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if converted is None:
        return []
    if isinstance(converted, list):
        return converted
    if isinstance(converted, tuple):
        return list(converted)
    return [converted]


def _load_model(model_cfg, *, dry_run: bool, device: str = "auto"):
    if dry_run:
        return None

    model_path = Path(str(model_cfg.model_path))
    if not (model_path / "model.safetensors").exists() and not (
        model_path / "model_state_dict" / "full_weights.pt"
    ).exists():
        raise FileNotFoundError(
            f"{model_path} is not an RLinf-loadable OpenPI checkpoint. Use a "
            "converted RLinf checkpoint directory or the RLinf Pi05 CALVIN SFT "
            "checkpoint layout before launching policy rollouts."
        )

    model = get_model(model_cfg)
    if model is None:
        raise ValueError(f"Unsupported model_type={model_cfg.model_type!r}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(torch.device(device))
    model.eval()
    return model


def _predict_actions(
    *,
    model,
    obs: dict[str, Any],
    env: CalvinSubtaskEnv,
    cfg,
) -> tuple[torch.Tensor, dict[str, Any]]:
    action_chunk_size = int(cfg.collector.action_chunk_size)
    action_dim = int(cfg.actor.model.action_dim)
    if cfg.collector.dry_run:
        actions = env.get_expert_action_chunk()
        actions = pad_or_trim_actions(actions, action_chunk_size, action_dim)
        return actions, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(actions.shape[0], -1).cpu().contiguous()
            },
        }

    with torch.no_grad():
        actions, result = model.predict_action_batch(
            env_obs=obs,
            mode=str(cfg.collector.model_mode),
            compute_values=bool(cfg.collector.compute_values),
        )
    actions = pad_or_trim_actions(actions.detach().cpu(), action_chunk_size, action_dim)
    forward_inputs = result.get("forward_inputs", {})
    if "action" not in forward_inputs:
        forward_inputs["action"] = actions.reshape(actions.shape[0], -1)
    return actions, result


def _optional_cpu(value):
    return value.detach().cpu().contiguous() if torch.is_tensor(value) else None


def _cpu_forward_inputs(forward_inputs: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in forward_inputs.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().contiguous()
        elif isinstance(value, dict):
            result[key] = _cpu_forward_inputs(value)
    return result


def _make_chunk(
    *,
    actions: torch.Tensor,
    model_result: dict[str, Any],
    curr_obs: dict[str, Any],
    next_obs: dict[str, Any],
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
) -> CollectedChunk:
    action_flat = model_result.get("forward_inputs", {}).get("action")
    if action_flat is None:
        action_flat = actions.reshape(actions.shape[0], -1)
    else:
        action_flat = action_flat.detach().cpu()

    dones = torch.logical_or(terminations, truncations)
    versions = torch.zeros((actions.shape[0], 1), dtype=torch.float32)
    return CollectedChunk(
        actions=action_flat.cpu().contiguous(),
        rewards=rewards.cpu().contiguous(),
        terminations=terminations.cpu().contiguous(),
        truncations=truncations.cpu().contiguous(),
        dones=dones.cpu().contiguous(),
        prev_logprobs=_optional_cpu(model_result.get("prev_logprobs")),
        prev_values=_optional_cpu(model_result.get("prev_values")),
        versions=versions,
        forward_inputs=_cpu_forward_inputs(model_result.get("forward_inputs", {})),
        curr_obs=obs_for_storage(curr_obs),
        next_obs=obs_for_storage(next_obs),
    )


def _write_jsonl(path: Path, record: dict[str, Any]):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="calvin_subtask_collect_pi05",
)
def main(cfg) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    set_collection_seed(int(cfg.collector.seed))
    output_dir = Path(str(cfg.collector.output_dir))
    replay_dir = output_dir / "replay_buffer"
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = output_dir / "subtask_rollout_manifest.jsonl"
    if manifest_jsonl.exists() and bool(cfg.collector.overwrite_manifest):
        manifest_jsonl.unlink()

    segments = load_calvin_subtask_segments(
        cfg.collector.data_root,
        split=str(cfg.collector.split),
        canonical_tasks=_as_plain_list(cfg.collector.canonical_tasks),
        segment_indices=_as_plain_list(cfg.collector.segment_indices),
        max_segments_per_task=cfg.collector.max_segments_per_task,
        max_total_segments=cfg.collector.max_total_segments,
        target_window_size=int(cfg.collector.target_window_size),
        next_window_size=int(cfg.collector.next_window_size),
        resolve_scene_from_scene_info=bool(cfg.collector.resolve_scene_from_scene_info),
    )
    if not segments:
        raise ValueError("No CALVIN subtask segments selected for collection")

    model = _load_model(
        cfg.actor.model,
        dry_run=bool(cfg.collector.dry_run),
        device=str(cfg.collector.model_device),
    )
    model_weights_id = make_model_weights_id(
        cfg.actor.model.model_path,
        str(cfg.collector.get("model_tag", "")),
    )
    replay_buffer = TrajectoryReplayBuffer(
        seed=int(cfg.collector.seed),
        enable_cache=False,
        auto_save=True,
        auto_save_path=str(replay_dir),
        trajectory_format="pt",
    )

    env = CalvinSubtaskEnv(
        data_root=cfg.collector.data_root,
        split=str(cfg.collector.split),
        task_suite_name=str(cfg.calvin.task_suite_name),
        scene=cfg.calvin.scene,
        action_chunk_size=int(cfg.collector.action_chunk_size),
        max_subtask_steps=int(cfg.collector.max_subtask_steps),
        action_key=str(cfg.collector.action_key),
        target_window_size=int(cfg.collector.target_window_size),
        include_next_window_in_dense=bool(cfg.collector.include_next_window_in_dense),
        dense_beta=float(cfg.reward.dense_beta),
        dense_state_weight=float(cfg.reward.dense_state_weight),
        dense_visual_weight=float(cfg.reward.dense_visual_weight),
        oracle_success_bonus=float(cfg.reward.oracle_success_bonus),
    )

    rollout_count = 0
    try:
        for segment in segments:
            for repeat_index in range(int(cfg.collector.rollouts_per_subtask)):
                obs = env.reset(segment)
                chunks: list[CollectedChunk] = []
                final_info = None
                max_chunks = ceil_div(env.max_steps, int(cfg.collector.action_chunk_size))
                for _ in range(max_chunks):
                    actions, model_result = _predict_actions(
                        model=model,
                        obs=obs,
                        env=env,
                        cfg=cfg,
                    )
                    next_obs, rewards, terminations, truncations, info = env.step_chunk(actions)
                    chunks.append(
                        _make_chunk(
                            actions=actions,
                            model_result=model_result,
                            curr_obs=obs,
                            next_obs=next_obs,
                            rewards=rewards,
                            terminations=terminations,
                            truncations=truncations,
                        )
                    )
                    obs = next_obs
                    final_info = info
                    if bool(torch.logical_or(terminations, truncations).any()):
                        break

                trajectory = build_trajectory(
                    chunks,
                    max_episode_length=env.max_steps,
                    model_weights_id=model_weights_id,
                )
                replay_buffer.add_trajectories([trajectory])
                record = {
                    "rollout_index": rollout_count,
                    "repeat_index": repeat_index,
                    "num_chunks": len(chunks),
                    "output_replay_dir": str(replay_dir),
                    **(final_info or {}),
                }
                _write_jsonl(manifest_jsonl, record)
                print(json.dumps(record, sort_keys=True))
                rollout_count += 1
    finally:
        env.close()
        replay_buffer.close(wait=True)

    print(
        f"Saved {rollout_count} CALVIN subtask rollouts to {replay_dir}; "
        f"metadata: {manifest_jsonl}"
    )


if __name__ == "__main__":
    main()
