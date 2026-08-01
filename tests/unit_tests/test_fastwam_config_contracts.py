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

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

MODULE_PATH = Path(__file__).resolve().parents[2] / "rlinf/config_contracts.py"
SPEC = importlib.util.spec_from_file_location("fastwam_config_contracts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recompute_kv_requires_every_update_weight_sync() -> None:
    MODULE.validate_fastwam_kv_weight_sync("recompute", 1)
    MODULE.validate_fastwam_kv_weight_sync("stored", 4)

    with pytest.raises(ValueError, match="weight_sync_interval: 1"):
        MODULE.validate_fastwam_kv_weight_sync("recompute", 2)


def test_pi05_critic_artifact_config_requires_path_and_hex_digest() -> None:
    MODULE.validate_pi05_critic_artifact_config("/tmp/pi05", "a" * 64)

    with pytest.raises(ValueError, match="non-empty checkpoint path"):
        MODULE.validate_pi05_critic_artifact_config("", "a" * 64)
    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        MODULE.validate_pi05_critic_artifact_config("/tmp/pi05", "z" * 64)


def test_terminal_reward_config_rejects_repeated_success_reward_mode() -> None:
    MODULE.validate_libero_terminal_reward_config(
        ignore_terminations=False,
        use_rel_reward=False,
    )
    MODULE.validate_libero_terminal_reward_config(
        ignore_terminations=True,
        use_rel_reward=True,
    )

    with pytest.raises(ValueError, match="terminal success rewards would repeat"):
        MODULE.validate_libero_terminal_reward_config(
            ignore_terminations=True,
            use_rel_reward=False,
        )


def test_fastwam_resume_uses_consistent_payload_step() -> None:
    assert (
        MODULE.validate_fastwam_resume_steps(
            [7, 7],
            "/tmp/global_step_7",
        )
        == 7
    )

    with pytest.raises(ValueError, match="directory/payload step mismatch"):
        MODULE.validate_fastwam_resume_steps([7, 7], "/tmp/global_step_8")

    with pytest.raises(ValueError, match="ranks disagree"):
        MODULE.validate_fastwam_resume_steps([7, 8], "/tmp/global_step_7")

    with pytest.raises(ValueError, match="did not return payload steps"):
        MODULE.validate_fastwam_resume_steps([7, None], "/tmp/global_step_7")


def _checkpoint_cfg():
    return OmegaConf.create(
        {
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "global_batch_size": 4,
                "training_backend": "fsdp",
                "enable_offload": False,
                "model": {"model_type": "fastwam_adaptive", "precision": "bf16"},
                "fsdp_config": {
                    "sharding_strategy": "no_shard",
                    "use_orig_params": True,
                    "ignore_frozen_parameters": True,
                },
                "optim": {"gate_lr": 1e-4, "lora_lr": 1e-5, "value_lr": 1e-4},
            },
            "algorithm": {"loss_type": "fastwam_dual_ppo"},
            "rollout": {
                "generation_backend": "huggingface",
                "recompute_logprobs": False,
                "unnorm_key": "libero_10",
                "enable_offload": False,
                "pipeline_stage_num": 1,
                "collect_prev_infos": True,
                "enable_cuda_graph": False,
            },
            "env": {
                "train": {
                    "env_type": "libero",
                    "task_suite_name": "libero_10",
                    "total_num_envs": 2,
                    "rollout_epoch": 1,
                    "group_size": 1,
                    "auto_reset": True,
                    "max_episode_steps": 32,
                    "max_steps_per_rollout_epoch": 32,
                    "seed": 0,
                }
            },
            "runner": {
                "max_steps": 1,
                "weight_sync_interval": 1,
                "overlap_env_bootstrap": False,
                "use_training_pipeline": False,
            },
            "weight_syncer": {
                "type": "patch",
                "patch": {"init_sync": {"enabled": True}},
            },
            "cluster": {"component_placement": {"actor": "0-1"}},
        }
    )


def test_checkpoint_contract_covers_continuation_semantics_not_run_length() -> None:
    cfg = _checkpoint_cfg()
    baseline = MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2)
    assert baseline["schema"] == "fastwam-adaptive-checkpoint-contract-v2"
    assert baseline["actor"]["seed"] == 42
    assert baseline["weight_syncer"]["patch"]["init_sync"]["enabled"] is True

    cfg.runner.max_steps = 2
    assert MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2) == baseline

    cfg.actor.seed = 43
    assert MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2) != baseline
