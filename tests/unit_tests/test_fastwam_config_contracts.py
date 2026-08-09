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


def test_checkpoint_contract_normalizes_validate_cfg_inserted_defaults() -> None:
    unvalidated = _checkpoint_cfg()
    del unvalidated.runner.weight_sync_interval
    del unvalidated.runner.overlap_env_bootstrap

    validated = _checkpoint_cfg()
    validated.actor.fsdp_config.amp_autocast = {
        "enabled": False,
        "precision": "bf16",
    }
    validated.actor.fsdp_config.grad_scaler = {
        "enabled": False,
        "init_scale": None,
        "growth_interval": None,
    }

    assert MODULE.build_fastwam_checkpoint_contract(
        unvalidated,
        world_size=2,
    ) == MODULE.build_fastwam_checkpoint_contract(validated, world_size=2)


def _eval_model_cfg():
    return OmegaConf.create(
        {
            "model_type": "fastwam_adaptive",
            "precision": "bf16",
            "init_device": "cpu",
            "action_dim": 7,
            "num_action_chunks": 10,
            "actor_checkpoint": "/parents/fastwam.pt",
            "actor_checkpoint_sha256": "a" * 64,
            "model_path": "/parents/fastwam.pt",
            "fastwam": {
                "_target_": "fastwam.runtime.create_fastwam_idm",
                "load_text_encoder": True,
                "action_dit_config": {"num_layers": 30, "hidden_dim": 1024},
            },
            "uncond_lora": {
                "rank": 16,
                "alpha": 16.0,
                "target_groups": ["self_attention_qkvo", "ffn"],
            },
            "gate": {
                "hidden_dim": 256,
                "ffn_multiplier": 4,
                "share_blocks": False,
                "denoise_last_n": 1,
                "layer_taps": {
                    "mode": "all",
                    "last_n": None,
                    "indices": None,
                },
            },
            "gate_epsilon": 0.1,
            "gate_temperature": 1.0,
            "eval_routing_mode": "learned_threshold",
            "eval_idm_threshold": 0.5,
            "eval_random_idm_probability": None,
            "eval_routing_seed": 0,
            "eval_microbatch_size": 1,
            "kv_replay": {"backend": "stored"},
            "flow_sde": {"enabled": True, "noise_level": 0.5},
            "runtime": {
                "generation_horizon": 32,
                "execution_horizon": 10,
                "num_video_frames": 9,
                "reset_wait_steps": 30,
                "max_episode_steps": 700,
                "num_inference_steps": 10,
                "seeded_noise_device": "cpu",
                "text_embedding_cache_dir": None,
                "camera_resize_mode": "official_pil_center_crop",
                "binarize_gripper": False,
            },
            "critic": {
                "load_for_eval": False,
                "input_dim": 2048,
                "hidden_sizes": [1024, 512, 256],
                "backbone_checkpoint_sha256": "b" * 64,
                "backbone": {"model_path": "/parents/pi05"},
            },
        }
    )


def test_eval_model_contract_allows_only_declared_runtime_differences() -> None:
    saved = _eval_model_cfg()
    live = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    live.init_device = "cuda"
    live.actor_checkpoint = "/mounted/fastwam.pt"
    live.model_path = "/mounted/fastwam.pt"
    live.fastwam.load_text_encoder = False
    live.runtime.text_embedding_cache_dir = "/cache/text"
    live.critic.load_for_eval = False
    live.critic.backbone.model_path = ""
    live.critic.backbone_checkpoint_sha256 = ""
    live.gate_epsilon = 0.0
    live.eval_routing_mode = "matched_random"
    live.eval_idm_threshold = 0.25
    live.eval_random_idm_probability = 0.75
    live.eval_routing_seed = 17
    live.eval_microbatch_size = 2
    live.eval_timing_cuda_synchronize = True
    live.eval_without_critic = True

    expected = MODULE.build_fastwam_eval_model_contract(saved, load_critic=False)
    actual = MODULE.build_fastwam_eval_model_contract(live, load_critic=False)

    assert actual == expected
    assert "critic" not in actual["model"]
    assert "eval_without_critic" not in actual["model"]
    assert "eval_timing_cuda_synchronize" not in actual["model"]
    assert "gate_temperature" in actual["model"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("gate.layer_taps.mode", "indices"),
        ("gate.share_blocks", True),
        ("gate.denoise_last_n", 2),
        ("gate.hidden_dim", 128),
        ("gate.ffn_multiplier", 2),
        ("uncond_lora.rank", 8),
        ("uncond_lora.alpha", 8.0),
        ("uncond_lora.target_groups", ["ffn"]),
        ("fastwam.action_dit_config.num_layers", 29),
        ("gate_temperature", 0.5),
        ("runtime.generation_horizon", 16),
        ("runtime.execution_horizon", 8),
        ("runtime.num_video_frames", 33),
        ("runtime.reset_wait_steps", 15),
        ("runtime.max_episode_steps", 512),
        ("runtime.num_inference_steps", 20),
        ("runtime.seeded_noise_device", "model"),
        ("runtime.camera_resize_mode", "torch_bilinear"),
    ],
)
def test_eval_model_contract_rejects_structural_differences(path, value) -> None:
    saved = _eval_model_cfg()
    live = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    OmegaConf.update(live, path, value, merge=False)

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        MODULE.validate_fastwam_eval_model_contract(
            saved,
            live,
            load_critic=False,
        )


def test_eval_model_contract_includes_critic_when_requested() -> None:
    saved = _eval_model_cfg()
    live = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    live.critic.load_for_eval = True
    live.critic.hidden_sizes = [512, 256]

    with pytest.raises(ValueError, match=r"critic\.hidden_sizes"):
        MODULE.validate_fastwam_eval_model_contract(
            saved,
            live,
            load_critic=True,
        )


def test_eval_checkpoint_contract_validates_parent_hashes_before_construction() -> None:
    saved = _eval_model_cfg()
    live = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    live.critic.load_for_eval = True
    payload = {
        "schema": "fastwam-adaptive-rl-checkpoint-v1",
        "parent_checkpoint_sha256": "a" * 64,
        "critic_parent_checkpoint_sha256": "b" * 64,
        "contract": {
            "model": OmegaConf.to_container(saved, resolve=True),
        },
    }

    contract = MODULE.validate_fastwam_eval_checkpoint_contract(
        payload,
        live,
        expected_parent_checkpoint_sha256="a" * 64,
        load_critic=True,
    )
    assert contract["model"]["critic"]["backbone_checkpoint_sha256"] == "b" * 64

    wrong_outer_critic = dict(payload)
    wrong_outer_critic["critic_parent_checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="pi0.5 evaluation checkpoint parent"):
        MODULE.validate_fastwam_eval_checkpoint_contract(
            wrong_outer_critic,
            live,
            expected_parent_checkpoint_sha256="a" * 64,
            load_critic=True,
        )

    wrong_contract_critic = {
        **payload,
        "contract": {
            "model": {
                **payload["contract"]["model"],
                "critic": {
                    **payload["contract"]["model"]["critic"],
                    "backbone_checkpoint_sha256": "c" * 64,
                },
            }
        },
    }
    with pytest.raises(ValueError, match="wrong critic parent hash"):
        MODULE.validate_fastwam_eval_checkpoint_contract(
            wrong_contract_critic,
            live,
            expected_parent_checkpoint_sha256="a" * 64,
            load_critic=True,
        )


def test_p6_eval_contract_rejects_outer_v1_and_accepts_v2_schema() -> None:
    saved = _eval_model_cfg()
    saved.uncond_visual_sidecar = {
        "enabled": True,
        "type": "dinov3_router_wan_value",
        "transport": {"contract_sha256": "c" * 64},
    }
    live = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
    payload = {
        "schema": "fastwam-adaptive-rl-checkpoint-v1",
        "parent_checkpoint_sha256": "a" * 64,
        "contract": {"model": OmegaConf.to_container(saved, resolve=True)},
    }

    with pytest.raises(ValueError, match="Unsupported FastWAM adaptive evaluation"):
        MODULE.validate_fastwam_eval_checkpoint_contract(
            payload,
            live,
            expected_parent_checkpoint_sha256="a" * 64,
            load_critic=False,
        )

    payload["schema"] = "fastwam-adaptive-rl-checkpoint-v2-p6"
    contract = MODULE.validate_fastwam_eval_checkpoint_contract(
        payload,
        live,
        expected_parent_checkpoint_sha256="a" * 64,
        load_critic=False,
    )
    assert contract["model"]["uncond_visual_sidecar"]["enabled"] is True
