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

import hashlib
import importlib.util
import json
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


def test_p8_readiness_gate_freeze_is_explicit_and_exact() -> None:
    MODULE.validate_p8_readiness_gate_ownership(
        p8_enabled=True,
        gate_trainable=False,
        readiness_endpoint=True,
        gate_lr=0.0,
        gate_loss_weight=0.0,
    )
    MODULE.validate_p8_readiness_gate_ownership(
        p8_enabled=False,
        gate_trainable=True,
        readiness_endpoint=False,
        gate_lr=3e-5,
        gate_loss_weight=1.0,
    )

    with pytest.raises(ValueError, match="restricted"):
        MODULE.validate_p8_readiness_gate_ownership(
            p8_enabled=True,
            gate_trainable=False,
            readiness_endpoint=False,
            gate_lr=0.0,
            gate_loss_weight=0.0,
        )
    with pytest.raises(ValueError, match="exact zero"):
        MODULE.validate_p8_readiness_gate_ownership(
            p8_enabled=True,
            gate_trainable=False,
            readiness_endpoint=True,
            gate_lr=1e-8,
            gate_loss_weight=0.0,
        )


def test_p8_frozen_gate_endpoints_are_explicit_and_mutually_exclusive() -> None:
    MODULE.validate_p8_readiness_gate_ownership(
        p8_enabled=True,
        gate_trainable=False,
        readiness_endpoint=False,
        stage2_systems_endpoint=True,
        gate_lr=0.0,
        gate_loss_weight=0.0,
    )
    MODULE.validate_p8_readiness_gate_ownership(
        p8_enabled=True,
        gate_trainable=False,
        readiness_endpoint=False,
        formal_stage2_endpoint=True,
        gate_lr=0.0,
        gate_loss_weight=0.0,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE.validate_p8_readiness_gate_ownership(
            p8_enabled=True,
            gate_trainable=False,
            readiness_endpoint=True,
            stage2_systems_endpoint=True,
            gate_lr=0.0,
            gate_loss_weight=0.0,
        )
    with pytest.raises(ValueError, match="frozen-Gate"):
        MODULE.validate_p8_readiness_gate_ownership(
            p8_enabled=True,
            gate_trainable=True,
            readiness_endpoint=False,
            stage2_systems_endpoint=True,
            gate_lr=3e-5,
            gate_loss_weight=1.0,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        MODULE.validate_p8_readiness_gate_ownership(
            p8_enabled=True,
            gate_trainable=False,
            readiness_endpoint=False,
            stage2_systems_endpoint=True,
            formal_stage2_endpoint=True,
            gate_lr=0.0,
            gate_loss_weight=0.0,
        )


def test_default_fastwam_model_does_not_change_v1_gate_contract() -> None:
    model_path = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/model/fastwam_adaptive.yaml"
    )
    model = OmegaConf.load(model_path)

    assert "gate_trainable" not in model


def test_p8_readiness_endpoint_rejects_identity_or_run_length_drift() -> None:
    values = {
        "max_steps": 2,
        "max_epochs": 1,
        "actor_total_training_steps": 2,
        "actor_seed": 424242,
        "global_batch_size": 1,
        "env_seed": 424242,
        "total_num_envs": 1,
        "task_id_filter": [0],
        "specific_reset_id": 1,
        "use_fixed_reset_state_ids": True,
        "training_route_override": "forced_uncond_after_initial",
        "load_text_encoder": False,
        "formal_training_authorized": False,
        "final_ledger_path": None,
    }
    MODULE.validate_p8_readiness_endpoint_contract(**values)

    for key, bad_value in (
        ("max_steps", 3),
        ("specific_reset_id", 0),
        ("task_id_filter", [1]),
        ("training_route_override", "none"),
        ("load_text_encoder", True),
        ("formal_training_authorized", True),
        ("final_ledger_path", "/forbidden/final_ledger.json"),
    ):
        invalid = dict(values)
        invalid[key] = bad_value
        with pytest.raises(ValueError, match="two-step B0/B1"):
            MODULE.validate_p8_readiness_endpoint_contract(**invalid)


def test_p8_stage2_systems_endpoint_is_distinct_and_fail_closed() -> None:
    values = {
        "max_steps": 1,
        "max_epochs": 1,
        "actor_total_training_steps": 1,
        "actor_seed": 20260731,
        "global_batch_size": 4,
        "env_seed": 20260801,
        "total_num_envs": 2,
        "task_id_filter": [0],
        "specific_reset_id": 0,
        "use_fixed_reset_state_ids": True,
        "training_route_override": "forced_uncond_after_initial",
        "load_text_encoder": False,
        "formal_training_authorized": False,
        "final_ledger_path": None,
        "replay_backend": "stored_native",
        "compile_enabled": False,
        "route_seed": 20260801,
    }
    MODULE.validate_p8_stage2_systems_endpoint_contract(**values)

    for key, bad_value in (
        ("max_steps", 2),
        ("global_batch_size", 1),
        ("env_seed", 424242),
        ("specific_reset_id", 1),
        ("total_num_envs", 4),
        ("replay_backend", "recompute_native"),
        ("compile_enabled", True),
        ("formal_training_authorized", True),
    ):
        invalid = dict(values)
        invalid[key] = bad_value
        with pytest.raises(ValueError, match="one-update reset-0"):
            MODULE.validate_p8_stage2_systems_endpoint_contract(**invalid)


def _write_p8_authorization(output_root: Path) -> str:
    output_root.mkdir(parents=True)
    stop_rules = list(MODULE.P8_FORMAL_STOP_RULES)
    payload = {
        "schema": "fastwam-p8-formal-training-authorization-v1",
        "status": "READY-AUTHORIZED",
        "candidate": "P8-A0/KV",
        "formal_training_authorized": True,
        "final_ledger_used": False,
        "output_root": str(output_root),
        "authorized_budget": {
            "runner_steps": 100,
            "optimizer_updates_per_runner_step": 10,
            "optimizer_updates": 1000,
            "environments": 4,
            "global_batch_size": 28,
            "seed": 42,
        },
        "candidate_evidence_sha256": {
            "p8": "8" * 64,
            "p6": "6" * 64,
            "p7": "7" * 64,
        },
        "code_revisions": {
            "outer": "a" * 40,
            "FastWAM": "b" * 40,
            "RLinf": "c" * 40,
        },
        "authorization_text_sha256": "d" * 64,
        "formal_config_sha256": "e" * 64,
        "asset_manifest_sha256": "f" * 64,
        "stop_rules": stop_rules,
        "stop_rules_sha256": MODULE._canonical_sha256(stop_rules),
        "resource_caps": {
            "gpu_used_bytes_per_device": 38 * 1024**3,
            "process_tree_rss_bytes": 128 * 1024**3,
            "output_bytes": 16 * 1024**3,
            "minimum_free_fraction": 0.20,
        },
    }
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    (output_root / "formal_training_authorization.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _p8_formal_contract_values(output_root: Path) -> dict[str, object]:
    authorization_sha256 = _write_p8_authorization(output_root)
    output_root_value = str(output_root)
    return {
        "max_steps": 100,
        "max_epochs": 100,
        "save_interval": 10,
        "optimizer_updates_per_runner_step": 10,
        "actor_total_training_steps": 1000,
        "actor_seed": 42,
        "micro_batch_size": 1,
        "global_batch_size": 28,
        "env_seed": 42,
        "total_num_envs": 4,
        "task_id_filter": None,
        "specific_reset_id": None,
        "use_fixed_reset_state_ids": True,
        "training_route_override": "forced_uncond_after_initial",
        "preserve_fixed_route_across_actor_updates": True,
        "load_text_encoder": False,
        "formal_training_authorized": True,
        "authorization_record_path": (
            f"{output_root_value}/formal_training_authorization.json"
        ),
        "authorization_record_sha256": authorization_sha256,
        "final_ledger_path": None,
        "replay_backend": "stored_native",
        "compile_enabled": False,
        "update_epoch": 1,
        "fixed_branch_cost_enabled": True,
        "fixed_branch_idm_cost": 0.1,
        "fixed_branch_uncond_cost": 0.0,
        "fixed_idm_differential_cost": 0.1,
        "runtime_cost_profile_sha256": (MODULE.P8_FORMAL_RUNTIME_COST_PROFILE_SHA256),
        "correction_bound": 1.0,
        "bound_semantics": "fixed_alpha_upper_bound",
        "precision": "bf16",
        "storage_dtype": "bfloat16",
        "refiner_layer_indices": [12],
        "refiner_query_rank": 32,
        "refiner_output_rank": 32,
        "refiner_temperature": 0.07,
        "refiner_alpha": 1.0,
        "lora_lr": 1.0e-5,
        "refiner_lr": 1.0e-5,
        "refiner_weight_decay": 0.0,
        "value_lr": 1.0e-4,
        "component_placement": {"actor": "0-1", "env,rollout": "2-3"},
        "output_root": output_root_value,
        "formal_stage2_mode": "training",
        "checkpoint_path": f"{output_root_value}/step_zero/actor",
        "bootstrap_checkpoint_dir": None,
        "resume_dir": None,
        "checkpoint_keep_last": 2,
        "checkpoint_atomic": True,
        "training_guard_enabled": False,
        "formal_action_audit": True,
    }


def test_p8_formal_stage2_endpoint_locks_the_authorized_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_parent = tmp_path / "p8-formal-training"
    monkeypatch.setattr(MODULE, "P8_FORMAL_OUTPUT_PARENT", output_parent)
    values = _p8_formal_contract_values(
        output_parent / "p8_a0_kv_stage2_seed42_20260810T010203Z_v1"
    )
    MODULE.validate_p8_formal_stage2_endpoint_contract(**values)

    for key, bad_value in (
        ("max_steps", 101),
        ("save_interval", 20),
        ("optimizer_updates_per_runner_step", 9),
        ("global_batch_size", 4),
        ("task_id_filter", [0]),
        ("specific_reset_id", 0),
        ("preserve_fixed_route_across_actor_updates", False),
        ("formal_training_authorized", False),
        ("authorization_record_path", "relative/authorization.json"),
        ("authorization_record_sha256", "bad"),
        ("replay_backend", "recompute_native"),
        ("compile_enabled", True),
        ("fixed_branch_cost_enabled", False),
        ("fixed_branch_idm_cost", 0.2),
        ("fixed_branch_uncond_cost", 0.1),
        ("fixed_idm_differential_cost", 0.2),
        ("runtime_cost_profile_sha256", "0" * 64),
        ("correction_bound", 2.0),
        ("bound_semantics", "dynamic_alpha"),
        ("refiner_layer_indices", [12, 21]),
        ("lora_lr", 3.0e-5),
        ("component_placement", {"actor,env,rollout": "all"}),
        ("checkpoint_keep_last", 3),
        ("checkpoint_atomic", False),
        ("formal_action_audit", False),
        ("final_ledger_path", "/forbidden/final_ledger.json"),
    ):
        invalid = dict(values)
        invalid[key] = bad_value
        with pytest.raises(ValueError, match="authorized 1000-update"):
            MODULE.validate_p8_formal_stage2_endpoint_contract(**invalid)


def test_p8_formal_step_zero_export_is_separate_from_training_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_parent = tmp_path / "p8-formal-training"
    monkeypatch.setattr(MODULE, "P8_FORMAL_OUTPUT_PARENT", output_parent)
    values = _p8_formal_contract_values(
        output_parent / "p8_a0_kv_stage2_seed42_20260810T010203Z_v1"
    )
    output_root = str(values["output_root"])
    values.update(
        {
            "formal_stage2_mode": "step_zero_export",
            "checkpoint_path": None,
            "bootstrap_checkpoint_dir": f"{output_root}/step_zero",
        }
    )
    MODULE.validate_p8_formal_stage2_endpoint_contract(**values)

    values["checkpoint_path"] = f"{output_root}/step_zero/actor"
    with pytest.raises(ValueError, match="step-zero export"):
        MODULE.validate_p8_formal_stage2_endpoint_contract(**values)


def test_p8_formal_authorization_record_is_content_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_parent = tmp_path / "p8-formal-training"
    monkeypatch.setattr(MODULE, "P8_FORMAL_OUTPUT_PARENT", output_parent)
    values = _p8_formal_contract_values(
        output_parent / "p8_a0_kv_stage2_seed42_20260810T010203Z_v1"
    )
    record = Path(str(values["authorization_record_path"]))
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["candidate_evidence_sha256"]["p6"] = "forged"
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    record.write_bytes(raw)
    values["authorization_record_sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(ValueError, match="evidence SHA-256"):
        MODULE.validate_p8_formal_stage2_endpoint_contract(**values)


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


def test_p8_checkpoint_contract_binds_frozen_endpoint_without_changing_v1() -> None:
    cfg = _checkpoint_cfg()
    v1 = MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2)
    assert "p8_stage2_systems_endpoint" not in v1["runner"]

    cfg.actor.model.uncond_visual_sidecar = {
        "enabled": True,
        "compile": False,
        "replay": {"backend": "stored_native"},
    }
    cfg.runner.p8_readiness_endpoint = False
    cfg.runner.p8_stage2_systems_endpoint = True
    cfg.runner.formal_training_authorized = False
    p8 = MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2)

    assert p8["runner"] == {
        **v1["runner"],
        "p8_readiness_endpoint": False,
        "p8_stage2_systems_endpoint": True,
        "p8_formal_stage2_endpoint": False,
        "formal_training_authorized": False,
        "formal_training_authorization_record": None,
        "formal_training_authorization_sha256": None,
    }
    cfg.runner.p8_stage2_systems_endpoint = False
    assert MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2) != p8

    cfg.runner.p8_formal_stage2_endpoint = True
    cfg.runner.formal_training_authorized = True
    cfg.runner.formal_training_authorization_record = "/tmp/authorization.json"
    cfg.runner.formal_training_authorization_sha256 = "c" * 64
    cfg.runner.formal_optimizer_updates_per_runner_step = 10
    cfg.runner.p8_formal_action_audit = True
    formal = MODULE.build_fastwam_checkpoint_contract(cfg, world_size=2)
    assert formal["runner"]["p8_formal_stage2_endpoint"] is True
    assert formal["runner"]["formal_training_authorization_sha256"] == "c" * 64
    assert formal["runner"]["formal_optimizer_updates_per_runner_step"] == 10
    assert formal["runner"]["p8_formal_action_audit"] is True


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
