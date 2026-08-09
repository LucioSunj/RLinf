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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "examples/embodiment/config"
BUILDER_ROOT = REPO_ROOT / "rlinf/models/embodiment/wam_policy"


def _load_builder_package():
    package_name = "fastwam_builder_under_test"
    spec = importlib.util.spec_from_file_location(
        package_name,
        BUILDER_ROOT / "__init__.py",
        submodule_search_locations=[str(BUILDER_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


_builder = _load_builder_package()


def _critic_config():
    return OmegaConf.create(
        {
            "input_dim": 2048,
            "hidden_sizes": [1024, 512, 256],
            "backbone": {
                "add_value_head": False,
                "strict_vlm_checkpoint": True,
                "openpi": {
                    "config_name": "pi05_libero",
                    "add_value_head": False,
                },
            },
        }
    )


def test_builder_rejects_non_exact_or_pretrained_critic_head() -> None:
    config = _critic_config()
    _builder._validate_exact_pi05_critic_config(config)

    config.backbone.openpi.add_value_head = True
    with pytest.raises(ValueError, match="without a value head"):
        _builder._validate_exact_pi05_critic_config(config)

    config = _critic_config()
    config.hidden_sizes = [512, 256]
    with pytest.raises(ValueError, match="2048 -> 1024"):
        _builder._validate_exact_pi05_critic_config(config)

    config = _critic_config()
    config.backbone.strict_vlm_checkpoint = False
    with pytest.raises(ValueError, match="strict_vlm_checkpoint"):
        _builder._validate_exact_pi05_critic_config(config)

    config = _critic_config()
    del config.backbone.strict_vlm_checkpoint
    with pytest.raises(ValueError, match="strict_vlm_checkpoint"):
        _builder._validate_exact_pi05_critic_config(config)


def test_builder_skips_critic_allocation_contract_in_standalone_eval(
    monkeypatch,
) -> None:
    def _unexpected_call(*_args, **_kwargs):
        pytest.fail("standalone evaluation must not inspect or load the critic")

    monkeypatch.setattr(
        _builder, "_validate_exact_pi05_critic_config", _unexpected_call
    )
    monkeypatch.setattr(_builder, "_validate_critic_parent_artifact", _unexpected_call)

    assert not _builder._validate_critic_build_config(
        OmegaConf.create({"eval_without_critic": True})
    )


def test_builder_actor_surface_fails_before_distributed_launch() -> None:
    complete = SimpleNamespace(
        action_expert=object(),
        mot=object(),
        infer_action_scheduler=object(),
        infer_video_scheduler=object(),
        vae=object(),
        load_checkpoint=lambda _path: None,
    )
    _builder._validate_fastwam_actor_surface(complete)

    with pytest.raises(TypeError, match="infer_video_scheduler"):
        _builder._validate_fastwam_actor_surface(
            SimpleNamespace(
                action_expert=object(),
                mot=object(),
                infer_action_scheduler=object(),
                vae=object(),
                load_checkpoint=lambda _path: None,
            )
        )


def test_fastwam_parent_payload_requires_complete_mot_and_proprio() -> None:
    actor = SimpleNamespace(mot=nn.Linear(2, 2), proprio_encoder=nn.Linear(2, 3))
    payload = {
        "mot": actor.mot.state_dict(),
        "proprio_encoder": actor.proprio_encoder.state_dict(),
    }
    _builder._validate_fastwam_parent_payload(actor, payload)

    incomplete = {"mot": {"weight": payload["mot"]["weight"]}}
    with pytest.raises(ValueError, match="MoT key mismatch"):
        _builder._validate_fastwam_parent_payload(actor, incomplete)

    with pytest.raises(ValueError, match="missing `proprio_encoder`"):
        _builder._validate_fastwam_parent_payload(
            actor,
            {"mot": actor.mot.state_dict()},
        )


def test_checkpoint_artifact_hash_is_content_bound(tmp_path) -> None:
    checkpoint = tmp_path / "critic.pt"
    checkpoint.write_bytes(b"critic-v1")
    first = _builder._sha256_artifact(checkpoint)
    assert first == hashlib.sha256(b"critic-v1").hexdigest()
    checkpoint.write_bytes(b"critic-v2")
    second = _builder._sha256_artifact(checkpoint)
    assert first != second

    directory = tmp_path / "critic-dir"
    directory.mkdir()
    (directory / "weights.bin").write_bytes(b"weights")
    directory_hash = _builder._sha256_artifact(directory)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    assert _builder._sha256_artifact(directory) != directory_hash


def test_eval_checkpoint_resolver_accepts_actor_directory(tmp_path) -> None:
    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    rank_zero = actor_dir / "rank_0.pt"
    rank_one = actor_dir / "rank_1.pt"
    rank_zero.write_bytes(b"rank-zero")
    rank_one.write_bytes(b"rank-one")

    assert (
        _builder.resolve_fastwam_adaptive_eval_checkpoint(
            actor_dir,
            rank=1,
        )
        == rank_one
    )
    assert (
        _builder.resolve_fastwam_adaptive_eval_checkpoint(
            actor_dir,
            rank=7,
        )
        == rank_zero
    )
    assert (
        _builder.resolve_fastwam_adaptive_eval_checkpoint(
            rank_one,
            rank=0,
        )
        == rank_one
    )


def test_builder_enforces_non_joint_positive_flow_sde() -> None:
    config = OmegaConf.create(
        {
            "enabled": True,
            "joint_logprob": False,
            "denoise_index_sampling": "uniform",
            "noise_level": 0.5,
            "ignore_last_transition": True,
        }
    )
    _builder._validate_flow_sde_config(config)

    config.ignore_last_transition = False
    _builder._validate_flow_sde_config(config)
    config.ignore_last_transition = True

    config.joint_logprob = True
    with pytest.raises(ValueError, match="joint_logprob"):
        _builder._validate_flow_sde_config(config)
    config.joint_logprob = False
    config.noise_level = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        _builder._validate_flow_sde_config(config)
    config.noise_level = float("nan")
    with pytest.raises(ValueError, match="strictly positive"):
        _builder._validate_flow_sde_config(config)


def test_libero_adaptive_config_composes_with_confirmed_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT_SHA256", "b" * 64)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="libero_10_ppo_fastwam_adaptive")

    assert cfg.actor.model.model_type == "fastwam_adaptive"
    assert cfg.rollout.model.model_type == "fastwam_adaptive"
    assert cfg.actor.model.model_path == "/tmp/fastwam.pt"
    assert cfg.actor.model.is_lora is False
    assert cfg.actor.model.add_value_head is True
    assert cfg.actor.fsdp_config.use_orig_params is True
    assert cfg.actor.fsdp_config.ignore_frozen_parameters is True
    assert cfg.weight_syncer.patch.init_sync.enabled is True
    assert cfg.actor.model.gate.layer_taps.mode == "all"
    assert cfg.actor.model.gate.denoise_last_n == 1
    assert cfg.actor.model.gate_epsilon == 0.1
    assert cfg.actor.model.eval_routing_mode == "learned_threshold"
    assert cfg.actor.model.eval_idm_threshold == 0.5
    assert cfg.actor.model.eval_random_idm_probability is None
    assert cfg.actor.model.eval_routing_seed == 0
    assert cfg.actor.model.eval_microbatch_size == 1
    assert cfg.actor.model.kv_replay.backend == "stored"
    assert cfg.actor.model.uncond_lora.target_groups == [
        "self_attention_qkvo",
        "cross_attention_qkvo",
        "ffn",
    ]
    assert cfg.actor.model.flow_sde.noise_level == 0.5
    assert cfg.actor.model.flow_sde.joint_logprob is False
    assert cfg.actor.model.flow_sde.ignore_last_transition is True
    assert cfg.algorithm.fixed_branch_cost.idm_cost == 0.01
    assert cfg.env.eval.reward_coef == cfg.algorithm.reward_coef
    assert cfg.algorithm.fixed_branch_cost.uncond_cost == 0.0
    assert cfg.env.train.use_step_penalty is False
    assert cfg.actor.model.critic.kind == "pi0_5_value_after_vlm"
    assert cfg.actor.model.critic.backbone_checkpoint_sha256 == "b" * 64
    assert cfg.actor.model.critic.load_for_eval is False
    assert cfg.actor.model.critic.backbone.add_value_head is False
    assert cfg.actor.model.critic.backbone.strict_vlm_checkpoint is True
    assert cfg.actor.model.critic.backbone.openpi.add_value_head is False
    assert cfg.actor.model.runtime.prompt_template.endswith("{task}")
    _builder._validate_exact_pi05_critic_config(cfg.actor.model.critic)


def test_p8_readiness_uses_only_the_pinned_text_cache(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT_SHA256", "b" * 64)
    monkeypatch.setenv("FASTWAM_DINOV3_SOURCE_ROOT", "/tmp/dinov3")
    monkeypatch.setenv("FASTWAM_DINOV3_WEIGHTS", "/tmp/dinov3.safetensors")
    monkeypatch.setenv("FASTWAM_TEXT_EMBEDDING_CACHE", "/tmp/text-cache")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="libero_10_ppo_fastwam_adaptive_p8_readiness")

    assert cfg.actor.model.fastwam.load_text_encoder is False
    assert cfg.rollout.model.fastwam.load_text_encoder is False
    assert cfg.actor.model.runtime.text_embedding_cache_dir == "/tmp/text-cache"
    assert cfg.rollout.model.runtime.text_embedding_cache_dir == "/tmp/text-cache"


def test_p8_stage2_systems_profile_passes_production_config_guard(monkeypatch) -> None:
    from rlinf.config import _validate_fastwam_adaptive_cfg

    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT_SHA256", "b" * 64)
    monkeypatch.setenv("FASTWAM_DINOV3_SOURCE_ROOT", "/tmp/dinov3")
    monkeypatch.setenv("FASTWAM_DINOV3_WEIGHTS", "/tmp/dinov3.safetensors")
    monkeypatch.setenv("FASTWAM_TEXT_EMBEDDING_CACHE", "/tmp/text-cache")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="libero_10_ppo_fastwam_adaptive_p8_readiness")
    with open_dict(cfg):
        cfg.runner.p8_readiness_endpoint = False
        cfg.runner.p8_stage2_systems_endpoint = True
        cfg.runner.max_steps = 1
        cfg.runner.l11_route_seed = 20260801
        cfg.actor.optim.total_training_steps = 1
        cfg.actor.seed = 20260731
        cfg.actor.global_batch_size = 4
        cfg.env.train.seed = 20260801
        cfg.env.train.total_num_envs = 2
        cfg.env.train.specific_reset_id = 0

    _validate_fastwam_adaptive_cfg(cfg, only_eval=False)
    cfg.runner.p8_readiness_endpoint = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_fastwam_adaptive_cfg(cfg, only_eval=False)


def test_hydra_overrides_select_recompute_and_gate_subsets(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT_SHA256", "b" * 64)

    overrides = []
    for owner in ("actor", "rollout"):
        overrides.extend(
            (
                f"{owner}.model.kv_replay.backend=recompute",
                f"{owner}.model.gate.layer_taps.mode=last_n",
                f"{owner}.model.gate.layer_taps.last_n=4",
                f"{owner}.model.eval_routing_mode=matched_random",
                f"{owner}.model.eval_random_idm_probability=0.375",
                f"{owner}.model.eval_routing_seed=19",
            )
        )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(
            config_name="libero_10_ppo_fastwam_adaptive",
            overrides=overrides,
        )

    assert cfg.actor.model.kv_replay.backend == "recompute"
    assert cfg.rollout.model.kv_replay.backend == "recompute"
    assert cfg.actor.model.gate.layer_taps.mode == "last_n"
    assert cfg.actor.model.gate.layer_taps.last_n == 4
    assert cfg.actor.model.eval_routing_mode == "matched_random"
    assert cfg.rollout.model.eval_routing_mode == "matched_random"
    assert cfg.actor.model.eval_random_idm_probability == 0.375
    assert cfg.rollout.model.eval_random_idm_probability == 0.375
    assert cfg.actor.model.eval_routing_seed == 19
    assert cfg.rollout.model.eval_routing_seed == 19


def test_standalone_eval_config_resolves_without_pi05_environment(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.delenv("PI05_CRITIC_CHECKPOINT", raising=False)
    monkeypatch.delenv("PI05_CRITIC_CHECKPOINT_SHA256", raising=False)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="libero_10_ppo_fastwam_adaptive")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["rollout"]["model"]["critic"]["backbone_checkpoint_sha256"] == ""
    assert resolved["rollout"]["model"]["critic"]["backbone"]["model_path"] == ""
