import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


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


def test_builder_enforces_non_joint_positive_flow_sde() -> None:
    config = OmegaConf.create(
        {
            "enabled": True,
            "joint_logprob": False,
            "denoise_index_sampling": "uniform",
            "noise_level": 0.5,
            "ignore_last_transition": False,
        }
    )
    _builder._validate_flow_sde_config(config)

    config.joint_logprob = True
    with pytest.raises(ValueError, match="joint_logprob"):
        _builder._validate_flow_sde_config(config)
    config.joint_logprob = False
    config.noise_level = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        _builder._validate_flow_sde_config(config)


def test_libero_adaptive_config_composes_with_confirmed_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="libero_10_ppo_fastwam_adaptive")

    assert cfg.actor.model.model_type == "fastwam_adaptive"
    assert cfg.rollout.model.model_type == "fastwam_adaptive"
    assert cfg.actor.model.model_path == "/tmp/fastwam.pt"
    assert cfg.actor.model.is_lora is False
    assert cfg.actor.model.add_value_head is True
    assert cfg.actor.fsdp_config.use_orig_params is True
    assert cfg.actor.model.gate.layer_taps.mode == "all"
    assert cfg.actor.model.gate.denoise_last_n == 1
    assert cfg.actor.model.gate_epsilon == 0.1
    assert cfg.actor.model.kv_replay.backend == "stored"
    assert cfg.actor.model.uncond_lora.target_groups == [
        "self_attention_qkvo",
        "cross_attention_qkvo",
        "ffn",
    ]
    assert cfg.actor.model.flow_sde.noise_level == 0.5
    assert cfg.actor.model.flow_sde.joint_logprob is False
    assert cfg.algorithm.fixed_branch_cost.idm_cost == 1.0
    assert cfg.algorithm.fixed_branch_cost.uncond_cost == 0.0
    assert cfg.env.train.use_step_penalty is False
    assert cfg.actor.model.critic.kind == "pi0_5_value_after_vlm"
    assert cfg.actor.model.critic.backbone.add_value_head is False
    assert cfg.actor.model.critic.backbone.openpi.add_value_head is False
    _builder._validate_exact_pi05_critic_config(cfg.actor.model.critic)


def test_hydra_overrides_select_recompute_and_gate_subsets(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("PI05_CRITIC_CHECKPOINT", "/tmp/pi05")

    overrides = []
    for owner in ("actor", "rollout"):
        overrides.extend(
            (
                f"{owner}.model.kv_replay.backend=recompute",
                f"{owner}.model.gate.layer_taps.mode=last_n",
                f"{owner}.model.gate.layer_taps.last_n=4",
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
