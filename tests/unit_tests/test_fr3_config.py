# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native Hydra FR3 task bindings reject incompatible model and budget inputs."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.models.embodiment.wam_policy.fr3_config import (
    validate_fr3_model_config,
    validate_fr3_training_config,
)

CONFIGS = Path(__file__).resolve().parents[2] / "examples/embodiment/config"


@pytest.fixture(
    params=[("drawer", 0, 16), ("insert_tubes", 1, 16), ("insert_tubes", 1, 32)]
)
def config(request, tmp_path):
    task, task_id, horizon = request.param
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name=f"realworld_fr3_fastwam_{task}"
            + ("_h32" if horizon == 32 else "")
        )
    assert cfg.fr3.ready is False
    assert cfg.fr3.data_contract_confirmed is False
    assert cfg.fr3.task_id == task_id
    for key, value in {
        "device": "cpu",
        "updates": 3,
        "warmup_updates": 1,
        "max_episode_actions": 35,
        "max_episode_seconds": 60.0,
        "output_dir": str(tmp_path / "run"),
        "statistics": str(tmp_path / "stats.json"),
        "training_config": str(tmp_path / "training.yaml"),
        "parent_checkpoint": str(tmp_path / "parent.pt"),
        "parent_checkpoint_sha256": "a" * 64,
    }.items():
        cfg.fr3[key] = value
    model = cfg.actor.model
    for key, value in {
        "actor_checkpoint": cfg.fr3.parent_checkpoint,
        "actor_checkpoint_sha256": cfg.fr3.parent_checkpoint_sha256,
        "model_path": cfg.fr3.parent_checkpoint,
        "runtime.processor_stats_path": cfg.fr3.statistics,
        "runtime.training_config": cfg.fr3.training_config,
        "runtime.easywam_root": str(tmp_path / "EasyWAM"),
        "runtime.task_name": task,
        "runtime.task_prompt": cfg.fr3.prompt,
    }.items():
        OmegaConf.update(model, key, value)
    cfg.algorithm.fixed_branch_cost.task_budget = {
        "profile": "fr3",
        "suite": "fr3",
        "targets": {str(task_id): 0.375},
        "task_names": {str(task_id): task},
        "calibration": {
            "status": "COMPLETE",
            "parameters": {
                "awe_threshold": 0.005,
                "sail_radius_m": 0.04,
                "sail_min_samples": 3,
                "execution_horizon": horizon,
            },
            "source": str(tmp_path / "budget_targets.json"),
            "data_version": "synthetic-fixture-v1",
        },
    }
    return cfg


def test_hydra_profiles_require_confirmation_and_keep_fr3_dimensions(config):
    validate_fr3_model_config(config.actor.model)
    with pytest.raises(ValueError, match="NOT_READY"):
        validate_fr3_training_config(config)
    config.fr3.data_contract_confirmed = True
    config.fr3.ready = True
    validate_fr3_training_config(config)
    assert config.actor.model.runtime.max_episode_steps == 35
    assert config.actor.model.runtime.processor is None
    assert config.actor.model.critic.backbone.action_dim == 8
    assert (
        config.actor.model.critic.backbone.num_action_chunks
        == config.fr3.action_horizon
    )
    assert config.actor.model.critic.backbone.openpi_data.profile == "fr3"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("actor.model.action_dim", 7, "action_dim"),
        ("actor.model.runtime.execution_horizon", 10, "execution_horizon"),
        ("actor.model.uncond_lora.rank", 16, "uncond_lora.rank"),
        ("actor.model.video_lora.alpha", 16, "video_lora.alpha"),
        ("actor.model.gate.current_mode_embedding", True, "current_mode_embedding"),
        ("actor.model.actor_checkpoint", "/wrong/parent.pt", "actor_checkpoint"),
        (
            "actor.model.runtime.processor_stats_path",
            "/wrong/stats.json",
            "processor_stats_path",
        ),
        ("actor.model.runtime.task_prompt", "wrong task", "task_prompt"),
        ("actor.model.route_neutral_online.critic_warmup.runner_updates", 2, "warm-up"),
        ("env.train.num_envs", 2, "single task-bound"),
        ("env.train.task_id_filter", [0, 1], "task_id_filter"),
        ("fr3.updates", 0, "updates"),
        ("env.train.execution_horizon", 8, "horizons"),
        ("fr3.action_horizon", 8, "horizons"),
        ("actor.model.critic.backbone.num_action_chunks", 8, "critic action shape"),
        ("fr3.max_episode_seconds", float("inf"), "max_episode_seconds"),
        (
            "algorithm.fixed_branch_cost.task_budget.calibration.status",
            "NOT_READY",
            "COMPLETE",
        ),
        (
            "algorithm.fixed_branch_cost.task_budget.calibration.parameters.sail_radius_m",
            0.02,
            "SAIL2.5/3/4cm",
        ),
    ],
)
def test_hydra_profiles_reject_contract_mismatch(config, key, value, message):
    config.fr3.data_contract_confirmed = True
    config.fr3.ready = True
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match=message):
        validate_fr3_training_config(config)


def test_hydra_profiles_reject_another_valid_task_budget(config):
    config.fr3.data_contract_confirmed = True
    config.fr3.ready = True
    other_id = 1 - config.fr3.task_id
    other_name = {0: "drawer", 1: "insert_tubes"}[other_id]
    config.fr3.task_id = other_id
    config.algorithm.fixed_branch_cost.task_budget.targets = {str(other_id): 0.375}
    config.algorithm.fixed_branch_cost.task_budget.task_names = {
        str(other_id): other_name
    }
    with pytest.raises(ValueError, match="task_id"):
        validate_fr3_training_config(config)


def test_budget_horizon_cannot_cross_h16_h32(config):
    config.fr3.ready = config.fr3.data_contract_confirmed = True
    parameters = config.algorithm.fixed_branch_cost.task_budget.calibration.parameters
    parameters.execution_horizon = 32 if config.fr3.action_horizon == 16 else 16
    with pytest.raises(ValueError, match="execution horizon"):
        validate_fr3_training_config(config)


@pytest.mark.parametrize("limits", [(None, None), (35, None), (None, 60.0)])
def test_explicit_optional_episode_limits_match_env_and_runtime(config, limits):
    config.fr3.ready = config.fr3.data_contract_confirmed = True
    config.fr3.max_episode_actions, config.fr3.max_episode_seconds = limits
    validate_fr3_training_config(config)
    assert config.env.train.max_episode_actions == limits[0]
    assert config.env.train.max_episode_seconds == limits[1]
    assert config.actor.model.runtime.max_episode_steps == limits[0]


@pytest.mark.parametrize("key", ["max_episode_actions", "max_episode_seconds"])
@pytest.mark.parametrize("value", [0, -1, True, "null", float("inf"), float("nan")])
def test_episode_limits_reject_invalid_values(config, key, value):
    config.fr3.ready = config.fr3.data_contract_confirmed = True
    config.fr3[key] = value
    with pytest.raises(ValueError, match=key):
        validate_fr3_training_config(config)


@pytest.mark.parametrize("key", ["max_episode_actions", "max_episode_seconds"])
def test_unlimited_learner_rejects_missing_or_mismatched_limit(config, key):
    config.fr3.ready = config.fr3.data_contract_confirmed = True
    config.fr3[key] = None
    config.env.train[key] = 60
    with pytest.raises(ValueError, match=key):
        validate_fr3_training_config(config)
    with open_dict(config.fr3):
        del config.fr3[key]
    with pytest.raises(ValueError, match=key):
        validate_fr3_training_config(config)
