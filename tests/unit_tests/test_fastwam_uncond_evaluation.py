# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Exercise native UNCOND evaluation, deterministic resumption, and collection."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_fastwam_evaluation_collector import (
    _IdentityEnv,
    _ledger,
    _outcome,
    _rollout,
)
from test_fastwam_uncond_rl import _compose, _obs, _policy

from rlinf.models.embodiment.wam_policy.uncond_rl_config import (
    validate_uncond_rl_config,
)
from rlinf.runners.uncond_rl_evaluation import UncondRLEvalCollector


def test_eval_without_critic_uses_ledger_seeds_after_native_reload(monkeypatch):
    policy = _policy(monkeypatch)
    policy.set_global_step(200)
    payload = copy.deepcopy(policy.trainable_state_dict())
    policy.critic = None
    obs = _obs()
    obs["_fastwam_action_noise_seeds"] = torch.tensor([71, 93])
    expected, result = policy.predict_action_batch(obs, mode="eval")
    assert result["forward_inputs"] == {}
    assert not result["prev_values"].any()
    assert not result["emitted_gate"].valid.any()
    assert not result["evaluation_selection"].effective_next_route.any()
    restored = _policy(monkeypatch)
    restored.critic = None
    assert (
        restored.load_eval_checkpoint(
            {
                "schema": "fastwam-adaptive-rl-checkpoint-v1",
                "step": 200,
                "parent_checkpoint_sha256": "a" * 64,
                "contract": {"model": {"actor_checkpoint_sha256": "a" * 64}},
                "policy": payload,
            },
            expected_parent_checkpoint_sha256="a" * 64,
        )
        == 200
    )
    # Intervening global RNG use and a new collector episode counter must not
    # alter an episode's actions when its immutable noise identity is reused.
    torch.randn(73)
    actual, _ = restored.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    obs["_fastwam_action_noise_seeds"] += 1
    other, _ = restored.predict_action_batch(obs, mode="eval")
    assert not torch.equal(other, expected)


def test_compiled_action_view_preserves_video_lora_and_eval_actions(monkeypatch):
    """The compiled Action copy must retain the live dual-LoRA Video path."""
    policy = _policy(monkeypatch)
    policy.actor.action_expert.freqs = torch.ones(1)
    policy.actor.video_expert.freqs = (torch.ones(1),)
    policy.actor.vae = torch.nn.Identity()
    policy.actor.vae.scale = [torch.ones(1)]
    policy.eval()
    observation = {key: value[:1] for key, value in _obs().items()}
    observation["_fastwam_action_noise_seeds"] = torch.tensor([71])
    expected, _ = policy.predict_action_batch(observation, mode="eval")
    monkeypatch.setattr(torch, "compile", lambda function, **_kwargs: function)
    policy.enable_torch_compile()
    actual, _ = policy.predict_action_batch(observation, mode="eval")
    view = policy._compiled_inference.runtime.actor
    assert view.video_expert is policy.actor.video_expert
    assert view.mot.mixtures["video"] is policy.actor.video_expert
    assert view.action_expert is not policy.actor.action_expert
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_plus_eval_preset_accepts_perturbed_long_task_ids(monkeypatch):
    _compose(monkeypatch)
    for name in ("OUTPUT_DIR", "RUN_ID", "LEDGER"):
        monkeypatch.setenv(f"FASTWAM_EVAL_{name}", f"/unit/{name}")
    config_dir = Path(__file__).parents[2] / "evaluations/libero"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="libero_plus_long_fastwam_uncond_rl_eval")
    OmegaConf.resolve(cfg)
    OmegaConf.update(cfg, "env.eval.task_id_filter", [0, 252, 2518], force_add=True)
    validate_uncond_rl_config(cfg, only_eval=True)
    assert cfg.rollout.model.critic.load_for_eval is False
    assert cfg.rollout.model.eval_without_critic is True
    assert cfg.rollout.model.runtime.num_video_frames == 1
    assert "gate" not in cfg.rollout.model
    cfg.env.eval.task_suite_name = "libero_goal"
    with pytest.raises(ValueError, match="LIBERO-Long"):
        validate_uncond_rl_config(cfg, only_eval=True)


def test_eval_entrypoint_validates_and_dispatches_pure_worker(monkeypatch):
    import rlinf.config as config_module
    import rlinf.models.embodiment.wam_policy.uncond_rl_lifecycle as lifecycle

    _compose(monkeypatch)
    for name in ("OUTPUT_DIR", "RUN_ID", "LEDGER"):
        monkeypatch.setenv(f"FASTWAM_EVAL_{name}", f"/unit/{name}")
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "uncond_eval_entry", root / "evaluations/eval_embodied_agent.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    with initialize_config_dir(
        config_dir=str(root / "evaluations/libero"), version_base="1.1"
    ):
        cfg = compose(config_name="libero_plus_long_fastwam_uncond_rl_eval")
    launched = []

    class Placement:
        def __init__(self, *_args):
            pass

        def get_world_size(self, _owner):
            return 1

        def get_strategy(self, owner):
            return owner

    class Group:
        def __init__(self, owner):
            self.owner = owner

        def launch(self, *_args, **_kwargs):
            launched.append(self.owner)
            return self

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs["cfg"].rollout.model.eval_without_critic
            assert kwargs["rollout"].owner == "uncond_rollout"

        def init_workers(self):
            launched.append("init")

        def run(self):
            launched.append("run")

    for module in (config_module, entry):
        monkeypatch.setattr(module, "Cluster", lambda **_kwargs: None)
        monkeypatch.setattr(module, "HybridComponentPlacement", Placement)
    for worker, owner in (
        (lifecycle.UncondRLRolloutWorker, "uncond_rollout"),
        (entry.EnvWorker, "env"),
    ):
        monkeypatch.setattr(
            worker, "create_group", lambda _cfg, name=owner: Group(name)
        )
    monkeypatch.setattr(entry, "EmbodiedEvalRunner", Runner)
    entry.main.__wrapped__(cfg)
    assert launched == ["uncond_rollout", "env", "init", "run"]


def test_collector_records_nonterminal_and_terminal_without_gate(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    _ledger(ledger_path)
    collector = UncondRLEvalCollector(
        output_dir=str(tmp_path),
        ledger_path=str(ledger_path),
        run_id="uncond-unit",
        rank=0,
        routing_mode="forced_uncond",
        idm_threshold=0.5,
        random_idm_probability=None,
        routing_seed=0,
    )
    for chunk_id, terminal in enumerate((False, True)):
        snapshot = collector.snapshot_before_step(
            stage_id=0, env=_IdentityEnv(), env_ids=torch.tensor([0])
        )
        rollout = _rollout(
            chunk_id=chunk_id,
            route_used=0,
            forced=True,
            source_chunk_id=-1,
            terminal=terminal,
        )
        rollout.emitted_gate.valid.zero_()
        rollout.emitted_gate.base_probability.zero_()
        rollout.emitted_gate.behavior_probability.zero_()
        rollout.emitted_gate.old_logprob.zero_()
        rollout.gate_latency_seconds = rollout.gate_h2d_seconds = None
        collector.record_chunk(
            snapshot=snapshot,
            rollout_result=rollout,
            env_output=_outcome(terminal=terminal, success=terminal),
            policy_latency_seconds=0.1,
            environment_latency_seconds=0.01,
        )
    artifact = collector.finalize()
    assert artifact.episode_record_count == 1
    episode = json.loads(Path(artifact.episode_path).read_text())
    assert episode["success"] is True
    assert episode["forced_initial_idm_count"] == 0
    assert all(
        item["route"] == "uncond" and not item["eligible_decision"]
        for item in map(json.loads, Path(artifact.chunk_path).read_text().splitlines())
    )
