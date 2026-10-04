# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only UNCOND PPO coverage with tiny real MoT/LoRA/Flow-SDE modules."""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastwam.adapters import (
    RegimeLoRAConfig,
    inject_action_dit_lora,
    inject_video_bc_dit_lora,
)
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock
from fastwam.uncond_bc_checkpoint import (
    DUAL_UNCOND_BC_SIDECAR_SCHEMA,
    build_lora_sidecar_payload,
)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.wam_policy.adaptive_policy import (
    FastWAMAdaptivePolicyConfig,
)
from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayConfig
from rlinf.models.embodiment.wam_policy.optimizer import (
    assert_fastwam_optimizer_update_resolution,
    fastwam_optimizer_gradient_norms,
    partition_fastwam_trainable_parameters,
)
from rlinf.models.embodiment.wam_policy.uncond_rl import (
    FastWAMUncondRLPolicy,
    compute_uncond_rl_loss,
)
from rlinf.models.embodiment.wam_policy.uncond_rl_config import (
    validate_uncond_rl_config,
)
from rlinf.models.embodiment.wam_policy.uncond_rl_runtime import UncondRLLiberoRuntime
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _compose(monkeypatch):
    config_dir = Path(__file__).parents[2] / "examples/embodiment/config"
    for name, value in {
        "EMBODIED_PATH": str(config_dir.parent),
        "FASTWAM_CHECKPOINT": "/parent.pt",
        "FASTWAM_CHECKPOINT_SHA256": "a" * 64,
        "FASTWAM_DATASET_STATS": "/stats.json",
        "FASTWAM_UNCOND_BC_SIDECAR": "/bc.pt",
        "FASTWAM_UNCOND_BC_SIDECAR_SHA256": "b" * 64,
        "FASTWAM_TEXT_CACHE": "/text-cache",
        "PI05_CRITIC_CHECKPOINT": "/critic",
        "PI05_CRITIC_CHECKPOINT_SHA256": "c" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(config_name="libero_10_ppo_fastwam_uncond")


def test_preset_is_long_only_flow_ppo(monkeypatch):
    cfg = _compose(monkeypatch)
    validate_uncond_rl_config(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert cfg.env.train.task_id_filter == list(range(10))
    assert cfg.env.train.task_suite_name == "libero_10"
    assert cfg.env.train.total_num_envs == cfg.env.eval.total_num_envs == 84
    assert cfg.actor.global_batch_size == 392
    assert cfg.actor.micro_batch_size == 4
    assert cfg.env.train.total_num_envs // 6 == 14
    assert cfg.actor.global_batch_size // cfg.actor.micro_batch_size == 98
    assert cfg.env.train.total_num_envs * 70 // cfg.actor.global_batch_size == 15
    assert list(cfg.actor.uncond_rl_execution.helper_ranks) == list(range(6))
    assert cfg.actor.model.uncond_lora.rank == cfg.actor.model.video_lora.rank == 128
    assert cfg.actor.model.runtime.num_video_frames == 1
    assert not any("idm" in key or "gate" in key for key in resolved["algorithm"])
    assert cfg.actor.optim.lora_lr == 1e-5
    assert cfg.actor.model.flow_sde.ignore_last_transition


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("algorithm.uncond_idm_bc", {"enabled": True, "loss_weight": 0.2}, "excludes"),
        ("algorithm.fixed_branch_cost", {"enabled": True}, "excludes"),
        ("actor.model.gate", {"hidden_dim": 256}, "excludes"),
        ("env.train.task_suite_name", "libero_spatial", "LIBERO-Long"),
        ("rollout.model.flow_sde.noise_level", 0.2, "differ"),
        ("actor.model.flow_sde.noise_level", 0.0, "positive"),
        ("env.train.total_num_envs", 42, "total_num_envs=84"),
        ("env.eval.total_num_envs", 42, "total_num_envs=84"),
        ("actor.global_batch_size", 196, "global_batch_size=392"),
        ("actor.micro_batch_size", 8, "micro_batch_size=4"),
    ],
)
def test_profile_rejects_mixed_inputs(monkeypatch, path, value, message):
    cfg = _compose(monkeypatch)
    OmegaConf.update(cfg, path, value, force_add=True)
    with pytest.raises(ValueError, match=message):
        validate_uncond_rl_config(cfg)


def test_fresh_run_requires_bc_but_resume_does_not_rebootstrap(monkeypatch):
    cfg = _compose(monkeypatch)
    cfg.runner.bootstrap_uncond_lora_sidecar = None
    cfg.runner.bootstrap_uncond_lora_sidecar_sha256 = None
    with pytest.raises(ValueError, match="Fresh UNCOND"):
        validate_uncond_rl_config(cfg)
    cfg.runner.resume_dir = "/native/checkpoint"
    validate_uncond_rl_config(cfg)


class _Expert(nn.Module):
    def __init__(self, video, checkpoint):
        super().__init__()
        self.video = video
        self.hidden_dim, self.num_heads, self.attn_head_dim = 8, 2, 4
        self.action_dim = 7
        self.use_gradient_checkpointing = checkpoint
        self.fuse_vae_embedding_in_latents = False
        self.time_embedding = nn.Identity()
        self.input = nn.Linear(3 if video else 7, 8)
        self.output = nn.Linear(8, 7)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim=8, attn_head_dim=4, num_heads=2, ffn_dim=16)
            for _ in range(2)
        )

    def pre_dit(self, *, context, context_mask, timestep, **kwargs):
        tokens = self.input(
            kwargs["x"][:, :, 0].flatten(2).transpose(1, 2)
            if self.video
            else kwargs["action_tokens"]
        )
        return {
            "tokens": tokens,
            "freqs": torch.ones(tokens.shape[1], 1, 2, dtype=torch.complex128),
            "t_mod": torch.zeros(tokens.shape[0], 6, 8),
            "context": context,
            "context_mask": context_mask[:, None, :].expand(-1, tokens.shape[1], -1),
            "meta": {"tokens_per_frame": tokens.shape[1]},
        }

    def post_dit(self, tokens, _pre):
        assert not self.video, "UNCOND must never predict video"
        return self.output(tokens)


class _Actor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.video_expert = _Expert(True, checkpoint)
        self.action_expert = _Expert(False, checkpoint)
        self.mot = MoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            mot_checkpoint_mixed_attn=checkpoint,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=1000, shift=5.0
        )

    def _encode_input_image_latents_tensor(self, image, *, tiled):
        return image.unsqueeze(2)

    def _build_mot_attention_mask(self, *, video_seq_len, action_seq_len, **kwargs):
        mask = torch.ones(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
            dtype=torch.bool,
        )
        mask[:video_seq_len, video_seq_len:] = False
        return mask

    def _video_denoise_step_compiled(self, **kwargs):
        raise AssertionError("UNCOND must never call future prediction")


class _Critic(nn.Module):
    replay_feature_key = "critic_prefix"

    def __init__(self):
        super().__init__()
        self.value_head = nn.Linear(8, 1)

    def predict_value_batch(self, obs, return_prefix=False):
        features = obs["states"].detach()
        values = self.value_from_features(features)
        return (values, features) if return_prefix else values

    def value_from_features(self, features):
        return self.value_head(features.detach())


def _policy(monkeypatch, *, dual=True, checkpoint=False):
    torch.manual_seed(42)
    actor = _Actor(checkpoint)
    action = inject_action_dit_lora(
        actor.action_expert, RegimeLoRAConfig(rank=4, alpha=4)
    )
    video = (
        inject_video_bc_dit_lora(
            actor.video_expert,
            RegimeLoRAConfig(rank=4, alpha=4),
            regime_context=action.regime_context,
        )
        if dual
        else None
    )
    runtime = UncondRLLiberoRuntime(
        actor=actor,
        lora_adapter=action,
        video_lora_adapter=video,
        generation_horizon=3,
        execution_horizon=2,
        num_video_frames=1,
        max_episode_steps=4,
        num_inference_steps=3,
        flow_sde_ignore_last_transition=True,
    )
    monkeypatch.setattr(
        runtime,
        "_encode_condition",
        lambda obs: (obs["images"], obs["context"], obs["context_mask"]),
    )
    monkeypatch.setattr(runtime, "critic_observation", lambda *, env_obs: env_obs)
    monkeypatch.setattr(
        runtime, "_denormalize_action_stages", lambda actions, **kw: (actions, None)
    )
    return FastWAMUncondRLPolicy(
        actor=actor,
        runtime=runtime,
        lora_adapter=action,
        video_lora_adapter=video,
        gate=None,
        critic=_Critic(),
        config=FastWAMAdaptivePolicyConfig(
            formal_training_sampling_seed=42,
            kv_replay=GateKVReplayConfig(backend="recompute"),
        ),
    )


def _obs():
    return {
        "states": torch.randn(2, 8),
        "images": torch.randn(2, 3, 2, 2),
        "context": torch.randn(2, 4, 8),
        "context_mask": torch.ones(2, 4, dtype=torch.bool),
        "_fastwam_env_ids": torch.tensor([1, 7]),
        "_fastwam_reset_mask": torch.ones(2, dtype=torch.bool),
    }


def _micro(sample):
    return {
        **sample,
        "flow_advantages": torch.tensor([[1.0], [-0.3]]),
        "flow_valid_mask": torch.ones(2, dtype=torch.bool),
        "returns": sample["prev_values"] + 1.0,
        "loss_mask": torch.ones(2, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "dual,checkpoint", [(False, False), (True, False), (True, True)]
)
def test_rollout_replay_gradients_and_warmup(monkeypatch, dual, checkpoint):
    torch.set_num_threads(1)
    cfg = _compose(monkeypatch)
    policy = _policy(monkeypatch, dual=dual, checkpoint=checkpoint)
    frozen = {
        name: value.detach().clone()
        for name, value in policy.actor.named_parameters()
        if not value.requires_grad
    }
    actions, sample = policy.predict_action_batch(_obs())
    assert actions.shape == (2, 2, 7)
    assert not sample["route_info"].route_used.any()
    assert not sample["emitted_gate"].valid.any()
    assert not any(
        "idm" in key or "teacher" in key or "gate" in key
        for key in sample["forward_inputs"]
    )
    assert (sample["forward_inputs"]["denoise_indices"] < 2).all()
    policy.train()
    output = policy.default_forward(sample["forward_inputs"])
    torch.testing.assert_close(
        output["flow_logprobs"], sample["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    loss, metrics = compute_uncond_rl_loss(
        cfg=cfg, actor_version=0, micro_batch=_micro(sample), output_dict=output
    )
    loss.backward()
    assert all(param.grad is None for param in policy.lora_parameters())
    assert policy.critic.value_head.weight.grad.abs().sum() > 0
    assert not any("gate" in key or "bc" in key or "idm" in key for key in metrics)
    policy.zero_grad(set_to_none=True)
    output = policy.default_forward(sample["forward_inputs"])
    loss, _ = compute_uncond_rl_loss(
        cfg=cfg, actor_version=10, micro_batch=_micro(sample), output_dict=output
    )
    loss.backward()
    for adapter in policy.lora_adapters.values():
        assert (
            sum(
                p.grad.abs().sum()
                for p in adapter.lora_parameters()
                if p.grad is not None
            )
            > 0
        )
    groups = partition_fastwam_trainable_parameters(
        policy.named_parameters(), require_gate=False
    )
    assert set(groups) == {"uncond_lora", "value_head"}
    optimizer = torch.optim.AdamW(
        policy.optimizer_parameter_groups(lora_lr=1e-5, value_lr=1e-4), weight_decay=0
    )
    assert (
        fastwam_optimizer_gradient_norms(optimizer, require_gate=False)["uncond_lora"]
        > 0
    )
    optimizer.step()
    assert all(
        torch.equal(frozen[name], value)
        for name, value in policy.actor.named_parameters()
        if name in frozen
    )


def test_sampling_state_reset_sync_and_checkpoint_roundtrip(monkeypatch):
    policy = _policy(monkeypatch)
    obs = _obs()
    policy.predict_action_batch(obs)
    obs["_fastwam_reset_mask"].zero_()
    saved = copy.deepcopy(policy.trainable_state_dict())
    expected, _ = policy.predict_action_batch(obs)
    restored = _policy(monkeypatch)
    restored.load_trainable_state_dict(saved)
    actual, sample = restored.predict_action_batch(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sample["route_info"].chunk_ids.tolist() == [1, 1]
    restored.set_global_step(10)
    obs["_fastwam_reset_mask"][0] = True
    _, sample = restored.predict_action_batch(obs)
    assert not sample["route_info"].route_used.any()
    assert sample["route_info"].episode_ids.tolist() == [1, 0]
    assert sample["route_info"].chunk_ids.tolist() == [0, 2]
    assert "gate" not in saved and "actor" not in saved
    runtime = restored.rollout_runtime_state_dict()
    policy.load_rollout_runtime_state_dict(runtime)
    assert policy.rollout_runtime_state_dict() == runtime
    _, evaluation = policy.predict_action_batch(obs, mode="eval", compute_values=False)
    assert evaluation["forward_inputs"] == {}
    assert evaluation["prev_logprobs"].shape == (2, 0)


def test_native_bc_bootstrap_initializes_both_adapters_only(monkeypatch, tmp_path):
    from test_fastwam_actor_checkpoint import _checkpoint_worker

    policy = _policy(monkeypatch)
    worker = _checkpoint_worker()
    worker.model, worker.optimizer_steps = policy, 0
    before = copy.deepcopy(policy.critic.state_dict())
    payload = {"schema": DUAL_UNCOND_BC_SIDECAR_SCHEMA}
    for branch, adapter in policy.lora_adapters.items():
        payload[branch] = build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256="a" * 64,
            extra_metadata={"bc_step": 1942, "bc_config_sha256": "c" * 64},
        )
        for value in payload[branch]["state_dict"].values():
            value.fill_(0.25 if branch == "action" else 0.5)
    sidecar = tmp_path / "bc.pt"
    torch.save(payload, sidecar)
    worker.bootstrap_fastwam_uncond_lora(
        str(sidecar), hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )
    for branch, adapter in policy.lora_adapters.items():
        for name, value in adapter.lora_state_dict().items():
            assert torch.equal(value, payload[branch]["state_dict"][name])
    assert all(
        torch.equal(value, before[name])
        for name, value in policy.critic.state_dict().items()
    )
    worker.version = policy.actor_version = 3
    worker.save_checkpoint(str(tmp_path / "native"), 3)
    with torch.no_grad():
        next(policy.lora_parameters()).add_(1)
    assert worker.load_checkpoint(str(tmp_path / "native")) == 3
    assert policy.actor_version == 3


@pytest.mark.parametrize("resume", [False, True])
def test_actor_init_bootstraps_bc_only_for_fresh_run(monkeypatch, resume):
    cfg = _compose(monkeypatch)
    cfg.runner.resume_dir = "/native" if resume else None
    calls = []
    worker = SimpleNamespace(
        cfg=cfg,
        _rank=0,
        enable_offload=False,
        setup_model_and_optimizer=lambda: calls.append("setup"),
        bootstrap_fastwam_uncond_lora=lambda *args: calls.append(args),
    )
    EmbodiedFSDPActor.init_worker(worker)
    assert calls == (["setup"] if resume else ["setup", ("/bc.pt", "b" * 64)])


def test_value_only_resolution_does_not_require_lora_gradients():
    lora, value = nn.Parameter(torch.ones(4)), nn.Parameter(torch.ones(4))
    value.grad = torch.ones_like(value)
    optimizer = torch.optim.AdamW(
        [
            {"name": "uncond_lora", "params": [lora]},
            {"name": "value_head", "params": [value]},
        ],
        lr=1e-4,
    )
    report = assert_fastwam_optimizer_update_resolution(
        optimizer, minimum_half_ulp_ratio=1, require_gate=False, value_only=True
    )
    assert set(report) == {"value_head"}


def test_shared_builder_omits_gate_allocation(monkeypatch):
    import fastwam.adapters
    import fastwam.models.wan22.gate_transformer as gate_module
    import hydra.utils

    from rlinf.models.embodiment.wam_policy import get_model

    cfg = _compose(monkeypatch).actor.model
    OmegaConf.set_struct(cfg, False)
    cfg.eval_without_critic = True
    cfg.runtime.processor = None
    cfg.runtime.processor_stats_path = None
    cfg.runtime.text_embedding_cache_dir = None
    cfg.fastwam.action_dit_config.num_layers = 2
    cfg.fastwam.action_dit_config.num_heads = 2
    cfg.fastwam.action_dit_config.attn_head_dim = 4
    actor = _Actor(False)
    actor.infer_video_scheduler = actor.infer_action_scheduler
    actor.vae = nn.Identity()
    actor.load_checkpoint = lambda path: {"mot": actor.mot.state_dict()}
    instantiate = hydra.utils.instantiate

    def build(config, **kwargs):
        if config._target_ == "fastwam.runtime.create_fastwam":
            return actor
        return instantiate(config, **kwargs)

    def forbidden_gate(*args, **kwargs):
        pytest.fail("UNCOND builder allocated a Gate")

    monkeypatch.setattr(hydra.utils, "instantiate", build)
    monkeypatch.setattr(fastwam.adapters, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(gate_module, "GateTransformer", forbidden_gate)
    policy = get_model(cfg, torch.float32)
    assert isinstance(policy, FastWAMUncondRLPolicy)
    assert policy.gate is None
    assert policy.video_lora_adapter is not None
    assert isinstance(policy.runtime, UncondRLLiberoRuntime)


def test_fsdp_optimizer_warmup_keeps_lora_moments_and_weights(monkeypatch):
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager

    cfg = _compose(monkeypatch)
    policy = _policy(monkeypatch)
    optimizer = torch.optim.AdamW(
        policy.optimizer_parameter_groups(lora_lr=1e-5, value_lr=1e-4),
        weight_decay=0.1,
    )
    worker = SimpleNamespace(
        _cfg=cfg.actor,
        version=0,
        optimizer_steps=0,
        optimizer=optimizer,
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        model=policy,
        _strategy=SimpleNamespace(
            clip_grad_norm_=lambda model: torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
        ),
        _logger=SimpleNamespace(info=lambda message: None),
        _fastwam_update_resolution_checked=False,
        critic_warmup_steps=0,
    )
    before = [p.detach().clone() for p in policy.lora_parameters()]
    for parameter in policy.parameters():
        if parameter.requires_grad:
            # FSDP can return zero views for the inactive LoRA parameters.
            parameter.grad = torch.zeros_like(parameter)
    for parameter in policy.critic.value_head.parameters():
        parameter.grad.fill_(1)
    FSDPModelManager.optimizer_step(worker)
    assert all(p not in optimizer.state for p in policy.lora_parameters())
    assert all(
        torch.equal(saved, p) for saved, p in zip(before, policy.lora_parameters())
    )
    assert not worker._fastwam_update_resolution_checked
    worker.version = 10
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    FSDPModelManager.optimizer_step(worker)
    assert worker._fastwam_update_resolution_checked
    assert all(optimizer.state[p]["step"] == 1 for p in policy.lora_parameters())


def test_all_uncond_chunks_keep_flow_credit_and_terminal_mask(monkeypatch):
    from rlinf.algorithms.advantages import align_fastwam_policy_advantages
    from rlinf.models.embodiment.wam_policy.contracts import (
        ChunkRouteRecord,
        GateDecisionRecord,
    )

    policy = _policy(monkeypatch)
    obs = _obs()
    _, first = policy.predict_action_batch(obs)
    obs["_fastwam_reset_mask"].zero_()
    _, second = policy.predict_action_batch(obs)
    valid = torch.tensor([[[True], [True]], [[False], [True]]])
    advantages = torch.tensor([[[2.0], [3.0]], [[999.0], [5.0]]])
    alignment = align_fastwam_policy_advantages(
        advantages=advantages,
        route=ChunkRouteRecord.stack([first["route_info"], second["route_info"]]),
        emitted=GateDecisionRecord.stack(
            [first["emitted_gate"], second["emitted_gate"]]
        ),
        dones=torch.tensor([[[False], [False]], [[True], [False]], [[True], [True]]]),
        rollout_epoch=1,
        carry_pending_across_epochs=False,
        loss_mask=valid,
    )
    assert torch.equal(alignment.flow_valid_mask, valid.squeeze(-1))
    assert not alignment.gate_valid_mask.any()
    assert torch.equal(
        alignment.flow_advantages[alignment.flow_valid_mask].reshape(-1),
        torch.tensor([2.0, 3.0, 5.0]),
    )
