# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Independent model builder for PAD-Frozen selected by config."""

from __future__ import annotations

from typing import Any


def build_pad_frozen_model(cfg: Any, torch_dtype):
    """Construct two immutable ActionDiTs plus fresh Gate/value heads."""

    import torch
    from fastwam.adapters import load_frozen_uncond_action_artifact, sha256_file
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.gate_transformer import (
        GateTransformerConfig,
        LayerTapConfig,
    )
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.wam_policy import (
        _load_strict_fastwam_parent,
        _validate_critic_build_config,
        _validate_fastwam_actor_surface,
        _validate_fastwam_current_frame_critic_config,
    )
    from rlinf.models.embodiment.wam_policy.adaptive_policy import (
        FastWAMAdaptivePolicyConfig,
    )
    from rlinf.models.embodiment.wam_policy.critic import (
        CriticKind,
        FastWAMCurrentFrameValueCritic,
        FastWAMValueTransformerConfig,
    )
    from rlinf.models.embodiment.wam_policy.kv_replay import GateKVReplayConfig
    from rlinf.models.embodiment.wam_policy.pi05_critic import (
        Pi05ValueAfterVLMCritic,
    )

    from .config import PAD_FROZEN_POLICY_TARGET, PAD_FROZEN_RUNTIME_TARGET
    from .gate import PadCurrentStepGate
    from .policy import PadFrozenPolicy

    if torch_dtype is None:
        raise ValueError("PAD-Frozen requires an explicit model precision.")
    if str(cfg.get("builder_target", "")) != (
        "rlinf.models.embodiment.wam_policy.pad_rv.builder.build_pad_frozen_model"
    ):
        raise ValueError("PAD-Frozen builder must be selected explicitly by config.")
    if str(cfg.runtime.get("_target_", "")) != PAD_FROZEN_RUNTIME_TARGET:
        raise ValueError("PAD-Frozen requires its dedicated runtime target.")
    if (
        str(cfg.get("policy_target", PAD_FROZEN_POLICY_TARGET))
        != PAD_FROZEN_POLICY_TARGET
    ):
        raise ValueError("PAD-Frozen policy target changed unexpectedly.")
    flow = cfg.flow_sde
    if bool(flow.get("enabled", False)) or float(flow.get("noise_level", 0.0)) != 0.0:
        raise ValueError("PAD-Frozen disables stochastic Flow-SDE training replay.")
    if bool(flow.get("joint_logprob", False)):
        raise ValueError("PAD-Frozen has no joint action log-probability.")
    replay = cfg.get("kv_replay", {})
    if str(replay.get("backend", "")) != "condition":
        raise ValueError("PAD-Frozen requires condition-only replay.")
    if replay.get("gate_kv_sample_budget") is not None:
        raise ValueError("PAD condition replay does not subsample action K/V.")

    actor_checkpoint = str(cfg.get("actor_checkpoint", "")).strip()
    expected_actor_hash = str(cfg.get("actor_checkpoint_sha256", "")).lower()
    if not actor_checkpoint or len(expected_actor_hash) != 64:
        raise ValueError("PAD-Frozen requires the pinned parent and SHA-256.")
    actual_actor_hash = sha256_file(actor_checkpoint)
    if actual_actor_hash != expected_actor_hash:
        raise ValueError(
            f"PAD parent SHA-256 mismatch: {actual_actor_hash} != {expected_actor_hash}."
        )
    experts = cfg.get("route_action_experts")
    if experts is None or str(experts.get("idm_source", "")) != "parent_checkpoint":
        raise ValueError(
            "PAD IDM expert must come directly from the parent checkpoint."
        )
    merged_path = str(experts.get("uncond_merged_checkpoint", "")).strip()
    merged_hash = str(experts.get("uncond_merged_checkpoint_sha256", "")).lower()
    source_lora_hash = str(experts.get("source_lora_sidecar_sha256", "")).lower()
    if not merged_path or sha256_file(merged_path) != merged_hash:
        raise ValueError("PAD merged Warm-U artifact or SHA-256 is invalid.")

    layer_payload = OmegaConf.to_container(cfg.gate.get("layer_taps", {}), resolve=True)
    if layer_payload.get("indices") is not None:
        layer_payload["indices"] = tuple(layer_payload["indices"])
    layer_taps = LayerTapConfig(**layer_payload)
    gate_payload = OmegaConf.to_container(cfg.gate, resolve=True)
    gate_payload.pop("layer_taps", None)
    action_cfg = cfg.fastwam.action_dit_config
    gate_contract = GateTransformerConfig(
        num_mot_layers=int(action_cfg.num_layers),
        source_num_heads=int(action_cfg.num_heads),
        source_head_dim=int(action_cfg.attn_head_dim),
        layer_taps=layer_taps,
        **gate_payload,
    )
    if gate_contract.current_mode_embedding or gate_contract.denoise_timestep_embedding:
        raise ValueError("PAD current-step Gate excludes route/timestep shortcuts.")
    gate_features = FastWAMValueTransformerConfig(
        num_mot_layers=gate_contract.num_mot_layers,
        source_num_heads=gate_contract.source_num_heads,
        source_head_dim=gate_contract.source_head_dim,
        layer_indices=gate_contract.layer_taps.resolve(gate_contract.num_mot_layers),
        sources=("current_frame_video", "text_state_context"),
        hidden_dim=gate_contract.hidden_dim,
        num_query_tokens=gate_contract.num_query_tokens,
        ffn_multiplier=gate_contract.ffn_multiplier,
        share_blocks=gate_contract.share_blocks,
        layer_index_embedding=gate_contract.layer_index_embedding,
        pooling="mean_token",
    )
    load_critic = _validate_critic_build_config(cfg)
    critic_kind = CriticKind.parse(
        cfg.critic.get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
    )
    has_processor = cfg.runtime.get("processor") is not None
    has_stats = bool(cfg.runtime.get("processor_stats_path"))
    if has_processor != has_stats:
        raise ValueError("PAD runtime processor and statistics must be paired.")

    # The inherited policy config carries evaluation and sampling semantics.
    # Its replay object is unreachable in PAD, so it stays on a valid legacy
    # backend while the public PAD config says `condition`.
    replay_config = GateKVReplayConfig(backend="recompute")
    policy_config = FastWAMAdaptivePolicyConfig(
        gate_epsilon=float(cfg.get("gate_epsilon", 0.1)),
        gate_temperature=float(cfg.get("gate_temperature", 1.0)),
        eval_routing_mode=str(cfg.get("eval_routing_mode", "learned_threshold")),
        eval_idm_threshold=float(cfg.get("eval_idm_threshold", 0.5)),
        eval_random_idm_probability=(
            None
            if cfg.get("eval_random_idm_probability") is None
            else float(cfg.eval_random_idm_probability)
        ),
        eval_random_lag1_autocorrelation=(
            None
            if cfg.get("eval_random_lag1_autocorrelation") is None
            else float(cfg.eval_random_lag1_autocorrelation)
        ),
        eval_routing_seed=cfg.get("eval_routing_seed", 0),
        eval_microbatch_size=int(cfg.get("eval_microbatch_size", 1)),
        eval_timing_cuda_synchronize=bool(
            cfg.get("eval_timing_cuda_synchronize", False)
        ),
        training_rollout_microbatch_size=(
            None
            if cfg.get("training_rollout_microbatch_size") is None
            else int(cfg.training_rollout_microbatch_size)
        ),
        formal_training_sampling_seed=(
            None
            if cfg.get("formal_training_sampling_seed") is None
            else int(cfg.formal_training_sampling_seed)
        ),
        decision_telemetry_enabled=bool(cfg.get("decision_telemetry_enabled", False)),
        kv_replay=replay_config,
    )

    init_device = str(cfg.get("init_device", "cpu"))
    actor = instantiate(cfg.fastwam, model_dtype=torch_dtype, device=init_device)
    _validate_fastwam_actor_surface(actor, require_value_kv=True)
    actual_contract = (
        int(actor.mot.num_layers),
        int(actor.mot.num_heads),
        int(actor.mot.attn_head_dim),
    )
    configured_contract = (
        gate_contract.num_mot_layers,
        gate_contract.source_num_heads,
        gate_contract.source_head_dim,
    )
    if actual_contract != configured_contract:
        raise ValueError(
            f"PAD MoT contract mismatch: {actual_contract} != {configured_contract}."
        )
    _load_strict_fastwam_parent(actor, actor_checkpoint)
    actor.requires_grad_(False)
    actor.eval()
    action_payload = OmegaConf.to_container(action_cfg, resolve=True)
    if not isinstance(action_payload, dict):
        raise TypeError("PAD ActionDiT config must resolve to a mapping.")
    uncond_expert = ActionDiT(**action_payload).to(
        device=init_device, dtype=torch_dtype
    )
    load_frozen_uncond_action_artifact(
        merged_path,
        action_dit=uncond_expert,
        expected_action_dit_config=action_payload,
        expected_parent_checkpoint_sha256=expected_actor_hash,
        expected_source_lora_sidecar_sha256=source_lora_hash,
    )
    uncond_expert.requires_grad_(False)
    uncond_expert.eval()
    gate = PadCurrentStepGate(gate_features).to(dtype=torch.float32)

    critic = None
    critic_features = None
    if load_critic:
        default_hidden = (
            (1024, 512, 256) if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM else ()
        )
        hidden_sizes = tuple(
            int(item) for item in cfg.critic.get("hidden_sizes", default_hidden)
        )
        activation = str(cfg.critic.get("activation", "relu"))
        bias_last = bool(cfg.critic.get("bias_last", True))
        if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM:
            from rlinf.models.embodiment.openpi import get_model as get_openpi_model

            backbone = get_openpi_model(cfg.critic.backbone, torch_dtype)
            critic = Pi05ValueAfterVLMCritic(
                backbone,
                input_dim=int(cfg.critic.get("input_dim", 2048)),
                hidden_sizes=hidden_sizes,
                activation=activation,
                bias_last=bias_last,
            )
        else:
            critic_features = _validate_fastwam_current_frame_critic_config(
                cfg.critic,
                num_layers=int(actor.mot.num_layers),
                source_num_heads=int(actor.mot.num_heads),
                source_head_dim=int(actor.mot.attn_head_dim),
            )
            critic = FastWAMCurrentFrameValueCritic(
                config=critic_features,
                hidden_sizes=hidden_sizes,
                activation=activation,
                bias_last=bias_last,
            )
    runtime = instantiate(
        cfg.runtime,
        actor=actor,
        lora_adapter=None,
        uncond_action_expert=uncond_expert,
        gate_feature_config=gate_features,
        gate_replay_backend="condition",
        gate_layer_indices=gate_features.layer_indices,
        gate_denoise_last_n=1,
        critic_feature_config=critic_features,
        flow_sde_noise_level=0.0,
        flow_sde_ignore_last_transition=False,
    )
    return PadFrozenPolicy(
        actor=actor,
        uncond_action_expert=uncond_expert,
        runtime=runtime,
        gate=gate,
        critic=critic,
        config=policy_config,
    )
