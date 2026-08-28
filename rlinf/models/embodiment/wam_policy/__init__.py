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

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

from .contracts import (
    AlignedGateDecisions,
    ChunkRouteRecord,
    GateDecisionRecord,
    GateKVMetadata,
    WAMMode,
    WAMRoute,
    shift_emitted_gate_decisions,
)
from .critic import (
    CriticKind,
    FastWAMValueTransformerConfig,
)
from .evaluation import (
    EvaluationRouteSelection,
    EvaluationRoutingConfig,
    EvaluationRoutingMode,
    select_evaluation_routes,
)


def _sha256_artifact(path: str | Path) -> str:
    """Hash one checkpoint file or a deterministic directory manifest."""

    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint artifact does not exist: {root}")
    digest = hashlib.sha256()
    if root.is_file():
        with root.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    files = [
        (candidate.relative_to(root).as_posix(), candidate)
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file()
    ]
    if not files:
        raise ValueError(f"Checkpoint directory contains no files: {root}")
    for relative_name, candidate in files:
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_critic_parent_artifact(cfg) -> str:
    expected = str(cfg.get("backbone_checkpoint_sha256", "")).lower()
    checkpoint_path = str(cfg.backbone.get("model_path", "")).strip()
    if not checkpoint_path:
        raise ValueError("The pi0.5 critic requires a non-empty checkpoint path.")
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "The pi0.5 critic requires a 64-character hexadecimal "
            "`backbone_checkpoint_sha256`."
        )
    actual = _sha256_artifact(checkpoint_path)
    if actual != expected:
        raise ValueError(
            f"pi0.5 critic parent SHA-256 mismatch: expected {expected}, got {actual}."
        )
    return actual


def _validate_fastwam_parent_payload(actor, payload: Mapping) -> None:
    mot_state = payload.get("mot")
    if not isinstance(mot_state, Mapping):
        raise ValueError(
            "Adaptive FastWAM parent checkpoint must contain `mot` weights."
        )
    expected_mot = set(actor.mot.state_dict())
    actual_mot = set(mot_state)
    missing_mot = sorted(expected_mot - actual_mot)
    unexpected_mot = sorted(actual_mot - expected_mot)
    if missing_mot or unexpected_mot:
        raise ValueError(
            "Adaptive FastWAM parent MoT key mismatch: "
            f"missing={missing_mot[:8]}, unexpected={unexpected_mot[:8]}."
        )

    proprio = getattr(actor, "proprio_encoder", None)
    proprio_state = payload.get("proprio_encoder")
    if proprio is not None:
        if not isinstance(proprio_state, Mapping):
            raise ValueError(
                "Adaptive FastWAM parent checkpoint is missing `proprio_encoder`."
            )
        expected_proprio = set(proprio.state_dict())
        actual_proprio = set(proprio_state)
        if expected_proprio != actual_proprio:
            raise ValueError(
                "Adaptive FastWAM parent proprio key mismatch: "
                f"missing={sorted(expected_proprio - actual_proprio)}, "
                f"unexpected={sorted(actual_proprio - expected_proprio)}."
            )
    elif proprio_state is not None:
        raise ValueError(
            "Adaptive FastWAM parent contains proprio weights but the configured "
            "actor has no proprio encoder."
        )


def resolve_fastwam_adaptive_eval_checkpoint(
    checkpoint_path: str | Path,
    *,
    rank: int,
) -> Path:
    """Resolve either a rank checkpoint file or its containing actor directory."""

    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"FastWAM evaluation checkpoint does not exist: {path}")

    rank_path = path / f"rank_{int(rank)}.pt"
    if rank_path.is_file():
        return rank_path
    rank_zero_path = path / "rank_0.pt"
    if rank_zero_path.is_file():
        return rank_zero_path
    candidates = sorted(path.glob("rank_*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "FastWAM evaluation checkpoint directory must contain the local rank file "
        "or a rank_0.pt fallback."
    )


def _load_strict_fastwam_parent(actor, checkpoint: str) -> None:
    payload = actor.load_checkpoint(checkpoint)
    if not isinstance(payload, Mapping):
        raise TypeError("FastWAM `load_checkpoint` must return its payload mapping.")
    _validate_fastwam_parent_payload(actor, payload)


def _validate_value_head_config(cfg, *, expected_input_dim: int) -> None:
    """Validate the shared configurable scalar value-head contract."""

    if cfg.get("_target_") is not None:
        raise ValueError(
            "FastWAM critic does not support arbitrary `_target_` classes."
        )
    input_dim = cfg.get("input_dim", expected_input_dim)
    if isinstance(input_dim, bool) or not isinstance(input_dim, int):
        raise TypeError("FastWAM critic `input_dim` must be an integer.")
    if input_dim != expected_input_dim:
        raise ValueError(
            "FastWAM critic input width does not match its feature backend: "
            f"expected {expected_input_dim}, got {input_dim}."
        )
    output_dim = cfg.get("output_dim", 1)
    if isinstance(output_dim, bool) or not isinstance(output_dim, int):
        raise TypeError("FastWAM critic `output_dim` must be an integer.")
    if output_dim != 1:
        raise ValueError("FastWAM critic `output_dim` is fixed at 1.")
    hidden_sizes = cfg.get("hidden_sizes", (1024, 512, 256))
    if hidden_sizes is None or isinstance(hidden_sizes, (str, bytes, Mapping)):
        raise TypeError("FastWAM critic `hidden_sizes` must be a sequence of integers.")
    try:
        hidden_sizes = tuple(hidden_sizes)
    except TypeError as exc:
        raise TypeError(
            "FastWAM critic `hidden_sizes` must be a sequence of integers."
        ) from exc
    for hidden_size in hidden_sizes:
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size < 1
        ):
            raise ValueError(
                "FastWAM critic hidden sizes must contain only positive integers."
            )
    activation = str(cfg.get("activation", "relu")).lower()
    if activation not in {"relu", "gelu", "tanh"}:
        raise ValueError(
            "FastWAM critic activation must be one of relu, gelu, or tanh."
        )
    if not isinstance(cfg.get("bias_last", True), bool):
        raise TypeError("FastWAM critic `bias_last` must be a boolean.")


def _validate_exact_pi05_critic_config(cfg) -> None:
    """Fail before loading if the Pi05 critic could restore an existing head."""

    _validate_value_head_config(cfg, expected_input_dim=2048)
    backbone = cfg.backbone
    config_name = str(backbone.openpi.get("config_name", ""))
    if not config_name.startswith("pi05_"):
        raise ValueError(
            f"FastWAM exact critic requires a pi0.5 config, got {config_name!r}."
        )
    if bool(backbone.get("add_value_head", False)) or bool(
        backbone.openpi.get("add_value_head", False)
    ):
        raise ValueError(
            "The pretrained pi0.5 critic backbone must be built without a value head "
            "so no existing RL critic weights can be loaded."
        )
    if not bool(backbone.get("strict_vlm_checkpoint", False)):
        raise ValueError(
            "The exact pi0.5 critic requires `strict_vlm_checkpoint: true`."
        )


def _validate_fastwam_current_frame_critic_config(
    cfg,
    *,
    num_layers: int,
    source_num_heads: int,
    source_head_dim: int,
) -> FastWAMValueTransformerConfig:
    """Validate and materialize the Gate-style value-transformer config."""

    if cfg.get("backbone") is not None:
        raise ValueError(
            "FastWAM current-frame critic must set `backbone: null`; it reuses "
            "the colocated frozen actor."
        )
    if cfg.get("backbone_checkpoint_sha256") not in {None, ""}:
        raise ValueError(
            "FastWAM current-frame critic must not configure an external critic "
            "parent hash."
        )
    feature = cfg.get("feature")
    if feature is None:
        raise ValueError("FastWAM current-frame critic requires a `feature` mapping.")
    transformer = cfg.get("transformer")
    if transformer is None:
        raise ValueError(
            "FastWAM current-frame critic requires a `transformer` mapping."
        )
    unknown_feature_keys = sorted(
        set(feature) - {"source_dim", "layer_indices", "sources"}
    )
    if unknown_feature_keys:
        raise ValueError(
            f"Unknown FastWAM critic feature fields: {unknown_feature_keys}."
        )
    unknown_transformer_keys = sorted(
        set(transformer)
        - {
            "hidden_dim",
            "num_query_tokens",
            "ffn_multiplier",
            "share_blocks",
            "layer_index_embedding",
            "pooling",
        }
    )
    if unknown_transformer_keys:
        raise ValueError(
            f"Unknown FastWAM value-transformer fields: {unknown_transformer_keys}."
        )
    expected_source_dim = source_num_heads * source_head_dim
    source_dim = feature.get("source_dim", expected_source_dim)
    if isinstance(source_dim, bool) or not isinstance(source_dim, int):
        raise TypeError("FastWAM critic `feature.source_dim` must be an integer.")
    if source_dim != expected_source_dim:
        raise ValueError(
            "FastWAM critic source width does not match Video/Context K/V: "
            f"expected {expected_source_dim}, got {source_dim}."
        )
    layer_indices = feature.get("layer_indices", (14,))
    if isinstance(layer_indices, (str, bytes, Mapping)):
        raise TypeError("FastWAM critic `feature.layer_indices` must be a sequence.")
    sources = feature.get(
        "sources",
        ("current_frame_video", "text_state_context"),
    )
    if isinstance(sources, (str, bytes, Mapping)):
        raise TypeError("FastWAM critic `feature.sources` must be a sequence.")
    result = FastWAMValueTransformerConfig(
        num_mot_layers=num_layers,
        source_num_heads=source_num_heads,
        source_head_dim=source_head_dim,
        layer_indices=tuple(layer_indices),
        sources=tuple(str(source) for source in sources),
        hidden_dim=transformer.get("hidden_dim", 256),
        num_query_tokens=transformer.get("num_query_tokens", 4),
        ffn_multiplier=transformer.get("ffn_multiplier", 4),
        share_blocks=transformer.get("share_blocks", False),
        layer_index_embedding=transformer.get("layer_index_embedding", True),
        pooling=str(transformer.get("pooling", "mean_token")),
    )
    _validate_value_head_config(cfg, expected_input_dim=result.hidden_dim)
    return result


def _validate_fastwam_actor_surface(
    actor,
    *,
    require_value_kv: bool = False,
) -> None:
    required = (
        "action_expert",
        "mot",
        "infer_action_scheduler",
        "infer_video_scheduler",
        "vae",
        "load_checkpoint",
    )
    missing = [name for name in required if not hasattr(actor, name)]
    if missing:
        raise TypeError(f"FastWAM actor is missing required adaptive APIs: {missing}.")
    if not require_value_kv:
        return
    if not hasattr(actor.action_expert, "text_embedding"):
        raise TypeError(
            "FastWAM ActionDiT must expose `text_embedding` for critic K/V."
        )
    if not hasattr(actor.mot, "read_condition_layer_kv"):
        raise TypeError(
            "FastWAM MoT must expose `read_condition_layer_kv` for critic K/V."
        )


def _validate_flow_sde_config(cfg) -> None:
    """Validate the v0 one-transition pi-RL Flow-SDE contract."""

    if not bool(cfg.get("enabled", True)):
        raise ValueError("FastWAM adaptive v0 requires Flow-SDE to be enabled.")
    if bool(cfg.get("joint_logprob", False)):
        raise ValueError("FastWAM adaptive v0 requires `joint_logprob: false`.")
    if str(cfg.get("denoise_index_sampling", "uniform")) != "uniform":
        raise ValueError(
            "FastWAM adaptive v0 requires uniform denoising-index sampling."
        )
    noise_level = float(cfg.get("noise_level", 0.0))
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError(
            "Training Flow-SDE requires a strictly positive `noise_level`."
        )


def _validate_critic_build_config(cfg) -> bool:
    """Return whether this model instance should allocate its configured critic."""

    if bool(cfg.get("eval_without_critic", False)):
        return False
    critic_kind = CriticKind.parse(
        cfg.critic.get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
    )
    if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM:
        _validate_exact_pi05_critic_config(cfg.critic)
        _validate_critic_parent_artifact(cfg.critic)
    else:
        action_config = cfg.fastwam.action_dit_config
        _validate_fastwam_current_frame_critic_config(
            cfg.critic,
            num_layers=int(action_config.num_layers),
            source_num_heads=int(action_config.num_heads),
            source_head_dim=int(action_config.attn_head_dim),
        )
    return True


def get_model(cfg, torch_dtype):
    """Build the composite policy from explicit FastWAM/OpenPi sub-configs."""

    import torch
    from fastwam.adapters import (
        RegimeLoRAConfig,
        inject_action_dit_lora,
        sha256_file,
    )
    from fastwam.models.wan22.gate_transformer import (
        GateTransformer,
        GateTransformerConfig,
        LayerTapConfig,
    )
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from .adaptive_policy import (
        FastWAMAdaptivePolicy,
        FastWAMAdaptivePolicyConfig,
    )
    from .critic import FastWAMCurrentFrameValueCritic
    from .kv_replay import GateKVReplayConfig
    from .pi05_critic import Pi05ValueAfterVLMCritic

    if torch_dtype is None:
        raise ValueError(
            "FastWAM adaptive policy requires an explicit model precision."
        )

    # Validate the complete lightweight contract before allocating either large
    # pretrained backbone.
    actor_checkpoint = cfg.get("actor_checkpoint")
    expected_actor_hash = cfg.get("actor_checkpoint_sha256")
    if not actor_checkpoint or not expected_actor_hash:
        raise ValueError(
            "FastWAM adaptive policy requires `actor_checkpoint` and "
            "`actor_checkpoint_sha256`."
        )
    actual_actor_hash = sha256_file(str(actor_checkpoint))
    if actual_actor_hash != str(expected_actor_hash).lower():
        raise ValueError(
            "FastWAM parent checkpoint SHA-256 mismatch: "
            f"expected {expected_actor_hash}, got {actual_actor_hash}."
        )
    lora_payload = OmegaConf.to_container(cfg.uncond_lora, resolve=True)
    lora_config = RegimeLoRAConfig(**lora_payload)
    if lora_config.dropout != 0.0:
        raise ValueError("FastWAM PPO requires deterministic LoRA dropout == 0.")
    layer_payload = OmegaConf.to_container(
        cfg.gate.get("layer_taps", {}),
        resolve=True,
    )
    if layer_payload.get("indices") is not None:
        layer_payload["indices"] = tuple(layer_payload["indices"])
    layer_taps = LayerTapConfig(**layer_payload)
    gate_payload = OmegaConf.to_container(cfg.gate, resolve=True)
    gate_payload.pop("layer_taps", None)
    action_dit_config = cfg.fastwam.action_dit_config
    gate_config = GateTransformerConfig(
        num_mot_layers=int(action_dit_config.num_layers),
        source_num_heads=int(action_dit_config.num_heads),
        source_head_dim=int(action_dit_config.attn_head_dim),
        layer_taps=layer_taps,
        **gate_payload,
    )
    replay_payload = OmegaConf.to_container(
        cfg.get("kv_replay", {}),
        resolve=True,
    )
    replay_config = GateKVReplayConfig(**replay_payload)
    _validate_flow_sde_config(cfg.flow_sde)
    load_critic = _validate_critic_build_config(cfg)
    critic_kind = CriticKind.parse(
        cfg.critic.get("kind", CriticKind.PI0_5_VALUE_AFTER_VLM)
    )
    inference_steps = int(cfg.runtime.get("num_inference_steps", 0))
    if gate_config.denoise_last_n > inference_steps:
        raise ValueError(
            "Gate `denoise_last_n` cannot exceed runtime `num_inference_steps`: "
            f"{gate_config.denoise_last_n} > {inference_steps}."
        )
    has_processor = cfg.runtime.get("processor") is not None
    has_processor_stats = bool(cfg.runtime.get("processor_stats_path"))
    if has_processor != has_processor_stats:
        raise ValueError(
            "FastWAM runtime processor and processor stats must be configured together."
        )
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
            else cfg.training_rollout_microbatch_size
        ),
        formal_training_sampling_seed=(
            None
            if cfg.get("formal_training_sampling_seed") is None
            else cfg.formal_training_sampling_seed
        ),
        decision_telemetry_enabled=bool(cfg.get("decision_telemetry_enabled", False)),
        kv_replay=replay_config,
    )

    init_device = str(cfg.get("init_device", "cpu"))
    actor = instantiate(
        cfg.fastwam,
        model_dtype=torch_dtype,
        device=init_device,
    )
    _validate_fastwam_actor_surface(
        actor,
        require_value_kv=(
            load_critic and critic_kind is CriticKind.FASTWAM_CURRENT_FRAME_VALUE
        ),
    )
    actual_mot_contract = (
        int(actor.mot.num_layers),
        int(actor.mot.num_heads),
        int(actor.mot.attn_head_dim),
    )
    configured_mot_contract = (
        gate_config.num_mot_layers,
        gate_config.source_num_heads,
        gate_config.source_head_dim,
    )
    if actual_mot_contract != configured_mot_contract:
        raise ValueError(
            "Constructed FastWAM MoT differs from the prevalidated Gate source "
            f"contract: configured={configured_mot_contract}, actual={actual_mot_contract}."
        )
    _load_strict_fastwam_parent(actor, str(actor_checkpoint))
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    lora_adapter = inject_action_dit_lora(
        actor.action_expert,
        lora_config,
    )
    # The Gate is trainable, so it stays in FP32 rather than adopting the frozen
    # model dtype. A BF16 Gate discards every Adam update below half a BF16 ULP,
    # which is where this Gate's updates land; see
    # `docs/BF16_PARAMETER_UPDATE_LOSS.md`. `DirectKVAttention` already casts the
    # stored K/V banks to the query dtype, so the read-only tap still works.
    gate = GateTransformer(gate_config).to(dtype=torch.float32)

    critic = None
    critic_feature_config = None
    if load_critic:
        default_hidden_sizes = (
            (1024, 512, 256) if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM else ()
        )
        hidden_sizes = tuple(
            int(item) for item in cfg.critic.get("hidden_sizes", default_hidden_sizes)
        )
        activation = str(cfg.critic.get("activation", "relu"))
        bias_last = bool(cfg.critic.get("bias_last", True))
        if critic_kind is CriticKind.PI0_5_VALUE_AFTER_VLM:
            from rlinf.models.embodiment.openpi import get_model as get_openpi_model

            critic_backbone = get_openpi_model(cfg.critic.backbone, torch_dtype)
            critic = Pi05ValueAfterVLMCritic(
                critic_backbone,
                input_dim=int(cfg.critic.get("input_dim", 2048)),
                hidden_sizes=hidden_sizes,
                activation=activation,
                bias_last=bias_last,
            )
        else:
            critic_feature_config = _validate_fastwam_current_frame_critic_config(
                cfg.critic,
                num_layers=int(actor.mot.num_layers),
                source_num_heads=int(actor.mot.num_heads),
                source_head_dim=int(actor.mot.attn_head_dim),
            )
            critic = FastWAMCurrentFrameValueCritic(
                config=critic_feature_config,
                hidden_sizes=hidden_sizes,
                activation=activation,
                bias_last=bias_last,
            )

    runtime = instantiate(
        cfg.runtime,
        actor=actor,
        lora_adapter=lora_adapter,
        gate_layer_indices=gate_config.layer_taps.resolve(gate_config.num_mot_layers),
        gate_denoise_last_n=gate_config.denoise_last_n,
        gate_replay_backend=replay_config.backend,
        critic_feature_config=critic_feature_config,
        flow_sde_noise_level=float(cfg.flow_sde.noise_level),
        flow_sde_ignore_last_transition=bool(
            cfg.flow_sde.get("ignore_last_transition", False)
        ),
    )
    return FastWAMAdaptivePolicy(
        actor=actor,
        runtime=runtime,
        lora_adapter=lora_adapter,
        gate=gate,
        critic=critic,
        config=policy_config,
    )


__all__ = [
    "AlignedGateDecisions",
    "ChunkRouteRecord",
    "EvaluationRouteSelection",
    "EvaluationRoutingConfig",
    "EvaluationRoutingMode",
    "GateDecisionRecord",
    "GateKVMetadata",
    "WAMMode",
    "WAMRoute",
    "get_model",
    "resolve_fastwam_adaptive_eval_checkpoint",
    "select_evaluation_routes",
    "shift_emitted_gate_decisions",
]
