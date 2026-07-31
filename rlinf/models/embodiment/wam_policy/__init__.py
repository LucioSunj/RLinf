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
            "pi0.5 critic parent SHA-256 mismatch: "
            f"expected {expected}, got {actual}."
        )
    return actual


def _validate_fastwam_parent_payload(actor, payload: Mapping) -> None:
    mot_state = payload.get("mot")
    if not isinstance(mot_state, Mapping):
        raise ValueError("Adaptive FastWAM parent checkpoint must contain `mot` weights.")
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


def _validate_exact_pi05_critic_config(cfg) -> None:
    """Fail before loading if the critic could restore an existing value head."""

    input_dim = int(cfg.get("input_dim", 2048))
    hidden_sizes = tuple(int(item) for item in cfg.get("hidden_sizes", (1024, 512, 256)))
    if input_dim != 2048 or hidden_sizes != (1024, 512, 256):
        raise ValueError(
            "The v0 exact pi0.5 critic requires ValueHead "
            "2048 -> 1024 -> 512 -> 256 -> 1."
        )
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


def _validate_fastwam_actor_surface(actor) -> None:
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


def _validate_flow_sde_config(cfg) -> None:
    """Validate the v0 one-transition pi-RL Flow-SDE contract."""

    if not bool(cfg.get("enabled", True)):
        raise ValueError("FastWAM adaptive v0 requires Flow-SDE to be enabled.")
    if bool(cfg.get("joint_logprob", False)):
        raise ValueError("FastWAM adaptive v0 requires `joint_logprob: false`.")
    if str(cfg.get("denoise_index_sampling", "uniform")) != "uniform":
        raise ValueError("FastWAM adaptive v0 requires uniform denoising-index sampling.")
    if bool(cfg.get("ignore_last_transition", False)):
        raise ValueError("Ignoring the final Flow-SDE transition is not implemented in v0.")
    noise_level = float(cfg.get("noise_level", 0.0))
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("Training Flow-SDE requires a strictly positive `noise_level`.")


def _validate_critic_build_config(cfg) -> bool:
    """Return whether this model instance should allocate the pi0.5 critic."""

    if bool(cfg.get("eval_without_critic", False)):
        return False
    _validate_exact_pi05_critic_config(cfg.critic)
    _validate_critic_parent_artifact(cfg.critic)
    return True


def get_model(cfg, torch_dtype):
    """Build the composite policy from explicit FastWAM/OpenPi sub-configs."""

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

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

    from .adaptive_policy import (
        FastWAMAdaptivePolicy,
        FastWAMAdaptivePolicyConfig,
    )
    from .kv_replay import GateKVReplayConfig
    from .pi05_critic import Pi05ValueAfterVLMCritic

    if torch_dtype is None:
        raise ValueError("FastWAM adaptive policy requires an explicit model precision.")

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
        eval_idm_threshold=float(cfg.get("eval_idm_threshold", 0.5)),
        eval_microbatch_size=int(cfg.get("eval_microbatch_size", 1)),
        kv_replay=replay_config,
    )

    init_device = str(cfg.get("init_device", "cpu"))
    actor = instantiate(
        cfg.fastwam,
        model_dtype=torch_dtype,
        device=init_device,
    )
    _validate_fastwam_actor_surface(actor)
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
    gate = GateTransformer(gate_config).to(dtype=torch_dtype)

    critic = None
    if load_critic:
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        critic_backbone = get_openpi_model(cfg.critic.backbone, torch_dtype)
        critic = Pi05ValueAfterVLMCritic(
            critic_backbone,
            input_dim=int(cfg.critic.get("input_dim", 2048)),
            hidden_sizes=tuple(cfg.critic.get("hidden_sizes", (1024, 512, 256))),
        )

    runtime = instantiate(
        cfg.runtime,
        actor=actor,
        lora_adapter=lora_adapter,
        gate_layer_indices=gate_config.layer_taps.resolve(gate_config.num_mot_layers),
        gate_denoise_last_n=gate_config.denoise_last_n,
        gate_replay_backend=replay_config.backend,
        flow_sde_noise_level=float(cfg.flow_sde.noise_level),
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
    "GateDecisionRecord",
    "GateKVMetadata",
    "WAMMode",
    "WAMRoute",
    "shift_emitted_gate_decisions",
    "resolve_fastwam_adaptive_eval_checkpoint",
    "get_model",
]
