# Copyright 2025 The RLinf Authors.
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
from pathlib import Path

import torch
from omegaconf import DictConfig


_ALLOWED_WAM_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_wam_dtype(value) -> torch.dtype:
    name = str(value).lower()
    if name not in _ALLOWED_WAM_DTYPES:
        raise ValueError(
            f"wam.dtype must be one of {sorted(_ALLOWED_WAM_DTYPES)}, got {name!r}"
        )
    return _ALLOWED_WAM_DTYPES[name]


def _is_none_like(value) -> bool:
    return value is None or str(value).lower() in {"", "none", "null"}


def _resolve_fastwam_path(value, *, fastwam_root: Path) -> Path | None:
    if _is_none_like(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = fastwam_root / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_stats_path(
    fcfg: DictConfig | None,
    wam_cfg: DictConfig,
    *,
    fastwam_root: Path,
) -> Path | None:
    stats_path = _resolve_fastwam_path(
        wam_cfg.get("dataset_stats_path", None), fastwam_root=fastwam_root
    )
    if stats_path is None and fcfg is not None:
        stats_path = _resolve_fastwam_path(
            fcfg.data.train.get("pretrained_norm_stats", None),
            fastwam_root=fastwam_root,
        )
    if stats_path is None and not _is_none_like(wam_cfg.get("ckpt", None)):
        ckpt_path = _resolve_fastwam_path(
            wam_cfg.get("ckpt"), fastwam_root=fastwam_root
        )
        for parent in (ckpt_path.parent, ckpt_path.parent.parent):
            candidate = parent / "dataset_stats.json"
            if candidate.is_file():
                stats_path = candidate.resolve()
                break
    if stats_path is not None and not stats_path.is_file():
        raise FileNotFoundError(f"FastWAM dataset stats not found: {stats_path}")
    if stats_path is None and bool(wam_cfg.get("require_dataset_stats", True)):
        raise FileNotFoundError(
            "GatePolicy requires the exact FastWAM dataset stats used by the "
            "checkpoint. Set actor.model.wam.dataset_stats_path, or place "
            "dataset_stats.json next to the checkpoint."
        )
    return stats_path


def build_wam_adapter(wam_cfg: DictConfig):
    """Build a frozen two-regime FastWAM and wrap it in WAMModeAdapter.

    Required `wam_cfg` keys:
      configs_dir        : path to FastWAM/configs (for hydra compose)
      task               : dual-regime IDM FastWAM task name
      backbone_kind      : must be "idm"
      ckpt               : path to the dual-regime checkpoint
    Optional:
      num_video_frames (9), generation_horizon (32), inference_steps (20),
      cost_table_path (None), device ("cuda"), dtype ("bfloat16")
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from fastwam.adaptive_gate import WAMModeAdapter

    backbone_kind = str(wam_cfg.get("backbone_kind", "idm")).lower()
    if backbone_kind != "idm":
        raise ValueError(
            "The adaptive gate now supports only UNCOND/full-IDM routing; "
            f"backbone_kind must be `idm`, got `{backbone_kind}`."
        )
    device = str(wam_cfg.get("device", "cuda"))
    dtype = _resolve_wam_dtype(wam_cfg.get("dtype", "bfloat16"))
    configs_dir = str(wam_cfg.configs_dir)
    fastwam_root = Path(configs_dir).expanduser().resolve().parent
    ckpt_path = _resolve_fastwam_path(
        wam_cfg.get("ckpt", None), fastwam_root=fastwam_root
    )
    if ckpt_path is None:
        raise ValueError(
            "actor.model.wam.ckpt is required; adaptive routing cannot use an "
            "untrained/random WAM."
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {ckpt_path}")
    cost_table_path = _resolve_fastwam_path(
        wam_cfg.get("cost_table_path", None), fastwam_root=fastwam_root
    )
    if cost_table_path is None and not bool(
        wam_cfg.get("allow_analytical_cost", False)
    ):
        raise ValueError(
            "actor.model.wam.cost_table_path is required. Profile the exact WAM "
            "checkpoint/resolution, or explicitly set allow_analytical_cost=true "
            "for a non-benchmark smoke test."
        )
    if cost_table_path is not None and not cost_table_path.is_file():
        raise FileNotFoundError(f"FastWAM cost profile not found: {cost_table_path}")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        fcfg = compose(
            config_name="train",
            overrides=[
                f"task={wam_cfg.task}",
                "model.load_text_encoder="
                + str(wam_cfg.get("allow_online_text_encoding", False)).lower(),
            ],
        )
    actual_context_len = int(fcfg.data.train.get("context_len", 128))
    configured_context_len = int(
        wam_cfg.get("context_len", actual_context_len)
    )
    if configured_context_len != actual_context_len:
        raise ValueError(
            "Configured wam.context_len does not match the FastWAM task: "
            f"{configured_context_len} vs {actual_context_len}."
        )
    stats_path = _resolve_dataset_stats_path(
        fcfg, wam_cfg, fastwam_root=fastwam_root
    )
    stats_fingerprint = None if stats_path is None else _sha256_file(stats_path)
    model = instantiate(fcfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(ckpt_path))
    model.eval()
    model.requires_grad_(False)  # WAM stays FROZEN; only the gate trains.

    adapter = WAMModeAdapter(
        model,
        backbone_kind=backbone_kind,
        task=str(wam_cfg.task),
        num_video_frames=int(wam_cfg.get("num_video_frames", 9)),
        generation_horizon=int(wam_cfg.get("generation_horizon", 32)),
        inference_steps=int(wam_cfg.get("inference_steps", 20)),
        context_len=actual_context_len,
        dataset_stats_fingerprint=stats_fingerprint,
        cost_table_path=None if cost_table_path is None else str(cost_table_path),
        cost_source=wam_cfg.get("cost_source", None),
        default_seed=int(wam_cfg.get("default_seed", 0)),
        allow_legacy_checkpoint=bool(wam_cfg.get("allow_legacy_checkpoint", False)),
    )
    return adapter, fcfg, fastwam_root, stats_path


def build_fastwam_processor(
    fcfg: DictConfig,
    wam_cfg: DictConfig,
    *,
    fastwam_root: Path,
    stats_path: Path | None = None,
):
    """Instantiate FastWAM's processor and attach dataset stats for norm conversion."""
    from hydra.utils import instantiate

    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    if stats_path is None:
        stats_path = _resolve_dataset_stats_path(
            fcfg, wam_cfg, fastwam_root=fastwam_root
        )
    if stats_path is None:
        return None

    processor = instantiate(fcfg.data.train.processor)
    processor.eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    return processor


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.gate_policy.gate_policy import GatePolicy

    load_wam = bool(cfg.get("load_wam", True))
    adapter = None
    obs_preprocessor = None
    world_feat_dim = cfg.get("world_feat_dim", None)
    gate_cfg = cfg.get("gate", {})
    text_feat_dim = int(gate_cfg.get("text_feat_dim", cfg.get("text_feat_dim", 64)))
    if load_wam:
        adapter, fcfg, fastwam_root, stats_path = build_wam_adapter(cfg.wam)
        actual_world_feat_dim = int(adapter.world_feat_dim)
        if world_feat_dim is not None and int(world_feat_dim) != actual_world_feat_dim:
            raise ValueError(
                "Configured world_feat_dim does not match the loaded WAM: "
                f"{world_feat_dim} vs {actual_world_feat_dim}."
            )
        actual_proprio_dim = getattr(adapter.model, "proprio_dim", None)
        if actual_proprio_dim is not None and int(cfg.proprio_dim) != int(actual_proprio_dim):
            raise ValueError(
                "Configured proprio_dim does not match the loaded WAM: "
                f"{cfg.proprio_dim} vs {actual_proprio_dim}."
            )
        configured_action_dim = cfg.wam.get("action_dim", None)
        actual_action_dim = int(adapter.model.action_expert.action_dim)
        if configured_action_dim is not None and int(configured_action_dim) != actual_action_dim:
            raise ValueError(
                "Configured wam.action_dim does not match the loaded WAM: "
                f"{configured_action_dim} vs {actual_action_dim}."
            )
        processor = build_fastwam_processor(
            fcfg, cfg.wam, fastwam_root=fastwam_root, stats_path=stats_path
        )
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from rlinf.models.embodiment.gate_policy.obs_preprocessor import (
            make_gate_obs_preprocessor,
        )

        raw_cache_dir = fcfg.data.train.get("text_embedding_cache_dir", None)
        text_cache_dir = _resolve_fastwam_path(
            raw_cache_dir, fastwam_root=fastwam_root
        )
        obs_preprocessor = make_gate_obs_preprocessor(
            adapter.model,
            suite=str(cfg.wam.get("suite", "libero")),
            processor=processor,
            prompt_template=DEFAULT_PROMPT,
            device=str(cfg.wam.get("device", "cuda")),
            text_feat_dim=text_feat_dim,
            text_embedding_cache_dir=text_cache_dir,
            context_len=int(fcfg.data.train.get("context_len", 128)),
            allow_online_text_encoding=bool(
                cfg.wam.get("allow_online_text_encoding", False)
            ),
            binarize_libero_gripper=bool(
                cfg.wam.get("binarize_gripper", True)
            ),
        )
        world_feat_dim = actual_world_feat_dim
    if world_feat_dim is None:
        world_feat_dim = (cfg.get("wam", {}) or {}).get("world_feat_dim", None)
    if world_feat_dim is None:
        raise ValueError(
            "GatePolicy needs world_feat_dim when load_wam=false. Set "
            "actor.model.world_feat_dim or actor.model.wam.world_feat_dim."
        )

    hidden = tuple(gate_cfg.get("hidden_sizes", (256, 256)))
    if bool(gate_cfg.get("use_last_action", False)):
        raise ValueError(
            "gate.use_last_action was removed: the rollout has no reset-aware "
            "per-environment mode history. Use a recurrent gate if history is needed."
        )
    kl_cfg = gate_cfg.get("kl_prior", {}) or {}
    kl_enabled = bool(kl_cfg.get("enabled", False))
    if not kl_enabled and any(
        float(kl_cfg.get(key, 0.0)) != 0.0 for key in ("beta", "beta_end")
    ):
        raise ValueError(
            "gate.kl_prior beta values are nonzero but kl_prior.enabled is false."
        )

    policy = GatePolicy(
        world_feat_dim=int(world_feat_dim),
        proprio_dim=int(cfg.proprio_dim),
        text_feat_dim=text_feat_dim,
        num_modes=int(gate_cfg.get("num_modes", 2)),
        hidden_sizes=hidden,
        add_value_head=bool(cfg.get("add_value_head", True)),
        activation=str(gate_cfg.get("activation", "tanh")),
        explore_eps=float(gate_cfg.get("explore_eps", 0.0)),
        force_mode=gate_cfg.get("force_mode", None),
        allow_legacy_gate_checkpoint=bool(
            gate_cfg.get("allow_legacy_gate_checkpoint", False)
        ),
        kl_prior_beta=float(kl_cfg.get("beta", 0.0)),
        kl_prior_beta_end=float(kl_cfg.get("beta_end", 0.0)),
        kl_prior_decay_steps=int(kl_cfg.get("decay_steps", 0)),
        wam_adapter=adapter,
        obs_preprocessor=obs_preprocessor,
    )

    wam_cfg = cfg.get("wam", {}) or {}
    configs_value = wam_cfg.get("configs_dir", None)
    fastwam_root_for_profile = (
        Path(str(configs_value)).expanduser().resolve().parent
        if not _is_none_like(configs_value)
        else Path.cwd()
    )
    profile_path = _resolve_fastwam_path(
        wam_cfg.get("cost_table_path", None),
        fastwam_root=fastwam_root_for_profile,
    )
    allow_analytical_cost = bool(wam_cfg.get("allow_analytical_cost", False))
    if profile_path is None and not allow_analytical_cost:
        raise ValueError(
            "GatePolicy training/evaluation requires wam.cost_table_path unless "
            "allow_analytical_cost=true is explicitly set for a smoke test."
        )
    loaded_profile_meta = dict(getattr(adapter, "_cost_meta", None) or {})
    stats_path = _resolve_dataset_stats_path(
        None,
        wam_cfg,
        fastwam_root=fastwam_root_for_profile,
    )
    stats_fingerprint = (
        getattr(adapter, "dataset_stats_fingerprint", None)
        if adapter is not None
        else (None if stats_path is None else _sha256_file(stats_path))
    )
    if profile_path is None and allow_analytical_cost:
        ckpt_path = _resolve_fastwam_path(
            wam_cfg.get("ckpt", None), fastwam_root=fastwam_root_for_profile
        )
        if ckpt_path is None or not ckpt_path.is_file():
            raise FileNotFoundError(
                "Analytical-cost smoke tests still require the exact wam.ckpt "
                "for file-identity provenance."
            )
        checkpoint_fingerprint = f"file-sha256:{_sha256_file(ckpt_path)}"
    else:
        checkpoint_fingerprint = (
            getattr(adapter.model, "_loaded_checkpoint_fingerprint", None)
            if adapter is not None
            else loaded_profile_meta.get("ckpt_fingerprint")
        )
    policy.bc_expected_provenance = {
        "task": None
        if _is_none_like(wam_cfg.get("task", None))
        else str(wam_cfg.get("task")),
        "backbone_kind": str(wam_cfg.get("backbone_kind", "idm")).lower(),
        "ckpt_fingerprint": checkpoint_fingerprint,
        "dataset_stats_fingerprint": stats_fingerprint,
        "num_video_frames": int(wam_cfg.get("num_video_frames", 9)),
        "inference_steps": int(wam_cfg.get("inference_steps", 20)),
        "context_len": int(
            getattr(adapter, "context_len", wam_cfg.get("context_len", 128))
        ),
        "model_dtype": str(
            getattr(
                getattr(adapter, "model", None),
                "torch_dtype",
                _resolve_wam_dtype(wam_cfg.get("dtype", "bfloat16")),
            )
        ),
        "exec_horizon": int(wam_cfg.get("exec_horizon", 0)),
        "action_horizon": int(wam_cfg.get("generation_horizon", 0)),
        "cost_table_path": None if profile_path is None else str(profile_path),
    }

    # OPTIONAL (off by default; the gate needs NO supervision): warm-start from a
    # BC checkpoint and/or attach it as the frozen KL prior. `runner.ckpt_path`
    # also loads BC weights into the actor; `bc_init_path` additionally covers
    # the rollout instance and eval-only runs.
    bc_init_path = gate_cfg.get("bc_init_path", None)
    if not _is_none_like(bc_init_path):
        from rlinf.models.embodiment.gate_policy.bc import load_gate_bc_state

        state = load_gate_bc_state(
            str(bc_init_path), expected_policy=policy
        )
        arch_hint = (
            f"gate BC init {bc_init_path} does not match the configured gate "
            "architecture. Re-run train_gate_bc.py with the same "
            "gate.hidden_sizes/activation/add_value_head."
        )
        try:
            missing, unexpected = policy.load_bc_init(state)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"{arch_hint} ({exc})") from exc
        if missing or unexpected:
            print(f"[gate_policy] BC init loaded with missing={missing} unexpected={unexpected}")

    if kl_enabled:
        from rlinf.models.embodiment.gate_policy.bc import load_gate_bc_state

        prior_path = kl_cfg.get("path", None)
        if _is_none_like(prior_path):
            prior_path = bc_init_path
        if _is_none_like(prior_path):
            raise ValueError(
                "gate.kl_prior.enabled=True needs gate.kl_prior.path or gate.bc_init_path."
            )
        policy.attach_bc_prior(
            load_gate_bc_state(str(prior_path), expected_policy=policy)
        )
    return policy
