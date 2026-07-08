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

from pathlib import Path

import torch
from omegaconf import DictConfig


def _is_none_like(value) -> bool:
    return value is None or str(value).lower() in {"", "none", "null"}


def _resolve_fastwam_path(value, *, fastwam_root: Path) -> Path | None:
    if _is_none_like(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = fastwam_root / path
    return path.resolve()


def build_wam_adapter(wam_cfg: DictConfig):
    """Build a FROZEN fast-wam dual-regime model and wrap it in WAMModeAdapter.

    Required `wam_cfg` keys:
      configs_dir        : path to FastWAM/configs (for hydra compose)
      task               : fastwam task name (e.g. libero_metric_adaptive_joint_2cam224_1e-4)
      backbone_kind      : "joint" | "idm"
      ckpt               : path to the dual-regime checkpoint
    Optional:
      num_video_frames (9), action_horizon (32), k_lo (4), k_hi (20),
      cost_table_path (None), device ("cuda"), dtype ("bfloat16")
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from fastwam.adaptive_gate import WAMModeAdapter

    device = str(wam_cfg.get("device", "cuda"))
    dtype = getattr(torch, str(wam_cfg.get("dtype", "bfloat16")))
    configs_dir = str(wam_cfg.configs_dir)
    fastwam_root = Path(configs_dir).expanduser().resolve().parent
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        fcfg = compose(
            config_name="train",
            overrides=[
                f"task={wam_cfg.task}",
                f"model.load_text_encoder={str(wam_cfg.get('load_text_encoder', True)).lower()}",
            ],
        )
    model = instantiate(fcfg.model, model_dtype=dtype, device=device)
    ckpt = wam_cfg.get("ckpt", None)
    if ckpt:
        model.load_checkpoint(str(ckpt))
    model.eval()
    model.requires_grad_(False)  # WAM stays FROZEN; only the gate trains.

    adapter = WAMModeAdapter(
        model,
        backbone_kind=str(wam_cfg.backbone_kind),
        num_video_frames=int(wam_cfg.get("num_video_frames", 9)),
        action_horizon=int(wam_cfg.get("action_horizon", 32)),
        k_lo=int(wam_cfg.get("k_lo", 4)),
        k_hi=int(wam_cfg.get("k_hi", 20)),
        cost_table_path=wam_cfg.get("cost_table_path", None),
        cost_source=wam_cfg.get("cost_source", None),
        default_seed=wam_cfg.get("default_seed", None),
    )
    return adapter, fcfg, fastwam_root


def build_fastwam_processor(fcfg: DictConfig, wam_cfg: DictConfig, *, fastwam_root: Path):
    """Instantiate FastWAM's processor and attach dataset stats for norm conversion."""
    from hydra.utils import instantiate

    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    stats_path = _resolve_fastwam_path(
        wam_cfg.get("dataset_stats_path", None),
        fastwam_root=fastwam_root,
    )
    if stats_path is None:
        stats_path = _resolve_fastwam_path(
            fcfg.data.train.get("pretrained_norm_stats", None),
            fastwam_root=fastwam_root,
        )
    if stats_path is None and not _is_none_like(wam_cfg.get("ckpt", None)):
        ckpt_path = _resolve_fastwam_path(wam_cfg.get("ckpt"), fastwam_root=fastwam_root)
        for parent in (ckpt_path.parent, ckpt_path.parent.parent):
            candidate = parent / "dataset_stats.json"
            if candidate.exists():
                stats_path = candidate.resolve()
                break

    require_stats = bool(wam_cfg.get("require_dataset_stats", True))
    if stats_path is None:
        if require_stats:
            raise FileNotFoundError(
                "GatePolicy requires FastWAM dataset stats to normalize proprio and "
                "denormalize actions. Set actor.model.wam.dataset_stats_path, or place "
                "dataset_stats.json next to the checkpoint."
            )
        return None
    if not stats_path.exists():
        raise FileNotFoundError(f"FastWAM dataset stats not found: {stats_path}")

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
    if load_wam:
        adapter, fcfg, fastwam_root = build_wam_adapter(cfg.wam)
        processor = build_fastwam_processor(fcfg, cfg.wam, fastwam_root=fastwam_root)
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from rlinf.models.embodiment.gate_policy.obs_preprocessor import (
            make_gate_obs_preprocessor,
        )

        obs_preprocessor = make_gate_obs_preprocessor(
            adapter.model,
            suite=str(cfg.wam.get("suite", "libero")),
            processor=processor,
            prompt_template=DEFAULT_PROMPT,
            device=str(cfg.wam.get("device", "cuda")),
        )
        world_feat_dim = adapter.world_feat_dim
    if world_feat_dim is None:
        world_feat_dim = cfg.wam.get("world_feat_dim", None)
    if world_feat_dim is None:
        raise ValueError(
            "GatePolicy needs world_feat_dim when load_wam=false. Set "
            "actor.model.world_feat_dim or actor.model.wam.world_feat_dim."
        )

    gate_cfg = cfg.get("gate", {})
    hidden = tuple(gate_cfg.get("hidden_sizes", (256, 256)))

    policy = GatePolicy(
        world_feat_dim=int(world_feat_dim),
        proprio_dim=int(cfg.proprio_dim),
        num_modes=int(gate_cfg.get("num_modes", 3)),
        hidden_sizes=hidden,
        add_value_head=bool(cfg.get("add_value_head", True)),
        use_last_action=bool(gate_cfg.get("use_last_action", False)),
        activation=str(gate_cfg.get("activation", "tanh")),
        explore_eps=float(gate_cfg.get("explore_eps", 0.0)),
        wam_adapter=adapter,
        obs_preprocessor=obs_preprocessor,
    )

    # OPTIONAL (off by default; the gate needs NO supervision): warm-start from a
    # BC checkpoint and/or attach it as the frozen KL prior. `runner.ckpt_path`
    # also loads BC weights into the actor; `bc_init_path` additionally covers
    # the rollout instance and eval-only runs.
    bc_init_path = gate_cfg.get("bc_init_path", None)
    if not _is_none_like(bc_init_path):
        from rlinf.models.embodiment.gate_policy.bc import load_gate_bc_state

        state = load_gate_bc_state(str(bc_init_path))
        arch_hint = (
            f"gate BC init {bc_init_path} does not match the configured gate "
            "architecture. Re-run train_gate_bc.py with the same "
            "gate.hidden_sizes/activation/add_value_head."
        )
        try:
            missing, unexpected = policy.load_state_dict(state, strict=False)
        except RuntimeError as exc:  # same key, different shape (hidden_sizes drift)
            raise ValueError(f"{arch_hint} ({exc})") from exc
        # value-head keys may legitimately differ (BC w/o head -> PPO w/ head);
        # anything else missing means an architecture mismatch with the BC run.
        bad_missing = [k for k in missing if not k.startswith("value_head.")]
        if bad_missing:
            raise ValueError(f"{arch_hint} Missing keys: {bad_missing}.")
        if missing or unexpected:
            print(f"[gate_policy] BC init loaded with missing={missing} unexpected={unexpected}")

    kl_cfg = gate_cfg.get("kl_prior", {}) or {}
    if bool(kl_cfg.get("enabled", False)):
        from rlinf.models.embodiment.gate_policy.bc import load_gate_bc_state

        prior_path = kl_cfg.get("path", None)
        if _is_none_like(prior_path):
            prior_path = bc_init_path
        if _is_none_like(prior_path):
            raise ValueError(
                "gate.kl_prior.enabled=True needs gate.kl_prior.path or gate.bc_init_path."
            )
        policy.attach_bc_prior(load_gate_bc_state(str(prior_path)))
    return policy
