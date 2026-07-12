# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OPTIONAL BC (SFT) warm-start for the adaptive-prediction gate (M3).

The gate's default training path is pure GRPO and consumes NO supervision (see
adaptive_gate_README.md); nothing here is on that path. This module exists as
an accelerator/ablation: it trains the SAME `GatePolicy` the RL stage uses
(via `GatePolicy.mode_logits`) with cross-entropy on self-generated oracle mode
labels from raw VLA data (`FastWAM/scripts/generate_gate_oracle_labels.py`) —
no human annotation, no simulator, no WAM at SFT time (the shards already carry
the gate inputs `world_feat` + `proprio`).

Recipe (optional SFT -> RL):
  1. fastwam: generate oracle-label shards (heavy, frozen WAM, once).
  2. here:    `train_gate_bc` -> a RAW GatePolicy state_dict on disk.
  3. GRPO:    point `runner.ckpt_path` (and/or `actor.model.gate.bc_init_path`)
              at that file; optionally enable `gate.kl_prior` + a decaying
              an actor-side decaying KL so early RL stays near the BC gate.

The checkpoint written by `save_gate_bc_checkpoint` is a bare `state_dict` —
exactly what `EmbodiedFSDPActor.model_provider_func` feeds to
`model.load_state_dict(torch.load(runner.ckpt_path))` — plus a `<path>.meta.json`
sidecar for provenance. Construct the BC policy with the SAME `hidden_sizes` /
`activation` / `add_value_head` as the RL config or the strict load will fail.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

# Keys the BC stage needs from each oracle-label shard record.
BC_DATA_KEYS = ("world_feat", "proprio", "text_feat", "label")


def _mode_names(mode_order) -> list[str]:
    return [str(getattr(mode, "value", mode)) for mode in mode_order]


def _validate_oracle_shard_semantics(
    data: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
    *,
    world_feat_layout: str,
    text_feat_layout: str,
    mode_order,
) -> None:
    """Reject shape-compatible shards whose feature semantics differ online."""
    expected_modes = _mode_names(mode_order)
    if expected_modes != ["uncond", "idm"]:
        raise RuntimeError(
            f"FastWAM exposes unexpected adaptive-gate modes {expected_modes}; "
            "RLinf requires ['uncond', 'idm']."
        )
    expected_layouts = {
        "world_feat_layout": str(world_feat_layout),
        "text_feat_layout": str(text_feat_layout),
    }
    for key, expected in expected_layouts.items():
        actual = meta.get(key)
        if actual != expected:
            raise ValueError(
                f"oracle shard meta.{key}={actual!r} does not match the online "
                f"FastWAM layout {expected!r}; regenerate the labels."
            )
    actual_modes = list(meta.get("mode_order", []))
    if actual_modes != expected_modes:
        raise ValueError(
            f"oracle shard meta.mode_order={actual_modes!r} does not match "
            f"{expected_modes!r}."
        )

    dimensions = {
        "world_feat": "world_feat_dim",
        "proprio": "proprio_dim",
        "text_feat": "text_feat_dim",
    }
    for tensor_key, meta_key in dimensions.items():
        tensor = data.get(tensor_key)
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            shape = None if not torch.is_tensor(tensor) else tuple(tensor.shape)
            raise ValueError(f"oracle shard `{tensor_key}` must be [N,D], got {shape}")
        if meta_key not in meta:
            raise ValueError(f"oracle shard metadata is missing `{meta_key}`")
        declared = int(meta[meta_key])
        actual = int(tensor.shape[-1])
        if declared != actual:
            raise ValueError(
                f"oracle shard meta.{meta_key}={declared} does not match "
                f"`{tensor_key}` dimension {actual}."
            )


class GateOracleLabelDataset(torch.utils.data.Dataset):
    """(world_feat, proprio, task text) -> oracle mode label.

    Build from shard files with `from_shards` (delegates parsing/validation to
    `fastwam.adaptive_gate` so the format has a single source of truth); tests
    and ad-hoc callers can pass tensors directly.
    """

    def __init__(self, data: dict[str, torch.Tensor], meta: Optional[dict[str, Any]] = None):
        missing = [k for k in BC_DATA_KEYS if k not in data]
        if missing:
            raise ValueError(f"gate BC dataset is missing keys: {missing}")
        n = int(data["label"].shape[0])
        for key in BC_DATA_KEYS:
            if int(data[key].shape[0]) != n:
                raise ValueError(
                    f"inconsistent leading dim for `{key}`: {int(data[key].shape[0])} vs {n}"
                )
        for key in ("world_feat", "proprio", "text_feat"):
            if data[key].ndim != 2:
                raise ValueError(f"`{key}` must be [N,D], got {tuple(data[key].shape)}")
        if data["label"].ndim != 1:
            raise ValueError(f"`label` must be [N], got {tuple(data['label'].shape)}")
        self.world_feat = data["world_feat"].float()
        self.proprio = data["proprio"].float()
        self.text_feat = data["text_feat"].float()
        self.label = data["label"].long()
        if self.label.numel() and (
            int(self.label.min()) < 0 or int(self.label.max()) > 1
        ):
            raise ValueError("two-mode gate labels must be UNCOND=0 or IDM=1")
        self.group_id = None
        for key in ("episode_id", "group_id", "task_id"):
            value = data.get(key)
            if torch.is_tensor(value) and int(value.shape[0]) == n:
                candidate = value.reshape(n).long()
                if bool((candidate >= 0).all()):
                    self.group_id = candidate
                    self.group_key = key
                else:
                    warnings.warn(
                        f"`{key}` contains unknown negative ids; disabling group-wise "
                        "BC splitting so unrelated unknown rows are not treated as one episode.",
                        stacklevel=2,
                    )
                break
        self.meta = dict(meta or {})

    @classmethod
    def from_shards(
        cls,
        shards: str | Sequence[str],
        *,
        relabel: Optional[dict[str, Any]] = None,
        max_best_err: Optional[float] = None,
        max_idm_err: Optional[float] = None,
        quality_warn_threshold: float = 0.5,
    ) -> "GateOracleLabelDataset":
        """Load shard file(s) (glob/path/list). Optional `relabel` kwargs
        (metric/exec_horizon/tol_abs/tol_rel) re-derive labels offline from the
        stored error curves; rows whose masked window is empty are dropped."""
        try:
            from fastwam.adaptive_gate import (
                MODE_ORDER,
                TEXT_FEAT_LAYOUT,
                WORLD_FEAT_LAYOUT,
                load_label_shards,
                quality_metadata,
                relabel_from_steps,
            )
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "loading oracle-label shards requires the fastwam package "
                "(`pip install -e FastWAM`), which defines the shard format."
            ) from exc

        data, meta = load_label_shards(shards)
        _validate_oracle_shard_semantics(
            data,
            meta,
            world_feat_layout=WORLD_FEAT_LAYOUT,
            text_feat_layout=TEXT_FEAT_LAYOUT,
            mode_order=MODE_ORDER,
        )
        keep = torch.ones(data["label"].shape[0], dtype=torch.bool)
        effective_relabel = None
        if relabel:
            effective_relabel = dict(relabel)
            if effective_relabel.get("exec_horizon") is None:
                source_horizon = meta.get("exec_horizon")
                if source_horizon is None:
                    raise ValueError(
                        "relabeling without an explicit exec_horizon requires "
                        "meta.exec_horizon in the oracle shards"
                    )
                effective_relabel["exec_horizon"] = int(source_horizon)
            labels, chunk_err, has_valid = relabel_from_steps(
                data["step_l1"],
                data["step_l2"],
                data["valid_steps"],
                **effective_relabel,
            )
            data["label"] = labels
            data["chunk_err"] = chunk_err
            data.update(quality_metadata(chunk_err))
            keep &= has_valid

        best_err = data["best_err"].float()
        idm_err = data["idm_err"].float()
        threshold = float(quality_warn_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("quality_warn_threshold must be finite and non-negative")
        for name, value in (("max_best_err", max_best_err), ("max_idm_err", max_idm_err)):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        low_quality = (~torch.isfinite(best_err)) | (best_err > threshold)
        quality_report = {
            "warn_threshold": threshold,
            "low_quality_count": int(low_quality.sum()),
            "low_quality_fraction": float(low_quality.float().mean()),
            "nonfinite_best_count": int((~torch.isfinite(best_err)).sum()),
            "mean_best_err": float(best_err[torch.isfinite(best_err)].mean())
            if bool(torch.isfinite(best_err).any())
            else float("inf"),
            "mean_idm_err": float(idm_err[torch.isfinite(idm_err)].mean())
            if bool(torch.isfinite(idm_err).any())
            else float("inf"),
        }
        if max_best_err is not None:
            keep &= torch.isfinite(best_err) & (best_err <= float(max_best_err))
        if max_idm_err is not None:
            keep &= torch.isfinite(idm_err) & (idm_err <= float(max_idm_err))
        if max_best_err is None and max_idm_err is None and bool(low_quality.any()):
            warnings.warn(
                "BC oracle data has no absolute-quality filter: "
                f"{quality_report['low_quality_fraction']:.1%} of rows have "
                f"best_err > {threshold:g} or non-finite error. Consider "
                "max_best_err/max_idm_err so states where both modes fail do not "
                "teach the warm-start.",
                stacklevel=2,
            )
        data = {key: value[keep] for key, value in data.items()}
        meta = {
            **meta,
            "source_exec_horizon": meta.get("exec_horizon"),
            "exec_horizon": (
                int(effective_relabel["exec_horizon"])
                if effective_relabel is not None
                else meta.get("exec_horizon")
            ),
            "relabel": effective_relabel,
            "quality_filter": {
                "max_best_err": max_best_err,
                "max_idm_err": max_idm_err,
            },
            "quality_report": quality_report,
            "num_samples_before_quality_filter": int(keep.numel()),
            "num_samples": int(keep.sum()),
        }
        if not bool(keep.any()):
            raise ValueError("oracle quality filters removed every BC sample")
        return cls(data, meta)

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "world_feat": self.world_feat[idx],
            "proprio": self.proprio[idx],
            "text_feat": self.text_feat[idx],
            "label": self.label[idx],
        }

    @property
    def world_feat_dim(self) -> int:
        return int(self.world_feat.shape[-1])

    @property
    def proprio_dim(self) -> int:
        return int(self.proprio.shape[-1])

    @property
    def text_feat_dim(self) -> int:
        return int(self.text_feat.shape[-1])


def class_balance_weights(
    labels: torch.Tensor,
    num_classes: int,
    *,
    power: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency^power CE weights, normalized to mean 1 over PRESENT classes.

    This is opt-in because inverse-frequency weighting changes posterior
    calibration and can make the expensive mode over-confident. ``power=0`` keeps
    ordinary maximum-likelihood CE.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    present = counts > 0
    if not bool(present.any()):
        raise ValueError("labels tensor is empty")
    weights = torch.zeros(num_classes)
    weights[present] = (1.0 / counts[present]) ** float(power)
    # normalize to mean 1 over present classes (keeps loss scale ~ unweighted CE)
    weights[present] = weights[present] * present.sum().float() / weights[present].sum()
    return weights


def expected_mode_cost(
    labels: torch.Tensor, cost_table: Optional[dict[str, float]]
) -> Optional[float]:
    """Mean cost of a mode sequence under the shard's cost table (IDM=1)."""
    if not cost_table:
        return None
    try:
        from fastwam.adaptive_gate import MODE_ORDER

        costs = torch.tensor([float(cost_table[m.value]) for m in MODE_ORDER])
    except Exception:
        costs = torch.tensor([float(cost_table[k]) for k in ("uncond", "idm")])
    return float(costs[labels.long()].mean().item())


@dataclass
class GateBCConfig:
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    val_fraction: float = 0.1
    class_weight_power: float = 0.0  # opt-in; weighting changes calibration
    label_smoothing: float = 0.0
    device: str = "cpu"
    seed: int = 0
    log_every_epochs: int = 1
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(float(self.val_fraction)) or not 0.0 <= float(
            self.val_fraction
        ) < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if not math.isfinite(float(self.class_weight_power)) or float(
            self.class_weight_power
        ) < 0.0:
            raise ValueError("class_weight_power must be finite and non-negative")
        if not math.isfinite(float(self.label_smoothing)) or not 0.0 <= float(
            self.label_smoothing
        ) < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not math.isfinite(float(self.log_every_epochs)) or float(
            self.log_every_epochs
        ) < 0.0:
            raise ValueError("log_every_epochs must be finite and non-negative")


@torch.no_grad()
def evaluate_gate_bc(
    policy,
    world_feat: torch.Tensor,
    proprio: torch.Tensor,
    text_feat: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 4096,
    cost_table: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Accuracy / per-mode recall / predicted-mode distribution (+ mean cost)."""
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    preds = []
    for start in range(0, labels.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        logits = policy.mode_logits(
            world_feat[sl].to(device),
            proprio[sl].to(device),
            text_feat[sl].to(device),
        )
        preds.append(logits.argmax(dim=-1).cpu())
    if was_training:
        policy.train()
    pred = torch.cat(preds)
    labels = labels.long().cpu()
    num_modes = int(policy.num_modes)
    metrics: dict[str, Any] = {"accuracy": float((pred == labels).float().mean().item())}
    for mode_idx in range(num_modes):
        mask = labels == mode_idx
        metrics[f"recall/mode_{mode_idx}"] = (
            float((pred[mask] == mode_idx).float().mean().item())
            if bool(mask.any())
            else float("nan")
        )
        metrics[f"pred_frac/mode_{mode_idx}"] = float((pred == mode_idx).float().mean().item())
        metrics[f"label_frac/mode_{mode_idx}"] = float(mask.float().mean().item())
    pred_cost = expected_mode_cost(pred, cost_table)
    oracle_cost = expected_mode_cost(labels, cost_table)
    if pred_cost is not None:
        metrics["mean_cost/pred"] = pred_cost
        metrics["mean_cost/oracle"] = oracle_cost
    return metrics


def train_gate_bc(policy, dataset: GateOracleLabelDataset, cfg: GateBCConfig) -> dict[str, Any]:
    """Cross-entropy BC of the gate on oracle labels; returns metrics history.

    Trains only what RL trains (`backbone` + `logits_head`; a value head, if
    constructed, is left at init for the RL critic to learn). Keeps the BEST
    val-accuracy weights and restores them before returning.
    """
    generator = torch.Generator().manual_seed(int(cfg.seed))
    n = len(dataset)
    if n < 2:
        raise ValueError(f"gate BC needs at least 2 samples, got {n}")
    n_val = int(math.floor(n * float(cfg.val_fraction)))
    if (
        n_val > 0
        and dataset.group_id is not None
        and torch.unique(dataset.group_id).numel() > 1
    ):
        groups = torch.unique(dataset.group_id)
        groups = groups[torch.randperm(groups.numel(), generator=generator)]
        n_val_groups = min(
            max(1, int(round(groups.numel() * float(cfg.val_fraction)))),
            groups.numel() - 1,
        )
        val_groups = groups[:n_val_groups]
        val_mask = torch.isin(dataset.group_id, val_groups)
        val_idx = torch.nonzero(val_mask, as_tuple=False).flatten()
        train_idx = torch.nonzero(~val_mask, as_tuple=False).flatten()
        split_kind = f"group:{dataset.group_key}"
    else:
        if n_val > 0:
            warnings.warn(
                "Oracle shards have no usable episode/group id; BC validation uses "
                "a row-wise split and may leak adjacent states.",
                stacklevel=2,
            )
        perm = torch.randperm(n, generator=generator)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        split_kind = "row"
    if train_idx.numel() == 0:
        raise ValueError("val_fraction leaves no training samples")

    device = torch.device(cfg.device)
    policy = policy.to(device)
    world_feat = dataset.world_feat
    proprio = dataset.proprio
    text_feat = dataset.text_feat
    labels = dataset.label
    train_labels = labels[train_idx]

    weights = None
    if float(cfg.class_weight_power) > 0.0:
        weights = class_balance_weights(
            train_labels, int(policy.num_modes), power=float(cfg.class_weight_power)
        ).to(device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), betas=(0.9, 0.95)
    )

    def _val_metrics() -> dict[str, Any]:
        idx = val_idx if val_idx.numel() > 0 else train_idx
        return evaluate_gate_bc(
            policy,
            world_feat[idx],
            proprio[idx],
            text_feat[idx],
            labels[idx],
            cost_table=dataset.meta.get("cost_table"),
        )

    history: list[dict[str, Any]] = []
    best = {"accuracy": -1.0}
    best_state = None
    steps_per_epoch = max(1, math.ceil(train_idx.numel() / int(cfg.batch_size)))
    for epoch in range(int(cfg.epochs)):
        policy.train()
        epoch_perm = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch_idx = epoch_perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            if batch_idx.numel() == 0:
                continue
            logits = policy.mode_logits(
                world_feat[batch_idx].to(device),
                proprio[batch_idx].to(device),
                text_feat[batch_idx].to(device),
            )
            loss = F.cross_entropy(
                logits,
                labels[batch_idx].to(device),
                weight=weights,
                label_smoothing=float(cfg.label_smoothing),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        val = _val_metrics()
        record = {"epoch": epoch, "train_loss": epoch_loss / steps_per_epoch, **val}
        history.append(record)
        if val["accuracy"] > best["accuracy"]:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        if cfg.log_every_epochs and (epoch % int(cfg.log_every_epochs) == 0):
            pred_fractions = [
                round(val[f"pred_frac/mode_{i}"], 3)
                for i in range(policy.num_modes)
            ]
            print(
                f"[gate-bc] epoch {epoch:03d} loss={record['train_loss']:.4f} "
                f"val_acc={val['accuracy']:.4f} "
                f"pred_frac={pred_fractions}"
            )

    if best_state is not None:
        policy.load_state_dict(best_state)
    return {
        "best": best,
        "history": history,
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
        "split_kind": split_kind,
        "config": asdict(cfg),
    }


def save_gate_bc_checkpoint(policy, path: str, *, meta: Optional[dict[str, Any]] = None) -> str:
    """Write the policy's RAW state_dict at `path` (+ `<path>.meta.json`).

    Raw on purpose: `runner.ckpt_path` and `gate.bc_init_path` both feed the file
    straight into `load_state_dict`.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    torch.save(state, path)
    if meta is not None:
        with open(path + ".meta.json", "w") as handle:
            json.dump(meta, handle, indent=2, default=str)
    return path


def _resolve_expected_bc_provenance(policy, *, checkpoint_path: str) -> dict[str, Any]:
    provenance = getattr(policy, "bc_expected_provenance", None)
    if not isinstance(provenance, Mapping):
        raise ValueError(
            f"{checkpoint_path}: cannot validate BC task/WAM provenance because the "
            "configured GatePolicy has no WAM provenance. Build it through get_model."
        )
    task = provenance.get("task")
    backbone_kind = str(provenance.get("backbone_kind", "")).lower()
    if not task or backbone_kind != "idm":
        raise ValueError(
            f"{checkpoint_path}: invalid expected WAM provenance "
            f"task={task!r}, backbone_kind={backbone_kind!r}; binary BC requires IDM."
        )
    fingerprint = provenance.get("ckpt_fingerprint")
    if not fingerprint:
        cost_path = provenance.get("cost_table_path")
        if not cost_path or not os.path.isfile(str(cost_path)):
            raise ValueError(
                f"{checkpoint_path}: BC loading needs a measured cost profile with "
                "meta.ckpt_fingerprint; configured profile is unavailable: "
                f"{cost_path!r}."
            )
        try:
            import yaml

            with open(str(cost_path), encoding="utf-8") as handle:
                cost_payload = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"{cost_path}: invalid cost profile: {exc}") from exc
        cost_meta = cost_payload.get("meta") if isinstance(cost_payload, dict) else None
        if not isinstance(cost_meta, dict):
            raise ValueError(f"{cost_path}: cost profile is missing its `meta` block")
        profile_identity = {
            "task": cost_meta.get("task"),
            "backbone_kind": cost_meta.get("backbone_kind"),
        }
        expected_identity = {"task": str(task), "backbone_kind": "idm"}
        if profile_identity != expected_identity:
            raise ValueError(
                f"{cost_path}: cost profile identity does not match the configured "
                f"gate (actual, expected): {profile_identity}, {expected_identity}"
            )
        fingerprint = cost_meta.get("ckpt_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(
            f"{checkpoint_path}: expected WAM checkpoint fingerprint is unavailable"
        )
    required_semantics = {
        "dataset_stats_fingerprint": provenance.get("dataset_stats_fingerprint"),
        "num_video_frames": provenance.get("num_video_frames"),
        "inference_steps": provenance.get("inference_steps"),
        "context_len": provenance.get("context_len"),
        "model_dtype": provenance.get("model_dtype"),
        "exec_horizon": provenance.get("exec_horizon"),
        "action_horizon": provenance.get("action_horizon"),
    }
    missing_semantics = []
    for key, value in required_semantics.items():
        if key in {"dataset_stats_fingerprint", "model_dtype"}:
            if not isinstance(value, str) or not value:
                missing_semantics.append(key)
            continue
        try:
            valid = int(value) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            missing_semantics.append(key)
    if missing_semantics:
        raise ValueError(
            f"{checkpoint_path}: configured gate is missing oracle semantics "
            f"{missing_semantics}; set exact stats/solver/frame/horizon values."
        )
    return {
        "task": str(task),
        "backbone_kind": "idm",
        "ckpt_fingerprint": fingerprint,
        "dataset_stats_fingerprint": str(
            required_semantics["dataset_stats_fingerprint"]
        ),
        "num_video_frames": int(required_semantics["num_video_frames"]),
        "inference_steps": int(required_semantics["inference_steps"]),
        "context_len": int(required_semantics["context_len"]),
        "model_dtype": str(required_semantics["model_dtype"]),
        "exec_horizon": int(required_semantics["exec_horizon"]),
        "action_horizon": int(required_semantics["action_horizon"]),
    }


def build_gate_policy_sidecar_metadata(
    policy, *, kind: str = "gate_rl", step: Optional[int] = None
) -> dict[str, Any]:
    """Build provenance for a learned gate state dict saved by RL training."""
    if kind != "gate_rl":
        raise ValueError(f"unsupported learned-gate sidecar kind {kind!r}")
    try:
        from fastwam.adaptive_gate import (
            MODE_ORDER,
            TEXT_FEAT_LAYOUT,
            WORLD_FEAT_LAYOUT,
        )
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("saving GatePolicy provenance requires fastwam") from exc
    gate = {
        "world_feat_dim": int(policy.world_feat_dim),
        "proprio_dim": int(policy.proprio_dim),
        "text_feat_dim": int(policy.text_feat_dim),
        "num_modes": int(policy.num_modes),
        "hidden_sizes": list(policy.hidden_sizes),
        "activation": str(policy.activation),
        "add_value_head": bool(policy.add_value_head),
    }
    wam_provenance = {
        "world_feat_layout": WORLD_FEAT_LAYOUT,
        "text_feat_layout": TEXT_FEAT_LAYOUT,
        "mode_order": _mode_names(MODE_ORDER),
        "world_feat_dim": gate["world_feat_dim"],
        "proprio_dim": gate["proprio_dim"],
        "text_feat_dim": gate["text_feat_dim"],
        **_resolve_expected_bc_provenance(policy, checkpoint_path="<gate-save>"),
    }
    return {
        "kind": kind,
        "schema_version": 1,
        "step": None if step is None else int(step),
        "gate": gate,
        "wam_provenance": wam_provenance,
    }


def save_gate_policy_sidecar(path: str, metadata: Mapping[str, Any]) -> str:
    """Write the metadata next to an already-saved gate state dict."""
    if metadata.get("kind") != "gate_rl":
        raise ValueError("learned GatePolicy sidecar must have kind='gate_rl'")
    sidecar_path = path + ".meta.json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, default=str)
    return sidecar_path


def validate_gate_bc_sidecar(path: str, policy) -> Optional[dict[str, Any]]:
    """Validate a BC checkpoint's architecture and feature provenance sidecar.

    Missing sidecars fail closed unless the policy explicitly enables the legacy
    escape hatch. A present sidecar is authoritative and must match exactly.
    """
    sidecar_path = path + ".meta.json"
    if not os.path.isfile(sidecar_path):
        if not bool(getattr(policy, "allow_legacy_gate_checkpoint", False)):
            raise ValueError(
                f"{path}: GatePolicy checkpoint has no provenance sidecar. Set "
                "gate.allow_legacy_gate_checkpoint=true only after manually "
                "verifying task/WAM/stats/horizon compatibility."
            )
        warnings.warn(
            f"{path}: GatePolicy checkpoint has no .meta.json provenance sidecar; "
            "loading as a legacy checkpoint after state_dict shape checks only.",
            stacklevel=2,
        )
        return None
    try:
        with open(sidecar_path) as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{sidecar_path}: invalid Gate BC metadata: {exc}") from exc
    kind = meta.get("kind") if isinstance(meta, dict) else None
    if kind not in {"gate_bc", "gate_rl"}:
        raise ValueError(
            f"{sidecar_path}: expected kind='gate_bc' or 'gate_rl' metadata"
        )
    gate_meta = meta.get("gate")
    if not isinstance(gate_meta, dict):
        raise ValueError(f"{sidecar_path}: metadata is missing the `gate` block")

    expected_gate = {
        "world_feat_dim": int(policy.world_feat_dim),
        "proprio_dim": int(policy.proprio_dim),
        "text_feat_dim": int(policy.text_feat_dim),
        "num_modes": int(policy.num_modes),
        "hidden_sizes": list(policy.hidden_sizes),
        "activation": str(policy.activation),
        "add_value_head": bool(policy.add_value_head),
    }
    missing = [key for key in expected_gate if key not in gate_meta]
    if missing:
        raise ValueError(
            f"{sidecar_path}: gate metadata is missing required keys {missing}"
        )
    normalized_gate = {
        "world_feat_dim": int(gate_meta["world_feat_dim"]),
        "proprio_dim": int(gate_meta["proprio_dim"]),
        "text_feat_dim": int(gate_meta["text_feat_dim"]),
        "num_modes": int(gate_meta["num_modes"]),
        "hidden_sizes": [int(size) for size in gate_meta["hidden_sizes"]],
        "activation": str(gate_meta["activation"]),
        "add_value_head": bool(gate_meta["add_value_head"]),
    }
    mismatches = {
        key: (normalized_gate[key], expected)
        for key, expected in expected_gate.items()
        if normalized_gate[key] != expected
    }
    if mismatches:
        raise ValueError(
            f"{sidecar_path}: gate architecture metadata does not match the "
            f"configured policy (actual, expected): {mismatches}"
        )

    provenance_key = "shard_meta" if kind == "gate_bc" else "wam_provenance"
    shard_meta = meta.get(provenance_key)
    if not isinstance(shard_meta, dict):
        raise ValueError(
            f"{sidecar_path}: metadata is missing the `{provenance_key}` block"
        )
    try:
        from fastwam.adaptive_gate import (
            MODE_ORDER,
            TEXT_FEAT_LAYOUT,
            WORLD_FEAT_LAYOUT,
        )
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "validating Gate BC provenance requires the fastwam package"
        ) from exc
    expected_modes = _mode_names(MODE_ORDER)
    expected_provenance = _resolve_expected_bc_provenance(
        policy, checkpoint_path=path
    )
    expected_shard = {
        "world_feat_layout": WORLD_FEAT_LAYOUT,
        "text_feat_layout": TEXT_FEAT_LAYOUT,
        "mode_order": expected_modes,
        "world_feat_dim": expected_gate["world_feat_dim"],
        "proprio_dim": expected_gate["proprio_dim"],
        "text_feat_dim": expected_gate["text_feat_dim"],
        **expected_provenance,
    }
    shard_missing = [key for key in expected_shard if key not in shard_meta]
    if shard_missing:
        raise ValueError(
            f"{sidecar_path}: shard metadata is missing required keys {shard_missing}"
        )
    normalized_shard = {
        "world_feat_layout": shard_meta["world_feat_layout"],
        "text_feat_layout": shard_meta["text_feat_layout"],
        "mode_order": list(shard_meta["mode_order"]),
        "world_feat_dim": int(shard_meta["world_feat_dim"]),
        "proprio_dim": int(shard_meta["proprio_dim"]),
        "text_feat_dim": int(shard_meta["text_feat_dim"]),
        "task": str(shard_meta["task"]),
        "backbone_kind": str(shard_meta["backbone_kind"]).lower(),
        "ckpt_fingerprint": str(shard_meta["ckpt_fingerprint"]),
        "dataset_stats_fingerprint": str(
            shard_meta["dataset_stats_fingerprint"]
        ),
        "num_video_frames": int(shard_meta["num_video_frames"]),
        "inference_steps": int(shard_meta["inference_steps"]),
        "context_len": int(shard_meta["context_len"]),
        "model_dtype": str(shard_meta["model_dtype"]),
        "exec_horizon": int(shard_meta["exec_horizon"]),
        "action_horizon": int(shard_meta["action_horizon"]),
    }
    shard_mismatches = {
        key: (normalized_shard[key], expected)
        for key, expected in expected_shard.items()
        if normalized_shard[key] != expected
    }
    if shard_mismatches:
        raise ValueError(
            f"{sidecar_path}: gate/WAM feature provenance does not match the online "
            f"gate (actual, expected): {shard_mismatches}"
        )
    return meta


def load_gate_bc_state(
    path: str, *, expected_policy=None
) -> dict[str, torch.Tensor]:
    """Load BC weights and optionally validate their semantic provenance."""
    if expected_policy is not None:
        validate_gate_bc_sidecar(path, expected_policy)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and not any(
        torch.is_tensor(v) for v in payload.values()
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(torch.is_tensor(v) for v in payload.values()):
        raise ValueError(f"{path}: not a GatePolicy state_dict")
    return payload
