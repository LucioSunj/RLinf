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
              `gate_reward.beta_kl_prior` so early RL stays near the BC gate.

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
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

# Keys the BC stage needs from each oracle-label shard record.
BC_DATA_KEYS = ("world_feat", "proprio", "label")


class GateOracleLabelDataset(torch.utils.data.Dataset):
    """(world_feat, proprio) -> oracle mode label, from in-memory tensors.

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
        self.world_feat = data["world_feat"].float()
        self.proprio = data["proprio"].float()
        self.label = data["label"].long()
        self.meta = dict(meta or {})

    @classmethod
    def from_shards(
        cls,
        shards: str | Sequence[str],
        *,
        relabel: Optional[dict[str, Any]] = None,
    ) -> "GateOracleLabelDataset":
        """Load shard file(s) (glob/path/list). Optional `relabel` kwargs
        (metric/exec_horizon/tol_abs/tol_rel) re-derive labels offline from the
        stored error curves; rows whose masked window is empty are dropped."""
        try:
            from fastwam.adaptive_gate import load_label_shards, relabel_from_steps
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "loading oracle-label shards requires the fastwam package "
                "(`pip install -e FastWAM`), which defines the shard format."
            ) from exc

        data, meta = load_label_shards(shards)
        if relabel:
            labels, chunk_err, has_valid = relabel_from_steps(
                data["step_l1"], data["step_l2"], data["valid_steps"], **relabel
            )
            keep = has_valid
            data = {k: v[keep] for k, v in data.items()}
            data["label"] = labels[keep]
            data["chunk_err"] = chunk_err[keep]
            meta = {**meta, "relabel": dict(relabel), "num_samples": int(keep.sum())}
        return cls(data, meta)

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "world_feat": self.world_feat[idx],
            "proprio": self.proprio[idx],
            "label": self.label[idx],
        }

    @property
    def world_feat_dim(self) -> int:
        return int(self.world_feat.shape[-1])

    @property
    def proprio_dim(self) -> int:
        return int(self.proprio.shape[-1])


def class_balance_weights(
    labels: torch.Tensor,
    num_classes: int,
    *,
    power: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency^power CE weights, normalized to mean 1 over PRESENT classes.

    Oracle labels are typically SKIP-heavy; unweighted CE then collapses to the
    majority mode and the warm-start teaches the RL stage nothing about WHEN to
    predict. power=1 fully rebalances, power=0 disables, in between softens.
    Absent classes get weight 0 (they cannot contribute loss anyway).
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


def expected_mode_cost(labels: torch.Tensor, cost_table: Optional[dict[str, float]]) -> Optional[float]:
    """Mean cost of a mode sequence under the shard's cost table (FULL=1)."""
    if not cost_table:
        return None
    try:
        from fastwam.adaptive_gate import MODE_ORDER

        costs = torch.tensor([float(cost_table[m.value]) for m in MODE_ORDER])
    except Exception:
        # fallback: assume canonical (skip, latent, full) key order
        costs = torch.tensor([float(cost_table[k]) for k in ("skip", "latent", "full")])
    return float(costs[labels.long()].mean().item())


@dataclass
class GateBCConfig:
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    val_fraction: float = 0.1
    class_weight_power: float = 1.0  # 0 disables class rebalancing
    label_smoothing: float = 0.0
    device: str = "cpu"
    seed: int = 0
    log_every_epochs: int = 1
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def evaluate_gate_bc(
    policy,
    world_feat: torch.Tensor,
    proprio: torch.Tensor,
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
        logits = policy.mode_logits(world_feat[sl].to(device), proprio[sl].to(device))
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
            float((pred[mask] == mode_idx).float().mean().item()) if bool(mask.any()) else float("nan")
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
    perm = torch.randperm(n, generator=generator)
    n_val = int(math.floor(n * float(cfg.val_fraction)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if train_idx.numel() == 0:
        raise ValueError("val_fraction leaves no training samples")

    device = torch.device(cfg.device)
    policy = policy.to(device)
    world_feat, proprio, labels = dataset.world_feat, dataset.proprio, dataset.label
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
                world_feat[batch_idx].to(device), proprio[batch_idx].to(device)
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
            print(
                f"[gate-bc] epoch {epoch:03d} loss={record['train_loss']:.4f} "
                f"val_acc={val['accuracy']:.4f} "
                f"pred_frac={[round(val[f'pred_frac/mode_{i}'], 3) for i in range(policy.num_modes)]}"
            )

    if best_state is not None:
        policy.load_state_dict(best_state)
    return {
        "best": best,
        "history": history,
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
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


def load_gate_bc_state(path: str) -> dict[str, torch.Tensor]:
    """Load a BC checkpoint; accepts a raw state_dict or a {'state_dict': ...} payload."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and not any(
        torch.is_tensor(v) for v in payload.values()
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(torch.is_tensor(v) for v in payload.values()):
        raise ValueError(f"{path}: not a GatePolicy state_dict")
    return payload
