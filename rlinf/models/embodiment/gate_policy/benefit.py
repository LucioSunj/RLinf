# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cross-fitted decision-level imagination-benefit training."""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.gate_policy.paired_data import (
    PAIRED_NUM_FOLDS,
    PAIRED_SCHEMA,
    load_paired_shards,
)


FEATURE_GROUPS = ("world", "proprio", "text")
TARGETS = ("helpful", "difficulty")


class GateBenefitDataset(torch.utils.data.Dataset):
    """Pre-decision features with paired U/I terminal outcomes."""

    def __init__(
        self,
        data: dict[str, torch.Tensor],
        records: list[dict[str, Any]],
        meta: dict[str, Any],
        *,
        target: str = "helpful",
        enabled_features: Sequence[str] = FEATURE_GROUPS,
    ):
        target = str(target).lower()
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        requested = tuple(
            dict.fromkeys(str(value).lower() for value in enabled_features)
        )
        unknown = sorted(set(requested) - set(FEATURE_GROUPS))
        if unknown or not requested:
            raise ValueError(
                f"enabled_features must be a non-empty subset of {FEATURE_GROUPS}; "
                f"unknown={unknown}"
            )
        enabled = tuple(name for name in FEATURE_GROUPS if name in requested)
        n = int(data["world_feat"].shape[0])
        if len(records) != n:
            raise ValueError("paired records/tensor length mismatch")
        self.world_feat = data["world_feat"].float()
        self.proprio = data["proprio"].float()
        self.text_feat = data["text_feat"].float()
        self.trajectory_id = data["trajectory_id"].reshape(n).long()
        self.fold_id = (
            data["fold_id"].reshape(n).long()
            if "fold_id" in data
            else None
        )
        self.task_id = data["task_id"].reshape(n).long()
        self.success_uncond = data["success_uncond"].reshape(n).bool()
        self.success_idm = data["success_idm"].reshape(n).bool()
        self.treatment_effect = (
            self.success_idm.float() - self.success_uncond.float()
        )
        self.helpful = self.success_idm & ~self.success_uncond
        self.harmful = self.success_uncond & ~self.success_idm
        self.target_name = target
        self.label = (
            self.helpful.float()
            if target == "helpful"
            else (~self.success_uncond).float()
        )
        self.enabled_features = enabled
        self.records = records
        self.meta = dict(meta)
        if n < 2:
            raise ValueError("benefit dataset requires at least two states")

    @classmethod
    def from_shards(
        cls,
        paths,
        *,
        target: str = "helpful",
        enabled_features: Sequence[str] = FEATURE_GROUPS,
    ) -> "GateBenefitDataset":
        data, records, meta = load_paired_shards(paths)
        return cls(
            data,
            records,
            meta,
            target=target,
            enabled_features=enabled_features,
        )

    def __len__(self) -> int:
        return int(self.label.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        world, proprio, text = self.features(torch.as_tensor([index]))
        return {
            "world_feat": world[0],
            "proprio": proprio[0],
            "text_feat": text[0],
            "label": self.label[index],
            "treatment_effect": self.treatment_effect[index],
            "trajectory_id": self.trajectory_id[index],
        }

    def features(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = []
        for name, value in (
            ("world", self.world_feat),
            ("proprio", self.proprio),
            ("text", self.text_feat),
        ):
            selected = value[indices]
            result.append(
                selected if name in self.enabled_features else torch.zeros_like(selected)
            )
        return tuple(result)  # type: ignore[return-value]

    @property
    def world_feat_dim(self) -> int:
        return int(self.world_feat.shape[1])

    @property
    def proprio_dim(self) -> int:
        return int(self.proprio.shape[1])

    @property
    def text_feat_dim(self) -> int:
        return int(self.text_feat.shape[1])


def binary_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Tie-aware Mann-Whitney AUROC without sklearn."""
    labels = labels.detach().cpu().reshape(-1).bool()
    scores = scores.detach().cpu().reshape(-1).double()
    if labels.numel() != scores.numel() or not bool(torch.isfinite(scores).all()):
        raise ValueError("AUROC labels/scores must be aligned and finite")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[labels].sum())
    statistic = rank_sum - positives * (positives + 1) / 2.0
    return statistic / float(positives * negatives)


def binary_auprc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Average precision (area under the step-wise PR curve)."""
    labels = labels.detach().cpu().reshape(-1).bool()
    scores = scores.detach().cpu().reshape(-1).double()
    if labels.numel() != scores.numel() or not bool(torch.isfinite(scores).all()):
        raise ValueError("AUPRC labels/scores must be aligned and finite")
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_labels = labels[order].float()
    ranked_scores = scores[order]
    true_positive = 0.0
    false_positive = 0.0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < ranked_scores.numel():
        end = start + 1
        while end < ranked_scores.numel() and ranked_scores[end] == ranked_scores[start]:
            end += 1
        group_positive = float(ranked_labels[start:end].sum())
        true_positive += group_positive
        false_positive += float(end - start) - group_positive
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return area


def benefit_metrics(
    labels: torch.Tensor,
    scores: torch.Tensor,
    treatment_effect: torch.Tensor,
    *,
    calibration_bins: int = 10,
) -> dict[str, float]:
    labels = labels.detach().cpu().reshape(-1).float()
    scores = scores.detach().cpu().reshape(-1).float()
    tau = treatment_effect.detach().cpu().reshape(-1).float()
    if labels.shape != scores.shape or tau.shape != scores.shape:
        raise ValueError("benefit metric tensors must have identical shapes")
    if not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("benefit scores must be finite probabilities")
    ece = 0.0
    for index in range(int(calibration_bins)):
        low, high = index / calibration_bins, (index + 1) / calibration_bins
        mask = (scores >= low) & (
            scores <= high if index == calibration_bins - 1 else scores < high
        )
        if bool(mask.any()):
            ece += float(mask.float().mean()) * abs(
                float(scores[mask].mean() - labels[mask].mean())
            )
    k = max(1, int(math.ceil(scores.numel() * 0.2)))
    order = torch.argsort(scores)
    bottom, top = order[:k], order[-k:]
    prevalence = float(labels.mean())
    auprc = binary_auprc(labels.bool(), scores)
    return {
        "auroc": binary_auroc(labels.bool(), scores),
        "auprc": auprc,
        "prevalence": prevalence,
        "auprc_above_prevalence": auprc - prevalence,
        "brier": float(((scores - labels) ** 2).mean()),
        "ece": ece,
        "top20_tau": float(tau[top].mean()),
        "bottom20_tau": float(tau[bottom].mean()),
        "top_bottom_tau_gap": float(tau[top].mean() - tau[bottom].mean()),
        "top20_helpful_rate": float(labels[top].mean()),
        "bottom20_helpful_rate": float(labels[bottom].mean()),
    }


@dataclass
class GateBenefitConfig:
    folds: int = 5
    epochs: int = 30
    batch_size: int = 512
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    class_weight_power: float = 0.0
    device: str = "cpu"
    seed: int = 0
    log_every_epochs: int = 0

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("benefit cross-fitting requires at least two folds")
        if min(self.epochs, self.batch_size) <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.class_weight_power) or self.class_weight_power < 0:
            raise ValueError("class_weight_power must be finite and non-negative")


def _fit_policy(
    policy,
    dataset: GateBenefitDataset,
    train_idx: torch.Tensor,
    cfg: GateBenefitConfig,
    *,
    seed: int,
    val_idx: Optional[torch.Tensor] = None,
):
    device = torch.device(cfg.device)
    policy = policy.to(device)
    trainable = [
        parameter
        for name, parameter in policy.named_parameters()
        if name.startswith(("backbone.", "logits_head.")) and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    labels = dataset.label
    train_labels = labels[train_idx]
    positives = float(train_labels.sum())
    negatives = float(train_labels.numel() - positives)
    pos_weight = None
    if cfg.class_weight_power > 0 and positives > 0 and negatives > 0:
        pos_weight = torch.tensor(
            (negatives / positives) ** cfg.class_weight_power, device=device
        )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(int(cfg.epochs)):
        policy.train()
        order = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        total_loss, batches = 0.0, 0
        for start in range(0, order.numel(), int(cfg.batch_size)):
            indices = order[start : start + int(cfg.batch_size)]
            world, proprio, text = dataset.features(indices)
            logits = policy.mode_logits(
                world.to(device), proprio.to(device), text.to(device)
            )
            benefit_logit = logits[:, 1] - logits[:, 0]
            loss = F.binary_cross_entropy_with_logits(
                benefit_logit,
                labels[indices].to(device),
                pos_weight=pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        val_loss = total_loss / max(1, batches)
        if val_idx is not None and val_idx.numel() > 0:
            scores = predict_benefit(policy, dataset, val_idx)
            targets = labels[val_idx]
            val_loss = float(
                F.binary_cross_entropy(
                    scores.clamp(1e-6, 1 - 1e-6), targets
                )
            )
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, batches), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in policy.state_dict().items()
            }
        if cfg.log_every_epochs and epoch % cfg.log_every_epochs == 0:
            print(
                f"[gate-benefit] epoch {epoch:03d} "
                f"train={history[-1]['train_loss']:.4f} val={val_loss:.4f}"
            )
    if best_state is not None:
        policy.load_state_dict(best_state)
    return policy.cpu(), history


@torch.no_grad()
def predict_benefit(
    policy,
    dataset: GateBenefitDataset,
    indices: Optional[torch.Tensor] = None,
    *,
    batch_size: int = 4096,
) -> torch.Tensor:
    if indices is None:
        indices = torch.arange(len(dataset))
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    scores = []
    for start in range(0, indices.numel(), batch_size):
        batch = indices[start : start + batch_size]
        world, proprio, text = dataset.features(batch)
        logits = policy.mode_logits(
            world.to(device), proprio.to(device), text.to(device)
        )
        scores.append(torch.sigmoid(logits[:, 1] - logits[:, 0]).cpu())
    if was_training:
        policy.train()
    return torch.cat(scores)


def cross_fit_gate_benefit(policy, dataset: GateBenefitDataset, cfg: GateBenefitConfig):
    """Five-fold-by-default OOF prediction, then a final all-data fit."""
    groups = torch.unique(dataset.trajectory_id, sorted=True)
    if groups.numel() < cfg.folds:
        raise ValueError(
            f"{cfg.folds}-fold cross-fitting needs at least {cfg.folds} complete "
            f"trajectories, got {groups.numel()}"
        )
    if dataset.fold_id is not None:
        if cfg.folds != PAIRED_NUM_FOLDS:
            raise ValueError(
                f"paired-v1 split contract requires folds={PAIRED_NUM_FOLDS}, "
                f"got {cfg.folds}"
            )
        group_fold = {}
        for group in groups.tolist():
            values = torch.unique(dataset.fold_id[dataset.trajectory_id == group])
            if values.numel() != 1:
                raise ValueError(
                    f"trajectory {group} crosses multiple paired-v1 folds"
                )
            fold = int(values[0])
            if not 0 <= fold < cfg.folds:
                raise ValueError(f"trajectory {group} has invalid fold {fold}")
            group_fold[int(group)] = fold
        fold_groups = [
            torch.tensor(
                [group for group, assigned in group_fold.items() if assigned == fold],
                dtype=torch.int64,
            )
            for fold in range(cfg.folds)
        ]
        if any(group.numel() == 0 for group in fold_groups):
            raise ValueError("paired-v1 split contract contains an empty fold")
    else:
        generator = torch.Generator().manual_seed(int(cfg.seed))
        groups = groups[torch.randperm(groups.numel(), generator=generator)]
        fold_groups = [chunk.clone() for chunk in torch.tensor_split(groups, cfg.folds)]
    initial_state = copy.deepcopy(policy.state_dict())
    oof_scores = torch.full((len(dataset),), float("nan"))
    fold_id = torch.full((len(dataset),), -1, dtype=torch.int64)
    folds = []
    for index, validation_groups in enumerate(fold_groups):
        val_mask = torch.isin(dataset.trajectory_id, validation_groups)
        val_idx = torch.nonzero(val_mask, as_tuple=False).flatten()
        train_idx = torch.nonzero(~val_mask, as_tuple=False).flatten()
        train_groups = torch.unique(dataset.trajectory_id[train_idx])
        if bool(torch.isin(train_groups, validation_groups).any()):
            raise RuntimeError("trajectory leakage detected in benefit cross-fit")
        if torch.unique(dataset.label[train_idx]).numel() != 2:
            raise ValueError(
                f"benefit fold {index} training partition has only one target "
                "class; collect more complete trajectories before fitting"
            )
        fold_policy = copy.deepcopy(policy)
        fold_policy.load_state_dict(initial_state)
        fold_policy, history = _fit_policy(
            fold_policy,
            dataset,
            train_idx,
            cfg,
            seed=cfg.seed + index + 1,
            val_idx=val_idx,
        )
        scores = predict_benefit(fold_policy, dataset, val_idx)
        oof_scores[val_idx] = scores
        fold_id[val_idx] = index
        folds.append(
            {
                "fold": index,
                "train_groups": train_groups.tolist(),
                "validation_groups": validation_groups.tolist(),
                "num_train": int(train_idx.numel()),
                "num_validation": int(val_idx.numel()),
                "history": history,
            }
        )
    if not bool(torch.isfinite(oof_scores).all()) or bool((fold_id < 0).any()):
        raise RuntimeError("cross-fitting did not produce exactly one OOF score per row")

    metrics = benefit_metrics(dataset.label, oof_scores, dataset.treatment_effect)
    metrics.update(
        helpful_prevalence=float(dataset.helpful.float().mean()),
        harmful_prevalence=float(dataset.harmful.float().mean()),
        neutral_prevalence=float(
            (~dataset.helpful & ~dataset.harmful).float().mean()
        ),
    )
    final_policy = copy.deepcopy(policy)
    final_policy.load_state_dict(initial_state)
    all_idx = torch.arange(len(dataset))
    final_policy, final_history = _fit_policy(
        final_policy,
        dataset,
        all_idx,
        cfg,
        seed=cfg.seed + 10_000,
    )
    return final_policy, {
        "metrics": metrics,
        "oof_scores": oof_scores,
        "fold_id": fold_id,
        "folds": folds,
        "final_history": final_history,
        "config": asdict(cfg),
    }


def build_gate_benefit_sidecar(
    policy,
    dataset: GateBenefitDataset,
    result: dict[str, Any],
) -> dict[str, Any]:
    meta = dataset.meta
    gate = {
        "world_feat_dim": int(policy.world_feat_dim),
        "proprio_dim": int(policy.proprio_dim),
        "text_feat_dim": int(policy.text_feat_dim),
        "num_modes": int(policy.num_modes),
        "hidden_sizes": list(policy.hidden_sizes),
        "activation": str(policy.activation),
        "add_value_head": bool(policy.add_value_head),
    }
    wam_keys = (
        "world_feat_layout",
        "text_feat_layout",
        "mode_order",
        "task",
        "backbone_kind",
        "ckpt_fingerprint",
        "ckpt_file_sha256",
        "dataset_stats_fingerprint",
        "num_video_frames",
        "inference_steps",
        "solver_fingerprint",
        "context_len",
        "model_dtype",
        "exec_horizon",
        "action_horizon",
    )
    wam_provenance = {key: meta[key] for key in wam_keys}
    wam_provenance.update(
        world_feat_dim=gate["world_feat_dim"],
        proprio_dim=gate["proprio_dim"],
        text_feat_dim=gate["text_feat_dim"],
    )
    return {
        "kind": "gate_uplift",
        "schema_version": 1,
        "gate": gate,
        "wam_provenance": wam_provenance,
        "paired_provenance": {
            "schema": PAIRED_SCHEMA,
            "paired_dataset_fingerprint": meta["paired_dataset_fingerprint"],
            "splits_sha256": meta.get("splits_sha256"),
            "episode_manifest_sha256": meta["episode_manifest_sha256"],
            "heldout_test_manifest_sha256": meta[
                "heldout_test_manifest_sha256"
            ],
            "libero_plus_commit": meta["libero_plus_commit"],
            "manifest_split": meta["manifest_split"],
            "snapshot_schema": meta["snapshot_schema"],
            "continuation_mode": meta["continuation_mode"],
            "reference_policy_mix": meta["reference_policy_mix"],
            "reference_policy_assignment": meta[
                "reference_policy_assignment"
            ],
            "reference_assignment_manifest_sha256": meta[
                "reference_assignment_manifest_sha256"
            ],
            "reference_assignment_sha256": meta[
                "reference_assignment_sha256"
            ],
            "logical_composite_source_fingerprint": (
                meta.get("logical_merge", {}).get("composite_source_fingerprint")
            ),
            "collector_seed": meta["collector_seed"],
            "max_reference_decisions": meta["max_reference_decisions"],
            "max_branch_decisions": meta["max_branch_decisions"],
            "sensitivity_fraction": meta["sensitivity_fraction"],
            "target": dataset.target_name,
            "enabled_features": list(dataset.enabled_features),
            "num_samples": len(dataset),
            "num_trajectories": int(torch.unique(dataset.trajectory_id).numel()),
            "folds": int(result["config"]["folds"]),
        },
        "oof_metrics": result["metrics"],
        "train": result["config"],
    }


def save_gate_benefit_checkpoint(
    policy,
    path: str,
    *,
    sidecar: dict[str, Any],
    oof: Optional[dict[str, torch.Tensor]] = None,
) -> str:
    if sidecar.get("kind") != "gate_uplift":
        raise ValueError("benefit checkpoint sidecar must have kind='gate_uplift'")
    if sidecar["paired_provenance"].get("manifest_split") != "train":
        raise ValueError(
            "deployable uplift checkpoints must be trained from a split=train "
            "frozen manifest, never validation/test episodes"
        )
    if sidecar["paired_provenance"].get("enabled_features") != list(FEATURE_GROUPS):
        raise ValueError(
            "feature-ablation benefit models are analysis-only; only the full "
            "world+proprio+text Gate may be exported for online RL"
        )
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(
        {key: value.detach().cpu() for key, value in policy.state_dict().items()}, path
    )
    with open(path + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, default=str)
    if oof is not None:
        torch.save({key: value.detach().cpu() for key, value in oof.items()}, path + ".oof.pt")
    return path
