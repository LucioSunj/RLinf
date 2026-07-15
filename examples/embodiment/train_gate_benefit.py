# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Train a cross-fitted Gate uplift scorer from paired-v1 state branches."""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--paired",
        required=True,
        nargs="+",
        help=(
            "validated paired-v1 dataset directory (with splits.json); for a "
            "multi-suite manifest pass the strict logical merged directory"
        ),
    )
    parser.add_argument("--out", required=True, help="output raw GatePolicy checkpoint")
    parser.add_argument("--target", choices=["helpful", "difficulty"], default="helpful")
    parser.add_argument(
        "--enabled-features",
        nargs="+",
        choices=["world", "proprio", "text"],
        default=["world", "proprio", "text"],
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--add-value-head", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every-epochs", type=int, default=1)
    args = parser.parse_args()

    from rlinf.models.embodiment.gate_policy.benefit import (
        FEATURE_GROUPS,
        GateBenefitConfig,
        GateBenefitDataset,
        build_gate_benefit_sidecar,
        cross_fit_gate_benefit,
        save_gate_benefit_checkpoint,
    )
    from rlinf.models.embodiment.gate_policy.gate_policy import GatePolicy

    dataset = GateBenefitDataset.from_shards(
        args.paired,
        target=args.target,
        enabled_features=args.enabled_features,
    )
    if not dataset.meta.get("splits_sha256") or dataset.fold_id is None:
        raise ValueError(
            "benefit training requires the validated paired-v1 directory contract "
            "with splits.json; pass --paired <dataset-directory>, not loose shards"
        )
    if torch.unique(dataset.label).numel() != 2:
        raise ValueError(
            f"target={args.target!r} has only one class in paired-v1; the "
            "predictability experiment is undefined and must not export a Gate"
        )
    policy = GatePolicy(
        world_feat_dim=dataset.world_feat_dim,
        proprio_dim=dataset.proprio_dim,
        text_feat_dim=dataset.text_feat_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        add_value_head=args.add_value_head,
        activation=args.activation,
        wam_adapter=None,
        obs_preprocessor=None,
    )
    cfg = GateBenefitConfig(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weight_power=args.class_weight_power,
        device=args.device,
        seed=args.seed,
        log_every_epochs=args.log_every_epochs,
    )
    final_policy, result = cross_fit_gate_benefit(policy, dataset, cfg)
    sidecar = build_gate_benefit_sidecar(final_policy, dataset, result)
    oof = {"score": result["oof_scores"], "fold_id": result["fold_id"]}
    full_features = list(dataset.enabled_features) == list(FEATURE_GROUPS)
    if args.target == "helpful" and full_features:
        save_gate_benefit_checkpoint(
            final_policy, args.out, sidecar=sidecar, oof=oof
        )
        artifact_kind = "deployable_gate_uplift"
    else:
        # Difficulty and feature ablations are scientific controls, not online
        # warm-starts.  Wrapping the state prevents accidental raw strict-load.
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "kind": "gate_uplift_analysis",
                "state_dict": final_policy.state_dict(),
                "oof": oof,
                "metadata": sidecar,
            },
            args.out,
        )
        artifact_kind = "analysis_only"
    print(
        json.dumps(
            {
                "artifact": args.out,
                "kind": artifact_kind,
                "target": args.target,
                "enabled_features": list(dataset.enabled_features),
                "num_samples": len(dataset),
                "metrics": result["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
