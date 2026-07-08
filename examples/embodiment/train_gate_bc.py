# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""BC (SFT) warm-start of the adaptive-prediction gate on oracle labels (M3).

Standalone (no Ray/cluster/simulator/WAM): reads oracle-label shards produced by
`FastWAM/scripts/generate_gate_oracle_labels.py`, trains the SAME `GatePolicy`
the GRPO stage uses, and writes a RAW state_dict ready for
`runner.ckpt_path` / `actor.model.gate.bc_init_path`.

Example:
  cd RLinf
  python examples/embodiment/train_gate_bc.py \
    --labels '/path/to/gate_oracle/libero_joint/shard_*.pt' \
    --out /path/to/ckpts/gate_bc_libero_joint.pt \
    --epochs 30 --device cuda

Then in the GRPO config (see adaptive_gate_README.md §M3):
  runner.ckpt_path: /path/to/ckpts/gate_bc_libero_joint.pt
  actor.model.gate.bc_init_path: /path/to/ckpts/gate_bc_libero_joint.pt
  actor.model.gate.kl_prior.enabled: True
  gate_reward.beta_kl_prior: 0.05    # decays to 0; see beta_kl_prior_decay_steps

IMPORTANT: --hidden-sizes/--activation/--add-value-head must match the RL
config's `actor.model.gate` (and `add_value_head`) or the strict load will fail.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, nargs="+",
                    help="oracle-label shard path(s) or glob(s) (quote globs)")
    ap.add_argument("--out", required=True, help="output checkpoint path (.pt)")
    # architecture — MUST match the RL config's actor.model.gate
    ap.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--activation", default="tanh")
    ap.add_argument("--num-modes", type=int, default=3)
    ap.add_argument("--add-value-head", action="store_true",
                    help="only for the PPO variant (GRPO configs use no value head)")
    # optimization
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--class-weight-power", type=float, default=1.0,
                    help="inverse-frequency^power CE weights; 0 disables")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    # optional offline relabeling from the stored error curves (no WAM re-run)
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--relabel-metric", choices=["l1", "l2"], default="l1")
    ap.add_argument("--relabel-exec-horizon", type=int, default=None)
    ap.add_argument("--relabel-tol-abs", type=float, default=0.02)
    ap.add_argument("--relabel-tol-rel", type=float, default=0.1)
    args = ap.parse_args()

    from rlinf.models.embodiment.gate_policy.bc import (
        GateBCConfig,
        GateOracleLabelDataset,
        save_gate_bc_checkpoint,
        train_gate_bc,
    )
    from rlinf.models.embodiment.gate_policy.gate_policy import GatePolicy

    relabel = None
    if args.relabel:
        relabel = dict(
            metric=args.relabel_metric,
            exec_horizon=args.relabel_exec_horizon,
            tol_abs=args.relabel_tol_abs,
            tol_rel=args.relabel_tol_rel,
        )
    dataset = GateOracleLabelDataset.from_shards(args.labels, relabel=relabel)
    print(f"loaded {len(dataset)} labeled states "
          f"(world_feat_dim={dataset.world_feat_dim}, proprio_dim={dataset.proprio_dim})")

    policy = GatePolicy(
        world_feat_dim=dataset.world_feat_dim,
        proprio_dim=dataset.proprio_dim,
        num_modes=args.num_modes,
        hidden_sizes=tuple(args.hidden_sizes),
        add_value_head=args.add_value_head,
        activation=args.activation,
        wam_adapter=None,        # SFT needs no WAM: inputs are precomputed in shards
        obs_preprocessor=None,
    )

    cfg = GateBCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        device=args.device,
        seed=args.seed,
    )
    result = train_gate_bc(policy, dataset, cfg)
    print(f"best val metrics: {json.dumps(result['best'], indent=2)}")

    meta = {
        "kind": "gate_bc",
        "labels": args.labels,
        "relabel": relabel,
        "shard_meta": dataset.meta,
        "gate": {
            "world_feat_dim": dataset.world_feat_dim,
            "proprio_dim": dataset.proprio_dim,
            "num_modes": args.num_modes,
            "hidden_sizes": list(args.hidden_sizes),
            "activation": args.activation,
            "add_value_head": bool(args.add_value_head),
        },
        "train": result["config"],
        "best": result["best"],
        "num_train": result["num_train"],
        "num_val": result["num_val"],
    }
    path = save_gate_bc_checkpoint(policy, args.out, meta=meta)
    print(f"wrote {path} (+ .meta.json)")
    print("wire into GRPO: runner.ckpt_path / actor.model.gate.bc_init_path "
          "(+ gate.kl_prior.enabled + gate_reward.beta_kl_prior for the KL-to-BC prior)")


if __name__ == "__main__":
    main()
