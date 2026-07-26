# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect paired UNCOND/IDM counterfactual states from frozen LIBERO-Plus.

The default driver directly composes RLinf's LiberoEnv and frozen FastWAM from
the selected Hydra config. ``--driver MODULE:FACTORY`` remains available for a
cluster-specific adapter; it receives ``(args=args, manifest=manifest)`` and
must implement ``PairedCollectorDriver``. Missing LIBERO-Plus identity, an
unfrozen BDDL, incomplete snapshots, or feature/WAM provenance fail closed.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("--driver must have the form python.module:factory")
    module_name, factory_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(f"{spec!r} does not resolve to a callable factory")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument(
        "--heldout-test-manifest",
        default=None,
        help=(
            "required when --episode-manifest has split=train/validation; the concrete "
            "driver audits zero overlap in seeds, resets, perturbations and assets"
        ),
    )
    parser.add_argument(
        "--disjoint-audit",
        default=None,
        help=(
            "committed dev-test-disjoint-audit-v1 JSON for the logical "
            "manifest being collected on. Paired collection is a training-side "
            "consumer, so it refuses to start without a verifiable committed "
            "audit and refuses any split=test manifest outright."
        ),
    )
    parser.add_argument(
        "--driver",
        default=(
            "rlinf.models.embodiment.gate_policy.libero_paired_driver:"
            "build_libero_fastwam_driver"
        ),
        help="python.module:factory",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "paired-v1 dataset directory; writes states.jsonl, outcomes.jsonl, "
            "splits.json, metadata.json and tensors/shard_00000.pt"
        ),
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="snapshot directory (default: <out>/snapshots)",
    )
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--collector-seed", type=int, default=0)
    parser.add_argument("--max-reference-decisions", type=int, default=70)
    parser.add_argument("--max-branch-decisions", type=int, default=70)
    parser.add_argument("--sensitivity-fraction", type=float, default=0.2)
    parser.add_argument(
        "--rlinf-config-dir", default="examples/embodiment/config"
    )
    parser.add_argument("--rlinf-config-name", default="libero_10_grpo_gate")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Hydra override for the concrete LIBERO/FastWAM driver",
    )
    parser.add_argument(
        "--progress-fn",
        default=None,
        help=(
            "required by the concrete driver: module:function implementing a "
            "preregistered task-specific pre-treatment progress metric"
        ),
    )
    parser.add_argument(
        "--driver-arg",
        action="append",
        default=[],
        help="opaque key=value arguments available to the injected factory",
    )
    args = parser.parse_args()
    if args.num_episodes is not None and args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.snapshot_dir is None:
        args.snapshot_dir = str(Path(args.out).expanduser().resolve() / "snapshots")

    from rlinf.envs.libero.episode_manifest import load_frozen_episode_manifest
    from rlinf.models.embodiment.gate_policy.paired_collector import (
        PairedStateCollector,
    )
    from rlinf.utils.test_set_guard import (
        assert_disjoint_audit,
        assert_training_manifest,
    )

    # Validates LIBERO_PLUS_ROOT, LIBERO_PLUS_COMMIT, git HEAD, every BDDL SHA,
    # explicit seeds/factors/levels and rejects LIBERO_SUFFIX=all.
    manifest = load_frozen_episode_manifest(args.episode_manifest)
    # Paired collection trains the uplift gate downstream: the held-out
    # split=test half is never a legal input here, with or without keys.
    assert_training_manifest(
        manifest, context="collect_gate_paired_states --episode-manifest"
    )
    assignment_manifest = (
        load_frozen_episode_manifest(manifest.parent_manifest_path)
        if manifest.parent_manifest_path is not None
        else manifest
    )
    assert_training_manifest(
        assignment_manifest,
        context="collect_gate_paired_states parent (logical) manifest",
    )
    # The committed audit is keyed on file_sha256, which for a per-suite
    # partition only exists for the logical parent manifest - audit that one.
    assert_disjoint_audit(
        assignment_manifest,
        args.disjoint_audit,
        context="collect_gate_paired_states --disjoint-audit",
    )
    factory = _load_factory(args.driver)
    driver = factory(args=args, manifest=manifest)
    required = (
        "reset_episode",
        "capture_snapshot",
        "restore_snapshot",
        "context",
        "features",
        "action",
        "step_chunk",
    )
    missing = [name for name in required if not callable(getattr(driver, name, None))]
    if missing or not isinstance(getattr(driver, "paired_metadata", None), dict):
        raise TypeError(
            f"paired collector driver is incomplete: methods={missing}, "
            f"paired_metadata={type(getattr(driver, 'paired_metadata', None)).__name__}"
        )

    episodes = [episode.to_dict() for episode in manifest.episodes]
    assignment_episodes = [
        episode.to_dict() for episode in assignment_manifest.episodes
    ]
    if args.num_episodes is not None:
        if manifest.parent_manifest_path is not None:
            raise ValueError(
                "--num-episodes is forbidden for suite-partitioned paired "
                "collection because it would break exact logical-manifest coverage"
            )
        episodes = episodes[: args.num_episodes]
        assignment_episodes = episodes
    if len(assignment_episodes) % 3 != 0:
        raise ValueError(
            "logical paired collection needs an episode count divisible by three so "
            "always-UNCOND, always-IDM and Bernoulli-0.5 source trajectories "
            "receive exactly one third each"
        )
    collector = PairedStateCollector(
        driver,
        collector_seed=args.collector_seed,
        max_reference_decisions=args.max_reference_decisions,
        max_branch_decisions=args.max_branch_decisions,
        sensitivity_fraction=args.sensitivity_fraction,
        snapshot_dir=args.snapshot_dir,
    )
    try:
        reference_assignments = collector.reference_assignment_map(
            assignment_episodes
        )
        path = collector.collect_to_path(
            episodes,
            args.out,
            reference_assignments=reference_assignments,
            reference_assignment_manifest_sha256=assignment_manifest.sha256,
        )
    finally:
        close = getattr(driver, "close", None)
        if callable(close):
            close()
    print(
        json.dumps(
            {
                "paired_dataset": path,
                "episode_manifest_sha256": manifest.sha256,
                "num_episodes": len(episodes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
