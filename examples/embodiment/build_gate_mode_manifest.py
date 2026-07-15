# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Preregister exact per-episode gate schedules for matched-budget evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=(
            "forced",
            "episode_mixture",
            "bernoulli",
            "random_k",
            "periodic_k",
            "reference_random_k",
            "reference_task_factor",
            "reference_phase",
        ),
    )
    parser.add_argument(
        "--reference-trace",
        default=None,
        help="canonical learned-eval JSONL required by reference_* kinds",
    )
    parser.add_argument("--max-decisions", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--p-idm", type=float, default=None)
    parser.add_argument("--mode", type=int, choices=(0, 1), default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"--checkpoint does not exist: {checkpoint}")

    from rlinf.envs.libero.episode_manifest import load_frozen_episode_manifest
    from rlinf.models.embodiment.gate_policy.mode_selectors import (
        REFERENCE_MATCH_METHODS,
        build_eval_mode_selector,
        load_canonical_reference_trace,
        make_mode_schedule_manifest,
        make_reference_matched_mode_manifest,
        sha256_file,
        write_json_atomic,
    )

    # This validates the LIBERO-Plus checkout/commit and every frozen BDDL SHA.
    episode_manifest = load_frozen_episode_manifest(args.episode_manifest)
    checkpoint_sha256 = sha256_file(checkpoint)
    episode_uids = [
        episode.episode_id for episode in episode_manifest.episodes
    ]
    if args.kind in REFERENCE_MATCH_METHODS:
        if args.reference_trace is None:
            parser.error(f"--reference-trace is required for {args.kind}")
        reference_path = Path(args.reference_trace).expanduser().resolve()
        records = load_canonical_reference_trace(reference_path)
        payload = make_reference_matched_mode_manifest(
            records=records,
            method=args.kind,
            episode_uids=episode_uids,
            checkpoint_sha256=checkpoint_sha256,
            episode_manifest_sha256=episode_manifest.sha256,
            reference_trace_sha256=sha256_file(reference_path),
            seed=args.seed,
            max_decisions=args.max_decisions,
        )
    else:
        if args.reference_trace is not None:
            parser.error("--reference-trace is accepted only by reference_* kinds")
        selector_cfg = {
            "kind": args.kind,
            "max_decisions": args.max_decisions,
            "seed": args.seed,
            "k": args.k,
            "p_idm": args.p_idm,
            "mode": args.mode,
        }
        selector = build_eval_mode_selector(selector_cfg)
        if not hasattr(selector, "schedule_for"):
            parser.error(f"{args.kind} cannot be materialized before observing states")
        payload = make_mode_schedule_manifest(
            selector=selector,
            episode_uids=episode_uids,
            checkpoint_sha256=checkpoint_sha256,
            episode_manifest_sha256=episode_manifest.sha256,
        )
    write_json_atomic(args.out, payload)
    print(
        json.dumps(
            {
                "mode_manifest": str(Path(args.out).expanduser().resolve()),
                "method": args.kind,
                "episodes": len(episode_manifest.episodes),
                "max_decisions": args.max_decisions,
                "checkpoint_sha256": checkpoint_sha256,
                "episode_manifest_sha256": episode_manifest.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
