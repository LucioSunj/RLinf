# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real LIBERO-Plus/FastWAM snapshot -> step -> restore smoke test."""

from __future__ import annotations

import argparse
import json

import torch


def _assert_observation_equal(expected, actual) -> None:
    for key in ("main_images", "wrist_images"):
        if key not in expected or key not in actual or not torch.equal(
            torch.as_tensor(expected[key]), torch.as_tensor(actual[key])
        ):
            raise RuntimeError(f"deterministic replay changed {key} pixels")
    before = torch.as_tensor(expected["states"]).float()
    after = torch.as_tensor(actual["states"]).float()
    if before.shape != after.shape or not torch.allclose(
        before, after, rtol=0.0, atol=1.0e-6
    ):
        error = float(torch.max(torch.abs(before - after)))
        raise RuntimeError(
            f"deterministic replay changed proprio/state; max_abs_error={error:g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument(
        "--heldout-test-manifest",
        default=None,
        help="required when the smoke manifest has split=train/validation",
    )
    parser.add_argument("--rlinf-config-dir", default="examples/embodiment/config")
    parser.add_argument("--rlinf-config-name", default="libero_10_grpo_gate")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--progress-fn", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-prefix-decisions", type=int, default=1)
    parser.add_argument("--driver-arg", action="append", default=[])
    args = parser.parse_args()
    if args.replay_prefix_decisions <= 0:
        parser.error("--replay-prefix-decisions must be positive")

    from rlinf.envs.libero.episode_manifest import load_frozen_episode_manifest
    from rlinf.models.embodiment.gate_policy.libero_paired_driver import (
        build_libero_fastwam_driver,
    )

    manifest = load_frozen_episode_manifest(args.episode_manifest)
    driver = build_libero_fastwam_driver(args=args, manifest=manifest)
    try:
        episode = manifest.episodes[0].to_dict()
        observation = driver.reset_episode(episode)
        prefix_actions = []
        for index in range(args.replay_prefix_decisions):
            action = driver.action(observation, mode=0, seed=args.seed + index)
            prefix_actions.append(action.detach().clone())
            result = driver.step_chunk(action)
            if result["done"] or result["success"]:
                raise RuntimeError(
                    "deterministic smoke prefix terminated before the target snapshot"
                )
            observation = result["observation"]
        target_observation = observation
        context_before = driver.context(target_observation)
        snapshot = driver.capture_snapshot()

        replay_observation = driver.reset_episode(episode)
        for action in prefix_actions:
            result = driver.step_chunk(action.detach().clone())
            if result["done"] or result["success"]:
                raise RuntimeError(
                    "replayed prefix terminated before the target snapshot"
                )
            replay_observation = result["observation"]
        _assert_observation_equal(target_observation, replay_observation)
        replay_context = driver.context(replay_observation)

        restored_target = driver.restore_snapshot(snapshot)
        action = driver.action(restored_target, mode=0, seed=args.seed + 10_000)
        driver.step_chunk(action)
        restored = driver.restore_snapshot(snapshot)
        context_after = driver.context(restored)
        identity_keys = (
            "episode_uid",
            "decision_index",
            "elapsed_steps",
            "task_id",
            "trial_id",
            "reset_state_id",
            "factor",
            "level",
            "perturbation_id",
        )
        mismatches = {
            key: (context_before.get(key), context_after.get(key))
            for key in identity_keys
            if context_before.get(key) != context_after.get(key)
        }
        if mismatches:
            raise RuntimeError(f"outer snapshot identity mismatch: {mismatches}")
        replay_mismatches = {
            key: (context_before.get(key), replay_context.get(key))
            for key in identity_keys
            if context_before.get(key) != replay_context.get(key)
        }
        if replay_mismatches:
            raise RuntimeError(
                f"reset+prefix replay identity mismatch: {replay_mismatches}"
            )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "snapshot_schema": snapshot["schema"],
                    "episode_uid": context_after["episode_uid"],
                    "replay_prefix_decisions": args.replay_prefix_decisions,
                    "reset_prefix_replay_verified": True,
                    "verified_identity_keys": list(identity_keys),
                },
                indent=2,
            )
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
