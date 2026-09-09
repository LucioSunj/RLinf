# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Four shared-executor evaluation paths with all-attempt timing summaries."""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

from .builder import build_real_robot_policy
from .collector import SequentialRobotCollector
from .runner import load_checkpoint, make_mock_env


def episode_summary(episode, *, method, checkpoint, cfg, replay_path, counters):
    proposals, receipts = episode.proposals, episode.receipts
    valid = [i for i, receipt in enumerate(receipts) if receipt.chunk_valid]
    primitive = sum(receipt.executed_count for receipt in receipts)
    idm = sum(proposals[i].route == 1 for i in valid)
    timing_keys = {key for proposal in proposals for key in proposal.timings}
    timings = {
        key: sum(proposal.timings.get(key, 0.0) for proposal in proposals)
        for key in timing_keys
    }
    timings["prefix_execution"] = sum(
        receipt.end_time - receipt.start_time for receipt in receipts
    )
    timings["rpc_acknowledgement_including_gripper"] = sum(
        step.acknowledge_time - step.send_time
        for receipt in receipts
        for step in receipt.steps
    )
    timings["first_action_send"] = next(
        (
            receipt.steps[0].send_time - proposal.snapshot.receive_time
            for proposal, receipt in zip(proposals, receipts, strict=True)
            if receipt.steps
        ),
        None,
    )
    kind = episode.outcome.kind
    return {
        "trial_id": episode.episode_id,
        "task_id": episode.task_id,
        "layout_id": episode.layout_id,
        "method": method,
        "checkpoint": str(checkpoint),
        "uncond_weights": "same_PAD_checkpoint_trained_U",
        "actor_version": proposals[0].actor_version if proposals else None,
        "success": kind == "success",
        "outcome": kind,
        "stage_score": episode.outcome.stage_score,
        "censored": kind in {"censored", "abort"},
        "intervention": kind == "intervention",
        "primitive_count": primitive,
        "chunk_count": len(proposals),
        "valid_chunk_count": len(valid),
        "idm_calls": sum(p.route == 1 for p in proposals),
        "valid_idm_rate": idm / len(valid) if valid else None,
        "prediction_seconds": timings.get("prediction_total", 0.0),
        "wallclock_seconds": episode.wallclock_seconds,
        "manual_reset_seconds": episode.manual_reset_seconds,
        "retries": 0,
        "success_within_time_budget": kind == "success"
        and episode.wallclock_seconds <= cfg["evaluation"]["time_budget_seconds"],
        "time_budget_seconds": cfg["evaluation"]["time_budget_seconds"],
        "timings": timings,
        "model_calls": counters,
        "replay_path": str(replay_path),
        "routing": {
            "mode": cfg["evaluation"]["learned_mode"]
            if method == "learned"
            else method,
            "epsilon": cfg["training"]["gate_epsilon"] if method == "learned" else None,
            "temperature": cfg["training"]["gate_temperature"]
            if method == "learned"
            else None,
            "routing_seed": cfg["evaluation"]["routing_seed"],
            "random_probability": cfg["evaluation"]["random_idm_probability"],
            "random_probability_source": cfg["evaluation"]["random_probability_source"],
        },
        "clock": "modeled_execution_plus_measured_CPU_prediction"
        if cfg["backend"] == "mock"
        else "monotonic",
        "real_assets": "REAL-ASSET-NOT-RUN"
        if cfg["model"]["kind"] == "tiny"
        else "USER_PROVIDED_ASSETS",
    }


def _interval(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = (
        z
        * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
        / (1 + z * z / count)
    )
    return [center - radius, center + radius]


def summarize_records(records):
    """One episode is one trial; retain failures, timeouts and unknown outcomes."""
    result = {}
    for method in sorted({row["method"] for row in records}):
        rows = [row for row in records if row["method"] == method]
        success = sum(row["success"] for row in rows)
        by_layout = {
            layout: {
                "attempts": sum(r["layout_id"] == layout for r in rows),
                "successes": sum(
                    r["layout_id"] == layout and r["success"] for r in rows
                ),
            }
            for layout in sorted({row["layout_id"] for row in rows})
        }
        result[method] = {
            "attempts": len(rows),
            "successes": success,
            "success_rate": success / len(rows),
            "success_rate_wilson95": _interval(success, len(rows)),
            "censored": sum(row["censored"] for row in rows),
            "interventions": sum(row["intervention"] for row in rows),
            "timeouts": sum(row["outcome"] == "timeout" for row in rows),
            "success_within_time_budget_rate": sum(
                row["success_within_time_budget"] for row in rows
            )
            / len(rows),
            "wallclock_p50_p95_seconds": np.percentile(
                [r["wallclock_seconds"] for r in rows], [50, 95]
            ).tolist(),
            "prediction_p50_p95_seconds": np.percentile(
                [r["prediction_seconds"] for r in rows], [50, 95]
            ).tolist(),
            "successful_wallclock_mean_seconds": float(
                np.mean([r["wallclock_seconds"] for r in rows if r["success"]])
            )
            if success
            else None,
            "per_layout": by_layout,
        }
    return result


def run_evaluation(
    cfg,
    run_dir,
    *,
    methods,
    tasks_path,
    make_env,
    confirm_start,
    checkpoint=None,
    wait_for_outcome=None,
):
    """Evaluate explicit environments; the caller owns every manual-ready event.

    ``make_env(trial)`` returns the shared synchronous executor. ``confirm_start``
    must wait for the site's manual reset/ready event before each physical trial.
    No driver startup, reset, or motion authorization is supplied by this runner.
    """
    allowed = {"always_idm", "always_uncond", "random", "learned"}
    if not methods or set(methods) - allowed or len(set(methods)) != len(methods):
        raise ValueError("Specify distinct supported evaluation methods.")
    run_dir = Path(run_dir)
    checkpoint = Path(checkpoint or cfg["evaluation"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Evaluation requires a PAD checkpoint (same trained U): {checkpoint}"
        )
    tasks = yaml.safe_load(Path(tasks_path).read_text())
    session = uuid.uuid4().hex[:12]
    output = run_dir / f"evaluation_{session}"
    output.mkdir(parents=True, exist_ok=False)
    schedule = []
    order = random.Random(cfg["evaluation"]["routing_seed"])
    for layout_index, layout in enumerate(cfg["evaluation"]["layouts"]):
        for task_index, task in enumerate(cfg["collection"]["task_ids"]):
            trial_methods = list(methods)
            order.shuffle(trial_methods)
            for method in trial_methods:
                schedule.append(
                    {
                        "trial_id": f"{session}:{len(schedule):03d}",
                        "method": method,
                        "layout": layout,
                        "task": task,
                        "noise_seed": cfg["seed"]
                        + 1009 * layout_index
                        + 31 * task_index,
                        "route_seed": cfg["evaluation"]["routing_seed"]
                        + 1009 * layout_index
                        + 31 * task_index,
                    }
                )
    (output / "method_mapping.json").write_text(json.dumps(schedule, indent=2))
    (output / "scoring_sheet.json").write_text(
        json.dumps(
            [
                {key: row[key] for key in ("trial_id", "layout", "task")}
                for row in schedule
            ],
            indent=2,
        )
    )
    policies = {}
    for method in methods:
        policy = build_real_robot_policy(
            cfg, evaluation_method=method, initialize=False
        )
        load_checkpoint(checkpoint, policy=policy, cfg=cfg, evaluation_method=method)
        policy.eval()
        policies[method] = policy
    rows = []
    for trial in schedule:
        policy = policies[trial["method"]]
        for name, offset in (("action", 29), ("idm", 47), ("gate", 11)):
            policy.streams[name].manual_seed(
                (trial["route_seed"] if name == "gate" else trial["noise_seed"])
                + offset
            )
        policy.runtime.calls = dict.fromkeys(policy.runtime.calls, 0)
        policy.gate_calls = 0
        env = make_env(trial)
        reset_started = time.perf_counter()
        ready = confirm_start(trial)
        reset_seconds = (
            0.0 if env.backend.is_mock else time.perf_counter() - reset_started
        )
        episode = SequentialRobotCollector(
            env, policy, wait_for_outcome=wait_for_outcome
        ).collect(
            [
                {
                    "episode_id": trial["trial_id"],
                    "task_id": trial["task"],
                    "instruction": tasks[trial["task"]]["instruction"],
                    "layout_id": trial["layout"],
                    "operator_ready": ready,
                    "manual_reset_seconds": reset_seconds,
                }
            ],
            mode="eval",
            method=trial["method"],
            random_idm_probability=cfg["evaluation"]["random_idm_probability"],
            delay_seconds=cfg["evaluation"]["delay_injection_seconds"],
        )[0]
        replay = output / (trial["trial_id"].replace(":", "_") + ".pt")
        torch.save(episode, replay)
        counters = {**policy.runtime.calls, "gate": policy.gate_calls, "critic": 0}
        row = episode_summary(
            episode,
            method=trial["method"],
            checkpoint=checkpoint,
            cfg=cfg,
            replay_path=replay,
            counters=counters,
        )
        rows.append(row)
        with (output / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    summary = {
        "status": "COMPLETE",
        "evaluation_dir": str(output),
        "checkpoint": str(checkpoint),
        "methods": summarize_records(rows),
        "scope": "CPU/mock integration; timing is not a hardware benchmark"
        if cfg["backend"] == "mock"
        else "Explicit synchronous physical trials",
        "real_assets": "REAL-ASSET-NOT-RUN"
        if cfg["model"]["kind"] == "tiny"
        else "USER_PROVIDED_ASSETS",
        "hardware": "HARDWARE-NOT-RUN"
        if cfg["backend"] == "mock"
        else "EXPLICIT_BINDING",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_mock_evaluation(cfg, run_dir, *, methods, tasks_path, checkpoint=None):
    """CPU/mock wrapper for the same scheduling, execution and summary path."""
    if cfg["backend"] != "mock":
        raise ValueError("Mock evaluation cannot instantiate a live backend.")
    return run_evaluation(
        cfg,
        run_dir,
        methods=methods,
        tasks_path=tasks_path,
        checkpoint=checkpoint,
        make_env=lambda trial: make_mock_env(cfg, run_id=Path(run_dir).name),
        confirm_start=lambda trial: True,
    )


def summarize_run(run_dir):
    run_dir = Path(run_dir)
    sources = sorted(run_dir.glob("evaluation_*/episodes.jsonl"))
    rows = [
        json.loads(line)
        for source in sources
        for line in source.read_text().splitlines()
        if line
    ]
    if not rows:
        raise FileNotFoundError(
            "No completed evaluation episodes under " + str(run_dir)
        )
    # Do not combine repeated attempts from different evaluation settings.
    summaries = {
        str(source.parent): summarize_records(
            [json.loads(line) for line in source.read_text().splitlines() if line]
        )
        for source in sources
    }
    result = {
        "evaluations": summaries,
        "scope": "CPU/mock results are not real-robot performance evidence",
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result
