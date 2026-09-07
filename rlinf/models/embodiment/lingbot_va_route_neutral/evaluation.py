# Copyright 2026 The RLinf Authors.
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

"""Frozen scene-paired six-method schedules and measured result aggregation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contracts import Route
from .data import TASKS

METHODS = (
    "always_idm",
    "always_uncond_bc",
    "always_uncond_rl",
    "random",
    "periodic",
    "learned_gate",
)


def make_schedule(scenes, development_rates, *, seed=42):
    """Freeze routing rates and randomize the six methods within each reset scene."""
    generator = np.random.default_rng(seed)
    rows = []
    for task in TASKS:
        task_scenes = [row for row in scenes if row["task"] == task]
        if len(task_scenes) != 20 or len({r["scene_id"] for r in task_scenes}) != 20:
            raise ValueError(f"{task} requires 20 distinct held-out reset scenes.")
        rate = float(development_rates[task])
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Freeze a valid development routing rate for each task.")
        for scene in task_scenes:
            if not scene.get("reset_photo") or not scene.get("reset_description"):
                raise ValueError(
                    "Each physical reset scene needs a reference photo and description."
                )
            for method in generator.permutation(METHODS):
                rows.append(
                    {
                        **scene,
                        "method": str(method),
                        "routing_rate": rate,
                        "seed": int(generator.integers(0, 2**31)),
                        "trial_id": f"{task}/{scene['scene_id']}/{method}",
                        "max_steps": TASKS[task]["max_steps"],
                    }
                )
    return {
        "seed": seed,
        "development_rates": development_rates,
        "trials": rows,
        "pairing": "physical_reset_condition",
        "status": "PREPARED_NOT_RUN",
    }


def evaluation_route(method, chunk_index, rate):
    if method == "always_idm":
        return {"route": Route.IDM}
    if method in {"always_uncond_bc", "always_uncond_rl"}:
        return {"route": Route.UNCOND}
    if method == "random":
        return {"routing_rate": rate}
    if method == "periodic":
        idm = math.floor((chunk_index + 1) * rate) > math.floor(chunk_index * rate)
        return {"route": Route.IDM if idm else Route.UNCOND}
    if method == "learned_gate":
        return {}
    raise ValueError(f"Unknown evaluation method {method}.")


def wilson_interval(successes, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def aggregate_results(schedule, episodes, chunks=()):
    expected = {r["trial_id"]: r for r in schedule["trials"]}
    received = {}
    groups = defaultdict(list)
    for episode in episodes:
        identity = episode["trial_id"]
        if identity not in expected or identity in received:
            raise ValueError(f"Unknown or repeated physical trial {identity}.")
        received[identity] = episode
        trial = expected[identity]
        groups[(trial["task"], trial["method"])].append(episode)
    summaries = []
    timings = defaultdict(lambda: defaultdict(list))
    for row in chunks:
        if row.get("trial_id") not in received:
            continue
        trial = expected[row["trial_id"]]
        values = dict(row["timings"])
        values["observation_to_action_s"] = row.get("observation_to_action_s")
        for name, value in values.items():
            if name.endswith("_s") and value is not None:
                timings[(trial["task"], trial["method"])][name].append(float(value))
    for (task, method), rows in sorted(groups.items()):
        successes = sum(row["success"] for row in rows)
        idm = sum(row["idm_chunks"] for row in rows)
        chunks = sum(row["chunks"] for row in rows)
        summaries.append(
            {
                "task": task,
                "method": method,
                "episodes": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "success_95ci": wilson_interval(successes, len(rows)),
                "idm_rate": idm / chunks if chunks else None,
                "mean_completion_s": float(np.mean([r["completion_s"] for r in rows])),
                "mean_stage_progress": float(
                    np.mean([r["stage_progress"] for r in rows])
                ),
                "takeovers": sum(not r["autonomous"] for r in rows),
                "chunk_timing_seconds": {
                    name: {
                        "mean": float(np.mean(values)),
                        "p50": float(np.median(values)),
                        "p95": float(np.quantile(values, 0.95)),
                        "count": len(values),
                    }
                    for name, values in timings[(task, method)].items()
                },
            }
        )
    return {
        "status": "COMPLETE" if set(received) == set(expected) else "INCOMPLETE",
        "completed_trials": len(received),
        "expected_trials": len(expected),
        "groups": summaries,
        "missing_trials": sorted(set(expected) - set(received)),
        "pairing": "reset conditions, not identical physical states",
    }


def write_aggregate(schedule_path, episode_paths, output):
    schedule = json.loads(Path(schedule_path).read_text())
    episodes = [json.loads(Path(path).read_text()) for path in episode_paths]
    chunk_files = sorted(
        {Path(path).parent.parent / "chunks.jsonl" for path in episode_paths}
    )
    chunks = [
        json.loads(line)
        for path in chunk_files
        if path.is_file()
        for line in path.read_text().splitlines()
        if line
    ]
    result = aggregate_results(schedule, episodes, chunks)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    with Path(output).with_suffix(".csv").open("w", newline="") as handle:
        fields = [
            "task",
            "method",
            "episodes",
            "successes",
            "success_rate",
            "success_ci_low",
            "success_ci_high",
            "idm_rate",
            "mean_completion_s",
            "mean_stage_progress",
            "takeovers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in result["groups"]:
            writer.writerow(
                {
                    **{k: group[k] for k in fields if k in group},
                    "success_ci_low": group["success_95ci"][0],
                    "success_ci_high": group["success_95ci"][1],
                }
            )
    return result
