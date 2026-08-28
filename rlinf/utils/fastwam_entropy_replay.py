# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build an honest old/new Gate-entropy replay from historical summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

FASTWAM_ROLLOUT_STATE_SENTINEL = "FASTWAM_SHORT_RL_ROLLOUT_STATE_AUDIT"


def binary_entropy(probability: float) -> float:
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Bernoulli probability must lie in [0, 1].")
    if probability in {0.0, 1.0}:
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log1p(-probability)
    )


def entropy_bounds_from_range_mean(
    *,
    minimum: float,
    mean: float,
    maximum: float,
) -> tuple[float, float]:
    """Bound mean Bernoulli entropy from logged support and mean.

    Concavity makes ``H(mean)`` the upper bound. The chord between the logged
    minimum and maximum is a distribution-free lower bound over that support.
    """

    minimum = float(minimum)
    mean = float(mean)
    maximum = float(maximum)
    if not 0.0 <= minimum <= mean <= maximum <= 1.0:
        raise ValueError("Probability range and mean are inconsistent.")
    upper = binary_entropy(mean)
    if minimum == maximum:
        return upper, upper
    minimum_weight = (maximum - mean) / (maximum - minimum)
    lower = minimum_weight * binary_entropy(minimum) + (
        1.0 - minimum_weight
    ) * binary_entropy(maximum)
    return lower, upper


def entropy_guard_trigger_intervals(threshold: float) -> list[dict[str, Any]]:
    """Return the base-probability intervals where ``H(p) < threshold``."""

    threshold = float(threshold)
    if not 0.0 < threshold < math.log(2.0):
        raise ValueError("Gate entropy threshold must lie in (0, log(2)).")
    lower = 0.0
    upper = 0.5
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binary_entropy(midpoint) < threshold:
            lower = midpoint
        else:
            upper = midpoint
    boundary = (lower + upper) / 2.0
    return [
        {
            "minimum": 0.0,
            "maximum": boundary,
            "minimum_inclusive": True,
            "maximum_inclusive": False,
        },
        {
            "minimum": 1.0 - boundary,
            "maximum": 1.0,
            "minimum_inclusive": False,
            "maximum_inclusive": True,
        },
    ]


def summarize_entropy_guard(
    rows: list[dict[str, float]],
    *,
    threshold: float = 0.35,
    epsilons: tuple[float, ...] = (0.05, 0.1, 0.25),
) -> dict[str, Any]:
    """Summarize exact implications and proxy crossings of aggregate history."""

    intervals = entropy_guard_trigger_intervals(threshold)
    possible = [
        int(row["update"])
        for row in rows
        if row["base_entropy_lower_bound"] < threshold
    ]
    guaranteed = [
        int(row["update"]) for row in rows if row["base_entropy_of_mean"] < threshold
    ]
    indeterminate = [
        int(row["update"])
        for row in rows
        if row["base_entropy_lower_bound"] < threshold <= row["base_entropy_of_mean"]
    ]
    epsilon_intervals = {str(float(epsilon)): intervals for epsilon in epsilons}
    return {
        "base_entropy_threshold": float(threshold),
        "trigger_comparator": "mean base Bernoulli entropy < threshold",
        "first_possible_same_rollout_trigger_update": (
            possible[0] if possible else None
        ),
        "first_guaranteed_same_rollout_trigger_update": (
            guaranteed[0] if guaranteed else None
        ),
        "first_entropy_of_mean_proxy_trigger_update": (
            guaranteed[0] if guaranteed else None
        ),
        "indeterminate_same_rollout_updates": indeterminate,
        "base_probability_trigger_intervals_by_epsilon": epsilon_intervals,
        "trigger_intervals_are_epsilon_independent": len(
            {json.dumps(value, sort_keys=True) for value in epsilon_intervals.values()}
        )
        == 1,
        "exact_historical_actor_training_first_trigger": "NOT-RUN",
        "exact_trigger_limitation": (
            "Historical actor-training per-decision base probabilities were not "
            "retained. Aggregate rollout support and mean prove a trigger by the "
            "first guaranteed update, but cannot resolve earlier indeterminate "
            "updates or bitwise actor-training values."
        ),
    }


def load_rollout_probability_summaries(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if FASTWAM_ROLLOUT_STATE_SENTINEL not in line:
                continue
            payload = json.loads(
                line.split(FASTWAM_ROLLOUT_STATE_SENTINEL, 1)[1].strip()
            )
            if payload.get("schema") != "fastwam-rollout-state-audit-v1":
                raise ValueError("FastWAM rollout-state audit schema mismatch.")
            records.append(payload)
    if not records:
        raise ValueError("No FastWAM rollout-state audits were found.")
    return records


def load_tensorboard_scalar(path: str | Path, tag: str) -> list[float]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        raise ValueError(f"TensorBoard scalar is missing: {tag}.")
    events = accumulator.Scalars(tag)
    expected_steps = list(range(len(events)))
    if [int(event.step) for event in events] != expected_steps:
        raise ValueError(f"TensorBoard scalar {tag} has non-contiguous steps.")
    return [float(event.value) for event in events]


def build_entropy_replay_rows(
    rollout_records: list[dict[str, Any]],
    observed_behavior_entropy: list[float],
) -> list[dict[str, float]]:
    if len(rollout_records) != len(observed_behavior_entropy):
        raise ValueError(
            "Rollout and training entropy histories have different lengths."
        )
    rows = []
    for index, (record, observed) in enumerate(
        zip(rollout_records, observed_behavior_entropy, strict=True)
    ):
        base = record["base_probability"]
        behavior = record["behavior_probability"]
        base_lower, base_upper = entropy_bounds_from_range_mean(
            minimum=base["minimum"],
            mean=base["mean"],
            maximum=base["maximum"],
        )
        rows.append(
            {
                "runner_step": float(index),
                "update": float(index + 1),
                "eligible_decision_count": float(
                    record["eligible_gate_decision_count"]
                ),
                "base_probability_mean": float(base["mean"]),
                "behavior_probability_mean": float(behavior["mean"]),
                "base_entropy_lower_bound": base_lower,
                "base_entropy_of_mean": base_upper,
                "behavior_entropy_of_mean": binary_entropy(behavior["mean"]),
                "observed_old_train_behavior_entropy": float(observed),
            }
        )
    return rows


def write_entropy_replay(
    *,
    rows: list[dict[str, float]],
    output_dir: str | Path,
    guard_threshold: float = 0.35,
) -> tuple[Path, Path, Path]:
    """Write chart-ready CSV, explicit methodology JSON, and a static curve."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fifth_run_gate_entropy_replay.csv"
    json_path = output_dir / "fifth_run_gate_entropy_replay.json"
    png_path = output_dir / "fifth_run_gate_entropy_replay.png"
    for path in (csv_path, json_path, png_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite replay artifact: {path}")

    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    guard_replay = summarize_entropy_guard(rows, threshold=guard_threshold)
    report = {
        "schema": "fastwam-gate-entropy-replay-v1",
        "record_count": len(rows),
        "exact_series": ["observed_old_train_behavior_entropy"],
        "same_rollout_replay_series": [
            "base_entropy_of_mean",
            "behavior_entropy_of_mean",
        ],
        "base_rollout_entropy_bounds": [
            "base_entropy_lower_bound",
            "base_entropy_of_mean",
        ],
        "guard_replay": guard_replay,
        "limitation": (
            "The completed run retained aggregate pre-update rollout probability "
            "summaries, not actor-training per-decision probabilities. The new "
            "base-entropy training metric therefore cannot be reconstructed "
            "bitwise; H(mean p_base) is a labeled upper-bound proxy, with a "
            "support-and-mean lower bound."
        ),
        "rows": rows,
    }
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    updates = [row["update"] for row in rows]
    base_upper = [row["base_entropy_of_mean"] for row in rows]
    base_lower = [row["base_entropy_lower_bound"] for row in rows]
    behavior_proxy = [row["behavior_entropy_of_mean"] for row in rows]
    old_observed = [row["observed_old_train_behavior_entropy"] for row in rows]

    figure, axis = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    figure.patch.set_facecolor("#FCFCFD")
    axis.set_facecolor("#FCFCFD")
    axis.fill_between(
        updates,
        base_lower,
        base_upper,
        color="#D8E7F7",
        alpha=0.8,
        label="Base rollout entropy bounds",
        linewidth=0,
    )
    axis.plot(
        updates,
        base_upper,
        color="#2F6FB0",
        linewidth=2.2,
        label="New base entropy proxy: H(mean base p)",
    )
    axis.plot(
        updates,
        behavior_proxy,
        color="#D4882B",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        label="Old behavior entropy proxy: H(mean behavior p)",
    )
    axis.plot(
        updates,
        old_observed,
        color="#343A40",
        linewidth=1.4,
        marker="o",
        markersize=2.8,
        label="Observed old train/gate/entropy",
    )
    axis.axhline(
        guard_threshold,
        color="#59636E",
        linewidth=1.2,
        linestyle=":",
        label="New base-entropy guard threshold (0.35)",
    )
    first_guaranteed = guard_replay["first_guaranteed_same_rollout_trigger_update"]
    if first_guaranteed is not None:
        axis.axvline(
            first_guaranteed,
            color="#7A5195",
            linewidth=1.1,
            linestyle="--",
            label=f"First guaranteed same-rollout trigger ({first_guaranteed})",
        )
    axis.set_xlim(1, len(rows))
    axis.set_ylim(0.0, 0.72)
    axis.set_xlabel("Runner update")
    axis.set_ylabel("Bernoulli entropy (nats)")
    axis.set_title(
        "Gate entropy replay across 30 updates",
        loc="left",
        fontsize=15,
        fontweight="semibold",
        color="#20262D",
        pad=28,
    )
    axis.text(
        0.0,
        1.02,
        "Same-rollout entropy-of-mean proxies; shaded band is the rigorous base "
        "support/mean envelope",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#59636E",
        va="bottom",
    )
    axis.grid(axis="y", color="#DDE1E6", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8A939D")
    axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=8.5,
        ncol=2,
    )
    figure.text(
        0.105,
        0.01,
        "Source: fifth 30-update run. Exact historical train entropy is shown "
        "separately because per-decision actor replay probabilities were not retained.",
        fontsize=8,
        color="#59636E",
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.99, 0.98))
    figure.savefig(png_path, bbox_inches="tight")
    plt.close(figure)
    return csv_path, json_path, png_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-stdout", type=Path, required=True)
    parser.add_argument("--tensorboard-event", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guard-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rollout_records = load_rollout_probability_summaries(args.training_stdout)
    old_entropy = load_tensorboard_scalar(
        args.tensorboard_event,
        "train/gate/entropy",
    )
    rows = build_entropy_replay_rows(rollout_records, old_entropy)
    paths = write_entropy_replay(
        rows=rows,
        output_dir=args.output_dir,
        guard_threshold=args.guard_threshold,
    )
    guard_replay = summarize_entropy_guard(
        rows,
        threshold=args.guard_threshold,
    )
    print(
        json.dumps(
            {
                "record_count": len(rows),
                "outputs": [str(path) for path in paths],
                "first": rows[0],
                "last": rows[-1],
                "guard_replay": guard_replay,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
