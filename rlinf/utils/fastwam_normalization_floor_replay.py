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

"""Replay FastWAM advantage-normalization floor usage from legacy audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlinf.utils.fastwam_fair_cost_replay import load_counterfactual_cost_audits

FASTWAM_NORMALIZATION_FLOOR_REPLAY_SCHEMA = "fastwam-normalization-floor-replay-v1"
LEGACY_NORMALIZATION_EPSILON = 1.0e-5


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Quantiles require at least one value.")
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "p10": _linear_quantile(values, 0.10),
        "p25": _linear_quantile(values, 0.25),
        "p50": _linear_quantile(values, 0.50),
        "p75": _linear_quantile(values, 0.75),
        "p90": _linear_quantile(values, 0.90),
        "maximum": max(values),
    }


def _sample_variance(summary: Mapping[str, Any]) -> float:
    count = int(summary["finite_count"])
    if count != int(summary["count"]) or count < 2:
        raise ValueError("Normalization replay requires at least two finite values.")
    total = float(summary["sum"])
    total_squares = float(summary["sum_of_squares"])
    variance = (total_squares - total * total / count) / (count - 1)
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("Normalization replay requires positive finite variance.")
    return variance


def infer_legacy_normalization_std(entry: Mapping[str, Any]) -> float:
    """Recover the batch std from paired raw/normalized affine summaries.

    The legacy transform was ``(x - mean) / (batch_std + 1e-5)``. Gate
    alignment selects a subset after that transform, but does not change the
    affine scale. Thus the ratio of raw to normalized sample standard
    deviations recovers the same divisor from any non-degenerate selected
    subset. The extrema ratio provides an independent consistency check.
    """

    raw = entry["gate_advantage"]["unnormalized"]
    normalized = entry["gate_advantage"]["normalized"]
    if int(raw["count"]) != int(normalized["count"]):
        raise ValueError("Raw and normalized Gate summaries have different counts.")

    moment_divisor = math.sqrt(_sample_variance(raw) / _sample_variance(normalized))
    raw_range = float(raw["maximum"]) - float(raw["minimum"])
    normalized_range = float(normalized["maximum"]) - float(normalized["minimum"])
    if raw_range <= 0.0 or normalized_range <= 0.0:
        raise ValueError("Normalization replay requires non-degenerate ranges.")
    range_divisor = raw_range / normalized_range
    if not math.isclose(
        moment_divisor,
        range_divisor,
        rel_tol=1.0e-6,
        abs_tol=1.0e-8,
    ):
        raise ValueError(
            "Paired audit summaries do not share the legacy affine normalization."
        )

    batch_std = moment_divisor - LEGACY_NORMALIZATION_EPSILON
    if not math.isfinite(batch_std) or batch_std < 0.0:
        raise ValueError("Recovered normalization standard deviation is invalid.")
    return batch_std


def replay_normalization_floor(
    records: Sequence[Mapping[str, Any]],
    *,
    std_floor: float = 0.15,
) -> dict[str, Any]:
    """Return the floor decisions the new normalizer would make per update."""

    std_floor = float(std_floor)
    if not records:
        raise ValueError("Normalization-floor replay requires at least one audit.")
    if not math.isfinite(std_floor) or std_floor < 0.0:
        raise ValueError("Normalization standard-deviation floor is invalid.")

    replayed = []
    for index, record in enumerate(records):
        runner_step = int(record.get("runner_step", index))
        if runner_step != index:
            raise ValueError(
                "Normalization-floor replay requires contiguous runner steps."
            )
        configured_cost = float(record["configured_idm_cost"])
        entries = [
            entry
            for entry in record["entries"]
            if float(entry["idm_cost"]) == configured_cost
        ]
        if len(entries) != 1:
            raise ValueError(
                "Counterfactual audit must contain one configured-cost entry."
            )
        batch_std = infer_legacy_normalization_std(entries[0])
        floor_hit = batch_std < std_floor
        replayed.append(
            {
                "runner_step": runner_step,
                "update": runner_step + 1,
                "configured_idm_cost": configured_cost,
                "inferred_batch_standard_deviation": batch_std,
                "normalization_std_floor": std_floor,
                "floor_hit_fraction": float(floor_hit),
                "floor_divisor_over_batch_standard_deviation": (
                    std_floor / batch_std if floor_hit else None
                ),
                "floored_advantage_amplitude_fraction_vs_legacy": (
                    (batch_std + LEGACY_NORMALIZATION_EPSILON) / std_floor
                    if floor_hit
                    else 1.0
                ),
            }
        )

    standard_deviations = [
        item["inferred_batch_standard_deviation"] for item in replayed
    ]
    hit_count = sum(int(item["floor_hit_fraction"]) for item in replayed)
    floor_scaling_factors = [
        item["floor_divisor_over_batch_standard_deviation"]
        for item in replayed
        if item["floor_divisor_over_batch_standard_deviation"] is not None
    ]
    return {
        "schema": FASTWAM_NORMALIZATION_FLOOR_REPLAY_SCHEMA,
        "method": (
            "Recover each legacy batch standard deviation from the paired "
            "unnormalized/normalized configured-cost Gate summaries. Their "
            "sample-standard-deviation ratio equals the legacy divisor "
            "(batch_std + 1e-5); the extrema ratio is checked independently."
        ),
        "normalization_std_floor": std_floor,
        "normalization_batch_scope": (
            "One complete actor rollout batch per runner update, before the "
            "optimizer splits that batch into training microbatches."
        ),
        "normalization_batches_per_runner_step": 1,
        "record_count": len(replayed),
        "floor_hit_count": hit_count,
        "floor_hit_fraction": hit_count / len(replayed),
        "batch_standard_deviation_distribution": _distribution(standard_deviations),
        "floor_divisor_over_batch_standard_deviation_distribution": _distribution(
            floor_scaling_factors
        ),
        "batch_standard_deviation_minimum": min(standard_deviations),
        "batch_standard_deviation_median": statistics.median(standard_deviations),
        "batch_standard_deviation_maximum": max(standard_deviations),
        "records": replayed,
    }


def write_normalization_floor_replay(
    replay: Mapping[str, Any],
    output: str | Path,
) -> tuple[Path, Path, Path]:
    """Write the replay JSON, chart-ready CSV, and per-update curves."""

    json_path = Path(output)
    if json_path.suffix != ".json":
        raise ValueError("Normalization replay output must use a .json suffix.")
    csv_path = json_path.with_suffix(".csv")
    png_path = json_path.with_suffix(".png")
    for path in (json_path, csv_path, png_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite replay artifact: {path}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(replay, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")

    rows = list(replay["records"])
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    updates = [row["update"] for row in rows]
    standard_deviations = [row["inferred_batch_standard_deviation"] for row in rows]
    floor_hits = [row["floor_hit_fraction"] for row in rows]
    scaling_factors = [
        row["floor_divisor_over_batch_standard_deviation"] for row in rows
    ]
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), dpi=180, sharex=True)
    figure.patch.set_facecolor("#FCFCFD")
    for axis in axes:
        axis.set_facecolor("#FCFCFD")
        axis.grid(axis="y", color="#DDE1E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].plot(updates, standard_deviations, color="#2F6FB0", marker="o")
    axes[0].axhline(
        replay["normalization_std_floor"],
        color="#C4473A",
        linestyle="--",
        label="Configured floor",
    )
    axes[0].set_ylabel("Recovered batch std")
    axes[0].legend(frameon=False)
    axes[1].step(updates, floor_hits, where="mid", color="#7A5195")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Floor hit")
    axes[2].plot(
        updates,
        [float("nan") if value is None else value for value in scaling_factors],
        color="#D4882B",
        marker="o",
    )
    axes[2].axhline(1.0, color="#59636E", linestyle=":")
    axes[2].set_ylabel("floor / observed std")
    axes[2].set_xlabel("Runner update")
    figure.suptitle("FastWAM normalization-floor replay", x=0.08, ha="left")
    figure.tight_layout()
    figure.savefig(png_path, bbox_inches="tight")
    plt.close(figure)
    return json_path, csv_path, png_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--std-floor", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    replay = replay_normalization_floor(
        load_counterfactual_cost_audits(args.source),
        std_floor=args.std_floor,
    )
    paths = write_normalization_floor_replay(replay, args.output)
    print(
        json.dumps(
            {
                **{key: replay[key] for key in replay if key != "records"},
                "outputs": [str(path) for path in paths],
            }
        )
    )


if __name__ == "__main__":
    main()
