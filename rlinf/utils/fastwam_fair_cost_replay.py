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

"""Replay lagged fair IDM costs from a completed counterfactual audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlinf.algorithms.advantages import (
    FASTWAM_COUNTERFACTUAL_COST_AUDIT_SCHEMA,
    FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL,
    compute_fastwam_break_even_idm_cost,
)
from rlinf.runners.fastwam_fair_cost import FastWAMFairCostController

FASTWAM_FAIR_COST_REPLAY_SCHEMA = "fastwam-fair-cost-replay-v1"


def _positive_change_factor(previous: float, current: float) -> float | None:
    if previous == current:
        return 1.0
    if previous <= 0.0 or current <= 0.0:
        return None
    return max(previous / current, current / previous)


def summarize_fair_cost_stability(
    fair_costs: Sequence[float],
) -> dict[str, Any]:
    """Describe open-loop adjacent changes without claiming closed-loop stability."""

    if not fair_costs:
        raise ValueError("Fair-cost stability summary requires at least one cost.")
    costs = [float(value) for value in fair_costs]
    factors = [
        _positive_change_factor(previous, current)
        for previous, current in zip(costs, costs[1:], strict=False)
    ]
    finite_pairs = [
        (index, factor) for index, factor in enumerate(factors) if factor is not None
    ]
    maximum_pair = max(finite_pairs, key=lambda item: item[1], default=None)
    post_bootstrap_pairs = finite_pairs[1:]
    maximum_post_bootstrap = max(
        post_bootstrap_pairs,
        key=lambda item: item[1],
        default=None,
    )
    directions = []
    for previous, current in zip(costs, costs[1:], strict=False):
        if current > previous:
            directions.append(1)
        elif current < previous:
            directions.append(-1)
    direction_reversals = sum(
        previous != current
        for previous, current in zip(directions, directions[1:], strict=False)
    )
    nondecreasing = all(
        current >= previous for previous, current in zip(costs, costs[1:], strict=False)
    )
    nonincreasing = all(
        current <= previous for previous, current in zip(costs, costs[1:], strict=False)
    )
    unbounded_change = any(factor is None for factor in factors)
    maximum_factor = None if maximum_pair is None else maximum_pair[1]
    return {
        "scope": "open-loop replay over historical trajectories",
        "closed_loop_stability_claim": False,
        "adjacent_change_factors": factors,
        "maximum_adjacent_change_factor": maximum_factor,
        "maximum_adjacent_change_from_runner_step": (
            None if maximum_pair is None else maximum_pair[0]
        ),
        "maximum_adjacent_change_to_runner_step": (
            None if maximum_pair is None else maximum_pair[0] + 1
        ),
        "maximum_post_bootstrap_adjacent_change_factor": (
            None if maximum_post_bootstrap is None else maximum_post_bootstrap[1]
        ),
        "maximum_post_bootstrap_change_from_runner_step": (
            None if maximum_post_bootstrap is None else maximum_post_bootstrap[0]
        ),
        "maximum_post_bootstrap_change_to_runner_step": (
            None if maximum_post_bootstrap is None else maximum_post_bootstrap[0] + 1
        ),
        "unbounded_adjacent_change": unbounded_change,
        "adjacent_change_exceeds_twofold": (
            unbounded_change or (maximum_factor is not None and maximum_factor > 2.0)
        ),
        "nondecreasing": nondecreasing,
        "nonincreasing": nonincreasing,
        "monotonic": nondecreasing or nonincreasing,
        "direction_reversal_count": direction_reversals,
        "direction_reversals_present": direction_reversals > 0,
    }


def load_counterfactual_cost_audits(path: str | Path) -> list[dict[str, Any]]:
    """Load either the audit JSONL or actor stdout sentinel records."""

    records = []
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL in line:
                encoded = line.split(
                    FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL,
                    1,
                )[1].strip()
            elif line.lstrip().startswith("{"):
                encoded = line.strip()
            else:
                continue
            try:
                payload = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if payload.get("schema") == FASTWAM_COUNTERFACTUAL_COST_AUDIT_SCHEMA:
                records.append(payload)
    if not records:
        raise ValueError(f"No FastWAM counterfactual cost audits found in {source}.")
    return records


def recompute_artifact_break_even(record: Mapping[str, Any]) -> float | None:
    """Use the production break-even function on one serialized audit."""

    points = []
    for entry in record["entries"]:
        idm = entry["idm_destination_gate_advantage"]["unnormalized"]
        uncond = entry["uncond_destination_gate_advantage"]["unnormalized"]
        points.append(
            (
                float(entry["idm_cost"]),
                float(idm["sum"]),
                int(idm["finite_count"]),
                float(uncond["sum"]),
                int(uncond["finite_count"]),
            )
        )
    return compute_fastwam_break_even_idm_cost(points)


def replay_fair_costs(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_idm_cost: float | None = None,
    window_size: int = 5,
) -> dict[str, Any]:
    """Return the costs that a lagged, PI-disabled controller would apply."""

    if not records:
        raise ValueError("Fair-cost replay requires at least one audit record.")
    if bootstrap_idm_cost is None:
        bootstrap_idm_cost = float(records[0]["configured_idm_cost"])
    controller = FastWAMFairCostController(
        {
            "enabled": True,
            "window_size": window_size,
            "pi": {
                "enabled": False,
                "target_idm_fraction": 0.5,
                "integral_gain": 0.05,
                "proportional_gain": 0.6,
            },
        },
        bootstrap_idm_cost=bootstrap_idm_cost,
    )
    replayed = []
    undefined_count = 0
    for index, source_record in enumerate(records):
        runner_step = int(source_record.get("runner_step", index))
        if runner_step != index:
            raise ValueError(
                "Fair-cost replay requires contiguous runner steps beginning at zero."
            )
        recorded_break_even = source_record.get("break_even_idm_cost")
        recomputed_break_even = recompute_artifact_break_even(source_record)
        if (recorded_break_even is None) != (recomputed_break_even is None) or (
            recorded_break_even is not None
            and not math.isclose(
                float(recorded_break_even),
                float(recomputed_break_even),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "Serialized break-even price disagrees with advantages.py at "
                f"runner step {runner_step}."
            )
        undefined_count += recorded_break_even is None
        eligible_count = int(source_record["eligible_gate_decision_count"])
        eligible_idm_count = int(source_record["eligible_idm_decision_count"])
        if eligible_count < 1 or not 0 <= eligible_idm_count <= eligible_count:
            raise ValueError("Counterfactual audit route counts are invalid.")
        observation = controller.observe_rollout(
            runner_step=runner_step,
            break_even_idm_cost=recomputed_break_even,
            idm_fraction=eligible_idm_count / eligible_count,
        )
        replayed.append(
            {
                "runner_step": runner_step,
                "update": runner_step + 1,
                "historical_configured_idm_cost": float(
                    source_record["configured_idm_cost"]
                ),
                "historical_actual_charge": float(source_record["configured_idm_cost"]),
                "historical_break_even_idm_cost": recorded_break_even,
                "break_even_defined": recorded_break_even is not None,
                "eligible_idm_fraction": eligible_idm_count / eligible_count,
                "counterfactual_fair_charge": observation["applied"][
                    "applied_idm_cost"
                ],
                "fair_cost_estimate": observation["applied"]["fair_idm_cost"],
                "fair_cost_applied": observation["applied"],
                "carried_break_even_idm_cost": observation[
                    "carried_break_even_idm_cost"
                ],
                "undefined_break_even_carried_forward": (
                    recorded_break_even is None
                    and observation["carried_break_even_idm_cost"] is not None
                ),
                "fair_cost_for_next_step": observation["next"],
            }
        )
    fair_costs = [record["counterfactual_fair_charge"] for record in replayed]
    return {
        "schema": FASTWAM_FAIR_COST_REPLAY_SCHEMA,
        "bootstrap_idm_cost": float(bootstrap_idm_cost),
        "window_size": int(window_size),
        "pi_enabled": False,
        "record_count": len(replayed),
        "undefined_break_even_count": int(undefined_count),
        "undefined_break_even_runner_steps": [
            record["runner_step"]
            for record in replayed
            if not record["break_even_defined"]
        ],
        "stability": summarize_fair_cost_stability(fair_costs),
        "limitation": (
            "This is an open-loop replay over trajectories collected under the "
            "historical fixed cost. It does not validate closed-loop stability."
        ),
        "records": replayed,
    }


def write_fair_cost_replay(
    replay: Mapping[str, Any],
    output: str | Path,
) -> tuple[Path, Path, Path]:
    """Write fair-cost JSON, a flat comparison table, and replay curves."""

    json_path = Path(output)
    if json_path.suffix != ".json":
        raise ValueError("Fair-cost replay output must use a .json suffix.")
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
    csv_fields = [
        "runner_step",
        "update",
        "historical_actual_charge",
        "counterfactual_fair_charge",
        "fair_cost_estimate",
        "historical_break_even_idm_cost",
        "break_even_defined",
        "carried_break_even_idm_cost",
        "undefined_break_even_carried_forward",
        "eligible_idm_fraction",
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in csv_fields} for row in rows)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    updates = [row["update"] for row in rows]
    actual = [row["historical_actual_charge"] for row in rows]
    fair = [row["counterfactual_fair_charge"] for row in rows]
    break_even = [
        float("nan")
        if row["historical_break_even_idm_cost"] is None
        else row["historical_break_even_idm_cost"]
        for row in rows
    ]
    factors = [float("nan"), *replay["stability"]["adjacent_change_factors"]]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), dpi=180, sharex=True)
    figure.patch.set_facecolor("#FCFCFD")
    for axis in axes:
        axis.set_facecolor("#FCFCFD")
        axis.grid(axis="y", color="#DDE1E6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].plot(updates, actual, color="#343A40", label="Historical charge")
    axes[0].plot(updates, fair, color="#2F6FB0", label="Lagged fair charge")
    axes[0].scatter(
        updates,
        break_even,
        color="#D4882B",
        s=14,
        label="Observed break-even",
    )
    axes[0].set_ylabel("IDM cost")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(updates, factors, color="#7A5195", marker="o")
    axes[1].axhline(2.0, color="#C4473A", linestyle="--", label="2x gate")
    axes[1].set_ylabel("Adjacent change factor")
    axes[1].set_xlabel("Runner update")
    axes[1].legend(frameon=False)
    figure.suptitle("Lagged fair-cost open-loop replay", x=0.08, ha="left")
    figure.tight_layout()
    figure.savefig(png_path, bbox_inches="tight")
    plt.close(figure)
    return json_path, csv_path, png_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-idm-cost", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = load_counterfactual_cost_audits(args.source)
    replay = replay_fair_costs(
        records,
        bootstrap_idm_cost=args.bootstrap_idm_cost,
        window_size=args.window_size,
    )
    paths = write_fair_cost_replay(replay, args.output)
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
