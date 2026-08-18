# Copyright 2025 The RLinf Authors.
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

"""Periodic static training diagnostics for the FastWAM adaptive policy."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class _LineSpec:
    label: str
    candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RangeSpec:
    label: str
    mean: tuple[str, ...]
    minimum: tuple[str, ...]
    maximum: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PanelSpec:
    title: str
    lines: tuple[_LineSpec, ...] = ()
    ranges: tuple[_RangeSpec, ...] = ()
    y_label: str = "value"


_OVERVIEW_PANELS = (
    _PanelSpec(
        "Sparse success",
        lines=(
            _LineSpec("success rate", ("env/success_once",)),
            _LineSpec("environment return", ("env/return",)),
        ),
        y_label="rate / return",
    ),
    _PanelSpec(
        "Reward and cost",
        lines=(
            _LineSpec("environment reward", ("env/reward",)),
            _LineSpec(
                "shaped reward",
                (
                    "rollout/fastwam/reward/shaped_chunk/mean",
                    "rollout/rewards_mean",
                    "rollout/rewards",
                ),
            ),
            _LineSpec(
                "raw chunk reward",
                ("rollout/fastwam/reward/raw_chunk/mean",),
            ),
            _LineSpec(
                "IDM cost",
                ("rollout/fastwam/cost/actual_chunk/mean",),
            ),
        ),
        y_label="reward per chunk",
    ),
    _PanelSpec(
        "Returns",
        ranges=(
            _RangeSpec(
                "return",
                ("rollout/returns_mean",),
                ("rollout/returns_min",),
                ("rollout/returns_max",),
            ),
        ),
        y_label="return",
    ),
    _PanelSpec(
        "Value estimates",
        ranges=(
            _RangeSpec(
                "value",
                ("rollout/values_mean",),
                ("rollout/values_min",),
                ("rollout/values_max",),
            ),
        ),
        y_label="critic value",
    ),
    _PanelSpec(
        "Advantages",
        ranges=(
            _RangeSpec(
                "all",
                ("rollout/advantages_mean",),
                ("rollout/advantages_min",),
                ("rollout/advantages_max",),
            ),
        ),
        lines=(
            _LineSpec("Gate", ("rollout/gate_advantages_mean",)),
            _LineSpec("UNCOND Flow", ("rollout/flow_advantages_mean",)),
        ),
        y_label="advantage",
    ),
    _PanelSpec(
        "Gate output and realized IDM use",
        lines=(
            _LineSpec(
                "base P(IDM)",
                ("rollout/fastwam/gate/base_idm_probability_mean",),
            ),
            _LineSpec(
                "behavior P(IDM)",
                ("rollout/fastwam/gate/behavior_idm_probability_mean",),
            ),
            _LineSpec(
                "eligible IDM fraction",
                (
                    "rollout/fastwam/route/eligible_idm_fraction",
                    "rollout/fastwam/eligible_idm_fraction",
                ),
            ),
            _LineSpec(
                "all executed IDM fraction",
                ("rollout/fastwam/route/executed_idm_fraction",),
            ),
        ),
        y_label="probability / fraction",
    ),
    _PanelSpec(
        "Gate PPO",
        lines=(
            _LineSpec("entropy", ("train/gate/entropy",)),
            _LineSpec("approx KL", ("train/gate/approx_kl",)),
            _LineSpec("clip fraction", ("train/gate/clip_fraction",)),
            _LineSpec("|ratio - 1|", ("train/gate/ratio_abs",)),
        ),
    ),
    _PanelSpec(
        "Policy losses",
        lines=(
            _LineSpec("Gate", ("train/gate/policy_loss",)),
            _LineSpec("UNCOND Flow", ("train/uncond_flow/policy_loss",)),
            _LineSpec(
                "regularized total",
                ("train/fastwam/regularized_policy_loss",),
            ),
        ),
        y_label="loss",
    ),
    _PanelSpec(
        "UNCOND Flow PPO",
        lines=(
            _LineSpec("approx KL", ("train/uncond_flow/approx_kl",)),
            _LineSpec("clip fraction", ("train/uncond_flow/clip_fraction",)),
            _LineSpec("|ratio - 1|", ("train/uncond_flow/ratio_abs",)),
        ),
    ),
    _PanelSpec(
        "Critic",
        lines=(
            _LineSpec("value loss", ("train/critic/value_loss",)),
            _LineSpec(
                "explained variance",
                ("train/critic/explained_variance",),
            ),
            _LineSpec("value clip ratio", ("train/critic/value_clip_ratio",)),
        ),
    ),
    _PanelSpec(
        "Optimization",
        lines=(
            _LineSpec("gradient norm", ("train/actor/grad_norm",)),
            _LineSpec(
                "Gate relative update",
                ("train/gate/relative_update_l2_norm",),
            ),
            _LineSpec("Gate update max", ("train/gate/update_max_abs",)),
        ),
    ),
    _PanelSpec(
        "Wall-clock time",
        lines=(
            _LineSpec("step", ("time/step",)),
            _LineSpec("rollout", ("time/generate_rollouts",)),
            _LineSpec("actor training", ("time/actor_training",)),
        ),
        y_label="seconds",
    ),
)


def _ema(values: Sequence[float], weight: float) -> list[float]:
    smoothed = []
    previous = None
    for value in values:
        if not math.isfinite(value):
            smoothed.append(float("nan"))
            continue
        previous = (
            value if previous is None else weight * previous + (1.0 - weight) * value
        )
        smoothed.append(previous)
    return smoothed


def _first_available(
    series: Mapping[str, Mapping[int, float]], candidates: Sequence[str]
) -> str | None:
    return next((tag for tag in candidates if tag in series), None)


def _xy(points: Mapping[int, float]) -> tuple[list[int], list[float]]:
    ordered = sorted(points.items())
    return [step for step, _ in ordered], [value for _, value in ordered]


def _plot_line(ax, points, *, label: str, smoothing: float, color=None):
    steps, values = _xy(points)
    if smoothing:
        ax.plot(steps, values, alpha=0.22, linewidth=1.0, color=color)
        (line,) = ax.plot(
            steps,
            _ema(values, smoothing),
            label=label,
            linewidth=1.8,
            color=color,
        )
    else:
        (line,) = ax.plot(
            steps,
            values,
            label=label,
            linewidth=1.6,
            color=color,
        )
    return line


class FastWAMTrainingPlotter:
    """Collect main-run scalars and periodically refresh static diagnostics."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        title: str,
        interval_steps: int = 5,
        smoothing: float = 0.6,
        dpi: int = 160,
        export_all_scalars_on_finish: bool = True,
    ) -> None:
        if interval_steps < 1:
            raise ValueError("FastWAM plot interval_steps must be at least 1.")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("FastWAM plot smoothing must lie in [0, 1).")
        if dpi < 72:
            raise ValueError("FastWAM plot dpi must be at least 72.")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.interval_steps = int(interval_steps)
        self.smoothing = float(smoothing)
        self.dpi = int(dpi)
        self.export_all_scalars_on_finish = bool(export_all_scalars_on_finish)
        self._series: dict[str, dict[int, float]] = defaultdict(dict)
        self._last_render_step: int | None = None
        self._finished = False

    def record(self, data: Mapping[str, Any], step: int) -> None:
        """Record one scalar mapping without retaining tensors or arrays."""

        if self._finished:
            return
        for tag, value in data.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            self._series[str(tag)][int(step)] = scalar

    def maybe_render(self, step: int, *, force: bool = False) -> bool:
        """Refresh the live overview when the configured interval elapses."""

        if self._finished or not self._series:
            return False
        step = int(step)
        if (
            not force
            and self._last_render_step is not None
            and step - self._last_render_step < self.interval_steps
        ):
            return False
        self._render_overview(step, export_pdf=False)
        self._write_summary(finalized=False)
        self._last_render_step = step
        return True

    def finish(self) -> None:
        """Write the final overview, full scalar PDF, CSV, and summary."""

        if self._finished:
            return
        if not self._series:
            self._finished = True
            return
        latest_step = max(max(points) for points in self._series.values())
        self._render_overview(latest_step, export_pdf=True)
        self._write_csv()
        if self.export_all_scalars_on_finish:
            self._render_all_scalars()
        self._finished = True
        self._write_summary(finalized=True)

    def _render_overview(self, step: int, *, export_pdf: bool) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(4, 3, figsize=(18, 19), constrained_layout=True)
        figure.suptitle(f"{self.title} (through step {step})", fontsize=18)
        for ax, panel in zip(axes.flat, _OVERVIEW_PANELS, strict=True):
            plotted = False
            for range_spec in panel.ranges:
                mean_tag = _first_available(self._series, range_spec.mean)
                if mean_tag is None:
                    continue
                line = _plot_line(
                    ax,
                    self._series[mean_tag],
                    label=range_spec.label,
                    smoothing=self.smoothing,
                )
                plotted = True
                minimum_tag = _first_available(self._series, range_spec.minimum)
                maximum_tag = _first_available(self._series, range_spec.maximum)
                if minimum_tag is not None and maximum_tag is not None:
                    means = self._series[mean_tag]
                    minimum = self._series[minimum_tag]
                    maximum = self._series[maximum_tag]
                    common_steps = sorted(
                        means.keys() & minimum.keys() & maximum.keys()
                    )
                    ax.fill_between(
                        common_steps,
                        [minimum[item] for item in common_steps],
                        [maximum[item] for item in common_steps],
                        color=line.get_color(),
                        alpha=0.14,
                        linewidth=0,
                    )
            for line_spec in panel.lines:
                tag = _first_available(self._series, line_spec.candidates)
                if tag is None:
                    continue
                _plot_line(
                    ax,
                    self._series[tag],
                    label=line_spec.label,
                    smoothing=self.smoothing,
                )
                plotted = True
            ax.set_title(panel.title)
            ax.set_xlabel("optimizer update / runner step")
            ax.set_ylabel(panel.y_label)
            ax.grid(True, alpha=0.25)
            if plotted:
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "metric not logged",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        self._save_figure(figure, self.output_dir / "overview.png", dpi=self.dpi)
        if export_pdf:
            self._save_figure(figure, self.output_dir / "overview.pdf")
        plt.close(figure)

    def _render_all_scalars(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        destination = self.output_dir / "all-scalars.pdf"
        temporary = destination.with_name(f".{destination.name}.tmp")
        tags = sorted(self._series)
        with PdfPages(temporary) as pdf:
            for offset in range(0, len(tags), 6):
                page_tags = tags[offset : offset + 6]
                figure, axes = plt.subplots(
                    3,
                    2,
                    figsize=(14, 12),
                    constrained_layout=True,
                )
                for ax, tag in zip(axes.flat, page_tags, strict=False):
                    _plot_line(
                        ax,
                        self._series[tag],
                        label=tag,
                        smoothing=self.smoothing,
                    )
                    ax.set_title(tag, fontsize=9)
                    ax.set_xlabel("optimizer update / runner step")
                    ax.grid(True, alpha=0.25)
                for ax in axes.flat[len(page_tags) :]:
                    ax.axis("off")
                pdf.savefig(figure)
                plt.close(figure)
        temporary.replace(destination)

    def _write_csv(self) -> None:
        destination = self.output_dir / "scalars.csv"
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("tag", "step", "value"))
            for tag in sorted(self._series):
                for step, value in sorted(self._series[tag].items()):
                    writer.writerow((tag, step, value))
        temporary.replace(destination)

    def _write_summary(self, *, finalized: bool) -> None:
        values = [
            (tag, step, value)
            for tag, points in self._series.items()
            for step, value in points.items()
        ]
        nonfinite = [
            {"tag": tag, "step": step, "value": repr(value)}
            for tag, step, value in values
            if not math.isfinite(value)
        ]
        payload = {
            "schema": "fastwam-live-training-curves-v1",
            "finalized": finalized,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tag_count": len(self._series),
            "point_count": len(values),
            "minimum_step": min(step for _, step, _ in values),
            "maximum_step": max(step for _, step, _ in values),
            "smoothing": self.smoothing,
            "interval_steps": self.interval_steps,
            "nonfinite_point_count": len(nonfinite),
            "nonfinite_points": nonfinite,
            "tags": {
                tag: {
                    "point_count": len(points),
                    "minimum_step": min(points),
                    "maximum_step": max(points),
                    "latest_value": points[max(points)],
                }
                for tag, points in sorted(self._series.items())
            },
        }
        destination = self.output_dir / "summary.json"
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @staticmethod
    def _save_figure(figure, destination: Path, *, dpi: int | None = None) -> None:
        temporary = destination.with_name(f".{destination.name}.tmp")
        figure.savefig(temporary, format=destination.suffix.lstrip("."), dpi=dpi)
        temporary.replace(destination)
