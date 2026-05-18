#!/usr/bin/env python
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
"""Diagnostic CALVIN handoff baselines for OpenPI policies.

This evaluator compares the next-skill success rate from a policy-generated
terminal state against an oracle symbolic reset to the expected next-skill
start state. It is intentionally single-process, matching ``calvin_eval.py``.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RolloutResult:
    """Result from one subtask rollout."""

    success: bool
    steps: int
    frames: list[np.ndarray]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_plain_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _condition_matches(state: dict[str, Any], condition: dict[str, Any]) -> bool:
    for key, expected in condition.items():
        actual = state.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def advance_symbolic_state(
    state: dict[str, Any],
    subtask: str,
    task_specs: dict[str, list[dict[str, dict[str, Any]]]],
) -> dict[str, Any]:
    """Apply one CALVIN symbolic task effect to a symbolic state.

    Args:
        state: CALVIN symbolic state, e.g. values for drawer, slider, blocks.
        subtask: Canonical CALVIN subtask name.
        task_specs: Mapping from subtask names to CALVIN condition/effect rules.

    Returns:
        The symbolic state after the subtask effect.

    Raises:
        KeyError: If ``subtask`` is not in ``task_specs``.
        ValueError: If the state matches zero or multiple effect rules.
    """

    if subtask not in task_specs:
        raise KeyError(f"No symbolic task spec found for subtask {subtask!r}")

    matching_specs = [
        spec
        for spec in task_specs[subtask]
        if _condition_matches(state, spec.get("condition", {}))
    ]
    if len(matching_specs) != 1:
        raise ValueError(
            f"Expected exactly one symbolic effect for {subtask!r} from state "
            f"{state!r}; got {len(matching_specs)}"
        )

    next_state = copy.deepcopy(state)
    next_state.update(matching_specs[0].get("effect", {}))
    return next_state


def compute_symbolic_state_prefixes(
    initial_state: dict[str, Any],
    task_sequence: Iterable[str],
    task_specs: dict[str, list[dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Return symbolic states before each task and after the final task."""

    states = [copy.deepcopy(initial_state)]
    state = copy.deepcopy(initial_state)
    for subtask in task_sequence:
        state = advance_symbolic_state(state, subtask, task_specs)
        states.append(state)
    return states


def parse_pair_positions(pair_positions: str, sequence_length: int) -> list[int]:
    """Parse pair positions for adjacent pairs in a task sequence."""

    max_pair_position = sequence_length - 2
    if max_pair_position < 0:
        return []
    if pair_positions == "all":
        return list(range(max_pair_position + 1))

    positions = []
    for raw_position in pair_positions.split(","):
        raw_position = raw_position.strip()
        if not raw_position:
            continue
        position = int(raw_position)
        if position < 0 or position > max_pair_position:
            raise ValueError(
                f"Pair position {position} is out of range for sequence length "
                f"{sequence_length}; expected 0..{max_pair_position}"
            )
        positions.append(position)
    return positions


def get_task_prompt(task_instructions: dict[str, Any], subtask: str) -> str:
    """Select the same current-instruction-only prompt as calvin_eval.py."""

    prompts = task_instructions[subtask]
    if isinstance(prompts, str):
        return prompts
    return str(prompts[0])


def make_policy_element(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Build one OpenPI CALVIN policy input from an env observation."""

    img = obs["rgb_obs"]["rgb_static"]
    wrist_img = obs["rgb_obs"]["rgb_gripper"]
    state = obs["robot_obs"][:7]
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "observation/state_ee_pos": obs["robot_obs"][:3],
        "observation/state_ee_rot": obs["robot_obs"][3:6],
        "observation/state_gripper": obs["robot_obs"][6:7],
        "prompt": prompt,
    }


def rollout_subtask(
    *,
    env: Any,
    policy: Any,
    task_reward: Any,
    task_instructions: dict[str, Any],
    subtask: str,
    max_steps: int,
    action_chunk: int,
    collect_frames: bool = False,
) -> RolloutResult:
    """Roll out one current-instruction-only CALVIN subtask."""

    start_info = env.get_info()
    action_plan: collections.deque[np.ndarray] = collections.deque()
    obs = env.get_obs()
    frames: list[np.ndarray] = []
    prompt = get_task_prompt(task_instructions, subtask)

    for step_idx in range(max_steps):
        if collect_frames:
            frames.append(np.asarray(obs["rgb_obs"]["rgb_static"]))

        if not action_plan:
            action_chunk_result = policy.infer(make_policy_element(obs, prompt))[
                "actions"
            ]
            if len(action_chunk_result) < action_chunk:
                raise ValueError(
                    f"Policy returned {len(action_chunk_result)} actions, but "
                    f"action_chunk={action_chunk} was requested."
                )
            action_plan.extend(action_chunk_result[:action_chunk])

        action = np.asarray(action_plan.popleft()).copy()
        action[-1] = 1 if action[-1] > 0 else -1
        obs, _, _, current_info = env.step(action)

        current_task_info = task_reward.get_task_info_for_set(
            start_info, current_info, {subtask}
        )
        if len(current_task_info) > 0:
            if collect_frames:
                frames.append(np.asarray(obs["rgb_obs"]["rgb_static"]))
            return RolloutResult(success=True, steps=step_idx + 1, frames=frames)

    return RolloutResult(success=False, steps=max_steps, frames=frames)


def reset_env_to_symbolic_state(
    *,
    env: Any,
    symbolic_state: dict[str, Any],
    get_env_state_for_initial_condition: Any,
) -> None:
    robot_obs, scene_obs = get_env_state_for_initial_condition(symbolic_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _summarize_record_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_pairs = len(records)
    current_success_count = sum(int(record["current_success"]) for record in records)
    policy_next_attempts = sum(
        int(record["policy_terminal_next_attempted"]) for record in records
    )
    policy_next_success_count = sum(
        int(record["policy_terminal_next_success"] is True) for record in records
    )
    oracle_next_success_count = sum(
        int(record["oracle_next_success"]) for record in records
    )
    oracle_next_success_after_current_success_count = sum(
        int(record["current_success"] and record["oracle_next_success"])
        for record in records
    )

    policy_next_rate = _rate(policy_next_success_count, policy_next_attempts)
    oracle_next_rate = _rate(oracle_next_success_count, num_pairs)
    matched_oracle_next_rate = _rate(
        oracle_next_success_after_current_success_count,
        current_success_count,
    )
    handoff_gap = (
        None
        if oracle_next_rate is None or policy_next_rate is None
        else oracle_next_rate - policy_next_rate
    )
    matched_handoff_gap = (
        None
        if matched_oracle_next_rate is None or policy_next_rate is None
        else matched_oracle_next_rate - policy_next_rate
    )

    return {
        "num_pairs": num_pairs,
        "current_success_count": current_success_count,
        "current_success_rate": _rate(current_success_count, num_pairs),
        "policy_terminal_next_attempts": policy_next_attempts,
        "policy_terminal_next_success_count": policy_next_success_count,
        "next_success_from_policy_terminal": policy_next_rate,
        "oracle_next_success_count": oracle_next_success_count,
        "next_success_from_oracle_start": oracle_next_rate,
        "oracle_next_success_given_current_success_count": (
            oracle_next_success_after_current_success_count
        ),
        "next_success_from_oracle_start_given_current_success": (
            matched_oracle_next_rate
        ),
        "handoff_gap": handoff_gap,
        "handoff_gap_given_current_success": matched_handoff_gap,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize overall and grouped CALVIN handoff diagnostics."""

    def grouped_by(key: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in records:
            grouped[str(record[key])].append(record)
        return {
            group_key: _summarize_record_group(group_records)
            for group_key, group_records in sorted(grouped.items())
        }

    return {
        "overall": _summarize_record_group(records),
        "by_pair_position": grouped_by("pair_position"),
        "by_current_task": grouped_by("current_task"),
        "by_next_task": grouped_by("next_task"),
    }


def maybe_save_video(
    *,
    output_dir: pathlib.Path,
    frames: list[np.ndarray],
    name: str,
    video_temp_subsample: int,
) -> pathlib.Path | None:
    if not frames:
        return None
    import imageio

    video_path = output_dir / "videos" / f"{name}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subsample = max(1, video_temp_subsample)
    imageio.mimwrite(
        video_path,
        [np.asarray(frame) for frame in frames[::subsample]],
        fps=max(1, 50 // subsample),
    )
    return video_path


def run_handoff_diagnostic(
    *,
    env: Any,
    policy: Any,
    task_definitions: list[tuple[dict[str, Any], list[str]]],
    task_instructions: dict[str, Any],
    task_reward: Any,
    task_specs: dict[str, list[dict[str, dict[str, Any]]]],
    get_env_state_for_initial_condition: Any,
    output_dir: pathlib.Path,
    pair_positions: str,
    max_steps_current: int,
    max_steps_next: int,
    action_chunk: int,
    num_save_videos: int,
    video_temp_subsample: int,
    logger: logging.Logger | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the diagnostic baseline and write JSONL plus summary outputs."""

    logger = logger or logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "handoff_trials.jsonl"
    summary_path = output_dir / "handoff_summary.json"
    records: list[dict[str, Any]] = []
    saved_videos = 0

    iterator: Iterable[tuple[int, tuple[dict[str, Any], list[str]]]] = enumerate(
        task_definitions
    )
    if progress:
        try:
            import tqdm

            iterator = tqdm.tqdm(iterator, total=len(task_definitions))
        except ImportError:
            logger.warning("tqdm is not installed; running without a progress bar.")

    with records_path.open("w", encoding="utf-8") as records_file:
        for episode_idx, (initial_state, task_sequence) in iterator:
            task_sequence = list(task_sequence)
            symbolic_states = compute_symbolic_state_prefixes(
                initial_state,
                task_sequence,
                task_specs,
            )
            positions = parse_pair_positions(pair_positions, len(task_sequence))
            logger.info(
                "Evaluating episode %s pairs %s for sequence %s",
                episode_idx,
                positions,
                task_sequence,
            )

            for pair_position in positions:
                current_task = task_sequence[pair_position]
                next_task = task_sequence[pair_position + 1]
                current_start_state = symbolic_states[pair_position]
                oracle_next_start_state = symbolic_states[pair_position + 1]
                should_save_video = saved_videos < num_save_videos

                reset_env_to_symbolic_state(
                    env=env,
                    symbolic_state=current_start_state,
                    get_env_state_for_initial_condition=(
                        get_env_state_for_initial_condition
                    ),
                )
                current_result = rollout_subtask(
                    env=env,
                    policy=policy,
                    task_reward=task_reward,
                    task_instructions=task_instructions,
                    subtask=current_task,
                    max_steps=max_steps_current,
                    action_chunk=action_chunk,
                    collect_frames=should_save_video,
                )

                policy_next_result = None
                if current_result.success:
                    policy_next_result = rollout_subtask(
                        env=env,
                        policy=policy,
                        task_reward=task_reward,
                        task_instructions=task_instructions,
                        subtask=next_task,
                        max_steps=max_steps_next,
                        action_chunk=action_chunk,
                        collect_frames=should_save_video,
                    )

                reset_env_to_symbolic_state(
                    env=env,
                    symbolic_state=oracle_next_start_state,
                    get_env_state_for_initial_condition=(
                        get_env_state_for_initial_condition
                    ),
                )
                oracle_next_result = rollout_subtask(
                    env=env,
                    policy=policy,
                    task_reward=task_reward,
                    task_instructions=task_instructions,
                    subtask=next_task,
                    max_steps=max_steps_next,
                    action_chunk=action_chunk,
                    collect_frames=should_save_video,
                )

                video_path = None
                if should_save_video:
                    frames = list(current_result.frames)
                    if policy_next_result is not None:
                        frames.extend(policy_next_result.frames)
                    frames.extend(oracle_next_result.frames)
                    video_path = maybe_save_video(
                        output_dir=output_dir,
                        frames=frames,
                        name=(
                            f"episode_{episode_idx:04d}_pair_{pair_position}_"
                            f"{current_task}_to_{next_task}"
                        ),
                        video_temp_subsample=video_temp_subsample,
                    )
                    if video_path is not None:
                        saved_videos += 1

                record = {
                    "episode_index": episode_idx,
                    "pair_position": pair_position,
                    "current_task": current_task,
                    "next_task": next_task,
                    "current_start_state": _to_plain_jsonable(current_start_state),
                    "oracle_next_start_state": _to_plain_jsonable(
                        oracle_next_start_state
                    ),
                    "current_success": bool(current_result.success),
                    "current_steps": int(current_result.steps),
                    "policy_terminal_next_attempted": bool(current_result.success),
                    "policy_terminal_next_success": None
                    if policy_next_result is None
                    else bool(policy_next_result.success),
                    "policy_terminal_next_steps": None
                    if policy_next_result is None
                    else int(policy_next_result.steps),
                    "oracle_next_success": bool(oracle_next_result.success),
                    "oracle_next_steps": int(oracle_next_result.steps),
                    "video_path": None if video_path is None else str(video_path),
                }
                records.append(record)
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_file.flush()

    summary = summarize_records(records)
    summary["records_path"] = str(records_path)
    summary["summary_path"] = str(summary_path)
    summary["num_saved_videos"] = saved_videos
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CALVIN Pi05 handoff diagnostic baselines."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs and diagnostic artifacts.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="calvin_pi05_handoff_diag",
        help="Experiment name used under log_dir.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_calvin",
        help="OpenPI config name, typically pi05_calvin for this diagnostic.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the RLinf Pi05 CALVIN checkpoint directory.",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="calvin_d",
        choices=("calvin_d", "calvin_abc", "calvin_abcd"),
        help="CALVIN task annotation suite.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="Number of standard CALVIN evaluation sequences to diagnose.",
    )
    parser.add_argument(
        "--max_steps_current",
        type=int,
        default=480,
        help="Maximum steps for current subtask G_i.",
    )
    parser.add_argument(
        "--max_steps_next",
        type=int,
        default=480,
        help="Maximum steps for next subtask G_{i+1}.",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=5,
        help="Number of predicted actions to execute before replanning.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of denoising steps sampled by the OpenPI policy.",
    )
    parser.add_argument(
        "--pair_positions",
        type=str,
        default="all",
        help="Adjacent pair positions to evaluate: 'all' or comma-separated 0-based indices.",
    )
    parser.add_argument(
        "--num_save_videos",
        type=int,
        default=10,
        help="Number of pair diagnostic videos to save.",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=10,
        help="Temporal subsampling for saved videos.",
    )
    return parser


def main(args: argparse.Namespace) -> dict[str, Any]:
    from calvin_agent.evaluation.multistep_sequences import tasks as task_specs
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from calvin_env.envs.play_table_env import get_env

    from rlinf.envs.calvin import ENV_CFG_DIR, _get_calvin_tasks_and_reward
    from toolkits.eval_scripts_openpi import setup_logger, setup_policy

    logger = setup_logger(args.exp_name, args.log_dir)
    output_dir = pathlib.Path(args.log_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up CALVIN environment.")
    env = get_env(ENV_CFG_DIR, show_gui=False)
    task_definitions, task_instructions, task_reward = _get_calvin_tasks_and_reward(
        args.num_trials,
        task_suite_name=args.task_suite_name,
    )

    logger.info("Loading policy from %s.", args.pretrained_path)
    policy = setup_policy(args)
    logger.info("Policy setup done.")

    try:
        summary = run_handoff_diagnostic(
            env=env,
            policy=policy,
            task_definitions=task_definitions,
            task_instructions=task_instructions,
            task_reward=task_reward,
            task_specs=task_specs,
            get_env_state_for_initial_condition=get_env_state_for_initial_condition,
            output_dir=output_dir,
            pair_positions=args.pair_positions,
            max_steps_current=args.max_steps_current,
            max_steps_next=args.max_steps_next,
            action_chunk=args.action_chunk,
            num_save_videos=args.num_save_videos,
            video_temp_subsample=args.video_temp_subsample,
            logger=logger,
            progress=True,
        )
    finally:
        env.close()

    logger.info("Handoff summary: %s", json.dumps(summary["overall"], indent=2))
    logger.info("Wrote records to %s", summary["records_path"])
    logger.info("Wrote summary to %s", summary["summary_path"])
    return summary


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
