# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

import copy
import glob
import hashlib
import importlib
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

if TYPE_CHECKING:
    from fastwam.causal_prediction import (
        CausalSamplingMetadataV2,
        CausalStateIdentityV2,
    )

from rlinf.envs.action_contract import (
    SUBMITTED_LIBERO_ACTION_STAGE,
    ActionStageStatistics,
    validate_action_stage_contract,
)
from rlinf.envs.libero.action_contract import (
    LiberoActionContract,
    inspect_libero_action_contract,
    merge_libero_action_contracts,
)
from rlinf.envs.libero.causal_snapshot import (
    CausalSnapshotV1,
    CausalSnapshotV2,
    capture_process_rng_state,
    restore_process_rng_state,
)
from rlinf.envs.libero.egl import instantiate_with_isolated_egl
from rlinf.envs.libero.reward_utils import mask_rewards_after_first_done
from rlinf.envs.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
    get_benchmark_overridden,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
    record_completed_episode_task_stats,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

logger = get_logger()

libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    from libero.libero.benchmark import Benchmark


class LiberoEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        *,
        global_environment_offset: int = 0,
        total_global_environments: int | None = None,
    ):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        if seed_offset == 0:
            self._log_evaluation_mode()
        self.stage_invariant_fixed_reset_ids = bool(
            cfg.get("stage_invariant_fixed_reset_ids", False)
        )
        self.global_environment_offset = int(global_environment_offset)
        self.total_global_environments = int(
            num_envs if total_global_environments is None else total_global_environments
        )
        self.formal_runner_step = 0
        self.seed = (
            int(self.cfg.seed)
            if self.stage_invariant_fixed_reset_ids
            else self.cfg.seed + seed_offset
        )
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)
        self.ordered_reset_state_ids = cfg.get("ordered_reset_state_ids", None)
        if self.ordered_reset_state_ids is not None:
            self.ordered_reset_state_ids = list(self.ordered_reset_state_ids)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.is_eval = cfg.get("is_eval", False)
        if self.stage_invariant_fixed_reset_ids:
            if self.is_eval:
                raise ValueError("stage_invariant_fixed_reset_ids is training-only.")
            if not self.use_fixed_reset_state_ids:
                raise ValueError(
                    "stage_invariant_fixed_reset_ids requires "
                    "use_fixed_reset_state_ids=true."
                )
            if self.group_size != 1:
                raise ValueError(
                    "stage_invariant_fixed_reset_ids currently requires group_size=1."
                )
            if str(cfg.get("libero_variant", "standard")) != "standard":
                raise ValueError(
                    "stage_invariant_fixed_reset_ids currently requires standard "
                    "LIBERO."
                )
            if (
                self.global_environment_offset < 0
                or self.total_global_environments < 1
                or self.global_environment_offset + self.num_envs
                > self.total_global_environments
            ):
                raise ValueError(
                    "Stage-invariant LIBERO global environment bounds are invalid."
                )
        reset_wait_steps = cfg.get("reset_wait_steps", 15)
        if (
            isinstance(reset_wait_steps, bool)
            or int(reset_wait_steps) != reset_wait_steps
            or int(reset_wait_steps) < 1
        ):
            raise ValueError("LIBERO reset_wait_steps must be a positive integer.")
        self.reset_wait_steps = int(reset_wait_steps)

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.is_eval:
            pool = self.reset_state_ids_all[self.seed_offset]
            self._eval_reset_pool = pool[pool >= 0].copy()
        else:
            self._eval_reset_pool = np.array([], dtype=np.int64)
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()
        self._action_submission_capture = None

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

    def _log_evaluation_mode(self):
        """Log the LIBERO evaluation mode banner (rank 0 env worker only)."""
        libero_type = get_libero_type()
        if libero_type == "pro":
            perturbation = os.environ.get("LIBERO_PERTURBATION", "all")
            logger.info(f"Evaluation Mode: LIBERO-PRO | Perturbation: {perturbation}")
        elif libero_type == "plus":
            suffix = os.environ.get("LIBERO_SUFFIX", "all")
            logger.info(f"Evaluation Mode: LIBERO-PLUS | Suffix: {suffix}")
        else:
            logger.info("Evaluation Mode: Standard LIBERO")

    def capture_causal_snapshot(
        self,
        *,
        snapshot_id: str,
        recent_history=(),
        policy_runtime_state=None,
        source_policy: str,
        previous_mode: str | None,
        chunk_index: int,
        remaining_budget: float,
    ) -> CausalSnapshotV1:
        """Capture a complete single-environment same-state fork point."""

        if self.num_envs != 1:
            raise RuntimeError(
                "CausalSnapshotV1 requires num_envs=1 so restoring a branch cannot "
                "mutate unrelated vector slots."
            )
        if self.current_raw_obs is None or len(self.current_raw_obs) != 1:
            raise RuntimeError("LIBERO must be reset before capturing a snapshot.")
        wrapper_state = {
            "task_ids": self.task_ids.copy(),
            "trial_ids": self.trial_ids.copy(),
            "reset_state_ids": self.reset_state_ids.copy(),
            "prev_step_reward": self.prev_step_reward.copy(),
            "success_once": self.success_once.copy(),
            "fail_once": self.fail_once.copy(),
            "returns": self.returns.copy(),
            "success_episode_len": self.success_episode_len.copy(),
            "elapsed_steps": self._elapsed_steps.copy(),
            "task_success_stats": copy.deepcopy(self._task_success_stats),
            "eval_seen_trials": copy.deepcopy(self._eval_seen_trials),
            "generator": copy.deepcopy(self._generator.bit_generator.state),
            "generator_ordered": copy.deepcopy(
                self._generator_ordered.bit_generator.state
            ),
            "start_idx": int(self.start_idx),
            "formal_runner_step": int(self.formal_runner_step),
            "is_start": bool(self._is_start),
        }
        worker_state = self.env.capture_causal_states(id=[0])[0]
        return CausalSnapshotV1(
            snapshot_id=snapshot_id,
            worker_state=worker_state,
            wrapper_state=wrapper_state,
            current_raw_observation=copy.deepcopy(self.current_raw_obs[0]),
            recent_history=tuple(copy.deepcopy(tuple(recent_history))),
            policy_runtime_state=copy.deepcopy(policy_runtime_state or {}),
            driver_rng_state=capture_process_rng_state(),
            source_policy=str(source_policy),
            previous_mode=None if previous_mode is None else str(previous_mode),
            chunk_index=int(chunk_index),
            remaining_budget=float(remaining_budget),
        )

    def observe_causal_task_state(self):
        """Return native goal predicates and task-object contact for batch one."""

        if self.num_envs != 1:
            raise RuntimeError("Causal task observation requires num_envs=1.")
        return self.env.observe_causal_task_states(id=[0])[0]

    def observe_causal_determinism_state(self):
        """Return physical and contact state for a single-environment audit."""

        if self.num_envs != 1:
            raise RuntimeError("Causal determinism observation requires num_envs=1.")
        return self.env.observe_causal_determinism_states(id=[0])[0]

    def restore_causal_snapshot(self, snapshot: CausalSnapshotV1 | CausalSnapshotV2):
        """Restore a fork point and return its exact stored policy observation."""

        if isinstance(snapshot, CausalSnapshotV2):
            snapshot = snapshot.runtime_snapshot
        if not isinstance(snapshot, CausalSnapshotV1):
            raise TypeError("`snapshot` must be CausalSnapshotV1 or CausalSnapshotV2.")
        if self.num_envs != 1:
            raise RuntimeError("CausalSnapshotV1 restore requires num_envs=1.")
        self.env.restore_causal_states([snapshot.worker_state], id=[0])
        state = snapshot.wrapper_state
        for name, target_name in (
            ("task_ids", "task_ids"),
            ("trial_ids", "trial_ids"),
            ("reset_state_ids", "reset_state_ids"),
            ("prev_step_reward", "prev_step_reward"),
            ("success_once", "success_once"),
            ("fail_once", "fail_once"),
            ("returns", "returns"),
            ("success_episode_len", "success_episode_len"),
            ("elapsed_steps", "_elapsed_steps"),
        ):
            setattr(self, target_name, np.asarray(state[name]).copy())
        self._task_success_stats = copy.deepcopy(state["task_success_stats"])
        self._eval_seen_trials = copy.deepcopy(state["eval_seen_trials"])
        self._generator.bit_generator.state = copy.deepcopy(state["generator"])
        self._generator_ordered.bit_generator.state = copy.deepcopy(
            state["generator_ordered"]
        )
        self.start_idx = int(state["start_idx"])
        self.formal_runner_step = int(state["formal_runner_step"])
        self._is_start = bool(state["is_start"])
        self.current_raw_obs = [copy.deepcopy(snapshot.current_raw_observation)]
        restore_process_rng_state(snapshot.driver_rng_state)
        return self._wrap_obs(self.current_raw_obs)

    def restore_causal_simulator_only_for_audit(
        self,
        snapshot: CausalSnapshotV1 | CausalSnapshotV2,
    ):
        """Restore only MuJoCo state for the Stage-C negative control."""

        if isinstance(snapshot, CausalSnapshotV2):
            snapshot = snapshot.runtime_snapshot
        if not isinstance(snapshot, CausalSnapshotV1):
            raise TypeError("`snapshot` must be CausalSnapshotV1 or CausalSnapshotV2.")
        if self.num_envs != 1:
            raise RuntimeError("MuJoCo-only audit restore requires num_envs=1.")
        self.env.restore_causal_simulators_only_for_audit(
            [snapshot.worker_state],
            id=[0],
        )

    def capture_causal_snapshot_v2(
        self,
        *,
        identity: CausalStateIdentityV2,
        sampling: CausalSamplingMetadataV2,
        recent_history=(),
        policy_runtime_state=None,
        source_route: str,
        previous_mode: str | None,
        remaining_budget: float,
        predicate_before: tuple[bool, ...],
        source_trace_summary,
        parent_checkpoint_identity: str,
        statistics_identity: str,
    ) -> CausalSnapshotV2:
        """Capture the exact runtime state plus v2 scientific provenance."""

        runtime_snapshot = self.capture_causal_snapshot(
            snapshot_id=identity.snapshot_id,
            recent_history=recent_history,
            policy_runtime_state=policy_runtime_state,
            source_policy=sampling.source_policy,
            previous_mode=previous_mode,
            chunk_index=identity.chunk_index,
            remaining_budget=remaining_budget,
        )
        return CausalSnapshotV2(
            runtime_snapshot=runtime_snapshot,
            identity=identity,
            sampling=sampling,
            source_route=str(source_route),
            previous_mode=previous_mode,
            remaining_budget=float(remaining_budget),
            predicate_before=tuple(bool(value) for value in predicate_before),
            source_trace_summary=copy.deepcopy(source_trace_summary),
            parent_checkpoint_identity=str(parent_checkpoint_identity),
            statistics_identity=str(statistics_identity),
        )

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    @property
    def action_contract(self) -> LiberoActionContract:
        """Return the exact contract from every currently instantiated worker."""

        payloads = self.env.get_env_attr("_rlinf_action_contract")
        if not payloads or any(payload is None for payload in payloads):
            raise RuntimeError(
                "LIBERO worker did not expose an exact live Action contract."
            )
        return merge_libero_action_contracts(payloads)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()
        egl_instantiation_target = self.cfg.get("egl_instantiation_target", None)
        if egl_instantiation_target is not None:
            egl_instantiation_target = str(egl_instantiation_target)

        for env_fn_param in env_fn_params:

            def env_fn(
                param=env_fn_param,
                _type_val=current_type_val,
                _egl_instantiation_target=egl_instantiation_target,
            ):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                if _egl_instantiation_target is None:
                    env = instantiate_with_isolated_egl(WorkerEnv, param)
                else:
                    from hydra.utils import get_method

                    instantiate_environment = get_method(_egl_instantiation_target)
                    env = instantiate_environment(WorkerEnv, param)
                env._rlinf_action_contract = inspect_libero_action_contract(
                    env
                ).to_artifact()
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            from libero.libero import get_libero_path

            bddl_root = get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path

            if variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = [
                    "_light",
                    "_language",
                    "_table",
                    "_add",
                    "_tb",
                    "_sample",
                    "_level",
                ]
                if plus_suffix == "all":
                    filter_perts = valid_perts
                elif plus_suffix is not None:
                    normalized = (
                        f"_{plus_suffix}"
                        if not plus_suffix.startswith("_")
                        else plus_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    clean_name = file_name.replace(".bddl", "")
                    for marker in valid_perts:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                            and any(
                                pert in os.path.basename(f) for pert in filter_perts
                            )
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": (
                        self._stage_invariant_environment_seed(env_id)
                        if self.stage_invariant_fixed_reset_ids
                        else self.seed
                    ),
                }
            )
            task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = []
            for tid in self.task_id_filter:
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(
                        f"task_id_filter must contain ints, got "
                        f"{type(tid).__name__}: {tid}"
                    )
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= num_tasks:
                    raise ValueError(
                        f"task_id {tid_int} in task_id_filter is out of range "
                        f"[0, {num_tasks - 1}]"
                    )
                validated_tids.append(tid_int)
            validated_tids = sorted(set(validated_tids))

            self._valid_reset_state_ids = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                self._valid_reset_state_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(self._valid_reset_state_ids)
        else:
            self._valid_reset_state_ids = None

        if self.ordered_reset_state_ids is not None:
            if not self.is_eval:
                raise ValueError("ordered_reset_state_ids is evaluation-only.")
            if get_libero_type() != "standard":
                raise ValueError(
                    "ordered_reset_state_ids is supported only for standard LIBERO."
                )
            if self.specific_reset_id is not None or self.task_id_filter is not None:
                raise ValueError(
                    "ordered_reset_state_ids cannot be combined with "
                    "specific_reset_id or task_id_filter."
                )
            validated_reset_ids = []
            for reset_id in self.ordered_reset_state_ids:
                if not isinstance(reset_id, (int, np.integer)):
                    raise ValueError(
                        "ordered_reset_state_ids must contain ints, got "
                        f"{type(reset_id).__name__}: {reset_id}"
                    )
                reset_id = int(reset_id)
                if reset_id < 0 or reset_id >= self.total_num_group_envs:
                    raise ValueError(
                        f"reset_state_id {reset_id} is out of range "
                        f"[0, {self.total_num_group_envs - 1}]."
                    )
                validated_reset_ids.append(reset_id)
            if len(set(validated_reset_ids)) != len(validated_reset_ids):
                raise ValueError("ordered_reset_state_ids must be unique.")
            if len(validated_reset_ids) < self.total_num_processes:
                raise ValueError(
                    "ordered_reset_state_ids must provide at least one episode "
                    "for every evaluation process."
                )
            self.ordered_reset_state_ids = np.asarray(
                validated_reset_ids, dtype=np.int64
            )

    def update_reset_state_ids(self):
        if self.stage_invariant_fixed_reset_ids:
            reset_state_ids = self._get_stage_invariant_reset_state_ids()
        elif self.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def set_formal_runner_step(self, runner_step: int) -> None:
        """Select the deterministic reset identities for one formal runner step."""

        if isinstance(runner_step, bool) or int(runner_step) != runner_step:
            raise TypeError("Formal runner step must be an integer.")
        if int(runner_step) < 0:
            raise ValueError("Formal runner step must be non-negative.")
        self.formal_runner_step = int(runner_step)
        if self.stage_invariant_fixed_reset_ids:
            self.update_reset_state_ids()

    def _get_stage_invariant_reset_state_ids(self) -> np.ndarray:
        """Derive one reset identity per global environment without shard state."""

        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((self.num_group,), dtype=np.int64)
        pool = (
            self._valid_reset_state_ids
            if self._valid_reset_state_ids is not None
            else np.arange(self.total_num_group_envs, dtype=np.int64)
        )
        if len(pool) < 1:
            raise ValueError("Stage-invariant LIBERO reset pool is empty.")
        reset_state_ids = []
        for local_environment_index in range(self.num_group):
            global_environment_index = (
                self.global_environment_offset + local_environment_index
            )
            payload = b"\0".join(
                (
                    b"fastwam-formal-libero-reset-v1",
                    str(int(self.cfg.seed)).encode("ascii"),
                    str(self.formal_runner_step).encode("ascii"),
                    str(global_environment_index).encode("ascii"),
                )
            )
            offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            reset_state_ids.append(int(pool[offset % len(pool)]))
        return np.asarray(reset_state_ids, dtype=np.int64)

    def _stage_invariant_environment_seed(self, local_environment_index: int) -> int:
        """Return the simulator seed attached to one global environment id."""

        if not 0 <= int(local_environment_index) < self.num_envs:
            raise ValueError("Local LIBERO environment index is out of range.")
        return (
            int(self.cfg.seed)
            + self.global_environment_offset
            + int(local_environment_index)
        )

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        elif self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                low=0, high=len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            reset_state_ids = self._valid_reset_state_ids[indices]
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        if self.is_eval:
            if self.ordered_reset_state_ids is not None:
                reset_state_ids = self.ordered_reset_state_ids.copy()
            elif self._valid_reset_state_ids is not None:
                reset_state_ids = self._valid_reset_state_ids.copy()
            else:
                reset_state_ids = build_interleaved_eval_reset_state_ids(
                    self.trial_id_bins, self.cumsum_trial_id_bins
                )
            return distribute_reset_state_ids_round_robin(
                reset_state_ids, self.total_num_processes
            )

        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        self._generator_ordered.shuffle(reset_state_ids)

        # Ensure we have enough IDs for all processes by tiling if needed
        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((num_reset_states,), dtype=int)

        if self.is_eval:
            pool = self._eval_reset_pool
            if self.start_idx >= len(pool):
                return np.full((num_reset_states,), -1, dtype=np.int64)
            end = min(self.start_idx + num_reset_states, len(pool))
            n_valid = end - self.start_idx
            result = np.full((num_reset_states,), -1, dtype=np.int64)
            if n_valid > 0:
                result[:n_valid] = pool[self.start_idx : end]
            self.start_idx = end
            return result

        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        self._task_success_stats: dict[int, dict[str, int]] = {}
        self._eval_seen_trials: set[tuple[int, int]] = set()

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        # Only accumulate returns while not yet succeeded
        self.returns += step_reward * (~self.success_once)
        # Record episode_len at first success
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]

        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not self.is_eval:
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        if self.stage_invariant_fixed_reset_ids:
            self.env.seed(
                [
                    self._stage_invariant_environment_seed(local_environment_index)
                    for local_environment_index in range(self.num_envs)
                ]
            )
        else:
            self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            if self.is_eval:
                self._task_success_stats = {}
                self._eval_seen_trials = set()
                self.start_idx = 0
                pool = self.reset_state_ids_all[self.seed_offset]
                self._eval_reset_pool = pool[pool >= 0].copy()
                self.update_reset_state_ids()
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(self.reset_wait_steps):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    @staticmethod
    def _normalize_active_mask(active_mask, *, batch_size):
        """Validate a batch-aligned mask for frozen-ledger evaluation slots."""

        if active_mask is None:
            return np.ones(int(batch_size), dtype=bool)
        if isinstance(active_mask, torch.Tensor):
            if active_mask.dtype != torch.bool:
                raise TypeError("LIBERO active mask must have boolean dtype.")
            mask = active_mask.detach().cpu().numpy()
        else:
            mask = np.asarray(active_mask)
            if mask.dtype != np.bool_:
                raise TypeError("LIBERO active mask must have boolean dtype.")
        if mask.shape != (int(batch_size),):
            raise ValueError(
                "LIBERO active mask must have shape "
                f"[{int(batch_size)}], got {tuple(mask.shape)}."
            )
        return mask.astype(bool, copy=True)

    @staticmethod
    def _normalize_contract_failure_mask(contract_failure_mask, *, batch_size):
        """Validate slots rejected before Action submission."""

        if contract_failure_mask is None:
            return np.zeros(int(batch_size), dtype=bool)
        if isinstance(contract_failure_mask, torch.Tensor):
            if contract_failure_mask.dtype != torch.bool:
                raise TypeError("LIBERO contract failure mask must have boolean dtype.")
            mask = contract_failure_mask.detach().cpu().numpy()
        else:
            mask = np.asarray(contract_failure_mask)
            if mask.dtype != np.bool_:
                raise TypeError("LIBERO contract failure mask must have boolean dtype.")
        if mask.shape != (int(batch_size),):
            raise ValueError(
                "LIBERO contract failure mask must have shape "
                f"[{int(batch_size)}], got {tuple(mask.shape)}."
            )
        return mask.astype(bool, copy=True)

    @staticmethod
    def _mask_action_statistics(
        statistics: ActionStageStatistics,
        active_mask: np.ndarray,
    ) -> ActionStageStatistics:
        """Zero compact counts for slots whose Actions were never submitted."""

        row_mask = torch.as_tensor(
            active_mask,
            dtype=torch.bool,
            device=statistics.minimum.device,
        ).reshape(-1, 1)
        return replace(
            statistics,
            minimum=statistics.minimum.masked_fill(~row_mask, 0.0),
            maximum=statistics.maximum.masked_fill(~row_mask, 0.0),
            finite_count=statistics.finite_count.masked_fill(~row_mask, 0),
            below_low_count=statistics.below_low_count.masked_fill(~row_mask, 0),
            above_high_count=statistics.above_high_count.masked_fill(~row_mask, 0),
            total_value_count=statistics.total_value_count.masked_fill(~row_mask, 0),
        )

    @staticmethod
    def _validate_submitted_actions(
        statistics: ActionStageStatistics,
        contract: LiberoActionContract,
    ) -> None:
        """Reject invalid active-slot Actions before the underlying env step."""
        validate_action_stage_contract(
            statistics,
            dimension_names=contract.dimension_names,
            low=contract.low,
            high=contract.high,
        )

    def step(self, actions=None, auto_reset=True, active_mask=None):
        """Step only active frozen-ledger slots while preserving batch alignment."""

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim < 2 or int(actions.shape[0]) != self.num_envs:
            raise ValueError(
                "LIBERO Actions must preserve the configured environment batch."
            )
        active_mask = self._normalize_active_mask(
            active_mask,
            batch_size=self.num_envs,
        )
        active_indices = np.flatnonzero(active_mask)

        capture = self._action_submission_capture
        if capture is not None:
            contract, records = capture
            statistics = ActionStageStatistics.from_values(
                stage=SUBMITTED_LIBERO_ACTION_STAGE,
                values=actions[:, None, :],
                low=contract.low,
                high=contract.high,
                gripper_dimension_index=contract.gripper_dimension_index,
                action_contract_sha256=contract.canonical_sha256,
            )
            submitted_statistics = (
                self._mask_action_statistics(statistics, active_mask)
                if not active_mask.all()
                else statistics
            )
            records.append(submitted_statistics)
            self._validate_submitted_actions(submitted_statistics, contract)

        self._elapsed_steps[active_indices] += 1
        if active_mask.all():
            raw_obs, _reward, terminations, info_lists = self.env.step(actions)
            self.current_raw_obs = raw_obs
            infos = list_of_dict_to_dict_of_list(info_lists)
        else:
            if self.current_raw_obs is None:
                raise RuntimeError(
                    "LIBERO inactive slots require a prior reset observation."
                )
            raw_obs = list(self.current_raw_obs)
            terminations = np.zeros(self.num_envs, dtype=bool)
            infos = {}
            if active_indices.size:
                active_obs, _reward, active_terminations, _info_lists = self.env.step(
                    actions[active_indices],
                    active_indices,
                )
                for result_index, env_index in enumerate(active_indices):
                    raw_obs[int(env_index)] = active_obs[result_index]
                terminations[active_indices] = np.asarray(
                    active_terminations,
                    dtype=bool,
                )
            self.current_raw_obs = raw_obs

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) & active_mask
        obs = self._wrap_obs(self.current_raw_obs)

        previous_step_reward = self.prev_step_reward.copy()
        step_reward = np.asarray(self._calc_step_reward(terminations))
        if not active_mask.all():
            self.prev_step_reward[~active_mask] = previous_step_reward[~active_mask]
            step_reward = step_reward.copy()
            step_reward[~active_mask] = 0

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos, _ = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step_with_action_trace(
        self,
        chunk_actions,
        action_contract: LiberoActionContract,
        active_mask=None,
        *,
        contract_failure_mask=None,
    ):
        """Execute a chunk while reducing only actually submitted Actions.

        Contract-failure slots are never passed to the underlying environment.
        They become zero-reward true terminations and are reset independently,
        so value bootstrapping is disabled while the other vector slots run.
        """

        if not isinstance(action_contract, LiberoActionContract):
            raise TypeError("LIBERO chunk tracing requires a typed Action contract.")
        if self._action_submission_capture is not None:
            raise RuntimeError("Nested LIBERO Action submission capture is forbidden.")
        normalized_mask = self._normalize_active_mask(
            active_mask,
            batch_size=self.num_envs,
        )
        normalized_failure_mask = self._normalize_contract_failure_mask(
            contract_failure_mask,
            batch_size=self.num_envs,
        )
        records: list[ActionStageStatistics] = []
        self._action_submission_capture = (action_contract, records)
        try:
            result = self.chunk_step(
                chunk_actions,
                active_mask=normalized_mask,
                contract_failure_mask=normalized_failure_mask,
            )
        finally:
            self._action_submission_capture = None
        expected_steps = int(chunk_actions.shape[1])
        if len(records) != expected_steps:
            raise RuntimeError(
                "Final LIBERO Action capture did not observe every primitive step: "
                f"{len(records)} != {expected_steps}."
            )
        return result, ActionStageStatistics.merge_time(records)

    def chunk_step(
        self,
        chunk_actions,
        active_mask=None,
        *,
        contract_failure_mask=None,
    ):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        if (
            len(chunk_actions.shape) != 3
            or int(chunk_actions.shape[0]) != self.num_envs
        ):
            raise ValueError(
                "LIBERO chunk Actions must have shape [num_envs, horizon, action_dim]."
            )
        chunk_size = int(chunk_actions.shape[1])
        if chunk_size < 1:
            raise ValueError("LIBERO Action chunks must contain at least one step.")
        active_mask = self._normalize_active_mask(
            active_mask,
            batch_size=self.num_envs,
        )
        contract_failure_mask = self._normalize_contract_failure_mask(
            contract_failure_mask,
            batch_size=self.num_envs,
        )
        if bool((contract_failure_mask & ~active_mask).any()):
            raise ValueError("LIBERO contract failures must be active slots.")
        if contract_failure_mask.any():
            if self._action_submission_capture is None:
                raise RuntimeError(
                    "LIBERO contract-failure termination requires Action tracing."
                )
        execution_mask = active_mask & ~contract_failure_mask
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions,
                auto_reset=False,
                active_mask=execution_mask,
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        rejected = torch.as_tensor(
            contract_failure_mask,
            dtype=torch.bool,
            device=raw_chunk_terminations.device,
        )
        if rejected.any():
            raw_chunk_terminations[rejected, -1] = True
        raw_chunk_dones = torch.logical_or(
            raw_chunk_terminations,
            raw_chunk_truncations,
        )
        chunk_rewards = mask_rewards_after_first_done(
            chunk_rewards,
            raw_chunk_dones,
        )

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1], eval_count_mask = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )
        elif rejected.any():
            obs_list[-1], infos_list[-1] = self._handle_contract_failure_reset(
                rejected.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations

            if eval_count_mask is not None:
                eval_count_mask = torch.tensor(
                    eval_count_mask,
                    dtype=torch.bool,
                    device=past_terminations.device,
                )
                chunk_terminations[:, -1] &= eval_count_mask
                chunk_truncations[:, -1] &= eval_count_mask
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        inactive = torch.as_tensor(
            ~active_mask,
            dtype=torch.bool,
            device=chunk_rewards.device,
        )
        if inactive.any():
            chunk_rewards[inactive] = 0
            chunk_terminations[inactive] = False
            chunk_truncations[inactive] = False
            chunk_truncations[inactive, -1] = True
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        if self.is_eval:
            return self._handle_eval_auto_reset(dones, _final_obs, infos)
        obs, infos = self._handle_train_auto_reset(dones, _final_obs, infos)
        return obs, infos, None

    def _handle_contract_failure_reset(self, failures, _final_obs, infos):
        """Reset only pre-submission failure slots in non-auto-reset training."""

        if getattr(self, "is_eval", False):
            raise RuntimeError("Contract-failure episode outcomes are training-only.")
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[failures]
        final_info = copy.deepcopy(infos)
        if self.use_fixed_reset_state_ids:
            if self.stage_invariant_fixed_reset_ids:
                reset_state_ids = self._get_stage_invariant_reset_state_ids()[env_idx]
            elif self.cfg.use_ordered_reset_state_ids:
                reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
            else:
                reset_state_ids = self._get_random_reset_state_ids(len(env_idx))
            self.reset_state_ids[env_idx] = reset_state_ids
            obs, reset_infos = self.reset(
                env_idx=env_idx,
                reset_state_ids=reset_state_ids,
            )
        else:
            obs, reset_infos = self.reset(env_idx=env_idx, reset_state_ids=None)
        reset_infos["final_observation"] = final_obs
        reset_infos["final_info"] = final_info
        reset_infos["_final_info"] = np.asarray(failures, dtype=bool)
        reset_infos["_final_observation"] = failures
        reset_infos["_elapsed_steps"] = failures
        return obs, reset_infos

    def abort_eval_episodes(self, active_mask):
        """Fail and reset selected eval episodes without submitting an Action."""

        if not self.is_eval or not self.auto_reset:
            raise RuntimeError(
                "Action-contract episode abort requires auto-reset evaluation."
            )
        mask = self._normalize_active_mask(active_mask, batch_size=self.num_envs)
        if not mask.any():
            raise ValueError("Action-contract episode abort selected no environment.")
        if self.current_raw_obs is None:
            raise RuntimeError("Cannot abort an evaluation episode before reset.")
        final_obs = self._wrap_obs(self.current_raw_obs)
        infos: dict = {}
        self._record_metrics(
            np.zeros(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            infos,
        )
        obs, reset_infos, count_mask = self._handle_eval_auto_reset(
            mask,
            final_obs,
            infos,
        )
        reset_infos["fastwam_contract_violation"] = np.asarray(mask, dtype=bool)
        return obs, reset_infos, count_mask

    def _handle_eval_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        count_mask = record_completed_episode_task_stats(
            env_idx,
            final_info,
            self.task_ids,
            self.trial_ids,
            self.num_envs,
            self._eval_seen_trials,
            self._task_success_stats,
        )

        new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
        valid_mask = new_reset_state_ids >= 0
        env_to_reset = env_idx[valid_mask]
        if len(env_to_reset) > 0:
            self.reset_state_ids[env_to_reset] = new_reset_state_ids[valid_mask]
            obs, infos = self.reset(
                env_idx=env_to_reset,
                reset_state_ids=self.reset_state_ids[env_to_reset],
            )
        else:
            obs = _final_obs
            infos = {}

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool) & count_mask
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos, count_mask

    def _handle_train_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        if self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
            obs, infos = self.reset(
                env_idx=env_idx,
                reset_state_ids=self.reset_state_ids[env_idx],
            )
        else:
            obs, infos = self.reset(env_idx=env_idx, reset_state_ids=None)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool)
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward
