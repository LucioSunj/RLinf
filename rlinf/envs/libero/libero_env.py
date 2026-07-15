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

import copy
import glob
import hashlib
import importlib
import os
import sys
from typing import Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.libero.episode_manifest import (
    FrozenEpisode,
    load_frozen_episode_manifest,
    validate_manifest_disjoint,
)
from rlinf.envs.libero.gate_phase import (
    load_gate_phase_callback,
)
from rlinf.envs.libero.gate_snapshot import (
    capture_process_rng_state,
    restore_process_rng_state,
)
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

libero_type = get_libero_type()
LIBERO_GATE_SNAPSHOT_SCHEMA = "libero-gate-snapshot-v1"


def _variant_module_names(variant: str) -> tuple[str, str]:
    if variant == "pro":
        return "liberopro", "liberopro.liberopro"
    if variant == "plus":
        package = os.environ.get("LIBERO_PLUS_IMPORT_MODULE", "libero")
        if "." in package:
            return package.split(".", 1)[0], package
        core = "libero.libero" if package == "libero" else f"{package}.{package}"
        return package, core
    return "libero", "libero.libero"


def _load_variant_modules(variant: str):
    package_name, core_name = _variant_module_names(variant)
    package = importlib.import_module(package_name)
    core = importlib.import_module(core_name)
    try:
        benchmark = importlib.import_module(f"{core_name}.benchmark")
    except ImportError:
        benchmark = importlib.import_module(f"{package_name}.benchmark")
    try:
        envs = importlib.import_module(f"{core_name}.envs")
    except ImportError:
        envs = importlib.import_module(f"{package_name}.envs")
    return package, core, benchmark, envs


if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    try:
        (
            real_libero_pkg,
            real_libero_core,
            real_libero_benchmark,
            real_libero_envs,
        ) = _load_variant_modules(libero_type)

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        _, requested_core = _variant_module_names(libero_type)
        print(
            f"[Main Process Routing Error] Failed to import "
            f"'{requested_core}'. Error: {e}"
        )
        raise

from libero.libero.benchmark import Benchmark


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.is_eval = cfg.get("is_eval", False)
        phase_spec = cfg.get("gate_phase_fn", None)
        self._gate_phase_spec = (
            None
            if phase_spec is None or str(phase_spec).lower() in {"", "none", "null"}
            else str(phase_spec)
        )
        # Validate the dotted callable before any rollout workers are launched.
        load_gate_phase_callback(self._gate_phase_spec)

        manifest_path = cfg.get("episode_manifest_path", None)
        import_module = cfg.get("libero_import_module", None)
        runtime_import_module = os.environ.get("LIBERO_PLUS_IMPORT_MODULE", "libero")
        if (
            get_libero_type() == "plus"
            and import_module is not None
            and str(import_module) != runtime_import_module
        ):
            raise ValueError(
                "libero_import_module must match LIBERO_PLUS_IMPORT_MODULE set "
                "before Python imports LiberoEnv"
            )
        self.episode_manifest = None
        if manifest_path:
            if get_libero_type() != "plus":
                raise ValueError(
                    "episode_manifest_path is supported only with LIBERO_TYPE=plus"
                )
            self.episode_manifest = load_frozen_episode_manifest(
                manifest_path, libero_import_module=import_module
            )
            manifest_suites = {
                entry.task_suite_name for entry in self.episode_manifest.episodes
            }
            configured_suite = str(cfg.task_suite_name)
            if manifest_suites != {configured_suite}:
                raise ValueError(
                    "one LiberoEnv instance can execute exactly one task suite; "
                    f"config={configured_suite!r}, manifest={sorted(manifest_suites)!r}. "
                    "Partition the logical Plus-Full manifest with "
                    "scripts/adaptive_gate/plus_suite_manifest.py and merge all "
                    "suite traces before analysis"
                )
            if self.episode_manifest.split == "train":
                if self.is_eval:
                    raise ValueError("a split=train manifest cannot drive eval envs")
                if not self.use_fixed_reset_state_ids or self.auto_reset:
                    raise ValueError(
                        "manifest-driven GRPO requires use_fixed_reset_state_ids=true "
                        "and auto_reset=false so each group keeps one frozen episode"
                    )
                test_manifest_path = cfg.get("test_episode_manifest_path", None)
                if not test_manifest_path:
                    raise ValueError(
                        "split=train episode_manifest_path requires "
                        "test_episode_manifest_path for a fail-closed held-out audit"
                    )
                test_manifest = load_frozen_episode_manifest(
                    test_manifest_path, libero_import_module=import_module
                )
                validate_manifest_disjoint(self.episode_manifest, test_manifest)
                self.test_episode_manifest = test_manifest
            elif self.episode_manifest.split == "validation":
                if not self.is_eval or self.group_size != 1:
                    raise ValueError(
                        "split=validation/test manifests require is_eval=true and "
                        "group_size=1"
                    )
                test_manifest_path = cfg.get("test_episode_manifest_path", None)
                if not test_manifest_path:
                    raise ValueError(
                        "split=validation episode_manifest_path requires the "
                        "Plus-Full test_episode_manifest_path for held-out audit"
                    )
                test_manifest = load_frozen_episode_manifest(
                    test_manifest_path, libero_import_module=import_module
                )
                validate_manifest_disjoint(self.episode_manifest, test_manifest)
                self.test_episode_manifest = test_manifest
            else:
                if not self.is_eval or self.group_size != 1:
                    raise ValueError(
                        "split=test manifests require is_eval=true and group_size=1"
                    )
                self.test_episode_manifest = None
        else:
            self.test_episode_manifest = None

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._manifest_cursor = 0
        self._last_manifest_batch: list[FrozenEpisode] = []
        self._manifest_entries: list[Optional[FrozenEpisode]] = [None] * self.num_envs
        if self.episode_manifest is not None:
            self._manifest_episode_ordinals = {
                entry.episode_id: index
                for index, entry in enumerate(self.episode_manifest.episodes)
            }
            self._manifest_pool = self.episode_manifest.shard(
                self.seed_offset, self.total_num_processes
            )
            self._manifest_order = np.arange(len(self._manifest_pool), dtype=np.int64)
            if self.episode_manifest.split == "train":
                self._generator.shuffle(self._manifest_order)
            if len(self._manifest_pool) < self.num_group:
                raise ValueError(
                    "frozen episode manifest shard has fewer episodes than this "
                    f"worker's {self.num_group} environment groups"
                )
        else:
            self._manifest_episode_ordinals = {}
            self._manifest_pool = ()
            self._manifest_order = np.asarray([], dtype=np.int64)

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.episode_manifest is not None:
            self._eval_reset_pool = np.asarray(
                [entry.reset_state_id for entry in self._manifest_pool],
                dtype=np.int64,
            )
        elif self.is_eval:
            pool = self.reset_state_ids_all[self.seed_offset]
            self._eval_reset_pool = pool[pool >= 0].copy()
        else:
            self._eval_reset_pool = np.array([], dtype=np.int64)
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_generation = np.zeros(self.num_envs, dtype=np.int64)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
                    try:
                        real_pkg, real_core, real_bench, real_envs = (
                            _load_variant_modules(_type_val)
                        )

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

                env = WorkerEnv(**param)
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
            plus_core = importlib.import_module("libero.libero")
            bddl_root = plus_core.get_libero_path("bddl_files")
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
            if self.episode_manifest is not None:
                entry = self._manifest_entries[env_id]
                if entry is None:
                    raise RuntimeError(
                        f"environment {env_id} has no assigned manifest episode"
                    )
                if entry.task_id != int(self.task_ids[env_id]):
                    raise RuntimeError(
                        f"manifest task {entry.task_id} does not match slot task "
                        f"{int(self.task_ids[env_id])}"
                    )
                env_fn_params.append(
                    {
                        **base_env_args,
                        "bddl_file_name": entry.bddl_path,
                        "seed": entry.env_seed,
                    }
                )
                task_descriptions.append(task.language)
                continue
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
                if plus_suffix == "all":
                    clean_name = file_name.replace(".bddl", "")
                    for marker in [
                        "_view",
                        "_initstate",
                        "_noise",
                        "_sample",
                        "_light",
                        "_table",
                        "_add_1",
                        "_lan",
                        "_language",
                        "_copy",
                        "_level",
                        "_tb",
                    ]:
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
                    "seed": self.seed,
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

    def _take_manifest_entries(self, count: int) -> list[FrozenEpisode]:
        if self.episode_manifest is None:
            raise RuntimeError("no frozen episode manifest is configured")
        count = int(count)
        entries = []
        selected_episode_ids: set[str] = set()
        while len(entries) < count:
            if self._manifest_cursor >= len(self._manifest_order):
                if self.episode_manifest.split != "train":
                    break
                self._manifest_cursor = 0
                self._generator.shuffle(self._manifest_order)
            index = int(self._manifest_order[self._manifest_cursor])
            self._manifest_cursor += 1
            entry = self._manifest_pool[index]
            # A batch can cross a shuffled epoch boundary.  Skip entries already
            # selected for this assignment so distinct GRPO groups never share
            # one frozen episode (slots inside a group still do by construction).
            if entry.episode_id in selected_episode_ids:
                continue
            selected_episode_ids.add(entry.episode_id)
            entries.append(entry)
        self._last_manifest_batch = entries
        return entries

    def _manifest_global_reset_id(self, entry: FrozenEpisode) -> int:
        """Map the manifest's task-local reset identity to the legacy flat id."""
        task_id = int(entry.task_id)
        reset_state_id = int(entry.reset_state_id)
        if not 0 <= task_id < len(self.trial_id_bins):
            raise ValueError(
                f"manifest episode {entry.episode_id!r} has task_id {task_id} "
                f"outside [0,{len(self.trial_id_bins)})"
            )
        if not 0 <= reset_state_id < int(self.trial_id_bins[task_id]):
            raise ValueError(
                f"manifest episode {entry.episode_id!r} has task-local "
                f"reset_state_id {reset_state_id} outside "
                f"[0,{int(self.trial_id_bins[task_id])})"
            )
        start = 0 if task_id == 0 else int(self.cumsum_trial_id_bins[task_id - 1])
        return start + reset_state_id

    def _assign_manifest_entries(self, env_idx, entries) -> np.ndarray:
        if len(entries) * self.group_size == len(env_idx):
            entries = [
                entry for entry in entries for _ in range(self.group_size)
            ]
        elif len(entries) != len(env_idx):
            raise ValueError(f"manifest entries do not align with {len(env_idx)} slots")
        global_reset_ids = []
        for env_id, entry in zip(env_idx, entries):
            if entry.task_suite_name != str(self.cfg.task_suite_name):
                raise ValueError(
                    f"manifest episode {entry.episode_id!r} belongs to suite "
                    f"{entry.task_suite_name!r}, not {self.cfg.task_suite_name!r}"
                )
            global_reset_id = self._manifest_global_reset_id(entry)
            task_ids, local_reset_ids = self._get_task_and_trial_ids_from_reset_state_ids(
                np.asarray([global_reset_id], dtype=np.int64)
            )
            if len(task_ids) != 1:
                raise ValueError(
                    f"manifest episode {entry.episode_id!r} has invalid reset_state_id "
                    f"{entry.reset_state_id}"
                )
            if (
                int(task_ids[0]) != entry.task_id
                or int(local_reset_ids[0]) != entry.reset_state_id
            ):
                raise ValueError(
                    f"manifest episode {entry.episode_id!r} task/reset identity "
                    "cannot be represented by the configured benchmark"
                )
            self._manifest_entries[int(env_id)] = entry
            global_reset_ids.append(global_reset_id)
        return np.asarray(global_reset_ids, dtype=np.int64)

    def update_reset_state_ids(self):
        if self.episode_manifest is not None:
            entries = self._take_manifest_entries(self.num_group)
            if len(entries) != self.num_group:
                raise ValueError("frozen episode manifest was exhausted during reset")
            reset_state_ids = self._assign_manifest_entries(
                np.arange(self.num_envs), entries
            )
        elif self.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = (
            reset_state_ids
            if self.episode_manifest is not None
            else reset_state_ids.repeat(self.group_size)
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
        if self.episode_manifest is not None:
            return np.asarray(
                [
                    [
                        self._manifest_global_reset_id(entry)
                        for entry in self._manifest_pool
                    ]
                ],
                dtype=np.int64,
            )
        if self.is_eval:
            if self._valid_reset_state_ids is not None:
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
        if self.episode_manifest is not None:
            entries = self._take_manifest_entries(num_reset_states)
            result = np.full((num_reset_states,), -1, dtype=np.int64)
            if entries:
                result[: len(entries)] = [entry.reset_state_id for entry in entries]
            return result
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

    def _episode_uid(self, env_id: int) -> str:
        entry = self._manifest_entries[env_id]
        if entry is not None:
            return entry.episode_id
        perturbation = os.environ.get(
            "LIBERO_SUFFIX", os.environ.get("LIBERO_PERTURBATION", "standard")
        )
        identity = "|".join(
            [
                str(self.cfg.task_suite_name),
                str(int(self.seed)),
                str(int(self.task_ids[env_id])),
                str(int(self.trial_ids[env_id])),
                str(int(self.reset_state_ids[env_id])),
                str(int(self._episode_generation[env_id])),
                str(perturbation),
            ]
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def _gate_context(self) -> dict[str, object]:
        configured_horizon = self.cfg.get("gate_exec_horizon", None)
        exec_horizon = 1 if configured_horizon is None else int(configured_horizon)
        if exec_horizon <= 0:
            raise ValueError("gate_exec_horizon must be positive")
        factors, levels, perturbation_ids, env_seeds = [], [], [], []
        manifest_trial_ids, manifest_reset_ids = [], []
        base_tasks, task_suite_names, asset_ids = [], [], []
        manifest_ids = []
        phase_contexts = []
        for env_id, entry in enumerate(self._manifest_entries):
            if entry is None:
                factors.append("unknown")
                levels.append("unknown")
                perturbation_ids.append("unknown")
                manifest_ids.append("")
                env_seeds.append(int(self.seed))
                manifest_trial_ids.append(int(self.trial_ids[env_id]))
                manifest_reset_ids.append(int(self.reset_state_ids[env_id]))
                base_tasks.append("unknown")
                task_suite_names.append(str(self.cfg.task_suite_name))
                asset_ids.append([])
            else:
                factors.append(entry.factor)
                levels.append(entry.level)
                perturbation_ids.append(entry.perturbation_id)
                manifest_ids.append(self.episode_manifest.sha256)
                env_seeds.append(entry.env_seed)
                manifest_trial_ids.append(entry.trial_id)
                manifest_reset_ids.append(entry.reset_state_id)
                base_tasks.append(entry.base_task)
                task_suite_names.append(entry.task_suite_name)
                asset_ids.append(list(entry.asset_ids))
            phase_contexts.append(
                {
                    "env_id": env_id,
                    "task_id": int(self.task_ids[env_id]),
                    "task_description": self.task_descriptions[env_id],
                    "elapsed_steps": int(self._elapsed_steps[env_id]),
                    "manifest_entry": (
                        None if entry is None else entry.to_dict()
                    ),
                }
            )
        if self._gate_phase_spec is None:
            phase_results = [("unknown", False)] * self.num_envs
        else:
            phase_results = self.env.evaluate_gate_phases(
                self._gate_phase_spec,
                phase_contexts,
                id=np.arange(self.num_envs),
            )
        phases = [str(result[0]) for result in phase_results]
        phase_reliable = [bool(result[1]) for result in phase_results]
        return {
            "episode_uid": [self._episode_uid(i) for i in range(self.num_envs)],
            "decision_index": torch.as_tensor(
                self._elapsed_steps // exec_horizon, dtype=torch.int64
            ),
            "elapsed_steps": torch.as_tensor(self._elapsed_steps, dtype=torch.int64),
            "task_id": torch.as_tensor(self.task_ids, dtype=torch.int64),
            "task_description": list(self.task_descriptions),
            "env_seed": torch.as_tensor(env_seeds, dtype=torch.int64),
            "trial_id": torch.as_tensor(manifest_trial_ids, dtype=torch.int64),
            "reset_state_id": torch.as_tensor(manifest_reset_ids, dtype=torch.int64),
            "factor": factors,
            "base_task": base_tasks,
            "task_suite_name": task_suite_names,
            "level": levels,
            "perturbation_id": perturbation_ids,
            "asset_ids": asset_ids,
            "episode_manifest_sha256": manifest_ids,
            "heldout_test_manifest_sha256": [
                (
                    self.test_episode_manifest.sha256
                    if self.test_episode_manifest is not None
                    else (self.episode_manifest.sha256 if self.episode_manifest else "")
                )
            ]
            * self.num_envs,
            # This callback runs while wrapping the current observation, before
            # the Gate chooses or executes its next action chunk.
            "phase": phases,
            "phase_reliable": torch.as_tensor(
                phase_reliable, dtype=torch.bool
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
            "gate_context": self._gate_context(),
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
            if self.episode_manifest is not None or task_changed or not self.is_eval:
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        if self.episode_manifest is None:
            self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        # A frozen Plus episode binds both its perturbed BDDL and a task-local
        # reset state.  OffScreenRenderEnv.reset() only applies the former; the
        # benchmark state must still be installed explicitly, just as in the
        # standalone frozen-manifest evaluator.  Keep the legacy dynamic Plus
        # path unchanged because those perturbations have no frozen reset
        # contract to verify.
        if variant != "plus" or self.episode_manifest is not None:
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
        manifest_entries: Optional[list[FrozenEpisode]] = None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.asarray(env_idx, dtype=np.int64).reshape(-1)

        if self.is_start:
            if self.is_eval:
                self._task_success_stats = {}
                self._eval_seen_trials = set()
                if self.episode_manifest is None:
                    self.start_idx = 0
                    pool = self.reset_state_ids_all[self.seed_offset]
                    self._eval_reset_pool = pool[pool >= 0].copy()
                    self.update_reset_state_ids()
            if self.episode_manifest is not None:
                reset_state_ids = self.reset_state_ids[env_idx]
            else:
                reset_state_ids = (
                    self.reset_state_ids if self.use_fixed_reset_state_ids else None
                )
            self._is_start = False

        if self.episode_manifest is not None:
            if manifest_entries is None and reset_state_ids is None:
                reset_state_ids = self.reset_state_ids[env_idx]
            if manifest_entries is not None:
                reset_state_ids = self._assign_manifest_entries(
                    env_idx, manifest_entries
                )
            if reset_state_ids is None:
                raise RuntimeError(
                    "manifest-backed LIBERO reset has no frozen episode assignment"
                )

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
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

        self._reset_metrics(env_idx)
        self._episode_generation[env_idx] += 1
        obs = self._wrap_obs(self.current_raw_obs)
        infos = {}
        return obs, infos

    def _normalize_snapshot_env_idx(self, env_idx) -> np.ndarray:
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        env_idx = np.asarray(env_idx, dtype=np.int64).reshape(-1)
        if len(env_idx) == 0 or len(set(env_idx.tolist())) != len(env_idx):
            raise ValueError("snapshot env_idx must be non-empty and unique")
        if bool(((env_idx < 0) | (env_idx >= self.num_envs)).any()):
            raise ValueError(f"snapshot env_idx is outside [0, {self.num_envs})")
        return env_idx

    def capture_gate_snapshot(self, env_idx=None) -> dict[str, object]:
        """Capture simulator, controller, RNG and outer-wrapper state.

        Call only between action chunks.  Restoring a subset also restores the
        shared outer RNG stream, so paired collectors should normally use one
        environment per ``LiberoEnv`` or snapshot all vector slots together.
        """
        env_idx = self._normalize_snapshot_env_idx(env_idx)
        main_process_rng = capture_process_rng_state()
        worker_snapshots = self.env.capture_gate_snapshots(id=env_idx)
        array_names = (
            "reset_state_ids",
            "task_ids",
            "trial_ids",
            "prev_step_reward",
            "success_once",
            "fail_once",
            "returns",
            "success_episode_len",
            "_elapsed_steps",
            "_episode_generation",
        )
        per_env = {
            name: copy.deepcopy(np.asarray(getattr(self, name))[env_idx])
            for name in array_names
        }
        return {
            "schema": LIBERO_GATE_SNAPSHOT_SCHEMA,
            "env_idx": env_idx.copy(),
            "worker": worker_snapshots,
            "outer": {
                "per_env": per_env,
                "task_descriptions": [
                    copy.deepcopy(self.task_descriptions[int(i)]) for i in env_idx
                ],
                "manifest_entries": [
                    copy.deepcopy(self._manifest_entries[int(i)]) for i in env_idx
                ],
                "generator_state": copy.deepcopy(self._generator.bit_generator.state),
                "ordered_generator_state": copy.deepcopy(
                    self._generator_ordered.bit_generator.state
                ),
                "manifest_cursor": int(self._manifest_cursor),
                "manifest_order": self._manifest_order.copy(),
                "last_manifest_batch": copy.deepcopy(self._last_manifest_batch),
                "start_idx": int(self.start_idx),
                "is_start": bool(self._is_start),
                "task_success_stats": copy.deepcopy(self._task_success_stats),
                "eval_seen_trials": copy.deepcopy(self._eval_seen_trials),
                "main_process_rng": main_process_rng,
            },
        }

    def restore_gate_snapshot(self, snapshot: dict[str, object]):
        """Restore a counterfactual branch and return the verified wrapped obs."""
        if snapshot.get("schema") != LIBERO_GATE_SNAPSHOT_SCHEMA:
            raise ValueError(
                f"unsupported LIBERO gate snapshot schema {snapshot.get('schema')!r}"
            )
        env_idx = self._normalize_snapshot_env_idx(snapshot.get("env_idx"))
        worker_snapshots = snapshot.get("worker")
        outer = snapshot.get("outer")
        if not isinstance(worker_snapshots, list) or not isinstance(outer, dict):
            raise ValueError("malformed LIBERO gate snapshot")
        if len(worker_snapshots) != len(env_idx):
            raise ValueError("LIBERO gate snapshot worker/env count mismatch")

        raw_obs = self.env.restore_gate_snapshots(worker_snapshots, id=env_idx)
        per_env = outer.get("per_env")
        if not isinstance(per_env, dict):
            raise ValueError("LIBERO gate snapshot is missing outer.per_env")
        for name, values in per_env.items():
            target = getattr(self, name, None)
            if not isinstance(target, np.ndarray) or len(values) != len(env_idx):
                raise ValueError(f"invalid outer snapshot array {name!r}")
            target[env_idx] = copy.deepcopy(values)
        descriptions = outer.get("task_descriptions")
        manifest_entries = outer.get("manifest_entries")
        if not isinstance(descriptions, list) or not isinstance(
            manifest_entries, list
        ):
            raise ValueError("LIBERO gate snapshot is missing slot identities")
        for offset, slot in enumerate(env_idx):
            self.task_descriptions[int(slot)] = copy.deepcopy(descriptions[offset])
            self._manifest_entries[int(slot)] = copy.deepcopy(manifest_entries[offset])
            if self.current_raw_obs is None:
                self.current_raw_obs = [None] * self.num_envs
            self.current_raw_obs[int(slot)] = raw_obs[offset]

        self._generator.bit_generator.state = copy.deepcopy(
            outer["generator_state"]
        )
        self._generator_ordered.bit_generator.state = copy.deepcopy(
            outer["ordered_generator_state"]
        )
        self._manifest_cursor = int(outer["manifest_cursor"])
        self._manifest_order = np.asarray(outer["manifest_order"], dtype=np.int64)
        self._last_manifest_batch = copy.deepcopy(outer["last_manifest_batch"])
        self.start_idx = int(outer["start_idx"])
        self._is_start = bool(outer["is_start"])
        self._task_success_stats = copy.deepcopy(outer["task_success_stats"])
        self._eval_seen_trials = copy.deepcopy(outer["eval_seen_trials"])
        observation = self._wrap_obs(self.current_raw_obs)
        restore_process_rng_state(outer["main_process_rng"])
        return observation

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

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

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1], eval_count_mask = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
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

    def _handle_eval_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        stats_trial_ids = self.trial_ids
        if self.episode_manifest is not None:
            stats_trial_ids = np.asarray(
                [
                    self._manifest_episode_ordinals[
                        self._manifest_entries[env_id].episode_id
                    ]
                    for env_id in range(self.num_envs)
                ],
                dtype=np.int64,
            )
        count_mask = record_completed_episode_task_stats(
            env_idx,
            final_info,
            self.task_ids,
            stats_trial_ids,
            self.num_envs,
            self._eval_seen_trials,
            self._task_success_stats,
        )

        new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
        manifest_entries = (
            list(self._last_manifest_batch)
            if self.episode_manifest is not None
            else None
        )
        valid_mask = new_reset_state_ids >= 0
        env_to_reset = env_idx[valid_mask]
        if len(env_to_reset) > 0:
            self.reset_state_ids[env_to_reset] = new_reset_state_ids[valid_mask]
            valid_manifest_entries = None
            if manifest_entries is not None:
                valid_manifest_entries = [
                    entry
                    for entry, valid in zip(manifest_entries, valid_mask)
                    if valid
                ]
            obs, infos = self.reset(
                env_idx=env_to_reset,
                reset_state_ids=self.reset_state_ids[env_to_reset],
                manifest_entries=valid_manifest_entries,
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
