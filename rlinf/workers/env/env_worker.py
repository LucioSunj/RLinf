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

import asyncio
import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.algorithms.rlt.transition import update_rlt_transitions
from rlinf.data.embodied_io_struct import (
    ACTOR_TRAJECTORY_CHANNEL_TAG,
    ChunkStepResult,
    EmbodiedLerobotRolloutResult,
    EmbodiedRolloutResult,
    EnvOutput,
    EvaluationRolloutControl,
    RolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_contract import (
    PREPARED_LIBERO_ACTION_STAGE,
    ActionExecutionTrace,
    ActionStageStatistics,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers import RecordVideo
from rlinf.models.embodiment.wam_policy.contracts import ChunkRouteRecord, WAMRoute
from rlinf.scheduler import (
    Channel,
    Cluster,
    CommMapper,
    Worker,
    merge_batches,
)
from rlinf.scheduler.cluster.p8_formal import (
    emit_p8_formal_worker_placement_audit,
)
from rlinf.utils.checkpoint_state import checkpoint_state_sha256
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict_to_chunk,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)
from rlinf.workers.env.history_manager import HistoryManager

FASTWAM_TRAINING_ACTION_AUDIT_SENTINEL = "FASTWAM_TRAINING_ACTION_AUDIT"
FASTWAM_TRAINING_ACTION_AUDIT_SCHEMA = "fastwam-training-action-audit-v1"
FASTWAM_P8_FORMAL_ACTION_AUDIT_SCHEMA = "fastwam-p8-formal-action-audit-v1"
FASTWAM_TRAINING_ACTION_FAILURE_SENTINEL = "FASTWAM_TRAINING_ACTION_FAILURE"
FASTWAM_TRAINING_ACTION_FAILURE_SCHEMA = "fastwam-training-action-failure-v1"


def _fastwam_training_action_audit_enabled(
    cfg: Any,
    *,
    eval_mode: bool = False,
) -> bool:
    """Resolve the generic or P8-formal typed Action audit fail-closed."""

    runner = cfg.get("runner", {})
    formal_endpoint = bool(runner.get("p8_formal_stage2_endpoint", False))
    formal_audit = bool(runner.get("p8_formal_action_audit", False))
    if formal_endpoint != formal_audit:
        raise ValueError(
            "P8 formal Stage2 and its typed Action audit must be enabled together."
        )
    generic_audit = bool(runner.get("fastwam_training_guard", {}).get("enabled", False))
    return bool(not eval_mode and (generic_audit or formal_audit))


def _fastwam_training_action_audit_identity(
    formal_runner_step: int | None,
) -> dict[str, Any]:
    """Build the schema/step identity without changing the generic payload."""

    if formal_runner_step is None:
        return {"schema": FASTWAM_TRAINING_ACTION_AUDIT_SCHEMA}
    if isinstance(formal_runner_step, bool) or not 1 <= int(formal_runner_step) <= 100:
        raise ValueError("P8 formal Action runner_step must be in 1..100.")
    return {
        "schema": FASTWAM_P8_FORMAL_ACTION_AUDIT_SCHEMA,
        "status": "PASS",
        "runner_step": int(formal_runner_step),
    }


def _batch_metadata_value(value: Any, index: int) -> int | None:
    """Return one compact integer metadata value from a batch container."""

    if value is None:
        return None
    tensor = torch.as_tensor(value).reshape(-1)
    if not 0 <= index < int(tensor.numel()):
        return None
    return int(tensor[index].item())


def _route_metadata_for_batch_index(
    route: ChunkRouteRecord | None,
    index: int,
) -> dict[str, Any] | None:
    """Return one route identity without retaining a route timeline."""

    if route is None:
        return None
    flattened = {
        name: getattr(route, name).reshape(-1)
        for name in (
            "route_used",
            "route_was_forced",
            "chunk_ids",
            "episode_ids",
            "route_source_chunk_ids",
            "actor_versions",
        )
    }
    if any(index >= int(value.numel()) for value in flattened.values()):
        return None
    route_value = int(flattened["route_used"][index].item())
    return {
        "route": WAMRoute(route_value).name,
        "route_was_forced": bool(flattened["route_was_forced"][index].item()),
        "chunk_id": int(flattened["chunk_ids"][index].item()),
        "episode_id": int(flattened["episode_ids"][index].item()),
        "route_source_chunk_id": int(flattened["route_source_chunk_ids"][index].item()),
        "actor_version": int(flattened["actor_versions"][index].item()),
    }


def build_fastwam_episode_identity_sha256(
    *,
    route: ChunkRouteRecord,
    task_ids: Any,
    trial_ids: Any,
    reset_state_ids: Any,
) -> str:
    """Hash one pre-submission environment identity without retaining raw state."""

    batch_size = int(route.route_used.numel())
    identity = []
    for index in range(batch_size):
        record = {
            "environment_index": index,
            "task_id": _batch_metadata_value(task_ids, index),
            "trial_id": _batch_metadata_value(trial_ids, index),
            "reset_state_id": _batch_metadata_value(reset_state_ids, index),
            "episode_id": int(route.episode_ids.reshape(-1)[index].item()),
        }
        if any(
            record[name] is None for name in ("task_id", "trial_id", "reset_state_id")
        ):
            raise ValueError(
                "FastWAM training environment identity metadata is incomplete."
            )
        identity.append(record)
    return checkpoint_state_sha256(identity)


def build_fastwam_action_failure_audit(
    *,
    trace: ActionExecutionTrace,
    contract: LiberoActionContract,
    route: ChunkRouteRecord | None,
    task_ids: Any,
    trial_ids: Any,
    reset_state_ids: Any,
    denoise_indices: Any,
    worker_rank: int,
    pipeline_stage_id: int,
) -> dict[str, Any]:
    """Build compact first-violation provenance for a rejected Action chunk."""

    violations = []
    for batch_index in range(trace.batch_size):
        for dimension_index in range(trace.stages[0].action_dim):
            first_stage = None
            stage_record = None
            for statistics in trace.stages:
                invalid = (
                    int(statistics.finite_count[batch_index, dimension_index])
                    != int(statistics.total_value_count[batch_index, dimension_index])
                    or int(statistics.below_low_count[batch_index, dimension_index]) > 0
                    or int(statistics.above_high_count[batch_index, dimension_index])
                    > 0
                )
                if invalid:
                    first_stage = statistics.stage
                    stage_record = statistics.record_for_batch_index(batch_index)[
                        "dimensions"
                    ][dimension_index]
                    break
            if first_stage is None:
                continue
            route_metadata = _route_metadata_for_batch_index(route, batch_index)
            denoise_index = _batch_metadata_value(denoise_indices, batch_index)
            if (
                route_metadata is not None
                and route_metadata["route"] == WAMRoute.UNCOND.name
                and denoise_index is None
            ):
                raise ValueError(
                    "Rejected UNCOND Action is missing its Flow-SDE denoise index."
                )
            violations.append(
                {
                    "environment_index": batch_index,
                    "dimension_index": dimension_index,
                    "dimension_name": contract.dimension_names[dimension_index],
                    "first_out_of_live_bounds_stage": first_stage,
                    "statistics": stage_record,
                    "low": float(contract.low[dimension_index]),
                    "high": float(contract.high[dimension_index]),
                    "task_id": _batch_metadata_value(task_ids, batch_index),
                    "trial_id": _batch_metadata_value(trial_ids, batch_index),
                    "reset_state_id": _batch_metadata_value(
                        reset_state_ids, batch_index
                    ),
                    "route_metadata": route_metadata,
                    "flow_sde_denoise_index": denoise_index,
                }
            )
    if not violations:
        raise ValueError(
            "Rejected submitted Action has no matching pre-submission violation."
        )
    return {
        "schema": FASTWAM_TRAINING_ACTION_FAILURE_SCHEMA,
        "worker_rank": int(worker_rank),
        "pipeline_stage_id": int(pipeline_stage_id),
        "stage_order": list(trace.stage_names),
        "action_contract_sha256": contract.canonical_sha256,
        "violations": violations,
        "no_silent_clamp": True,
    }


def summarize_fastwam_flow_sde_denoise_indices(
    values: list[torch.Tensor],
    *,
    num_inference_steps: int,
    ignore_last_transition: bool,
) -> dict[str, Any]:
    """Summarize selected stochastic transitions without retaining a timeline."""

    if num_inference_steps < 1:
        raise ValueError("FastWAM inference-step count must be positive.")
    if not values:
        raise ValueError("FastWAM training collected no Flow-SDE denoise indices.")
    indices = torch.cat(
        [torch.as_tensor(value, dtype=torch.long).reshape(-1) for value in values]
    )
    selected = indices[indices >= 0]
    final_index = num_inference_steps - 1
    final_count = int((selected == final_index).sum().item())
    if ignore_last_transition and final_count:
        raise ValueError(
            "FastWAM ignore-last contract observed a selected final transition."
        )
    return {
        "ignore_last_transition": bool(ignore_last_transition),
        "num_inference_steps": int(num_inference_steps),
        "selected_count": int(selected.numel()),
        "minimum": int(selected.min().item()) if selected.numel() else None,
        "maximum": int(selected.max().item()) if selected.numel() else None,
        "final_transition_count": final_count,
    }


def _merge_evaluation_rollout_results(items: list[Any]) -> Any:
    """Merge typed FastWAM eval shards while preserving legacy payloads."""

    if not items:
        raise ValueError("At least one evaluation rollout shard is required.")
    typed = [isinstance(item, RolloutResult) for item in items]
    if any(typed) and not all(typed):
        raise TypeError(
            "Cannot merge mixed typed and legacy evaluation rollout payloads."
        )
    if all(typed):
        return RolloutResult.merge_rollout_results(items)
    return merge_batches(items)


def _split_evaluation_rollout_control(
    control: EvaluationRolloutControl,
    split_sizes: list[int],
) -> list[EvaluationRolloutControl]:
    """Split a typed stop control according to the normal route plan."""

    if not isinstance(control, EvaluationRolloutControl):
        raise TypeError("Evaluation stop split requires typed control.")
    return control.split(split_sizes)


def _mark_terminal_gate_unused(
    rollout_result: RolloutResult,
    dones: torch.Tensor | np.ndarray,
) -> RolloutResult:
    """Invalidate emitted decisions that cannot control a terminal next chunk."""

    emitted = rollout_result.emitted_gate
    if emitted is None:
        return rollout_result
    batch_size = int(emitted.next_route.shape[0])
    done_tensor = torch.as_tensor(dones, dtype=torch.bool)
    if done_tensor.ndim < 1 or int(done_tensor.shape[0]) != batch_size:
        raise ValueError(
            "Terminal mask batch must match emitted Gate decisions; got "
            f"{tuple(done_tensor.shape)} and {tuple(emitted.next_route.shape)}."
        )
    terminal = done_tensor.reshape(batch_size, -1).any(dim=1)
    if rollout_result.route_info is not None and (
        rollout_result.route_info.route_used.shape != emitted.next_route.shape
    ):
        raise ValueError("Executed routes and emitted Gate decisions are misaligned.")
    if rollout_result.actions is not None and (
        int(rollout_result.actions.shape[0]) != batch_size
    ):
        raise ValueError("Actions and emitted Gate decisions are misaligned.")
    selection = rollout_result.evaluation_selection
    if selection is not None and (
        selection.effective_next_route.shape != emitted.next_route.shape
    ):
        raise ValueError(
            "Evaluation selections and emitted Gate decisions are misaligned."
        )
    terminal = terminal.to(device=emitted.valid.device)
    aligned_gate = replace(
        emitted,
        valid=emitted.valid & ~terminal,
    )
    return replace(rollout_result, emitted_gate=aligned_gate)


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        emit_p8_formal_worker_placement_audit(cfg, self, role="env")

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False
        self._p8_formal_last_action_runner_step = 0
        self._p8_formal_pending_action_runner_step: int | None = None

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self.evaluation_collector = None
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num
        self.enable_rlt = (
            OmegaConf.select(self.cfg, "algorithm.loss_type", default="") == "rlt_ac"
        )

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", False
        )
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.model_cfg = (
            self.cfg.rollout.model if self.only_eval else self.cfg.actor.model
        )
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            self.cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1

        self.train_enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            eval_env_cfg.get("enable_offload", False)
            if eval_env_cfg is not None
            else False
        )
        if self.enable_train:
            self.enable_online_lerobot = bool(
                OmegaConf.select(
                    self.cfg,
                    "algorithm.dagger.online_lerobot.enabled",
                    default=False,
                )
            )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
            self.train_batch_size = self.cfg.env.train.total_num_envs // self.stage_num
        else:
            self.enable_online_lerobot = False
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
            self.eval_batch_size = self.cfg.env.eval.total_num_envs // self.stage_num
        self.n_train_chunk_steps = 0
        if self.enable_train:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.actor_split_num = (
            1 if not self.enable_train else self.get_actor_split_num()
        )
        if self.use_training_pipeline and self.enable_train:
            self._init_pipeline_params()

        if self.enable_train:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # Init the batch_router for env decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "the world size of env must be greater than the world size of rollout in env_decoupled_mode"
            )

    def _prepare_rollout_results(self, rollout_results: list | None = None) -> list:
        if self.enable_online_lerobot and rollout_results is not None:
            for stage_rollout in rollout_results:
                stage_rollout.rewards.clear()
            return rollout_results

        collect_only_success = bool(
            OmegaConf.select(
                self.cfg,
                "algorithm.dagger.online_lerobot.only_success",
                default=False,
            )
        )
        max_episode_length = self.cfg.env.train.max_episode_steps
        if self.enable_online_lerobot:
            return [
                EmbodiedLerobotRolloutResult(
                    max_episode_length=max_episode_length,
                    num_envs=self.train_num_envs_per_stage,
                    only_success=collect_only_success,
                    num_action_chunks=self.model_cfg.num_action_chunks,
                    action_dim=self.model_cfg.action_dim,
                )
                for _ in range(self.stage_num)
            ]
        return [
            EmbodiedRolloutResult(max_episode_length=max_episode_length)
            for _ in range(self.stage_num)
        ]

    def set_p8_formal_action_runner_step(self, runner_step: int) -> None:
        """Bind the next formal Action audit to one explicit 1-based step."""

        runner = self.cfg.get("runner", {})
        if not bool(runner.get("p8_formal_stage2_endpoint", False)) or not bool(
            runner.get("p8_formal_action_audit", False)
        ):
            raise RuntimeError(
                "P8 formal Action runner steps may be set only by the formal endpoint."
            )
        if isinstance(runner_step, bool):
            raise TypeError("P8 formal Action runner_step must be an integer.")
        runner_step = int(runner_step)
        max_steps = int(runner.get("max_steps", -1))
        expected = self._p8_formal_last_action_runner_step + 1
        if (
            max_steps != 100
            or not 1 <= runner_step <= max_steps
            or runner_step != expected
            or self._p8_formal_pending_action_runner_step is not None
        ):
            raise RuntimeError(
                "P8 formal Action runner_step must be a unique contiguous value "
                f"in 1..100: got {runner_step}, expected {expected}."
            )
        self._p8_formal_pending_action_runner_step = runner_step

    def _consume_p8_formal_action_runner_step(self) -> int | None:
        """Consume the runner-supplied step exactly once for this rollout."""

        runner = self.cfg.get("runner", {})
        formal_endpoint = bool(runner.get("p8_formal_stage2_endpoint", False))
        if not formal_endpoint:
            return None
        pending = self._p8_formal_pending_action_runner_step
        expected = self._p8_formal_last_action_runner_step + 1
        if pending is None or pending != expected:
            raise RuntimeError(
                "P8 formal Action rollout is missing its unique runner_step."
            )
        self._p8_formal_pending_action_runner_step = None
        self._p8_formal_last_action_runner_step = pending
        return pending

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        collector_cfg = self.cfg.runner.get("evaluation_collector", None)
        if collector_cfg is not None:
            if not self.only_eval:
                raise ValueError(
                    "The FastWAM evaluation collector requires runner.only_eval=true."
                )
            self.evaluation_collector = instantiate(
                collector_cfg,
                rank=self._rank,
                routing_mode=self.model_cfg.eval_routing_mode,
                idm_threshold=self.model_cfg.eval_idm_threshold,
                random_idm_probability=self.model_cfg.get(
                    "eval_random_idm_probability", None
                ),
                routing_seed=self.model_cfg.eval_routing_seed,
            )
            if self.eval_rollout_epoch != 1:
                raise ValueError(
                    "Frozen-ledger evaluation requires exactly one rollout epoch."
                )

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            if self.train_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.env_list
                ), "train envs must have an offload method to enable offload!"

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            if self.eval_enable_offload:
                assert all(
                    callable(get_env_attr(env, "offload")) for env in self.eval_env_list
                ), "eval envs must have an offload method to enable offload!"

        if self.enable_train:
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]

        self._init_env()

    def update_env_cfg(self):
        if self.enable_train:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(self.cfg.env.train)
        if self.enable_eval:
            eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
            if eval_override_cfgs is not None:
                assert len(eval_override_cfgs) > self._rank, (
                    f"{len(eval_override_cfgs)=} > {self._rank=}"
                )

                general_eval_override_cfg = OmegaConf.to_container(
                    self.cfg.env.eval.get("override_cfg", {}), resolve=True
                )
                eval_override_cfg = OmegaConf.to_container(
                    eval_override_cfgs[self._rank], resolve=True
                ).copy()
                base_eval_cfg = {}
                base_eval_cfg = update_nested_cfg(
                    base_eval_cfg, general_eval_override_cfg
                )
                base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
                setattr(
                    self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg)
                )
            self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _init_pipeline_params(self):
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.shuffle_rollout = self.cfg.algorithm.get("shuffle_rollout", True)
        self.pipeline_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.pipeline_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.pipeline_actor_env_ranks = {
            actor_rank: sorted(
                {
                    logical_src_rank // self.stage_num
                    for logical_src_rank, _ in CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs,
                        src_world_size=logical_env_ws,
                        dst_world_size=actor_ws,
                        dst_rank=actor_rank,
                    )
                }
            )
            for actor_rank in range(actor_ws)
        }
        self.pipeline_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "pipeline_actor"
            )
            for actor_rank in local_actor_ranks
        }
        if self.shuffle_rollout:
            self.shuffle_generators = {
                actor_rank: torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + self._rank * actor_ws
                )
                for actor_rank in local_actor_ranks
            }

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.cfg.env.train.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                if self.train_enable_offload and self.cfg.env.train.get(
                    "enable_init_offload", True
                ):
                    get_env_attr(self.env_list[i], "offload")()
            if self.enable_eval:
                if self.eval_enable_offload:
                    get_env_attr(self.eval_env_list[i], "offload")()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        stage_id: int,
        action_execution_trace: ActionExecutionTrace | None = None,
        route_info: ChunkRouteRecord | None = None,
        flow_sde_denoise_indices: torch.Tensor | None = None,
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        exec_actions = prepare_actions(
            raw_chunk_actions=chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
            env_cfg=self.cfg.env.train,
        )
        if isinstance(chunk_actions, dict):
            chunk_actions["actions"] = exec_actions
        else:
            chunk_actions = exec_actions
        env_info = {}

        training_action_audit = _fastwam_training_action_audit_enabled(self.cfg)
        combined_action_trace = None
        if training_action_audit:
            if str(self.model_cfg.model_type) != "fastwam_adaptive":
                raise ValueError(
                    "FastWAM training Action audit requires fastwam_adaptive."
                )
            if action_execution_trace is None:
                raise ValueError(
                    "FastWAM guarded training requires the typed model Action trace."
                )
            action_contract = get_env_attr(self.env_list[stage_id], "action_contract")
            if not isinstance(action_contract, LiberoActionContract):
                raise TypeError(
                    "FastWAM training requires a typed live LIBERO Action contract."
                )
            prepared_statistics = ActionStageStatistics.from_values(
                stage=PREPARED_LIBERO_ACTION_STAGE,
                values=exec_actions,
                low=action_contract.low,
                high=action_contract.high,
                gripper_dimension_index=action_contract.gripper_dimension_index,
                action_contract_sha256=action_contract.canonical_sha256,
            )
            pre_submission_trace = ActionExecutionTrace.combine(
                action_execution_trace,
                ActionExecutionTrace(stages=(prepared_statistics,)),
            )
            try:
                chunk_result, submitted_statistics = self.env_list[
                    stage_id
                ].chunk_step_with_action_trace(exec_actions, action_contract)
            except ValueError as error:
                if "Refusing to submit Action values outside" not in str(error):
                    raise
                environment = self.env_list[stage_id]
                failure_audit = build_fastwam_action_failure_audit(
                    trace=pre_submission_trace,
                    contract=action_contract,
                    route=route_info,
                    task_ids=get_env_attr(environment, "task_ids"),
                    trial_ids=get_env_attr(environment, "trial_ids"),
                    reset_state_ids=get_env_attr(environment, "reset_state_ids"),
                    denoise_indices=flow_sde_denoise_indices,
                    worker_rank=int(getattr(self, "_rank", 0)),
                    pipeline_stage_id=stage_id,
                )
                print(
                    f"{FASTWAM_TRAINING_ACTION_FAILURE_SENTINEL} "
                    + json.dumps(failure_audit, sort_keys=True),
                    flush=True,
                )
                raise ValueError(
                    f"{error} FastWAM failure provenance: "
                    + json.dumps(failure_audit, sort_keys=True)
                ) from error
            environment_trace = ActionExecutionTrace(
                stages=(prepared_statistics, submitted_statistics)
            )
            combined_action_trace = ActionExecutionTrace.combine(
                action_execution_trace, environment_trace
            )
        else:
            chunk_result = self.env_list[stage_id].chunk_step(chunk_actions)
        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            chunk_result
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
            rlt_switch_flags=rlt_switch_flags,
        )
        chunk_step_payload = {
            "chunk_actions": exec_actions,
            "obs_list": obs_list,
            "terminations": chunk_terminations,
            "truncations": chunk_truncations,
            "infos_list": infos_list,
            "action_execution_trace": combined_action_trace,
        }
        return env_output, env_info, chunk_step_payload

    def env_evaluate_step(
        self,
        raw_actions: torch.Tensor,
        stage_id: int,
        *,
        active_mask: tuple[bool, ...] | None = None,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            env_cfg=self.cfg.env.eval,
        )
        action_execution_trace = None
        eval_env = self.eval_env_list[stage_id]
        if getattr(self, "evaluation_collector", None) is not None:
            action_contract = get_env_attr(eval_env, "action_contract")
            if not isinstance(action_contract, LiberoActionContract):
                raise TypeError(
                    "FastWAM evaluation requires a typed live LIBERO Action contract."
                )
            prepared_statistics = ActionStageStatistics.from_values(
                stage=PREPARED_LIBERO_ACTION_STAGE,
                values=chunk_actions,
                low=action_contract.low,
                high=action_contract.high,
                gripper_dimension_index=(action_contract.gripper_dimension_index),
                action_contract_sha256=(action_contract.canonical_sha256),
            )
            chunk_result, submitted_statistics = eval_env.chunk_step_with_action_trace(
                chunk_actions,
                action_contract,
                active_mask=active_mask,
            )
            action_execution_trace = ActionExecutionTrace(
                stages=(prepared_statistics, submitted_statistics)
            )
        else:
            if active_mask is None:
                chunk_result = eval_env.chunk_step(chunk_actions)
            else:
                chunk_result = eval_env.chunk_step(
                    chunk_actions,
                    active_mask=active_mask,
                )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            chunk_result
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones.any(dim=1)  # [num_envs] bool
        active_outcomes = torch.ones_like(current_dones)
        if active_mask is not None:
            active_outcomes = torch.as_tensor(
                active_mask,
                dtype=torch.bool,
                device=current_dones.device,
            )
            if active_outcomes.shape != current_dones.shape:
                raise ValueError(
                    "Evaluation ledger active mask does not match the environment "
                    "batch."
                )
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones & active_outcomes
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        rlt_switch_flags = (
            infos["rlt_switch_flags"] if "rlt_switch_flags" in infos else None
        )

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            dones=current_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            rlt_switch_flags=rlt_switch_flags,
            action_execution_trace=action_execution_trace,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer batch dim for routed shards; supports RolloutResult and plain tensor payloads.

        When the channel carries a non-``RolloutResult`` shard (e.g. reward tensor or eval
        actions) into a rollout recv, avoid assuming dataclass fields and delegate or use
        the leading dimension of dense arrays.
        """

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        if isinstance(data, RolloutResult):
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(data, field_name, None)
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
            forward_inputs = getattr(data, "forward_inputs", None)
            if forward_inputs:
                first_tensor = next(iter(forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return int(first_tensor.shape[0])
            raise ValueError("Cannot infer batch size from rollout result.")
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video:
                    flush_video = get_env_attr(self.env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video:
                    flush_video = get_env_attr(self.eval_env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None
        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None and getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
            reward_input.update({"dones": dones})

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)

        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_to(
            group_name=self.cfg.reward.group_name,
            channel=send_channel,
            data=reward_input,
            tag="train_reward_obs",
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        )
        reward_output = self.recv_from(
            group_name=self.cfg.reward.group_name,
            channel=recv_channel,
            tag="train_reward_obs",
            batch_size=self.train_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def assign_history_reward(self, stage_id: int, reward_model_output: torch.Tensor):
        reward_assign_lengths = [
            min(
                history_buffer_length[env_id]
                for history_buffer_length in self.history_lengths[stage_id].values()
            )
            for env_id in range(self.train_num_envs_per_stage)
        ]
        rollout_rewards = self.rollout_results[stage_id].rewards
        rollout_rewards_length = len(rollout_rewards)
        reward_assign_lengths = [
            min(reward_assign_length, rollout_rewards_length)
            for reward_assign_length in reward_assign_lengths
        ]
        if not any(reward_assign_lengths):
            return
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        for env_id, reward_assign_length in enumerate(reward_assign_lengths):
            for reward_assign_step in range(2, reward_assign_length + 1):
                rollout_rewards[-reward_assign_step][env_id] += reward[env_id]

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                if self.enable_online_lerobot:
                    rollout_results = getattr(self, "rollout_results", None)
                    if rollout_results is not None:
                        rollout_results[stage_id].reset_episode_buffers()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    env_infos=infos if isinstance(infos, dict) else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _build_rollout_input_data(
        self,
        env_batch: dict[str, Any],
        *,
        stage_id: int,
        eval_mode: bool = False,
        force_reset: bool = False,
    ) -> dict[str, Any]:
        data = {
            "obs": env_batch["obs"],
            "final_obs": env_batch["final_obs"],
        }
        if env_batch["obs"]:
            first_value = next(
                (value for value in env_batch["obs"].values() if value is not None),
                None,
            )
            if first_value is None:
                raise ValueError("Cannot infer batch size from empty observations.")
            batch_size = (
                int(first_value.shape[0])
                if isinstance(first_value, torch.Tensor)
                else len(first_value)
            )
            training_action_audit = _fastwam_training_action_audit_enabled(
                self.cfg,
                eval_mode=eval_mode,
            )
            if training_action_audit:
                if str(self.model_cfg.model_type) != "fastwam_adaptive":
                    raise ValueError(
                        "FastWAM training Action audit requires fastwam_adaptive."
                    )
                action_contract = get_env_attr(
                    self.env_list[stage_id], "action_contract"
                )
                if not isinstance(action_contract, LiberoActionContract):
                    raise TypeError(
                        "FastWAM training requires a typed live LIBERO Action contract."
                    )
                rollout_obs = dict(data["obs"])
                rollout_obs["_fastwam_action_contract_low"] = (
                    torch.tensor(action_contract.low, dtype=torch.float32)
                    .expand(batch_size, -1)
                    .clone()
                )
                rollout_obs["_fastwam_action_contract_high"] = (
                    torch.tensor(action_contract.high, dtype=torch.float32)
                    .expand(batch_size, -1)
                    .clone()
                )
                rollout_obs["_fastwam_action_gripper_indices"] = torch.full(
                    (batch_size,),
                    action_contract.gripper_dimension_index,
                    dtype=torch.long,
                )
                rollout_obs["_fastwam_action_contract_sha256"] = [
                    action_contract.canonical_sha256
                ] * batch_size
                data["obs"] = rollout_obs
            per_stage = (
                self.eval_num_envs_per_stage
                if eval_mode
                else self.train_num_envs_per_stage
            )
            namespace = 1 << 50 if eval_mode else 0
            start = namespace + (self._rank * self.stage_num + stage_id) * per_stage
            data["fastwam_env_ids"] = torch.arange(
                start,
                start + batch_size,
                dtype=torch.long,
            )
            dones = env_batch.get("dones")
            if force_reset:
                reset_mask = torch.ones(batch_size, dtype=torch.bool)
            elif dones is None:
                reset_mask = torch.zeros(batch_size, dtype=torch.bool)
            else:
                reset_mask = (
                    dones.to(dtype=torch.bool).reshape(batch_size, -1).any(dim=1)
                )
            data["fastwam_reset_mask"] = reset_mask
        if self.enable_rlt:
            data["rlt_switch_flags"] = env_batch.get("rlt_switch_flags", None)
            data["intervene_flags"] = env_batch.get("intervene_flags", None)
        return data

    def _send_train_bootstrap(
        self, rollout_channel: Channel, env_outputs: list[EnvOutput]
    ) -> None:
        for stage_id in range(self.stage_num):
            env_output: EnvOutput = env_outputs[stage_id]
            env_batch = env_output.to_dict()
            self.send_to(
                group_name=self.cfg.rollout.group_name,
                channel=rollout_channel,
                data=self._build_rollout_input_data(
                    env_batch,
                    stage_id=stage_id,
                    force_reset=not self.cfg.env.train.auto_reset,
                ),
                mode="train",
                tag="rollout_results",
                route_key=stage_id if not self.env_decoupled_mode else None,
                decoupled_mode=self.env_decoupled_mode,
            )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def record_env_metrics(
        self,
        env_metrics: dict[str, list],
        env_info: dict[str, Any],
    ):
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    @Worker.timer("env/send_rollout_trajectories")
    async def send_rollout_trajectories(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
        *,
        stage_id: int,
    ):
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        logical_env_world_size = env_world_size * self.stage_num
        logical_env_rank = self._rank * self.stage_num + int(stage_id)
        routes = CommMapper.get_dst_ranks(
            batch_size=int(self.cfg.env.train.total_num_envs),
            src_world_size=logical_env_world_size,
            dst_world_size=actor_world_size,
            src_rank=logical_env_rank,
        )
        trajectories: list[Trajectory] = (
            rollout_result.to_splited_trajectories_by_sizes(
                [batch_size for _, batch_size in routes]
            )
        )
        rollout_result.clear()
        works = []
        for (actor_rank, _), trajectory in zip(routes, trajectories, strict=True):
            key = CommMapper.build_channel_key(
                logical_env_rank,
                actor_rank,
                ACTOR_TRAJECTORY_CHANNEL_TAG,
            )
            work = channel.put(trajectory, key=key, async_op=True)
            if work is not None:
                works.append(work)
        for work in works:
            await work.async_wait()
        del trajectories
        gc.collect()

    @Worker.timer("env/send_lerobot_episodes")
    async def send_lerobot_episodes(
        self, episodes: list[list[dict]], channel: Channel
    ) -> None:
        if not episodes:
            return
        if self.actor_split_num <= 1:
            chunks = [episodes]
        else:
            chunks = split_list(
                episodes,
                self.actor_split_num,
                enforce_divisible_batch=False,
            )
        for chunk in chunks:
            if not chunk:
                continue
            channel.put(chunk, async_op=True)

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results = self._prepare_rollout_results(
            getattr(self, "rollout_results", None)
        )
        env_metrics = defaultdict(list)
        rlt_pending_obs: list[dict[str, Any] | None] = [None] * self.stage_num
        training_action_audit = _fastwam_training_action_audit_enabled(self.cfg)
        formal_action_runner_step = (
            self._consume_p8_formal_action_runner_step()
            if training_action_audit
            else None
        )
        training_action_traces: list[list[ActionExecutionTrace]] = [
            [] for _ in range(self.stage_num)
        ]
        training_action_episode_identity_hashes: list[list[str]] = [
            [] for _ in range(self.stage_num)
        ]
        training_flow_sde_denoise_indices: list[list[torch.Tensor]] = [
            [] for _ in range(self.stage_num)
        ]

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(rollout_channel)

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    rollout_result = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="train_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.train_batch_size,
                        merge_fn=RolloutResult.merge_rollout_results,
                        infer_batch_size_fn=self._infer_rollout_batch_size,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=(
                            rollout_result.prev_logprobs
                            if self.collect_prev_infos
                            else None
                        ),
                        prev_values=(
                            rollout_result.prev_values
                            if self.collect_prev_infos
                            else None
                        ),
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                        route_info=rollout_result.route_info,
                        emitted_gate=rollout_result.emitted_gate,
                    )

                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)
                    if rollout_result.intervene_flags is not None:
                        self.rollout_results[
                            stage_id
                        ].mark_last_step_with_intervene_flags(
                            rollout_result.intervene_flags
                        )
                    if self.enable_rlt and self.collect_transitions:
                        update_rlt_transitions(
                            stage_id,
                            rlt_pending_obs,
                            self.rollout_results,
                            rollout_result,
                            cache_current=True,
                        )

                    if training_action_audit:
                        environment = self.env_list[stage_id]
                        training_action_episode_identity_hashes[stage_id].append(
                            build_fastwam_episode_identity_sha256(
                                route=rollout_result.route_info,
                                task_ids=get_env_attr(environment, "task_ids"),
                                trial_ids=get_env_attr(environment, "trial_ids"),
                                reset_state_ids=get_env_attr(
                                    environment, "reset_state_ids"
                                ),
                            )
                        )
                        denoise_indices = rollout_result.forward_inputs.get(
                            "denoise_indices"
                        )
                        if denoise_indices is None:
                            raise ValueError(
                                "FastWAM guarded training requires Flow-SDE "
                                "denoise indices."
                            )
                        training_flow_sde_denoise_indices[stage_id].append(
                            denoise_indices.detach().cpu()
                        )
                    env_output, env_info, chunk_step_payload = self.env_interact_step(
                        rollout_result.actions,
                        stage_id,
                        action_execution_trace=rollout_result.action_execution_trace,
                        route_info=rollout_result.route_info,
                        flow_sde_denoise_indices=rollout_result.forward_inputs.get(
                            "denoise_indices"
                        ),
                    )
                    action_trace = chunk_step_payload.pop(
                        "action_execution_trace", None
                    )
                    if action_trace is not None:
                        training_action_traces[stage_id].append(action_trace)
                    stage_rollout = self.rollout_results[stage_id]
                    if isinstance(stage_rollout, EmbodiedLerobotRolloutResult):
                        stage_rollout.append_chunk_episode_data(
                            rollout_result=rollout_result,
                            **chunk_step_payload,
                        )
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=self._build_rollout_input_data(
                            env_batch,
                            stage_id=stage_id,
                        ),
                        mode="train",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    if self.collect_transitions and not self.enable_rlt:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                reward_model_output = None
                if reward_channel is not None:
                    last_run = epoch == self.rollout_epoch - 1
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                rollout_result = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    route_key=stage_id if not self.env_decoupled_mode else None,
                    batch_size=self.train_batch_size,
                    merge_fn=RolloutResult.merge_rollout_results,
                    infer_batch_size_fn=self._infer_rollout_batch_size,
                    decoupled_mode=self.env_decoupled_mode,
                )
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                chunk_step_result = ChunkStepResult(
                    actions=rollout_result.forward_inputs.get("action", None),
                    prev_logprobs=(
                        rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None
                    ),
                    prev_values=(
                        rollout_result.prev_values if self.collect_prev_infos else None
                    ),
                    forward_inputs=rollout_result.forward_inputs,
                    versions=rollout_result.versions,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                    route_info=rollout_result.route_info,
                    emitted_gate=rollout_result.emitted_gate,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                    and reward_model_output is not None
                ):
                    self.assign_history_reward(stage_id, reward_model_output)
                if self.enable_rlt and self.collect_transitions:
                    update_rlt_transitions(
                        stage_id,
                        rlt_pending_obs,
                        self.rollout_results,
                        rollout_result,
                        cache_current=False,
                    )

            if self.use_training_pipeline and actor_channel is not None:
                await self.send_rollout_trajectories_pipeline(
                    self.rollout_results, actor_channel
                )
                self.rollout_results = self._prepare_rollout_results(
                    getattr(self, "rollout_results", None)
                )

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if not self.use_training_pipeline and actor_channel is not None:
            if self.enable_online_lerobot:
                for stage_id in range(self.stage_num):
                    episodes = self.rollout_results[stage_id].drain_episodes()
                    await self.send_lerobot_episodes(episodes, actor_channel)
            else:
                for stage_id in range(self.stage_num):
                    await self.send_rollout_trajectories(
                        self.rollout_results[stage_id],
                        actor_channel,
                        stage_id=stage_id,
                    )

        if training_action_audit:
            for stage_id, traces in enumerate(training_action_traces):
                if not traces:
                    raise RuntimeError(
                        "FastWAM training Action audit collected no execution traces."
                    )
                merged_trace = ActionExecutionTrace.merge_time(traces)
                contract = get_env_attr(self.env_list[stage_id], "action_contract")
                if not isinstance(contract, LiberoActionContract):
                    raise TypeError(
                        "FastWAM training Action audit lost its live contract."
                    )
                identity_hashes = training_action_episode_identity_hashes[stage_id]
                if len(identity_hashes) != len(traces):
                    raise RuntimeError(
                        "FastWAM training Action identity/trace counts disagree."
                    )
                payload = {
                    **_fastwam_training_action_audit_identity(
                        formal_action_runner_step
                    ),
                    "worker_rank": int(self._rank),
                    "pipeline_stage_id": int(stage_id),
                    "stage_order": list(merged_trace.stage_names),
                    "action_contract": contract.to_artifact(),
                    "episode_identity_observation_count": len(identity_hashes),
                    "first_episode_identity_sha256": identity_hashes[0],
                    "episode_identity_sequence_sha256": checkpoint_state_sha256(
                        identity_hashes
                    ),
                    "flow_sde_denoise_indices": (
                        summarize_fastwam_flow_sde_denoise_indices(
                            training_flow_sde_denoise_indices[stage_id],
                            num_inference_steps=int(
                                self.model_cfg.runtime.num_inference_steps
                            ),
                            ignore_last_transition=bool(
                                self.model_cfg.flow_sde.get(
                                    "ignore_last_transition", False
                                )
                            ),
                        )
                    ),
                    "records": [
                        merged_trace.record_for_batch_index(index)
                        for index in range(merged_trace.batch_size)
                    ],
                }
                print(
                    f"{FASTWAM_TRAINING_ACTION_AUDIT_SENTINEL} "
                    + json.dumps(payload, sort_keys=True),
                    flush=True,
                )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.train_enable_offload:
                get_env_attr(env, "offload")()

        return env_metrics

    @Worker.timer("evaluate")
    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)
        pending_snapshots = [None] * self.stage_num
        policy_started_at = [None] * self.stage_num

        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                        env_infos=infos if isinstance(infos, dict) else None,
                    )
                    env_batch = env_output.to_dict()
                    rollout_input = self._build_rollout_input_data(
                        env_batch,
                        stage_id=stage_id,
                        eval_mode=True,
                        force_reset=True,
                    )
                    if self.evaluation_collector is not None:
                        snapshot = self.evaluation_collector.snapshot_before_step(
                            stage_id,
                            self.eval_env_list[stage_id],
                            rollout_input["fastwam_env_ids"],
                        )
                        rollout_input = self.evaluation_collector.augment_rollout_input(
                            rollout_input, snapshot
                        )
                        pending_snapshots[stage_id] = snapshot
                    policy_started_at[stage_id] = time.perf_counter()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=rollout_input,
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                evaluation_complete = False
                for stage_id in range(self.stage_num):
                    rollout_results = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        batch_size=self.eval_batch_size,
                        merge_fn=_merge_evaluation_rollout_results,
                        infer_batch_size_fn=self._infer_rollout_batch_size,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    policy_latency_seconds = (
                        None
                        if policy_started_at[stage_id] is None
                        else time.perf_counter() - policy_started_at[stage_id]
                    )
                    raw_chunk_actions = (
                        rollout_results.actions
                        if hasattr(rollout_results, "actions")
                        else rollout_results
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    snapshot = (
                        pending_snapshots[stage_id]
                        if self.evaluation_collector is not None
                        else None
                    )
                    if self.evaluation_collector is not None and snapshot is None:
                        raise TypeError(
                            "FastWAM evaluation collector requires a pending "
                            "identity snapshot."
                        )
                    environment_started_at = time.perf_counter()
                    if snapshot is None:
                        env_output, env_info = self.env_evaluate_step(
                            raw_chunk_actions, stage_id
                        )
                    else:
                        env_output, env_info = self.env_evaluate_step(
                            raw_chunk_actions,
                            stage_id,
                            active_mask=snapshot.active_mask,
                        )
                    environment_latency_seconds = (
                        time.perf_counter() - environment_started_at
                    )
                    if isinstance(rollout_results, RolloutResult):
                        rollout_results = _mark_terminal_gate_unused(
                            rollout_results, env_output.dones
                        )
                    if self.evaluation_collector is not None:
                        if not isinstance(rollout_results, RolloutResult):
                            raise TypeError(
                                "FastWAM evaluation collector requires typed "
                                "RolloutResult."
                            )
                        self.evaluation_collector.record_chunk(
                            snapshot=snapshot,
                            rollout_result=rollout_results,
                            env_output=env_output,
                            policy_latency_seconds=policy_latency_seconds,
                            environment_latency_seconds=environment_latency_seconds,
                        )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if (
                        self.evaluation_collector is not None
                        and self.evaluation_collector.is_complete
                    ):
                        if self.stage_num != 1:
                            raise RuntimeError(
                                "Ledger-complete FastWAM evaluation termination "
                                "currently requires one environment stage."
                            )
                        if self.env_decoupled_mode:
                            raise RuntimeError(
                                "Ledger-complete FastWAM evaluation termination "
                                "requires coupled env/rollout transport."
                            )
                        if eval_step < self.n_eval_chunk_steps - 1:
                            stop_control = (
                                self.evaluation_collector.build_rollout_stop_control(
                                    logical_batch_size=self.eval_batch_size
                                )
                            )
                            self.send_to(
                                group_name=self.cfg.rollout.group_name,
                                channel=rollout_channel,
                                data=stop_control,
                                mode="eval",
                                tag="rollout_results",
                                route_key=stage_id,
                                batch_size=self.eval_batch_size,
                                split_fn=_split_evaluation_rollout_control,
                            )
                        evaluation_complete = True
                        continue

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    rollout_input = self._build_rollout_input_data(
                        env_batch,
                        stage_id=stage_id,
                        eval_mode=True,
                    )
                    if self.evaluation_collector is not None:
                        snapshot = self.evaluation_collector.snapshot_before_step(
                            stage_id,
                            self.eval_env_list[stage_id],
                            rollout_input["fastwam_env_ids"],
                        )
                        rollout_input = self.evaluation_collector.augment_rollout_input(
                            rollout_input, snapshot
                        )
                        pending_snapshots[stage_id] = snapshot
                    policy_started_at[stage_id] = time.perf_counter()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data=rollout_input,
                        mode="eval",
                        tag="rollout_results",
                        route_key=stage_id if not self.env_decoupled_mode else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )

                if evaluation_complete:
                    break

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                get_env_attr(self.eval_env_list[stage_id], "offload")()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        if self.evaluation_collector is None:
            return eval_metrics
        return {
            "metrics": dict(eval_metrics),
            "evaluation_artifact_shard": asdict(self.evaluation_collector.finalize()),
        }

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def compute_advantages_and_returns(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Advantages/returns are rollout-level quantities, so compute them before
        # splitting. After this point each channel item is an actor micro-batch that can
        # be trained directly without reconstructing the full rollout batch on actor.
        assert not (
            self.use_training_pipeline and self.cfg.algorithm.adv_type == "opd"
        ), (
            "OPD does not support runner.use_training_pipeline=True because "
            "teacher_logprobs are computed on actor workers after rollout."
        )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "prev_logprobs": rollout_batch.get("prev_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get("normalize_advantages", True)
            and not self.use_training_pipeline,
        }
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def prepare_pipeline_batch(self, trajectory: Trajectory) -> dict[str, torch.Tensor]:
        batch = convert_trajectories_to_batch([trajectory])
        batch = preprocess_embodied_batch(
            batch,
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )
        return self.compute_advantages_and_returns(batch)

    def pack_pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        if self.shuffle_rollout:
            shuffle_id = torch.randperm(
                batch_size, generator=self.shuffle_generators[actor_rank]
            )
        else:
            shuffle_id = torch.arange(batch_size)

        flatten_batch = flatten_embodied_batch(batch, shuffle_id)
        micro_batch_size = self.cfg.actor.micro_batch_size
        assert batch_size % micro_batch_size == 0, (
            f"Batch size {batch_size} is not divisible by micro_batch_size {micro_batch_size}."
        )
        num_micro_batches = batch_size // micro_batch_size
        micro_batches = split_dict_to_chunk(flatten_batch, num_micro_batches, dim=0)
        return [pack_batch(micro_batch) for micro_batch in micro_batches]

    async def send_rollout_trajectories_pipeline(
        self,
        rollout_results: list[EmbodiedRolloutResult],
        channel: Channel,
    ) -> None:
        pending_batches: list[tuple[int, dict[str, torch.Tensor]]] = []
        batches_by_actor_rank: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(
            list
        )

        with self.worker_timer("prepare_micro_batches"):
            for stage_id, rollout_result in enumerate(rollout_results):
                actor_splits = self.pipeline_stage_actor_splits[stage_id]
                trajectories = rollout_result.to_splited_trajectories_by_sizes(
                    [split_size for _, split_size in actor_splits]
                )

                for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                    batch = self.prepare_pipeline_batch(trajectory)
                    pending_batches.append((actor_rank, batch))
                    batches_by_actor_rank[actor_rank].append(batch)

            if self.cfg.algorithm.get("normalize_advantages", True):
                for actor_rank, batches in sorted(batches_by_actor_rank.items()):
                    local_adv_stats = sum(
                        masked_stats(batch["advantages"], batch.get("loss_mask"))
                        for batch in batches
                    )
                    env_ranks = self.pipeline_actor_env_ranks[actor_rank]
                    global_adv_stats = sum(
                        self.broadcast(
                            local_adv_stats if self._rank == src_rank else None,
                            groups=[(self._group_name, env_ranks)],
                            src=(self._group_name, src_rank),
                        )
                        for src_rank in env_ranks
                    )
                    for batch in batches:
                        batch["advantages"] = normalize_from_stats(
                            batch["advantages"], global_adv_stats
                        )

            for actor_rank, batch in pending_batches:
                for micro_batch in self.pack_pipeline_micro_batches(batch, actor_rank):
                    channel.put(
                        micro_batch,
                        key=self.pipeline_actor_keys[actor_rank],
                        async_op=True,
                    )
