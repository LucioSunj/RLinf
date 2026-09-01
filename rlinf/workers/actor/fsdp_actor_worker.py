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

import hashlib
import json
import math
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.advantages import (
    FASTWAM_CHUNK_COST_AUDIT_SENTINEL,
    FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL,
    FASTWAM_GATE_UPDATE_AUDIT_SENTINEL,
    FASTWAM_REWARD_AUDIT_SENTINEL,
    FASTWAM_ROLLOUT_STATE_AUDIT_SENTINEL,
    FastWAMGateUpdateAudit,
    align_fastwam_policy_advantages,
    apply_fastwam_chunk_cost,
    compute_fastwam_unnormalized_gate_alignment,
    summarize_fastwam_chunk_cost,
    summarize_fastwam_counterfactual_costs,
    summarize_fastwam_environment_rewards,
    summarize_fastwam_rollout_state,
)
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.fastwam_dual_ppo import (
    compute_base_uncond_kl_loss,
    compute_fastwam_dual_ppo_loss,
    compute_gate_collapse_penalty,
    compute_gate_ppo_loss,
    finalize_fastwam_weighted_metrics,
    pop_fastwam_weighted_metric_sums,
)
from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.config_contracts import (
    FASTWAM_RESUME_MODE_N4_TO_THREE_ROLLOUT,
    build_fastwam_checkpoint_contract,
    validate_fastwam_training_checkpoint_contract,
)
from rlinf.data.embodied_io_struct import (
    ACTOR_TRAJECTORY_CHANNEL_TAG,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.wam_policy.contracts import WAMRoute
from rlinf.models.embodiment.wam_policy.critic import (
    critic_parent_checkpoint_sha256,
)
from rlinf.runners.fastwam_decision_telemetry import (
    append_fastwam_decision_telemetry_jsonl,
    build_fastwam_training_decision_records,
)
from rlinf.runners.fastwam_training_guard import (
    append_fastwam_counterfactual_cost_audit_jsonl,
)
from rlinf.scheduler import Channel, Cluster, CommMapper, Worker
from rlinf.utils.checkpoint_state import (
    FASTWAM_ACTOR_RESUME_AUDIT_SENTINEL,
    FASTWAM_RESUME_AUDIT_SCHEMA,
    checkpoint_state_sha256,
)
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_loss_mask,
    compute_rollout_metrics,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    flatten_time_batch,
    flatten_time_batch_consuming,
    map_nested_tensors,
    merge_rollout_epoch_batch,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    get_rng_state,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
    seed_everything,
    set_rng_state,
)
from rlinf.workers.actor.fastwam_selective_sync import (
    materialize_fastwam_sync_state,
    prepare_fastwam_sync_tensors,
)
from rlinf.workers.rollout.utils import RankMapper

_FASTWAM_BC_BOOTSTRAP_SCHEMA = "fastwam-uncond-bc-bootstrap-v1"
FASTWAM_ACCELERATION_SEMANTICS_AUDIT_SENTINEL = "FASTWAM_ACCELERATION_SEMANTICS_AUDIT"
FASTWAM_PREUPDATE_LOG_RATIO_AUDIT_SENTINEL = "FASTWAM_PREUPDATE_LOG_RATIO_AUDIT"
FASTWAM_GATE_KV_SAMPLE_AUDIT_SENTINEL = "FASTWAM_GATE_KV_SAMPLE_AUDIT"
FASTWAM_GATE_GRADIENT_CURVE_AUDIT_SENTINEL = "FASTWAM_GATE_GRADIENT_CURVE_AUDIT"
_FASTWAM_BC_BOOTSTRAP_KEYS = {
    "schema",
    "bc_step",
    "bc_config_sha256",
    "sidecar_sha256",
    "parent_checkpoint_sha256",
}


def fastwam_effective_gate_kv_mask(
    gate_valid_mask: torch.Tensor,
    gate_kv_sample_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Restrict only Gate replay while leaving every non-Gate mask untouched."""

    gate_valid = gate_valid_mask.bool()
    if gate_kv_sample_mask is None:
        return gate_valid
    sampled = gate_kv_sample_mask.bool()
    if sampled.shape != gate_valid.shape:
        raise ValueError("Gate-valid and Gate K/V sample masks must have equal shape.")
    return gate_valid & sampled


def summarize_fastwam_gate_kv_episode_contributions(
    *,
    episode_ids: torch.Tensor,
    gate_valid_mask: torch.Tensor,
    gate_kv_sample_mask: torch.Tensor,
) -> list[dict[str, int]]:
    """Count sampled K/V and usable Gate rows for each rollout trajectory."""

    if episode_ids.ndim != 2:
        raise ValueError("Gate K/V episode telemetry requires [time, batch] tensors.")
    if (
        gate_valid_mask.shape != episode_ids.shape
        or gate_kv_sample_mask.shape != episode_ids.shape
    ):
        raise ValueError("Gate K/V episode telemetry tensors must have equal shape.")
    gate_valid = gate_valid_mask.bool()
    sampled = gate_kv_sample_mask.bool()
    contributions = []
    for trajectory_column in range(int(episode_ids.shape[1])):
        column_episodes = episode_ids[:, trajectory_column]
        contributions.append(
            {
                "trajectory_column": trajectory_column,
                "initial_episode_id": int(column_episodes[0].item()),
                "observed_episode_id_count": int(torch.unique(column_episodes).numel()),
                "emitted_chunk_count": int(episode_ids.shape[0]),
                "sampled_kv_count": int(sampled[:, trajectory_column].sum().item()),
                "sampled_eligible_gate_count": int(
                    (sampled[:, trajectory_column] & gate_valid[:, trajectory_column])
                    .sum()
                    .item()
                ),
            }
        )
    return contributions


def _fastwam_sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256.")
    return normalized


def _validate_fastwam_bc_bootstrap_provenance(
    payload: object,
    *,
    expected_parent_checkpoint_sha256: str,
) -> dict:
    if not isinstance(payload, dict) or set(payload) != (_FASTWAM_BC_BOOTSTRAP_KEYS):
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"FastWAM BC bootstrap provenance keys changed: {keys}.")
    if payload.get("schema") != _FASTWAM_BC_BOOTSTRAP_SCHEMA:
        raise ValueError("Unsupported FastWAM BC bootstrap provenance schema.")
    bc_step = payload.get("bc_step")
    if isinstance(bc_step, bool) or not isinstance(bc_step, int) or bc_step <= 0:
        raise ValueError("FastWAM BC bootstrap bc_step must be a positive integer.")
    bc_config_sha256 = _validate_sha256(
        payload.get("bc_config_sha256"),
        name="FastWAM BC bootstrap bc_config_sha256",
    )
    sidecar_sha256 = _validate_sha256(
        payload.get("sidecar_sha256"),
        name="FastWAM BC bootstrap sidecar_sha256",
    )
    expected_parent = _validate_sha256(
        expected_parent_checkpoint_sha256,
        name="FastWAM parent checkpoint SHA-256",
    )
    parent_sha256 = _validate_sha256(
        payload.get("parent_checkpoint_sha256"),
        name="FastWAM BC bootstrap parent checkpoint SHA-256",
    )
    if parent_sha256 != expected_parent:
        raise ValueError(
            "FastWAM BC bootstrap parent hash mismatch: "
            f"expected {expected_parent}, got {parent_sha256}."
        )
    return {
        "schema": _FASTWAM_BC_BOOTSTRAP_SCHEMA,
        "bc_step": bc_step,
        "bc_config_sha256": bc_config_sha256,
        "sidecar_sha256": sidecar_sha256,
        "parent_checkpoint_sha256": parent_sha256,
    }


def _raise_fastwam_collective_checkpoint_error(
    local_error: Exception | None,
    *,
    context: str,
) -> None:
    """Make rank-local checkpoint failures visible to every actor rank."""

    if not torch.distributed.is_initialized():
        if local_error is not None:
            raise local_error
        return
    local_message = (
        None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    )
    errors: list[str | None] = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(errors, local_message)
    failed = {rank: error for rank, error in enumerate(errors) if error is not None}
    if failed:
        raise RuntimeError(
            f"FastWAM {context} failed collectively: {failed}."
        ) from local_error


_MISSING_FASTWAM_FSDP_ROOT = object()


def _snapshot_fastwam_fsdp_lazy_root_state(
    model,
    *,
    fsdp_cls=None,
):
    """Capture private FSDP lazy-root flags before adaptive checkpoint load.

    Loading rank-local trainable and optimizer state before the first forward
    may trigger PyTorch FSDP bookkeeping on nested wrappers. Those flags are
    execution state rather than checkpoint state and must remain exactly as
    they were before restore so the first resumed root forward can initialize
    the hierarchy normally.
    """

    if fsdp_cls is None:
        from torch.distributed.fsdp import FullyShardedDataParallel

        fsdp_cls = FullyShardedDataParallel
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        return []
    seen = set()
    snapshot = []
    for module in modules_fn():
        identity = id(module)
        if identity in seen or not isinstance(module, fsdp_cls):
            continue
        seen.add(identity)
        snapshot.append(
            (module, getattr(module, "_is_root", _MISSING_FASTWAM_FSDP_ROOT))
        )
    return snapshot


def _restore_fastwam_fsdp_lazy_root_state(snapshot) -> None:
    """Restore private FSDP lazy-root flags captured before checkpoint load."""

    for module, original in snapshot:
        if original is _MISSING_FASTWAM_FSDP_ROOT:
            if hasattr(module, "_is_root"):
                delattr(module, "_is_root")
        else:
            module._is_root = original


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    return merge_rollout_epoch_batch(nested_dict, rollout_epoch)


def process_nested_dict_for_train(nested_dict, shuffle_id, *, consume=False):
    """Flatten and shuffle a rollout batch for actor training.

    Args:
        nested_dict: Time-major rollout batch.
        shuffle_id: Shared flattened-sample permutation.
        consume: Pop source fields while shuffling. This bounds peak memory for
            one-way stored-replay handoff without changing the permutation.

    Returns:
        Flattened and shuffled training batch.
    """

    ret_dict = {}
    for key in list(nested_dict):
        value = nested_dict.pop(key) if consume else nested_dict[key]
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
            continue

        flatten = flatten_time_batch_consuming if consume else flatten_time_batch
        ret_dict[key] = flatten(value, shuffle_id, field_name=key)
        del value
    return ret_dict


def trim_nested_tensor_time_dim(value, target_steps: int, key_path=()):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assert value.shape[0] in {target_steps, target_steps + 1}, (
            f"Cannot trim field {'.'.join(key_path)!r} with shape "
            f"{tuple(value.shape)} to {target_steps} OPD training steps."
        )
        return value[:target_steps]

    def trim_tensor(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[0] in {target_steps, target_steps + 1}, (
            f"Cannot trim field {'.'.join(key_path)!r} with shape "
            f"{tuple(tensor.shape)} to {target_steps} OPD training steps."
        )
        return tensor[:target_steps]

    return map_nested_tensors(value, trim_tensor)


def flatten_nested_tensor_time_batch(value, key_path=()):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assert value.dim() >= 2, (
            f"Cannot flatten field {'.'.join(key_path)!r} with shape "
            f"{tuple(value.shape)} across time and batch."
        )
        return value.reshape(-1, *value.shape[2:])

    def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() >= 2, (
            f"Cannot flatten field {'.'.join(key_path)!r} with shape "
            f"{tuple(tensor.shape)} across time and batch."
        )
        return tensor.reshape(-1, *tensor.shape[2:])

    return map_nested_tensors(value, flatten_tensor)


def compute_rollout_train_kl(
    m_batch: dict, loss_mask: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute the masked mean of absolute difference between rollout and training logprobs.

    Args:
        m_batch: Dictionary containing 'rollout_logprobs' and 'recomputed_logprobs'.
        loss_mask: Mask tensor for computing weighted mean.

    Returns:
        Masked mean of abs(recomputed_logprobs - rollout_logprobs), or None if keys are missing.
    """
    if "rollout_logprobs" not in m_batch or "recomputed_logprobs" not in m_batch:
        return None
    rollout_logprobs = m_batch["rollout_logprobs"]
    recomputed_logprobs = m_batch["recomputed_logprobs"]
    kl = torch.abs(recomputed_logprobs - rollout_logprobs)
    return masked_mean(kl, loss_mask)


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)
        self.variable_seq_lengths = self.cfg.actor.model.get(
            "variable_seq_lengths", False
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    @Worker.timer("actor/sync_model_to_rollout")
    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_to_fixed_len=not self.variable_seq_lengths,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits

        logits.div_(self.cfg.algorithm.sampling_params.temperature)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)

        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)

            if self.enable_dynamic_batch_size or self.variable_seq_lengths:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]

            return logprobs, entropy

        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        num_sequences: int,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=num_sequences
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        recomputed_logprobs, ref_logprobs = None, None

        # Recompute logprobs
        recomputed_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == recomputed_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {recomputed_logprobs.size(0)}"
            )
            recomputed_logprobs = recomputed_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return recomputed_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    recomputed_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result.num_sequence, compute_ref_logprobs
                    )

                rollout_result.recomputed_logprobs = recomputed_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    @Worker.timer("training_step")
    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            # Prefer recomputed_logprobs (from actor inference), fallback to rollout_logprobs
            old_logprobs = m_batch.get("recomputed_logprobs")
            if old_logprobs is None:
                old_logprobs = m_batch["rollout_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                if (
                    "rollout_logprobs" not in m_batch
                    or "recomputed_logprobs" not in m_batch
                ):
                    raise ValueError(
                        "importance_sampling_fix requires both rollout_logprobs and recomputed_logprobs"
                    )
                rollout_logprobs = m_batch["rollout_logprobs"]
                recomputed_logprobs = m_batch["recomputed_logprobs"]
                advantages = advantages * torch.clamp(
                    (recomputed_logprobs - rollout_logprobs).exp(),
                    max=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # display the degree of mismatch between training and rollout
        rollout_train_kl = compute_rollout_train_kl(m_batch, loss_mask)

        # aggregate metrics across micro-batches
        explained_variance_stats = pop_critic_explained_variance_stats(mbs_metrics_list)
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        if rollout_train_kl is not None:
            mean_metric_dict["actor/rollout_train_kl"] = rollout_train_kl

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    @Worker.timer("run_training")
    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        assert (
            "recomputed_logprobs" in global_batch or "rollout_logprobs" in global_batch
        )

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    @Worker.timer("compute_advantages_and_returns")
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                logprob = batch.get("recomputed_logprobs")
                if logprob is None:
                    logprob = batch.get("rollout_logprobs")
                logprob = logprob.to(Worker.torch_device_type)

                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=logprob,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages
        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )
        self._fastwam_kv_request_channel = None
        self._fastwam_kv_response_channel = None
        self._fastwam_kv_request_id = 0
        self._fastwam_kv_executor: ThreadPoolExecutor | None = None
        self._fastwam_kv_prefetch_stream: torch.cuda.Stream | None = None
        self._fastwam_kv_h2d_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._fastwam_kv_prefetch_wait_seconds = 0.0
        self._fastwam_kv_h2d_bytes = 0
        self._fastwam_kv_use_counts: dict[int, int] = {}

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        bootstrap_output_dir = self.cfg.runner.get(
            "bootstrap_project_checkpoint_dir",
            None,
        )
        if bootstrap_output_dir is not None:
            seed_everything(int(self.cfg.actor.seed) + int(self._rank))

        self.setup_model_and_optimizer()

        model_type = str(self.cfg.actor.model.get("model_type", ""))
        bootstrap_path = self.cfg.runner.get("ckpt_path", None)
        if (
            model_type == SupportedModel.FASTWAM_ADAPTIVE.value
            and bootstrap_path is not None
        ):
            if self.cfg.runner.get("resume_dir", None) is not None:
                raise ValueError(
                    "FastWAM training cannot set both runner.ckpt_path and "
                    "runner.resume_dir. Use ckpt_path only for native step-zero "
                    "bootstrap and resume_dir for paired nonzero resume."
                )
            loaded_step = self.load_checkpoint(str(bootstrap_path))
            if loaded_step != 0:
                raise ValueError(
                    "FastWAM runner.ckpt_path training bootstrap requires a "
                    f"native step-zero checkpoint, got step {loaded_step}."
                )

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        model_type = str(self.cfg.actor.model.get("model_type", ""))
        if self.cfg.runner.get("ckpt_path", None) and (
            model_type != SupportedModel.FASTWAM_ADAPTIVE.value
        ):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        if model_type == SupportedModel.FASTWAM_ADAPTIVE.value:
            if not bool(self.cfg.actor.fsdp_config.get("use_orig_params", False)):
                raise RuntimeError(
                    "FastWAM selective FSDP sync requires use_orig_params=True."
                )
            if not bool(
                self.cfg.actor.fsdp_config.get("ignore_frozen_parameters", False)
            ):
                raise RuntimeError(
                    "FastWAM FSDP requires ignore_frozen_parameters=True."
                )
            local_device = torch.device(
                Worker.torch_device_type,
                int(os.environ.get("LOCAL_RANK", 0)),
            )
            self._fastwam_rollout_sync_tensors = prepare_fastwam_sync_tensors(
                model,
                device=local_device,
            )

        return model

    def get_rollout_state_dict(self) -> dict:
        model_type = str(self.cfg.actor.model.get("model_type", ""))
        if model_type == SupportedModel.FASTWAM_ADAPTIVE.value:
            captured = getattr(self, "_fastwam_rollout_sync_tensors", None)
            if captured is None:
                raise RuntimeError(
                    "FastWAM rollout sync tensors were not captured before FSDP wrap."
                )
            return materialize_fastwam_sync_state(
                captured,
                self.param_names_need_sync,
            )
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    def _fastwam_policy_module(self):
        model = self.model
        visited = set()
        while id(model) not in visited:
            visited.add(id(model))
            if hasattr(model, "trainable_state_dict") and hasattr(
                model, "load_trainable_state_dict"
            ):
                return model
            next_model = getattr(model, "module", None)
            if next_model is None:
                next_model = getattr(model, "_fsdp_wrapped_module", None)
            if next_model is None:
                break
            model = next_model
        raise TypeError(
            "Could not unwrap the FastWAM adaptive policy from its FSDP wrapper."
        )

    def _capture_fastwam_gate_parameters(self) -> dict[str, torch.Tensor]:
        """Clone trainable Gate parameters to CPU for an audit-only delta."""

        gate = self._fastwam_policy_module().gate
        state = {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in gate.named_parameters()
            if parameter.requires_grad
        }
        if not state:
            raise RuntimeError("FastWAM Gate update audit found no trainable tensors.")
        return state

    @staticmethod
    def _fastwam_gate_parameter_audit_due(
        *,
        actor_version: int,
        interval_updates: int,
    ) -> bool:
        """Return whether the completed runner update is an audit boundary."""

        if actor_version < 0:
            raise ValueError("FastWAM actor version must be non-negative.")
        if interval_updates < 1:
            raise ValueError("FastWAM Gate parameter audit interval must be positive.")
        return (int(actor_version) + 1) % int(interval_updates) == 0

    @staticmethod
    def _summarize_fastwam_gate_update(
        *,
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor],
        optimizer_steps_before: int,
        optimizer_steps_after: int,
    ) -> FastWAMGateUpdateAudit:
        """Build finite aggregate update evidence from two CPU snapshots."""

        if set(before) != set(after):
            raise RuntimeError("FastWAM Gate parameter names changed during training.")
        before_square_sum = 0.0
        update_square_sum = 0.0
        update_max_abs = 0.0
        finite_count = 0
        nonfinite_count = 0
        parameter_count = 0
        for name in sorted(before):
            before_tensor = before[name]
            after_tensor = after[name]
            if (
                before_tensor.shape != after_tensor.shape
                or before_tensor.dtype != after_tensor.dtype
            ):
                raise RuntimeError(f"FastWAM Gate tensor {name!r} changed metadata.")
            before_float = before_tensor.float()
            update = after_tensor.float() - before_float
            finite = torch.isfinite(update)
            finite_count += int(finite.sum().item())
            nonfinite_count += int((~finite).sum().item())
            parameter_count += int(update.numel())
            if bool(finite.any().item()):
                finite_update = update[finite]
                update_square_sum += float(
                    finite_update.square().to(torch.float64).sum().item()
                )
                update_max_abs = max(
                    update_max_abs,
                    float(finite_update.abs().max().item()),
                )
            before_finite = before_float[torch.isfinite(before_float)]
            before_square_sum += float(
                before_finite.square().to(torch.float64).sum().item()
            )
        before_l2 = math.sqrt(before_square_sum)
        update_l2 = math.sqrt(update_square_sum)
        return FastWAMGateUpdateAudit(
            optimizer_steps_before=int(optimizer_steps_before),
            optimizer_steps_after=int(optimizer_steps_after),
            tensor_count=len(before),
            parameter_count=parameter_count,
            before_sha256=checkpoint_state_sha256(before),
            after_sha256=checkpoint_state_sha256(after),
            before_l2_norm=before_l2,
            update_l2_norm=update_l2,
            update_max_abs=update_max_abs,
            relative_update_l2_norm=(
                update_l2 / before_l2 if before_l2 > 0.0 else float("inf")
            ),
            finite_update_count=finite_count,
            nonfinite_update_count=nonfinite_count,
        )

    @staticmethod
    def _checkpoint_cpu_clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: EmbodiedFSDPActor._checkpoint_cpu_clone(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [EmbodiedFSDPActor._checkpoint_cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(
                EmbodiedFSDPActor._checkpoint_cpu_clone(item) for item in value
            )
        return value

    def _fastwam_checkpoint_contract(self) -> dict:
        return build_fastwam_checkpoint_contract(
            self.cfg,
            world_size=int(self._world_size),
        )

    def bootstrap_fastwam_uncond_lora(
        self,
        sidecar_path: str,
        expected_sidecar_sha256: str,
    ) -> dict:
        """Load one trained UNCOND LoRA before a pristine RL step-zero save."""

        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is not SupportedModel.FASTWAM_ADAPTIVE
        ):
            raise ValueError("BC LoRA bootstrap requires fastwam_adaptive.")
        runner_cfg = getattr(self.cfg, "runner", None)
        if getattr(runner_cfg, "resume_dir", None) is not None:
            raise ValueError("BC LoRA bootstrap is forbidden together with resume.")

        policy = self._fastwam_policy_module()
        if int(self.version) != 0 or int(policy.actor_version) != 0:
            raise ValueError(
                "BC LoRA bootstrap requires RL step/version and actor_version 0."
            )
        if int(self.optimizer_steps) != 0:
            raise ValueError("BC LoRA bootstrap requires optimizer_steps == 0.")

        expected_hash = _validate_sha256(
            expected_sidecar_sha256,
            name="Expected BC LoRA sidecar SHA-256",
        )
        resolved_path = os.path.abspath(os.path.expanduser(str(sidecar_path)))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"BC LoRA sidecar does not exist: {resolved_path}")
        actual_hash = _fastwam_sha256_file(resolved_path)
        if actual_hash != expected_hash:
            raise ValueError(
                "BC LoRA sidecar hash mismatch: "
                f"expected {expected_hash}, got {actual_hash}."
            )

        adapter = getattr(policy, "lora_adapter", None)
        if adapter is None:
            raise TypeError("FastWAM adaptive policy has no LoRA adapter.")
        previous_lora = adapter.lora_state_dict()
        try:
            sidecar_payload = torch.load(
                resolved_path,
                map_location="cpu",
                weights_only=True,
            )
            expected_lora = sidecar_payload.get("state_dict")
            if not isinstance(expected_lora, dict) or not expected_lora:
                raise TypeError("BC LoRA sidecar has no tensor state_dict.")
            metadata = adapter.load_sidecar(
                resolved_path,
                expected_parent_checkpoint_sha256=str(
                    self.cfg.actor.model.actor_checkpoint_sha256
                ).lower(),
                strict=True,
            )
            extra = metadata.get("extra") if isinstance(metadata, dict) else None
            if not isinstance(extra, dict):
                raise ValueError("BC LoRA sidecar requires mapping metadata.extra.")
            provenance = _validate_fastwam_bc_bootstrap_provenance(
                {
                    "schema": _FASTWAM_BC_BOOTSTRAP_SCHEMA,
                    "bc_step": extra.get("bc_step"),
                    "bc_config_sha256": extra.get("bc_config_sha256"),
                    "sidecar_sha256": actual_hash,
                    "parent_checkpoint_sha256": metadata.get(
                        "parent_checkpoint_sha256"
                    ),
                },
                expected_parent_checkpoint_sha256=str(
                    self.cfg.actor.model.actor_checkpoint_sha256
                ).lower(),
            )
            loaded_lora = adapter.lora_state_dict()
            if set(loaded_lora) != set(expected_lora):
                raise ValueError("BC LoRA bootstrap tensor names changed on load.")
            for name, expected_tensor in expected_lora.items():
                loaded_tensor = loaded_lora[name]
                if (
                    not isinstance(expected_tensor, torch.Tensor)
                    or loaded_tensor.shape != expected_tensor.shape
                    or loaded_tensor.dtype != expected_tensor.dtype
                    or not torch.equal(loaded_tensor.cpu(), expected_tensor.cpu())
                ):
                    raise ValueError(
                        f"BC LoRA bootstrap tensor {name!r} is not bitwise equal."
                    )
        except Exception:
            adapter.load_lora_state_dict(previous_lora, strict=True)
            raise

        self._fastwam_bc_bootstrap = provenance
        return dict(provenance)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is not SupportedModel.FASTWAM_ADAPTIVE
        ):
            return super().save_checkpoint(save_path, step)

        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)
        try:
            local_error: Exception | None = None
            try:
                policy = self._fastwam_policy_module()
                policy.set_global_step(int(step))
                self.version = int(step)
                payload = {
                    "schema": "fastwam-adaptive-rl-checkpoint-v1",
                    "step": int(step),
                    "optimizer_steps": int(self.optimizer_steps),
                    "parent_checkpoint_sha256": str(
                        self.cfg.actor.model.actor_checkpoint_sha256
                    ).lower(),
                    "critic_parent_checkpoint_sha256": (
                        critic_parent_checkpoint_sha256(self.cfg.actor.model.critic)
                    ),
                    "contract": self._fastwam_checkpoint_contract(),
                    "policy": policy.trainable_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "grad_scaler": self.grad_scaler.state_dict(),
                    "rng": get_rng_state(),
                }
                bc_bootstrap = getattr(self, "_fastwam_bc_bootstrap", None)
                if bc_bootstrap is not None:
                    payload["bc_bootstrap"] = _validate_fastwam_bc_bootstrap_provenance(
                        bc_bootstrap,
                        expected_parent_checkpoint_sha256=payload[
                            "parent_checkpoint_sha256"
                        ],
                    )
                payload = self._checkpoint_cpu_clone(payload)
                os.makedirs(save_path, exist_ok=True)
                target = os.path.join(save_path, f"rank_{self._rank}.pt")
                temporary = f"{target}.tmp"
                try:
                    torch.save(payload, temporary)
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            except Exception as error:
                local_error = error
            _raise_fastwam_collective_checkpoint_error(
                local_error,
                context="actor checkpoint save",
            )
        finally:
            if restore_weight_offload:
                self.offload_param_and_grad()
            if restore_optimizer_offload:
                self.offload_optimizer()

    def load_checkpoint(self, load_path: str) -> int | None:
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is not SupportedModel.FASTWAM_ADAPTIVE
        ):
            return super().load_checkpoint(load_path)

        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)
        fsdp_lazy_root_state = _snapshot_fastwam_fsdp_lazy_root_state(self.model)
        try:
            local_error: Exception | None = None
            loaded_version: int | None = None
            try:
                checkpoint_path = os.path.join(load_path, f"rank_{self._rank}.pt")
                payload = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                expected_keys = {
                    "schema",
                    "step",
                    "optimizer_steps",
                    "parent_checkpoint_sha256",
                    "critic_parent_checkpoint_sha256",
                    "contract",
                    "policy",
                    "optimizer",
                    "lr_scheduler",
                    "grad_scaler",
                    "rng",
                }
                checkpoint_keys = set(payload)
                allowed_keys = (expected_keys, expected_keys | {"bc_bootstrap"})
                if checkpoint_keys not in allowed_keys:
                    raise ValueError(
                        "FastWAM adaptive RL checkpoint keys changed: "
                        f"{sorted(payload)}."
                    )
                if payload.get("schema") != "fastwam-adaptive-rl-checkpoint-v1":
                    raise ValueError(
                        "Unsupported FastWAM adaptive RL checkpoint schema."
                    )
                expected_parent = str(
                    self.cfg.actor.model.actor_checkpoint_sha256
                ).lower()
                if payload.get("parent_checkpoint_sha256") != expected_parent:
                    raise ValueError(
                        "FastWAM checkpoint parent hash mismatch: "
                        f"expected {expected_parent}, got "
                        f"{payload.get('parent_checkpoint_sha256')}."
                    )
                bc_bootstrap = None
                if "bc_bootstrap" in payload:
                    bc_bootstrap = _validate_fastwam_bc_bootstrap_provenance(
                        payload["bc_bootstrap"],
                        expected_parent_checkpoint_sha256=expected_parent,
                    )
                expected_critic_parent = critic_parent_checkpoint_sha256(
                    self.cfg.actor.model.critic
                )
                if (
                    payload.get("critic_parent_checkpoint_sha256")
                    != expected_critic_parent
                ):
                    raise ValueError(
                        "critic checkpoint parent hash mismatch: "
                        f"expected {expected_critic_parent}, got "
                        f"{payload.get('critic_parent_checkpoint_sha256')}."
                    )
                expected_contract = self._fastwam_checkpoint_contract()
                resume_contract = validate_fastwam_training_checkpoint_contract(
                    payload.get("contract"),
                    expected_contract,
                    allow_n4_to_three_rollout_expansion=bool(
                        getattr(
                            self.cfg.runner,
                            "fastwam_n4_to_three_rollout_resume",
                            False,
                        )
                    ),
                    owner="actor",
                )
                if resume_contract[
                    "mode"
                ] == FASTWAM_RESUME_MODE_N4_TO_THREE_ROLLOUT and (
                    int(payload.get("step", -1)) != 100
                    or int(payload.get("optimizer_steps", -1)) != 1000
                ):
                    raise ValueError(
                        "FastWAM N=4 capacity resume requires the step-100 / "
                        "optimizer-step-1000 actor checkpoint."
                    )

                policy = self._fastwam_policy_module()
                saved_policy = payload["policy"]
                if "route_tracker" not in saved_policy:
                    raise ValueError(
                        "FastWAM actor checkpoint omits delayed-route state."
                    )
                saved_route_sha256 = checkpoint_state_sha256(
                    saved_policy["route_tracker"]
                )
                saved_rng_sha256 = checkpoint_state_sha256(payload["rng"])
                policy.load_trainable_state_dict(payload["policy"])
                self.optimizer.load_state_dict(payload["optimizer"])
                self.lr_scheduler.load_state_dict(payload["lr_scheduler"])
                self.grad_scaler.load_state_dict(payload["grad_scaler"])
                self.optimizer_steps = int(payload["optimizer_steps"])
                self.version = int(payload["step"])
                if policy.actor_version != self.version:
                    raise ValueError(
                        "FastWAM checkpoint policy version does not match its step."
                    )
                restored_route_sha256 = checkpoint_state_sha256(
                    policy.route_tracker.state_dict()
                )
                if restored_route_sha256 != saved_route_sha256:
                    raise ValueError(
                        "FastWAM actor delayed-route state changed during load."
                    )
                set_rng_state(payload["rng"])
                restored_rng_sha256 = checkpoint_state_sha256(get_rng_state())
                if restored_rng_sha256 != saved_rng_sha256:
                    raise ValueError("FastWAM actor RNG state changed during load.")
                resume_audit = {
                    "schema": FASTWAM_RESUME_AUDIT_SCHEMA,
                    "owner": "actor",
                    "rank": int(self._rank),
                    "step": int(self.version),
                    "optimizer_steps": int(self.optimizer_steps),
                    "actor_version": int(policy.actor_version),
                    "route_state_sha256": restored_route_sha256,
                    "rng_sha256": restored_rng_sha256,
                    "status": "PASS",
                }
                if resume_contract["mode"] == FASTWAM_RESUME_MODE_N4_TO_THREE_ROLLOUT:
                    resume_audit.update(
                        {
                            "resume_mode": resume_contract["mode"],
                            "source_world_size": resume_contract["source_world_size"],
                            "target_world_size": resume_contract["target_world_size"],
                            "source_environment_count": resume_contract[
                                "source_environment_count"
                            ],
                            "target_environment_count": resume_contract[
                                "target_environment_count"
                            ],
                            "route_state_mode": "source_exact",
                            "rng_mode": "source_exact",
                        }
                    )
                print(
                    f"{FASTWAM_ACTOR_RESUME_AUDIT_SENTINEL} "
                    + json.dumps(resume_audit, sort_keys=True),
                    flush=True,
                )
                self._fastwam_bc_bootstrap = bc_bootstrap
                loaded_version = self.version
            except Exception as error:
                local_error = error
            _raise_fastwam_collective_checkpoint_error(
                local_error,
                context="actor checkpoint load",
            )
            if loaded_version is None:
                raise RuntimeError("FastWAM actor checkpoint load returned no version.")
            return loaded_version
        finally:
            _restore_fastwam_fsdp_lazy_root_state(fsdp_lazy_root_state)
            if restore_weight_offload:
                self.offload_param_and_grad()
            if restore_optimizer_offload:
                self.offload_optimizer()

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        if self._uses_fastwam_handle_replay():
            self._initialize_fastwam_fsdp_for_handle_replay()
        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        self._release_consumed_rollout_batch_before_receive()
        clear_memory(sync=False)

        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        logical_env_world_size = env_world_size * self.stage_num
        routes = CommMapper.get_src_ranks(
            batch_size=int(self.cfg.env.train.total_num_envs),
            src_world_size=logical_env_world_size,
            dst_world_size=actor_world_size,
            dst_rank=self._rank,
        )
        works = [
            input_channel.get(
                key=CommMapper.build_channel_key(
                    logical_env_rank,
                    self._rank,
                    ACTOR_TRAJECTORY_CHANNEL_TAG,
                ),
                async_op=True,
            )
            for logical_env_rank, _ in routes
        ]
        recv_list: list[Trajectory] = [await work.async_wait() for work in works]

        self.rollout_batch = convert_trajectories_to_batch(recv_list, consume=True)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _release_consumed_rollout_batch_before_receive(self) -> None:
        """Optional scheme hook before the next rollout transfer starts."""

        return None

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type is SupportedModel.FASTWAM_ADAPTIVE:
            # Packed K/V layer indices are static schema metadata, not a batch
            # tensor. Trajectory concatenation cannot preserve that distinction,
            # so training reconstructs them from GateKVMetadata instead.
            rollout_batch.get("forward_inputs", {}).pop("gate_kv_layer_indices", None)

        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def _fastwam_effective_idm_cost(self, cost_cfg: Any) -> float:
        """Return the effective IDM cost through the legacy scalar interface."""

        return self._fastwam_effective_branch_costs(cost_cfg)[0]

    def _fastwam_effective_branch_costs(self, cost_cfg: Any) -> tuple[float, float]:
        """Return runner-published branch costs or unchanged configured values."""

        configured = float(cost_cfg.get("idm_cost", 0.0))
        configured_uncond = float(cost_cfg.get("uncond_cost", 0.0))
        fair_cost = cost_cfg.get("fair_cost", {})
        runtime_control_enabled = cost_cfg.get("controller") is not None or bool(
            fair_cost.get("enabled", False)
        )
        if not runtime_control_enabled:
            return configured, configured_uncond
        runtime_step = getattr(self, "_fastwam_runtime_branch_cost_step", None)
        if runtime_step != int(self.version):
            raise RuntimeError(
                "FastWAM branch costs were not published for the current runner "
                f"step {self.version}."
            )
        runtime_cost = float(getattr(self, "_fastwam_runtime_idm_cost", math.nan))
        runtime_uncond = float(getattr(self, "_fastwam_runtime_uncond_cost", math.nan))
        if any(
            not math.isfinite(cost) or cost < 0.0
            for cost in (runtime_cost, runtime_uncond)
        ):
            raise ValueError("FastWAM runtime branch costs must be non-negative.")
        return runtime_cost, runtime_uncond

    @staticmethod
    def _fastwam_charge_scope(cost_cfg: Any) -> str:
        controller = cost_cfg.get("controller")
        if controller is None:
            return "all_valid_idm"
        return str(controller.get("charge_scope", "all_valid_idm")).lower()

    @staticmethod
    def _fastwam_charge_mask(
        *,
        charge_scope: str,
        route: Any,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if charge_scope == "all_valid_idm":
            return None
        if charge_scope not in {"eligible_nonforced_idm", "eligible_nonforced"}:
            raise ValueError(f"Unsupported FastWAM charge scope {charge_scope!r}.")
        if valid_mask is None:
            valid_chunks = torch.ones_like(
                route.route_was_forced,
                dtype=torch.bool,
            )
        elif valid_mask.ndim == route.route_was_forced.ndim:
            valid_chunks = valid_mask
        else:
            valid_chunks = valid_mask.reshape(*route.shape, -1).any(dim=-1)
        return valid_chunks & ~route.route_was_forced

    def _align_fastwam_training_advantages(self, **kwargs):
        """Extension point for route contracts; legacy keeps delayed alignment."""

        return align_fastwam_policy_advantages(**kwargs)

    def _summarize_fastwam_rollout_state(self, **kwargs):
        """Extension point for replay-specific rollout audit semantics."""

        return summarize_fastwam_rollout_state(**kwargs)

    def _summarize_fastwam_counterfactual_costs(self, **kwargs):
        """Extension point for route-contract-specific cost diagnostics."""

        return summarize_fastwam_counterfactual_costs(**kwargs)

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if self.cfg.algorithm.adv_type == "opd":
            self.compute_opd_teacher_logprobs()

        reward_audit = None
        cost_audit = None
        counterfactual_cost_audit = None
        rollout_state_audit = None
        decision_telemetry_count = 0
        decision_telemetry_enabled = bool(
            self.cfg.actor.model.get("decision_telemetry_enabled", False)
        )
        if model_type is SupportedModel.FASTWAM_ADAPTIVE:
            if self.cfg.algorithm.reward_type != "chunk_level":
                raise ValueError(
                    "FastWAM fixed-route rewards require reward_type=chunk_level."
                )
            if "route_info" not in self.rollout_batch:
                raise KeyError("FastWAM advantage computation requires route_info.")
            if "emitted_gate" not in self.rollout_batch:
                raise KeyError("FastWAM advantage computation requires emitted_gate.")
            short_canary_guard = bool(
                self.cfg.runner.get(
                    "short_rl_canary_require_success_signal",
                    False,
                )
            )
            training_guard = self.cfg.runner.get("fastwam_training_guard", {})
            scientific_guard = bool(training_guard.get("enabled", False))
            audit_enabled = short_canary_guard or scientific_guard
            cost_audit_cfg = training_guard.get("cost_audit", {})
            cost_audit_enabled = scientific_guard and bool(
                cost_audit_cfg.get("enabled", False)
            )
            raw_environment_rewards = self.rollout_batch["rewards"]
            if audit_enabled:
                reward_audit = summarize_fastwam_environment_rewards(
                    environment_rewards=raw_environment_rewards,
                    route_used=self.rollout_batch["route_info"].route_used,
                    valid_mask=self.rollout_batch.get("loss_mask", None),
                )
                print(
                    f"{FASTWAM_REWARD_AUDIT_SENTINEL} "
                    + json.dumps(reward_audit.to_artifact(), sort_keys=True),
                    flush=True,
                )
                if short_canary_guard:
                    reward_audit.require_success_signal()
            cost_cfg = self.cfg.algorithm.get("fixed_branch_cost", {})
            configured_idm_cost, configured_uncond_cost = (
                self._fastwam_effective_branch_costs(cost_cfg)
            )
            charge_scope = self._fastwam_charge_scope(cost_cfg)
            charge_mask = self._fastwam_charge_mask(
                charge_scope=charge_scope,
                route=self.rollout_batch["route_info"],
                valid_mask=self.rollout_batch.get("loss_mask", None),
            )
            if bool(cost_cfg.get("enabled", False)):
                if "fastwam_branch_costs" in self.rollout_batch:
                    raise RuntimeError("FastWAM branch cost was already applied.")
                cost_result = apply_fastwam_chunk_cost(
                    environment_rewards=raw_environment_rewards,
                    route_used=self.rollout_batch["route_info"].route_used,
                    idm_cost=configured_idm_cost,
                    uncond_cost=configured_uncond_cost,
                    valid_mask=self.rollout_batch.get("loss_mask", None),
                    charge_mask=charge_mask,
                )
                self.rollout_batch["rewards"] = cost_result.rewards
                self.rollout_batch["fastwam_branch_costs"] = cost_result.costs
                if cost_audit_enabled:
                    cost_audit = summarize_fastwam_chunk_cost(
                        environment_rewards=raw_environment_rewards,
                        route=self.rollout_batch["route_info"],
                        cost_result=cost_result,
                        idm_cost=configured_idm_cost,
                        uncond_cost=configured_uncond_cost,
                        valid_mask=self.rollout_batch.get("loss_mask", None),
                        charge_mask=charge_mask,
                        charge_scope=charge_scope,
                    )
                    print(
                        f"{FASTWAM_CHUNK_COST_AUDIT_SENTINEL} "
                        + json.dumps(cost_audit.to_artifact(), sort_keys=True),
                        flush=True,
                    )
            elif cost_audit_enabled:
                raise ValueError(
                    "FastWAM cost audit requires fixed_branch_cost.enabled=true."
                )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "prev_logprobs": self.rollout_batch.get("prev_logprobs", None),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": bool(
                self.cfg.algorithm.get("normalize_advantages", True)
            ),
        }
        normalization_statistics = None
        normalization_std_floor = self.cfg.algorithm.get(
            "advantage_normalization_std_floor", None
        )
        if (
            model_type is SupportedModel.FASTWAM_ADAPTIVE
            and normalization_std_floor is not None
        ):
            normalization_statistics = {}
            kwargs.update(
                {
                    "normalization_std_floor": float(normalization_std_floor),
                    "normalization_statistics": normalization_statistics,
                }
            )

        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        if normalization_statistics is not None:
            if "floor_hit_fraction" not in normalization_statistics:
                raise RuntimeError(
                    "FastWAM advantage normalization did not report floor usage."
                )
            self._fastwam_advantage_normalization_statistics = dict(
                normalization_statistics
            )

        self.rollout_batch.update(advantages_and_returns)
        if model_type is SupportedModel.FASTWAM_ADAPTIVE:
            alignment = self._align_fastwam_training_advantages(
                advantages=advantages_and_returns["advantages"],
                route=self.rollout_batch["route_info"],
                emitted=self.rollout_batch["emitted_gate"],
                dones=self.rollout_batch["dones"],
                rollout_epoch=int(self.cfg.env.train.rollout_epoch),
                carry_pending_across_epochs=bool(self.cfg.env.train.auto_reset),
                loss_mask=self.rollout_batch.get("loss_mask", None),
            )
            self.rollout_batch.update(
                {
                    "flow_advantages": alignment.flow_advantages,
                    "flow_valid_mask": alignment.flow_valid_mask,
                    "gate_advantages": alignment.gate_advantages,
                    "gate_valid_mask": alignment.gate_valid_mask,
                }
            )
            if decision_telemetry_enabled:
                if bool(cost_cfg.get("enabled", False)):
                    configured_rewards = self.rollout_batch["rewards"]
                    telemetry_idm_cost = configured_idm_cost
                else:
                    configured_rewards = raw_environment_rewards.sum(
                        dim=-1, keepdim=True
                    )
                    telemetry_idm_cost = 0.0
                unnormalized_alignment = compute_fastwam_unnormalized_gate_alignment(
                    rewards=configured_rewards,
                    route=self.rollout_batch["route_info"],
                    emitted=self.rollout_batch["emitted_gate"],
                    dones=self.rollout_batch["dones"],
                    values=self.rollout_batch["prev_values"],
                    valid_mask=self.rollout_batch.get("loss_mask", None),
                    gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
                    gae_lambda=float(self.cfg.algorithm.get("gae_lambda", 1.0)),
                    rollout_epoch=int(self.cfg.env.train.rollout_epoch),
                    carry_pending_across_epochs=bool(self.cfg.env.train.auto_reset),
                )
                if not torch.equal(
                    unnormalized_alignment.gate_valid_mask,
                    alignment.gate_valid_mask,
                ):
                    raise ValueError(
                        "Normalized and unnormalized Gate telemetry eligibility "
                        "disagree."
                    )
                decision_records = build_fastwam_training_decision_records(
                    emitted=self.rollout_batch["emitted_gate"],
                    gate_valid_mask=alignment.gate_valid_mask,
                    unnormalized_gate_advantages=(
                        unnormalized_alignment.gate_advantages
                    ),
                    normalized_gate_advantages=alignment.gate_advantages,
                    runner_step=int(self.version),
                    rank=int(self._rank),
                    run_id=str(self.cfg.runner.logger.experiment_name),
                    task_suite=str(
                        self.cfg.env.train.get(
                            "task_suite_name",
                            self.cfg.env.train.get("env_type", "unknown"),
                        )
                    ),
                    configured_idm_cost=telemetry_idm_cost,
                )
                decision_path = (
                    Path(str(self.cfg.runner.logger.log_path))
                    / str(self.cfg.runner.logger.experiment_name)
                    / f"audits/training_decisions.rank-{self._rank}.jsonl"
                )
                append_fastwam_decision_telemetry_jsonl(
                    decision_path,
                    decision_records,
                )
                decision_telemetry_count = len(decision_records)
            if cost_audit_enabled:
                counterfactual_idm_costs = [
                    float(item)
                    for item in cost_audit_cfg.get("counterfactual_idm_costs", [])
                ]
                if cost_cfg.get("controller") is not None or bool(
                    cost_cfg.get("fair_cost", {}).get("enabled", False)
                ):
                    counterfactual_idm_costs.append(configured_idm_cost)
                counterfactual_cost_audit = (
                    self._summarize_fastwam_counterfactual_costs(
                        environment_rewards=raw_environment_rewards,
                        route=self.rollout_batch["route_info"],
                        emitted=self.rollout_batch["emitted_gate"],
                        dones=self.rollout_batch["dones"],
                        values=self.rollout_batch["prev_values"],
                        valid_mask=self.rollout_batch.get("loss_mask", None),
                        charge_mask=charge_mask,
                        idm_costs=tuple(sorted(set(counterfactual_idm_costs))),
                        configured_idm_cost=configured_idm_cost,
                        uncond_cost=configured_uncond_cost,
                        configured_gate_advantages=alignment.gate_advantages,
                        gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
                        gae_lambda=float(self.cfg.algorithm.get("gae_lambda", 1.0)),
                        rollout_epoch=int(self.cfg.env.train.rollout_epoch),
                        carry_pending_across_epochs=bool(self.cfg.env.train.auto_reset),
                    )
                )
                counterfactual_artifact = counterfactual_cost_audit.to_artifact()
                print(
                    f"{FASTWAM_COUNTERFACTUAL_COST_AUDIT_SENTINEL} "
                    + json.dumps(counterfactual_artifact, sort_keys=True),
                    flush=True,
                )
                if self._rank == 0:
                    audit_path = (
                        Path(str(self.cfg.runner.logger.log_path))
                        / str(self.cfg.runner.logger.experiment_name)
                        / "audits/counterfactual_cost_audit.jsonl"
                    )
                    append_fastwam_counterfactual_cost_audit_jsonl(
                        audit_path,
                        runner_step=int(self.version),
                        artifact=counterfactual_artifact,
                    )
            if audit_enabled:
                kv_cfg = self.cfg.actor.model.kv_replay
                rollout_state_audit = self._summarize_fastwam_rollout_state(
                    route=self.rollout_batch["route_info"],
                    emitted=self.rollout_batch["emitted_gate"],
                    eligible_gate_mask=alignment.gate_valid_mask,
                    valid_mask=self.rollout_batch.get("loss_mask", None),
                    kv_replay_backend=str(kv_cfg.backend),
                    max_bytes_per_sample=kv_cfg.get("max_bytes_per_sample", None),
                )
                print(
                    f"{FASTWAM_ROLLOUT_STATE_AUDIT_SENTINEL} "
                    + json.dumps(rollout_state_audit.to_artifact(), sort_keys=True),
                    flush=True,
                )
            formal_execution_profile = self.cfg.runner.get(
                "formal_execution_profile", None
            )
            if formal_execution_profile is not None:
                route = self.rollout_batch["route_info"]
                emitted = self.rollout_batch["emitted_gate"]
                kv_metadata = emitted.kv_metadata
                rollout_state = {
                    "actions": self.rollout_batch.get("actions"),
                    "raw_environment_rewards": raw_environment_rewards,
                    "dones": self.rollout_batch.get("dones"),
                    "terminations": self.rollout_batch.get("terminations"),
                    "truncations": self.rollout_batch.get("truncations"),
                    "prev_logprobs": self.rollout_batch.get("prev_logprobs"),
                    "prev_values": self.rollout_batch.get("prev_values"),
                    "versions": self.rollout_batch.get("versions"),
                    "loss_mask": self.rollout_batch.get("loss_mask"),
                    "denoise_indices": self.rollout_batch.get("forward_inputs", {}).get(
                        "denoise_indices"
                    ),
                    "route": {
                        name: getattr(route, name)
                        for name in (
                            "route_used",
                            "route_was_forced",
                            "chunk_ids",
                            "episode_ids",
                            "route_source_chunk_ids",
                            "actor_versions",
                        )
                    },
                    "emitted_gate": {
                        name: getattr(emitted, name)
                        for name in (
                            "next_route",
                            "base_probability",
                            "behavior_probability",
                            "old_logprob",
                            "epsilon",
                            "temperature",
                            "valid",
                            "source_chunk_ids",
                            "episode_ids",
                            "actor_versions",
                            "exploration_forced",
                            "mode_flip_delta",
                            "environment_ids",
                            "task_ids",
                            "trial_ids",
                            "reset_state_ids",
                        )
                    },
                    "kv_metadata": (
                        None
                        if kv_metadata is None
                        else {
                            "layer_indices": kv_metadata.layer_indices,
                            "denoise_timesteps": kv_metadata.denoise_timesteps,
                            "total_bytes": kv_metadata.total_bytes,
                            "storage_dtype": kv_metadata.storage_dtype,
                            "tensor_shapes": kv_metadata.tensor_shapes,
                            "payload_reference_ids": (
                                kv_metadata.payload_reference_ids
                            ),
                        }
                    ),
                }
                gae_state = {
                    name: self.rollout_batch.get(name)
                    for name in (
                        "advantages",
                        "returns",
                        "flow_advantages",
                        "flow_valid_mask",
                        "gate_advantages",
                        "gate_valid_mask",
                        "loss_mask",
                        "loss_mask_sum",
                    )
                }
                semantics_audit = {
                    "schema": "fastwam-acceleration-semantics-audit-v1",
                    "runner_step": int(self.version),
                    "execution_profile_sha256": str(
                        formal_execution_profile.sha256
                    ).lower(),
                    "rollout_sha256": checkpoint_state_sha256(rollout_state),
                    "gae_sha256": checkpoint_state_sha256(gae_state),
                    "sample_coverage": {
                        "valid_chunk_count": int(
                            self.rollout_batch["loss_mask"].sum().item()
                        ),
                        "gate_sample_count": int(
                            alignment.gate_valid_mask.sum().item()
                        ),
                        "uncond_flow_sample_count": int(
                            alignment.flow_valid_mask.sum().item()
                        ),
                    },
                }
                print(
                    f"{FASTWAM_ACCELERATION_SEMANTICS_AUDIT_SENTINEL} "
                    + json.dumps(semantics_audit, sort_keys=True),
                    flush=True,
                )
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        if reward_audit is not None:
            rollout_metrics.update(reward_audit.to_metrics())
        if rollout_state_audit is not None:
            rollout_metrics.update(rollout_state_audit.to_metrics())
        if cost_audit is not None:
            rollout_metrics.update(cost_audit.to_metrics())
        if counterfactual_cost_audit is not None:
            rollout_metrics.update(counterfactual_cost_audit.to_metrics())
        if decision_telemetry_count:
            rollout_metrics["fastwam/decision_telemetry/count"] = float(
                decision_telemetry_count
            )
        return rollout_metrics

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    def _optimizer_metrics(
        self,
        grad_norm: float,
        lr_list: list[float],
    ) -> dict[str, float]:
        data = {"actor/grad_norm": grad_norm}
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is SupportedModel.FASTWAM_ADAPTIVE
        ):
            lr_by_name = {
                group.get("name"): float(group["lr"])
                for group in self.optimizer.param_groups
            }
            expected = {"gate", "uncond_lora", "value_head"}
            if set(lr_by_name) != expected:
                raise RuntimeError(
                    "FastWAM adaptive optimizer groups changed unexpectedly: "
                    f"{sorted(lr_by_name)}"
                )
            gradient_norms = dict(self._fastwam_last_gradient_norms)
            if set(gradient_norms) != expected:
                raise RuntimeError(
                    "FastWAM adaptive gradient-norm groups changed unexpectedly: "
                    f"{sorted(gradient_norms)}"
                )
            data.update(
                {
                    "gate/lr": lr_by_name["gate"],
                    "uncond_flow/lora_lr": lr_by_name["uncond_lora"],
                    "critic/lr": lr_by_name["value_head"],
                    "gate/grad_norm": gradient_norms["gate"],
                    "uncond_lora/grad_norm": gradient_norms["uncond_lora"],
                    "value_head/grad_norm": gradient_norms["value_head"],
                }
            )
            return data
        data["actor/lr"] = lr_list[0]
        if len(lr_list) > 1:
            data["critic/lr"] = lr_list[1]
        return data

    def _uses_fastwam_handle_replay(self) -> bool:
        return (
            SupportedModel(self.cfg.actor.model.model_type)
            is SupportedModel.FASTWAM_ADAPTIVE
            and str(self.cfg.actor.model.kv_replay.backend) == "stored"
        )

    def _initialize_fastwam_fsdp_for_handle_replay(self) -> None:
        """Finish FSDP lazy init before the asynchronous K/V CUDA stream starts.

        Classic FSDP initializes its flat-parameter runtime storage on the
        first forward.  Handle replay starts a dedicated CUDA prefetch stream
        before that forward, so initialize the hierarchy synchronously and
        restore the ``use_orig_params`` views against the resulting flat
        storage first.  Rebinding preserves the original Parameter identities
        held by the optimizer.
        """

        if getattr(self, "_fastwam_handle_replay_fsdp_initialized", False):
            return
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        if not isinstance(self.model, FullyShardedDataParallel):
            raise TypeError("FastWAM handle replay requires a classic FSDP actor.")
        _lazy_init(self.model, self.model)
        handles = self._strategy._iter_fsdp_handles(self.model)
        if not handles:
            raise RuntimeError("FastWAM handle replay found no FSDP handles.")
        for handle in handles:
            self._strategy._rebind_handle_views(handle)
        self._fastwam_handle_replay_fsdp_initialized = True

    def _restore_fastwam_fsdp_parameter_views_after_backward(self) -> int:
        """Restore original Parameters after a conditionally unused handle.

        Classic FSDP exposes autograd-tracked Tensor views during backward.
        When a whole ``NO_SHARD`` handle is unused, PyTorch 2.7 returns before
        restoring the original ``use_orig_params`` Parameter objects.  The
        next forward then mistakes those shaped views for externally replaced
        parameters.  Rebind only that exact same-storage Tensor-view state and
        fail closed for any genuine storage or Parameter replacement.

        Returns:
            Number of FSDP handles whose original Parameter views were restored.
        """

        restored_handles = 0
        for handle in self._strategy._iter_fsdp_handles(self.model):
            flat_param = handle.flat_param
            flat_storage = flat_param.untyped_storage().data_ptr()
            restore_handle = False
            for parameter, shard_info, (name, owner, module_name) in zip(
                flat_param._params,
                flat_param._shard_param_infos,
                flat_param._param_infos,
                strict=True,
            ):
                if not shard_info.in_shard:
                    continue
                current = getattr(owner, name, None)
                if current is parameter:
                    if parameter.untyped_storage().data_ptr() != flat_storage:
                        raise RuntimeError(
                            "FastWAM FSDP Parameter detached from flat storage "
                            f"after backward: {module_name}.{name}."
                        )
                    continue
                recoverable_view = (
                    not handle.uses_sharded_strategy
                    and isinstance(current, torch.Tensor)
                    and not isinstance(current, nn.Parameter)
                    and current.shape == parameter.shape
                    and current.dtype == parameter.dtype
                    and current.device == parameter.device
                    and current.untyped_storage().data_ptr() == flat_storage
                    and parameter.untyped_storage().data_ptr() == flat_storage
                )
                if not recoverable_view:
                    raise RuntimeError(
                        "FastWAM FSDP observed a non-recoverable parameter "
                        "replacement after backward: "
                        f"{module_name}.{name}, current={type(current).__qualname__}, "
                        f"shape={getattr(current, 'shape', None)}."
                    )
                restore_handle = True
            if restore_handle:
                self._strategy._rebind_handle_views(handle)
                restored_handles += 1
        return restored_handles

    def _next_fastwam_kv_request_id(self) -> int:
        request_id = self._fastwam_kv_request_id
        self._fastwam_kv_request_id += 1
        return request_id

    def _post_fastwam_kv_request(
        self,
        *,
        source_rank: int,
        command: str,
        handles: tuple[int, ...],
        expect_response: bool,
    ):
        from rlinf.models.embodiment.wam_policy.tiered_kv_store import (
            GateKVStoreRequest,
            gate_kv_request_key,
            gate_kv_response_key,
        )

        if self._fastwam_kv_request_channel is None:
            raise RuntimeError("Gate K/V request channel is not configured.")
        request_id = self._next_fastwam_kv_request_id()
        response_work = None
        if expect_response:
            if self._fastwam_kv_response_channel is None:
                raise RuntimeError("Gate K/V response channel is not configured.")
            response_work = self._fastwam_kv_response_channel.get(
                key=gate_kv_response_key(
                    actor_rank=self._rank,
                    request_id=request_id,
                ),
                async_op=True,
            )
        request = GateKVStoreRequest(
            command=command,
            actor_rank=self._rank,
            request_id=request_id,
            handles=handles,
        )
        put_work = self._fastwam_kv_request_channel.put_via_ray(
            request,
            key=gate_kv_request_key(source_rank),
            async_op=True,
        )
        if put_work is not None:
            put_work.wait()
        return response_work

    @staticmethod
    def _group_fastwam_handles(handles: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
        from rlinf.models.embodiment.wam_policy.tiered_kv_store import (
            decode_gate_kv_handle,
        )

        grouped: dict[int, list[int]] = {}
        for handle in handles:
            source_rank, _, _ = decode_gate_kv_handle(handle)
            grouped.setdefault(source_rank, []).append(handle)
        return {rank: tuple(values) for rank, values in grouped.items()}

    def _start_fastwam_handle_replay(
        self,
        *,
        request_channel: Channel,
        response_channel: Channel,
        update_epoch: int,
        episode_contributions: list[dict[str, int]],
    ) -> None:
        emitted = self.rollout_batch.get("emitted_gate")
        metadata = None if emitted is None else emitted.kv_metadata
        if metadata is None or metadata.payload_reference_ids is None:
            raise ValueError(
                "Stored Gate replay trajectories must carry payload references."
            )
        gate_valid = self.rollout_batch.get("gate_valid_mask")
        if (
            gate_valid is None
            or gate_valid.shape != metadata.payload_reference_ids.shape
        ):
            raise ValueError(
                "Gate-valid mask and K/V payload references must have equal shape."
            )
        references = metadata.payload_reference_ids
        sample_mask = references >= 0
        self.rollout_batch["gate_kv_sample_mask"] = sample_mask
        all_handles = tuple(
            int(value) for value in references[sample_mask].detach().cpu().reshape(-1)
        )
        effective_gate_valid = fastwam_effective_gate_kv_mask(gate_valid, sample_mask)
        eligible_handles = tuple(
            int(value)
            for value in references[effective_gate_valid].detach().cpu().reshape(-1)
        )
        if len(set(all_handles)) != len(all_handles):
            raise ValueError(
                "One rollout update repeated a Gate K/V payload reference."
            )
        self._fastwam_kv_request_channel = request_channel
        self._fastwam_kv_response_channel = response_channel
        self._fastwam_kv_prefetch_wait_seconds = 0.0
        self._fastwam_kv_h2d_bytes = 0
        self._fastwam_kv_h2d_events = []
        self._fastwam_kv_use_counts = {
            handle: int(update_epoch) for handle in eligible_handles
        }
        self._fastwam_kv_all_handles = all_handles
        emitted_count = int(references.numel())
        sampled_count = int(sample_mask.sum().item())
        configured_budget = self.cfg.actor.model.kv_replay.get(
            "gate_kv_sample_budget", None
        )
        expected_sampled_count = (
            emitted_count
            if configured_budget is None
            else min(int(configured_budget), emitted_count)
        )
        if sampled_count != expected_sampled_count:
            raise RuntimeError(
                "Global Gate K/V sample count disagrees with its budget: "
                f"{sampled_count} != {expected_sampled_count}."
            )
        self._fastwam_gate_kv_sample_probability = (
            sampled_count / emitted_count if emitted_count else 1.0
        )
        full_eligible_count = int(gate_valid.bool().sum().item())
        effective_gate_gradient_count = int(effective_gate_valid.sum().item())
        candidate_eligibility_rate = (
            effective_gate_gradient_count / sampled_count if sampled_count else 0.0
        )
        candidate_to_effective_gap_fraction = 1.0 - candidate_eligibility_rate
        recommended_candidate_budget = configured_budget
        if (
            configured_budget is not None
            and candidate_to_effective_gap_fraction > 0.05
            and candidate_eligibility_rate > 0.0
        ):
            recommended_candidate_budget = math.ceil(
                int(configured_budget) / candidate_eligibility_rate
            )
        sample_audit = {
            "schema": "fastwam-gate-kv-sample-audit-v1",
            "actor_version": int(self.version),
            "configured_budget": configured_budget,
            "configured_seed": int(
                self.cfg.actor.model.kv_replay.get("gate_kv_sample_seed", 0)
            ),
            "emitted_candidate_count": emitted_count,
            "sampled_kv_count": sampled_count,
            "actual_sample_rate": self._fastwam_gate_kv_sample_probability,
            "full_eligible_gate_count": full_eligible_count,
            "sampled_eligible_gate_count": effective_gate_gradient_count,
            "effective_gate_gradient_count": effective_gate_gradient_count,
            "candidate_eligibility_rate": candidate_eligibility_rate,
            "candidate_to_effective_gap_fraction": (
                candidate_to_effective_gap_fraction
            ),
            "recommended_candidate_budget": recommended_candidate_budget,
            "episode_contributions": episode_contributions,
        }
        print(
            f"{FASTWAM_GATE_KV_SAMPLE_AUDIT_SENTINEL} "
            + json.dumps(sample_audit, sort_keys=True),
            flush=True,
        )
        self._fastwam_gate_kv_sampling_metrics = {
            "kv_cache/sampled_kv_samples": float(sampled_count),
            "kv_cache/emitted_kv_candidates": float(emitted_count),
            "kv_cache/actual_sample_rate": float(
                self._fastwam_gate_kv_sample_probability
            ),
            "kv_cache/full_eligible_gate_samples": float(full_eligible_count),
            "kv_cache/sampled_eligible_gate_samples": float(
                effective_gate_gradient_count
            ),
            "kv_cache/effective_gate_gradient_count": float(
                effective_gate_gradient_count
            ),
            "kv_cache/candidate_eligibility_rate": float(candidate_eligibility_rate),
            "kv_cache/candidate_to_effective_gap_fraction": float(
                candidate_to_effective_gap_fraction
            ),
        }
        if configured_budget is not None:
            self._fastwam_gate_kv_sampling_metrics["kv_cache/configured_budget"] = (
                float(configured_budget)
            )
            self._fastwam_gate_kv_sampling_metrics[
                "kv_cache/recommended_candidate_budget"
            ] = float(recommended_candidate_budget)
        self._fastwam_kv_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"gate-kv-prefetch-{self._rank}",
        )
        if torch.cuda.is_available():
            self._fastwam_kv_prefetch_stream = torch.cuda.Stream(device=self.device)

        grouped = self._group_fastwam_handles(eligible_handles)
        responses = []
        for source_rank in self._rollout_all_ranks:
            responses.append(
                self._post_fastwam_kv_request(
                    source_rank=source_rank,
                    command="retain",
                    handles=grouped.get(source_rank, ()),
                    expect_response=True,
                )
            )
        retained = 0
        for response_work in responses:
            response = response_work.wait()
            retained += int(response["retained"])
        if retained != len(eligible_handles):
            raise RuntimeError(
                "Rollout stores retained a different eligible K/V count: "
                f"{retained} != {len(eligible_handles)}."
            )

    @staticmethod
    def _pin_fastwam_response_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        if tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def _finish_fastwam_kv_prefetch(
        self,
        *,
        response_works: list[Any],
        handles: tuple[int, ...],
        batch_indices: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.cuda.Event | None]:
        from rlinf.models.embodiment.wam_policy.tiered_kv_store import (
            GATE_KV_BATCH_INDICES,
            GATE_KV_FORWARD_KEYS,
            GATE_KV_RESPONSE_HANDLES,
        )

        responses = [work.wait() for work in response_works]
        locations: dict[int, tuple[dict[str, torch.Tensor], int]] = {}
        for response in responses:
            response_handles = response[GATE_KV_RESPONSE_HANDLES].tolist()
            for index, handle in enumerate(response_handles):
                locations[int(handle)] = (response, index)
        missing = [handle for handle in handles if handle not in locations]
        if missing:
            raise KeyError(f"Gate K/V prefetch omitted handles {missing[:8]}.")

        host_payload = {}
        for key in GATE_KV_FORWARD_KEYS:
            ordered = [
                locations[handle][0][key][
                    locations[handle][1] : locations[handle][1] + 1
                ]
                for handle in handles
            ]
            host_payload[key] = self._pin_fastwam_response_tensor(
                torch.cat(ordered, dim=0)
            )
        host_payload[GATE_KV_BATCH_INDICES] = batch_indices
        self._fastwam_kv_h2d_bytes += sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in host_payload.values()
        )

        if self._fastwam_kv_prefetch_stream is None:
            return {
                key: value.to(self.device) for key, value in host_payload.items()
            }, None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._fastwam_kv_prefetch_stream):
            start_event.record()
            device_payload = {
                key: value.to(
                    self.device,
                    non_blocking=value.device.type == "cpu" and value.is_pinned(),
                )
                for key, value in host_payload.items()
            }
            end_event.record()
        self._fastwam_kv_h2d_events.append((start_event, end_event))
        return device_payload, end_event

    def _schedule_fastwam_kv_prefetch(self, micro_batch: dict) -> Future:
        if self._fastwam_kv_executor is None:
            raise RuntimeError("Gate K/V prefetch executor is not running.")
        emitted = micro_batch["emitted_gate"]
        metadata = emitted.kv_metadata
        gate_valid = fastwam_effective_gate_kv_mask(
            micro_batch["gate_valid_mask"],
            micro_batch.get("gate_kv_sample_mask"),
        ).reshape(-1)
        references = metadata.payload_reference_ids.reshape(-1)
        batch_indices = gate_valid.nonzero(as_tuple=False).reshape(-1).cpu()
        handles = tuple(
            int(value) for value in references[gate_valid].detach().cpu().reshape(-1)
        )
        if not handles:
            from rlinf.models.embodiment.wam_policy.tiered_kv_store import (
                GATE_KV_BATCH_INDICES,
            )

            future: Future = Future()
            future.set_result(
                (
                    {
                        GATE_KV_BATCH_INDICES: torch.empty(
                            0,
                            dtype=torch.long,
                            device=self.device,
                        )
                    },
                    None,
                    handles,
                )
            )
            return future
        response_works = []
        for source_rank, grouped_handles in self._group_fastwam_handles(
            handles
        ).items():
            response_works.append(
                self._post_fastwam_kv_request(
                    source_rank=source_rank,
                    command="fetch",
                    handles=grouped_handles,
                    expect_response=True,
                )
            )

        def finish():
            payload, event = self._finish_fastwam_kv_prefetch(
                response_works=response_works,
                handles=handles,
                batch_indices=batch_indices,
            )
            return payload, event, handles

        return self._fastwam_kv_executor.submit(finish)

    def _consume_fastwam_kv_prefetch(
        self,
        micro_batch: dict,
        future: Future,
    ) -> tuple[int, ...]:
        wait_start = time.perf_counter()
        payload, event, handles = future.result()
        self._fastwam_kv_prefetch_wait_seconds += time.perf_counter() - wait_start
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
        forward_inputs = dict(micro_batch.get("forward_inputs", {}))
        forward_inputs.update(payload)
        micro_batch["forward_inputs"] = forward_inputs
        return handles

    def _release_consumed_fastwam_kv(self, handles: tuple[int, ...]) -> None:
        releasable: dict[int, list[int]] = {}
        for handle in handles:
            remaining = self._fastwam_kv_use_counts[handle] - 1
            self._fastwam_kv_use_counts[handle] = remaining
            if remaining == 0:
                source_rank = next(iter(self._group_fastwam_handles((handle,))))
                releasable.setdefault(source_rank, []).append(handle)
        for source_rank, values in releasable.items():
            self._post_fastwam_kv_request(
                source_rank=source_rank,
                command="release",
                handles=tuple(values),
                expect_response=False,
            )

    def _prepare_fastwam_gate_diagnostic_forward_inputs(
        self,
        *,
        micro_batch: dict,
        forward_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Allow an opt-in actor subclass to add existing replay-only inputs."""

        del micro_batch
        return forward_inputs

    def _fastwam_gate_gradient_diagnostic_config(
        self,
    ) -> tuple[tuple[int, ...], int, int] | None:
        raw = self.cfg.runner.get("fastwam_gate_gradient_diagnostic", {})
        if not bool(raw.get("enabled", False)):
            return None
        if self._world_size != 1:
            raise ValueError(
                "Gate gradient diagnostics currently require one actor rank."
            )
        if (
            self.cfg.actor.model.kv_replay.get("gate_kv_sample_budget", None)
            is not None
        ):
            raise ValueError(
                "Gate gradient diagnostics require full resident K/V replay."
            )
        sample_sizes = tuple(int(value) for value in raw.get("sample_sizes", ()))
        if (
            not sample_sizes
            or any(value < 1 for value in sample_sizes)
            or len(set(sample_sizes)) != len(sample_sizes)
        ):
            raise ValueError(
                "Gate gradient diagnostic sample_sizes must be unique positive integers."
            )
        repeats = raw.get("repeats", 0)
        seed = raw.get("seed", 0)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("Gate gradient diagnostic repeats must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "Gate gradient diagnostic seed must be a non-negative integer."
            )
        return sample_sizes, int(repeats), int(seed)

    def _fastwam_optimizer_group_parameters(
        self, group_name: str
    ) -> tuple[nn.Parameter, ...]:
        matching = [
            group
            for group in self.optimizer.param_groups
            if str(group.get("name", "")) == group_name
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"Expected one FastWAM optimizer group {group_name!r}, "
                f"got {len(matching)}."
            )
        parameters = tuple(matching[0]["params"])
        if not parameters:
            raise RuntimeError(f"FastWAM optimizer group {group_name!r} is empty.")
        return parameters

    @staticmethod
    def _capture_fastwam_gradient_vector(
        parameters: tuple[nn.Parameter, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        captured = []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                captured.append(None)
                continue
            if gradient.is_sparse:
                raise TypeError("Gate gradient diagnostics require dense gradients.")
            gradient = gradient.detach().float()
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(
                    "Gate gradient diagnostics produced a non-finite gradient."
                )
            captured.append(gradient.cpu().contiguous().clone())
        return tuple(captured)

    @staticmethod
    def _fastwam_gradient_cosine(
        reference: tuple[torch.Tensor | None, ...],
        estimate: tuple[torch.Tensor | None, ...],
    ) -> tuple[float, float, float]:
        if len(reference) != len(estimate):
            raise ValueError("Gate gradient vectors have different tensor counts.")
        dot = 0.0
        reference_square = 0.0
        estimate_square = 0.0
        for reference_tensor, estimate_tensor in zip(reference, estimate):
            if reference_tensor is not None:
                reference_square += float(
                    reference_tensor.square().sum(dtype=torch.float64).item()
                )
            if estimate_tensor is not None:
                estimate_square += float(
                    estimate_tensor.square().sum(dtype=torch.float64).item()
                )
            if reference_tensor is not None and estimate_tensor is not None:
                if reference_tensor.shape != estimate_tensor.shape:
                    raise ValueError(
                        "Gate gradient tensor shapes changed across passes."
                    )
                dot += float(
                    (reference_tensor * estimate_tensor).sum(dtype=torch.float64).item()
                )
        reference_norm = math.sqrt(reference_square)
        estimate_norm = math.sqrt(estimate_square)
        if reference_norm <= 0.0 or estimate_norm <= 0.0:
            raise RuntimeError("Gate gradient diagnostic found a zero gradient norm.")
        cosine = dot / (reference_norm * estimate_norm)
        return float(cosine), float(reference_norm), float(estimate_norm)

    def _compute_fastwam_gate_gradient_for_indices(
        self,
        indices: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor | None, ...], dict[str, int]]:
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("Gate gradient diagnostic indices must be non-empty 1-D.")
        selected_count = int(indices.numel())

        def select(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.index_select(0, indices.to(tensor.device)).contiguous()

        selected_batch = map_nested_tensors(self.rollout_batch, select)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        chunks = math.ceil(selected_count / micro_batch_size)
        micro_batches = split_dict_to_chunk(selected_batch, chunks)
        gate_parameters = self._fastwam_optimizer_group_parameters("gate")
        non_gate_parameters = self._fastwam_optimizer_group_parameters(
            "uncond_lora"
        ) + self._fastwam_optimizer_group_parameters("value_head")
        gate_cfg = self.cfg.algorithm.gate_ppo
        self.optimizer.zero_grad()
        restored_handles = 0
        pending_prefetches: deque[Future] = deque()
        next_prefetch_index = 0
        prefetch_depth = min(
            int(self.cfg.actor.model.kv_replay.prefetch_depth),
            len(micro_batches),
        )
        while next_prefetch_index < prefetch_depth:
            pending_prefetches.append(
                self._schedule_fastwam_kv_prefetch(micro_batches[next_prefetch_index])
            )
            next_prefetch_index += 1
        for index, micro_batch in enumerate(micro_batches):
            prefetch = pending_prefetches.popleft()
            if next_prefetch_index < len(micro_batches):
                pending_prefetches.append(
                    self._schedule_fastwam_kv_prefetch(
                        micro_batches[next_prefetch_index]
                    )
                )
                next_prefetch_index += 1
            self._consume_fastwam_kv_prefetch(micro_batch, prefetch)
            micro_batch = put_tensor_device(micro_batch, self.device)
            emitted_gate = micro_batch["emitted_gate"]
            forward_inputs = dict(micro_batch["forward_inputs"])
            if emitted_gate.kv_metadata is None:
                raise ValueError("Gate gradient diagnostics require K/V metadata.")
            forward_inputs["gate_kv_layer_indices"] = torch.tensor(
                emitted_gate.kv_metadata.layer_indices,
                dtype=torch.long,
                device=self.device,
            )
            forward_inputs = self._prepare_fastwam_gate_diagnostic_forward_inputs(
                micro_batch=micro_batch,
                forward_inputs=forward_inputs,
            )
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(index + 1) == len(micro_batches),
            )
            with self.amp_context:
                output_dict = self.model(
                    forward_inputs=forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=True,
                    compute_values=False,
                    use_cache=False,
                    route_info=micro_batch["route_info"],
                    emitted_gate=emitted_gate,
                    compute_base_logprobs=False,
                )
                gate_loss, _ = compute_gate_ppo_loss(
                    logprobs=output_dict["gate_logprobs"].float(),
                    old_logprobs=emitted_gate.old_logprob.float(),
                    advantages=micro_batch["gate_advantages"].float(),
                    valid_mask=micro_batch["gate_valid_mask"].bool(),
                    clip_ratio_low=float(gate_cfg.clip_ratio_low),
                    clip_ratio_high=float(gate_cfg.clip_ratio_high),
                    behavior_probabilities=output_dict[
                        "gate_behavior_probabilities"
                    ].float(),
                    entropy_coefficient=float(gate_cfg.entropy_coefficient),
                    selected_loss_scale=1.0 / selected_count,
                )
                gate_loss = float(gate_cfg.get("loss_weight", 1.0)) * gate_loss
            with backward_ctx:
                gate_loss.backward()
            restored_handles += (
                self._restore_fastwam_fsdp_parameter_views_after_backward()
            )
            micro_batches[index] = None
            del micro_batch, output_dict, gate_loss
        gradient = self._capture_fastwam_gradient_vector(gate_parameters)
        non_gate_nonzero = 0
        for parameter in non_gate_parameters:
            if parameter.grad is not None:
                values = parameter.grad.detach()
                if not bool(torch.isfinite(values).all().item()):
                    raise FloatingPointError(
                        "Gate diagnostic produced a non-finite non-Gate gradient."
                    )
                non_gate_nonzero += int(bool((values != 0).any().item()))
        self.optimizer.zero_grad()
        return gradient, {
            "selected_count": selected_count,
            "non_gate_nonzero_parameter_count": non_gate_nonzero,
            "fsdp_view_restore_handles": restored_handles,
        }

    def _run_fastwam_gate_gradient_diagnostic(
        self,
        config: tuple[tuple[int, ...], int, int],
    ) -> dict[str, float]:
        sample_sizes, repeats, base_seed = config
        effective_mask = fastwam_effective_gate_kv_mask(
            self.rollout_batch["gate_valid_mask"],
            self.rollout_batch.get("gate_kv_sample_mask"),
        ).reshape(-1)
        eligible_indices = effective_mask.nonzero(as_tuple=False).reshape(-1).cpu()
        full_count = int(eligible_indices.numel())
        if full_count < 1:
            raise ValueError(
                "Gate gradient diagnostic requires at least one effective Gate sample."
            )
        rng_state = get_rng_state()
        optimizer_steps_before = int(self.optimizer_steps)
        results = []
        metrics: dict[str, float] = {}
        try:
            full_gradient, full_metadata = (
                self._compute_fastwam_gate_gradient_for_indices(eligible_indices)
            )
            _, full_norm, _ = self._fastwam_gradient_cosine(
                full_gradient, full_gradient
            )
            if full_metadata["non_gate_nonzero_parameter_count"] != 0:
                raise RuntimeError(
                    "Full Gate diagnostic gradient reached a non-Gate parameter."
                )
            metrics["gate_gradient_curve/full_effective_count"] = float(full_count)
            metrics["gate_gradient_curve/full_gradient_norm"] = full_norm
            for sample_size_index, sample_size in enumerate(sample_sizes):
                effective_count = min(sample_size, full_count)
                reused_full_gradient = effective_count == full_count
                repeat_results = []
                for repeat in range(repeats):
                    seed = base_seed + sample_size_index * repeats + repeat
                    if reused_full_gradient:
                        cosine = 1.0
                        estimate_norm = full_norm
                        metadata = full_metadata
                    else:
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(seed)
                        selected = eligible_indices[
                            torch.randperm(full_count, generator=generator)[
                                :effective_count
                            ]
                        ]
                        estimate, metadata = (
                            self._compute_fastwam_gate_gradient_for_indices(selected)
                        )
                        cosine, _, estimate_norm = self._fastwam_gradient_cosine(
                            full_gradient, estimate
                        )
                        if metadata["non_gate_nonzero_parameter_count"] != 0:
                            raise RuntimeError(
                                "Subsampled Gate diagnostic gradient reached a "
                                "non-Gate parameter."
                            )
                        del estimate
                    repeat_results.append(
                        {
                            "seed": seed,
                            "requested_effective_count": sample_size,
                            "effective_count": effective_count,
                            "capped_to_full": sample_size > full_count,
                            "reused_full_gradient": reused_full_gradient,
                            "cosine": cosine,
                            "gradient_norm": estimate_norm,
                            **metadata,
                        }
                    )
                cosines = [result["cosine"] for result in repeat_results]
                summary = {
                    "requested_effective_count": sample_size,
                    "effective_count": effective_count,
                    "capped_to_full": sample_size > full_count,
                    "reused_full_gradient": reused_full_gradient,
                    "cosine_mean": float(sum(cosines) / len(cosines)),
                    "cosine_min": float(min(cosines)),
                    "cosine_max": float(max(cosines)),
                    "repeats": repeat_results,
                }
                results.append(summary)
                prefix = f"gate_gradient_curve/n_{sample_size}"
                metrics[f"{prefix}/cosine_mean"] = summary["cosine_mean"]
                metrics[f"{prefix}/cosine_min"] = summary["cosine_min"]
                metrics[f"{prefix}/cosine_max"] = summary["cosine_max"]
                metrics[f"{prefix}/effective_count"] = float(effective_count)
                metrics[f"{prefix}/requested_effective_count"] = float(sample_size)
            del full_gradient
        finally:
            self.optimizer.zero_grad()
            set_rng_state(rng_state)
        if int(self.optimizer_steps) != optimizer_steps_before:
            raise RuntimeError("Gate gradient diagnostic changed optimizer steps.")
        if checkpoint_state_sha256(get_rng_state()) != checkpoint_state_sha256(
            rng_state
        ):
            raise RuntimeError("Gate gradient diagnostic did not restore RNG state.")
        audit = {
            "schema": "fastwam-gate-gradient-curve-audit-v1",
            "status": "PASS",
            "actor_version": int(self.version),
            "full_effective_count": full_count,
            "full_gradient_norm": full_norm,
            "sample_sizes": results,
            "optimizer_steps_before": optimizer_steps_before,
            "optimizer_steps_after": int(self.optimizer_steps),
            "rng_restored": True,
            "non_gate_gradient_nonzero_parameter_count": 0,
        }
        print(
            f"{FASTWAM_GATE_GRADIENT_CURVE_AUDIT_SENTINEL} "
            + json.dumps(audit, sort_keys=True),
            flush=True,
        )
        return metrics

    def _stop_fastwam_handle_replay(self) -> dict[str, float]:
        for source_rank in self._rollout_all_ranks:
            self._post_fastwam_kv_request(
                source_rank=source_rank,
                command="stop",
                handles=(),
                expect_response=False,
            )
        if self._fastwam_kv_executor is not None:
            self._fastwam_kv_executor.shutdown(wait=True)
            self._fastwam_kv_executor = None
        h2d_seconds = 0.0
        for start_event, end_event in self._fastwam_kv_h2d_events:
            end_event.synchronize()
            h2d_seconds += float(start_event.elapsed_time(end_event)) / 1000.0
        process_memory = psutil.Process(os.getpid()).memory_full_info()
        free_gpu = peak_gpu = 0
        if torch.cuda.is_available():
            free_gpu, _ = torch.cuda.mem_get_info(self.device)
            peak_gpu = torch.cuda.max_memory_allocated(self.device)
        metrics = {
            "kv_cache/prefetch_wait_seconds": self._fastwam_kv_prefetch_wait_seconds,
            "kv_cache/prefetch_wait_time": self._fastwam_kv_prefetch_wait_seconds,
            "kv_cache/h2d_bytes": float(self._fastwam_kv_h2d_bytes),
            "kv_cache/h2d_seconds": h2d_seconds,
            "kv_cache/d2d_bytes": 0.0,
            "kv_cache/d2d_seconds": 0.0,
            "kv_cache/actor_mig_free_bytes": float(free_gpu),
            "kv_cache/actor_mig_peak_allocated_bytes": float(peak_gpu),
            "kv_cache/node_available_bytes": float(psutil.virtual_memory().available),
            "kv_cache/actor_rss_bytes": float(process_memory.rss),
            "kv_cache/actor_uss_bytes": float(
                getattr(process_memory, "uss", process_memory.rss)
            ),
        }
        metrics.update(getattr(self, "_fastwam_gate_kv_sampling_metrics", {}))
        return metrics

    @Worker.timer("run_training")
    def run_training(
        self,
        kv_request_channel: Channel | None = None,
        kv_response_channel: Channel | None = None,
    ) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        if self._uses_fastwam_handle_replay():
            self._initialize_fastwam_fsdp_for_handle_replay()

        gate_update_before = None
        gate_optimizer_steps_before = None
        gate_parameter_audit_interval = None
        training_guard = self.cfg.runner.get("fastwam_training_guard", {})
        cost_audit_enabled = bool(
            training_guard.get("cost_audit", {}).get("enabled", False)
        )
        configured_gate_parameter_audit_interval = int(
            training_guard.get("gate_parameter_audit_interval_updates", 1)
        )
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is SupportedModel.FASTWAM_ADAPTIVE
            and cost_audit_enabled
            and self._fastwam_gate_parameter_audit_due(
                actor_version=int(self.version),
                interval_updates=configured_gate_parameter_audit_interval,
            )
        ):
            gate_update_before = self._capture_fastwam_gate_parameters()
            gate_optimizer_steps_before = int(self.optimizer_steps)
            gate_parameter_audit_interval = configured_gate_parameter_audit_interval

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is SupportedModel.FASTWAM_ADAPTIVE
            and str(self.cfg.actor.model.kv_replay.backend) == "recompute"
        ):
            self._fastwam_policy_module().capture_gate_recompute_reference()

        self.model.train()
        batch_reference = self._training_batch_reference(self.rollout_batch)
        rollout_size = batch_reference.shape[0] * batch_reference.shape[1]
        gate_kv_episode_contributions = []
        if self._uses_fastwam_handle_replay():
            emitted_gate = self.rollout_batch.get("emitted_gate")
            metadata = None if emitted_gate is None else emitted_gate.kv_metadata
            references = None if metadata is None else metadata.payload_reference_ids
            gate_valid = self.rollout_batch.get("gate_valid_mask")
            if references is None or gate_valid is None:
                raise ValueError(
                    "Stored Gate replay requires K/V references and a valid mask."
                )
            gate_kv_episode_contributions = (
                summarize_fastwam_gate_kv_episode_contributions(
                    episode_ids=emitted_gate.episode_ids,
                    gate_valid_mask=gate_valid,
                    gate_kv_sample_mask=references >= 0,
                )
            )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch,
                shuffle_id,
                consume=(
                    SupportedModel(self.cfg.actor.model.model_type)
                    is SupportedModel.FASTWAM_ADAPTIVE
                    and str(self.cfg.actor.model.kv_replay.backend) == "stored"
                ),
            )
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        gate_gradient_diagnostic_config = (
            self._fastwam_gate_gradient_diagnostic_config()
            if self._uses_fastwam_handle_replay()
            else None
        )
        if self._uses_fastwam_handle_replay():
            if kv_request_channel is None or kv_response_channel is None:
                raise ValueError(
                    "Handle-based stored Gate replay requires request and "
                    "response channels."
                )
            self._start_fastwam_handle_replay(
                request_channel=kv_request_channel,
                response_channel=kv_response_channel,
                update_epoch=update_epoch,
                episode_contributions=gate_kv_episode_contributions,
            )
        gate_gradient_diagnostic_metrics = {}
        if gate_gradient_diagnostic_config is not None:
            gate_gradient_diagnostic_metrics = (
                self._run_fastwam_gate_gradient_diagnostic(
                    gate_gradient_diagnostic_config
                )
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self._training_batch_reference(self.rollout_batch).size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        if gate_gradient_diagnostic_metrics:
            append_to_dict(metrics, gate_gradient_diagnostic_metrics)
        preupdate_log_ratio_audit_pending = (
            SupportedModel(self.cfg.actor.model.model_type)
            is SupportedModel.FASTWAM_ADAPTIVE
        )
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = self._training_batch_reference(
                    train_global_batch
                ).shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )
                selected_loss_scales = None
                counts = None
                if (
                    SupportedModel(self.cfg.actor.model.model_type)
                    is SupportedModel.FASTWAM_ADAPTIVE
                ):
                    route_info = train_global_batch["route_info"]
                    effective_gate_mask = fastwam_effective_gate_kv_mask(
                        train_global_batch["gate_valid_mask"],
                        train_global_batch.get("gate_kv_sample_mask"),
                    )
                    gate_count = effective_gate_mask.sum()
                    full_gate_count = train_global_batch["gate_valid_mask"].bool().sum()
                    flow_count = (
                        train_global_batch["flow_valid_mask"].bool()
                        & (route_info.route_used == int(WAMRoute.UNCOND))
                    ).sum()
                    counts = torch.tensor(
                        [
                            float(gate_count),
                            float(flow_count),
                            float(full_gate_count),
                        ],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    torch.distributed.all_reduce(
                        counts,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                    scale_numerator = float(
                        self.gradient_accumulation * torch.distributed.get_world_size()
                    )
                    selected_loss_scales = {
                        "gate": (
                            scale_numerator
                            / (
                                counts[2].item()
                                * float(
                                    getattr(
                                        self,
                                        "_fastwam_gate_kv_sample_probability",
                                        1.0,
                                    )
                                )
                            )
                            if counts[2].item() > 0
                            else 0.0
                        ),
                        "flow": (
                            scale_numerator / counts[1].item()
                            if counts[1].item() > 0
                            else 0.0
                        ),
                    }

                self.optimizer.zero_grad()
                pending_prefetches: deque[Future] = deque()
                next_prefetch_index = 0
                if self._uses_fastwam_handle_replay():
                    prefetch_depth = min(
                        int(self.cfg.actor.model.kv_replay.prefetch_depth),
                        len(train_micro_batch),
                    )
                    while next_prefetch_index < prefetch_depth:
                        pending_prefetches.append(
                            self._schedule_fastwam_kv_prefetch(
                                train_micro_batch[next_prefetch_index]
                            )
                        )
                        next_prefetch_index += 1
                for idx, batch in enumerate(train_micro_batch):
                    consumed_handles: tuple[int, ...] = ()
                    if self._uses_fastwam_handle_replay():
                        prefetch = pending_prefetches.popleft()
                        if next_prefetch_index < len(train_micro_batch):
                            pending_prefetches.append(
                                self._schedule_fastwam_kv_prefetch(
                                    train_micro_batch[next_prefetch_index]
                                )
                            )
                            next_prefetch_index += 1
                        consumed_handles = self._consume_fastwam_kv_prefetch(
                            batch,
                            prefetch,
                        )
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                        selected_loss_scales=selected_loss_scales,
                    )
                    if consumed_handles:
                        self._release_consumed_fastwam_kv(consumed_handles)
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                if preupdate_log_ratio_audit_pending:
                    if counts is None:
                        raise RuntimeError(
                            "FastWAM pre-update log-ratio audit lacks branch counts."
                        )
                    prefixes = ("gate", "uncond_flow")
                    local_maxima = []
                    for prefix in prefixes:
                        values = metrics.get(f"{prefix}/log_ratio_max_abs", [])
                        if not values:
                            raise RuntimeError(
                                "FastWAM pre-update log-ratio audit lacks "
                                f"{prefix} metrics."
                            )
                        local_maxima.append(
                            torch.stack(
                                [
                                    torch.as_tensor(
                                        value,
                                        dtype=torch.float32,
                                        device=self.device,
                                    ).detach()
                                    for value in values
                                ]
                            ).max()
                        )
                    preupdate_maxima = torch.stack(local_maxima)
                    torch.distributed.all_reduce(
                        preupdate_maxima,
                        op=torch.distributed.ReduceOp.MAX,
                    )
                    gate_samples = int(counts[0].item())
                    flow_samples = int(counts[1].item())
                    gate_max = float(preupdate_maxima[0].item())
                    flow_max = float(preupdate_maxima[1].item())
                    audit = {
                        "schema": "fastwam-preupdate-log-ratio-audit-v1",
                        "actor_rank": int(self._rank),
                        "actor_version": int(self.version),
                        "optimizer_steps_before": int(self.optimizer_steps),
                        "gate": {
                            "sample_count": gate_samples,
                            "max_abs_log_ratio": gate_max,
                        },
                        "uncond_flow": {
                            "sample_count": flow_samples,
                            "max_abs_log_ratio": flow_max,
                        },
                    }
                    if self._rank == 0:
                        print(
                            f"{FASTWAM_PREUPDATE_LOG_RATIO_AUDIT_SENTINEL} "
                            + json.dumps(audit, sort_keys=True),
                            flush=True,
                        )
                    append_to_dict(
                        metrics,
                        {
                            "gate/preupdate_log_ratio_max_abs": gate_max,
                            "gate/preupdate_sample_count": float(gate_samples),
                            "uncond_flow/preupdate_log_ratio_max_abs": flow_max,
                            "uncond_flow/preupdate_sample_count": float(flow_samples),
                        },
                    )
                    preupdate_log_ratio_audit_pending = False

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = self._optimizer_metrics(grad_norm, lr_list)
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        if gate_update_before is not None:
            gate_update_after = self._capture_fastwam_gate_parameters()
            gate_update_audit = self._summarize_fastwam_gate_update(
                before=gate_update_before,
                after=gate_update_after,
                optimizer_steps_before=int(gate_optimizer_steps_before),
                optimizer_steps_after=int(self.optimizer_steps),
            )
            gate_update_artifact = gate_update_audit.to_artifact()
            gate_update_artifact.update(
                {
                    "runner_update": int(self.version) + 1,
                    "interval_updates": int(gate_parameter_audit_interval),
                }
            )
            print(
                f"{FASTWAM_GATE_UPDATE_AUDIT_SENTINEL} "
                + json.dumps(gate_update_artifact, sort_keys=True),
                flush=True,
            )
            append_to_dict(
                metrics,
                {
                    "gate/update_l2_norm": gate_update_audit.update_l2_norm,
                    "gate/update_max_abs": gate_update_audit.update_max_abs,
                    "gate/relative_update_l2_norm": (
                        gate_update_audit.relative_update_l2_norm
                    ),
                    "gate/update_nonfinite_count": float(
                        gate_update_audit.nonfinite_update_count
                    ),
                },
            )
        if self._uses_fastwam_handle_replay():
            kv_metrics = self._stop_fastwam_handle_replay()
            append_to_dict(metrics, kv_metrics)
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        weighted_sums = {}
        weighted_maxima = {}
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is SupportedModel.FASTWAM_ADAPTIVE
        ):
            weighted_sums, weighted_maxima = pop_fastwam_weighted_metric_sums(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )
        if weighted_sums:
            reduced_weighted_sums = all_reduce_dict(
                weighted_sums, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict.update(
                finalize_fastwam_weighted_metrics(reduced_weighted_sums)
            )
        if weighted_maxima:
            mean_metric_dict.update(
                all_reduce_dict(
                    weighted_maxima,
                    op=torch.distributed.ReduceOp.MAX,
                )
            )
        if "gate/total_loss" in mean_metric_dict:
            mean_metric_dict["fastwam_dual/total_policy_loss"] = (
                float(self.cfg.algorithm.gate_ppo.get("loss_weight", 1.0))
                * mean_metric_dict["gate/total_loss"]
                + float(self.cfg.algorithm.uncond_flow_ppo.get("loss_weight", 1.0))
                * mean_metric_dict["uncond_flow/total_loss"]
            )

        return mean_metric_dict

    def _compute_fastwam_loss(
        self,
        *,
        micro_batch: dict,
        output_dict: dict[str, torch.Tensor],
        selected_loss_scales: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute independent Gate/Flow PPO losses and the fresh critic loss."""

        required_fields = (
            "route_info",
            "emitted_gate",
            "flow_advantages",
            "flow_valid_mask",
            "gate_advantages",
            "gate_valid_mask",
            "prev_logprobs",
            "returns",
            "prev_values",
        )
        missing = [field for field in required_fields if field not in micro_batch]
        if missing:
            raise KeyError(f"FastWAM training batch is missing fields: {missing}.")

        gate_cfg = self.cfg.algorithm.gate_ppo
        flow_cfg = self.cfg.algorithm.uncond_flow_ppo
        route_info = micro_batch["route_info"]
        emitted_gate = micro_batch["emitted_gate"]
        selected_loss_scales = selected_loss_scales or {}
        effective_gate_mask = fastwam_effective_gate_kv_mask(
            micro_batch["gate_valid_mask"],
            micro_batch.get("gate_kv_sample_mask"),
        )
        policy_loss_value, metrics = compute_fastwam_dual_ppo_loss(
            gate_logprobs=output_dict["gate_logprobs"].float(),
            gate_old_logprobs=emitted_gate.old_logprob.float(),
            gate_advantages=micro_batch["gate_advantages"].float(),
            gate_valid_mask=effective_gate_mask,
            gate_clip_ratio_low=float(gate_cfg.clip_ratio_low),
            gate_clip_ratio_high=float(gate_cfg.clip_ratio_high),
            gate_base_probabilities=(
                output_dict["gate_base_probabilities"].float()
                if str(gate_cfg.get("entropy_metric_source", "behavior")).lower()
                == "base"
                else None
            ),
            gate_behavior_probabilities=output_dict[
                "gate_behavior_probabilities"
            ].float(),
            gate_entropy_coefficient=float(gate_cfg.entropy_coefficient),
            gate_loss_coefficient=float(gate_cfg.get("loss_weight", 1.0)),
            flow_logprobs=output_dict["flow_logprobs"].float(),
            flow_old_logprobs=micro_batch["prev_logprobs"].float(),
            flow_advantages=micro_batch["flow_advantages"].float(),
            route_used=route_info.route_used,
            flow_valid_mask=micro_batch["flow_valid_mask"].bool(),
            flow_clip_ratio_low=float(flow_cfg.clip_ratio_low),
            flow_clip_ratio_high=float(flow_cfg.clip_ratio_high),
            flow_entropy=output_dict.get("flow_entropy", None),
            flow_entropy_coefficient=float(flow_cfg.entropy_coefficient),
            flow_loss_coefficient=float(flow_cfg.get("loss_weight", 1.0)),
            gate_selected_loss_scale=selected_loss_scales.get("gate"),
            flow_selected_loss_scale=selected_loss_scales.get("flow"),
        )
        regularization_cfg = self.cfg.algorithm.get("regularization", {})
        base_kl_cfg = regularization_cfg.get("base_uncond_kl", {})
        base_kl_enabled = bool(base_kl_cfg.get("enabled", False))
        base_kl_log_metric = bool(base_kl_cfg.get("log_metric", False))
        if base_kl_enabled or base_kl_log_metric:
            if "base_uncond_kl" not in output_dict:
                raise KeyError(
                    "Base-UNCOND KL logging requires analytic transition KL replay."
                )
            base_kl_loss, base_kl_metrics = compute_base_uncond_kl_loss(
                kl_values=output_dict["base_uncond_kl"].float(),
                route_used=route_info.route_used,
                valid_mask=micro_batch["flow_valid_mask"].bool(),
                selected_loss_scale=selected_loss_scales.get("flow"),
            )
            if base_kl_enabled:
                policy_loss_value = (
                    policy_loss_value
                    + float(base_kl_cfg.get("coefficient", 0.0)) * base_kl_loss
                )
            metrics.update(base_kl_metrics)

        collapse_cfg = regularization_cfg.get("collapse", {})
        if bool(collapse_cfg.get("enabled", False)):
            collapse_loss, collapse_metrics = compute_gate_collapse_penalty(
                base_idm_probabilities=output_dict["gate_base_probabilities"].float(),
                episode_ids=emitted_gate.episode_ids,
                valid_mask=effective_gate_mask,
                tau_calls=float(collapse_cfg.get("tau_calls", 1.0)),
                target_floor=(
                    None
                    if collapse_cfg.get("target_floor", None) is None
                    else float(collapse_cfg.target_floor)
                ),
                scope=str(collapse_cfg.get("scope", "microbatch")),
            )
            policy_loss_value = (
                policy_loss_value
                + float(collapse_cfg.get("coefficient", 0.0)) * collapse_loss
            )
            metrics.update(collapse_metrics)
        metrics["fastwam/regularized_policy_loss"] = policy_loss_value.detach()

        critic_cfg = self.cfg.algorithm.critic_loss
        critic_loss, critic_metrics = compute_ppo_critic_loss(
            values=output_dict["values"].float(),
            returns=micro_batch["returns"].float(),
            prev_values=micro_batch["prev_values"].float(),
            value_clip=float(critic_cfg.value_clip),
            huber_delta=float(critic_cfg.huber_delta),
            loss_mask=micro_batch.get("loss_mask", None),
            loss_mask_sum=micro_batch.get("loss_mask_sum", None),
            max_episode_steps=self.cfg.env.train.max_episode_steps,
        )
        loss = (
            policy_loss_value + float(critic_cfg.get("loss_weight", 1.0)) * critic_loss
        )
        metrics.update(critic_metrics)
        metrics["fastwam/total_loss"] = loss.detach()
        return loss, {
            key: value.detach().item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }

    def _training_batch_reference(
        self,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Return the batch-shaped field used by the generic PPO loop."""

        reference = batch.get("prev_logprobs")
        if not isinstance(reference, torch.Tensor):
            raise KeyError("PPO training requires tensor prev_logprobs.")
        return reference

    def _allows_absent_action_logprobs(self) -> bool:
        """Whether a policy has no independently trainable Action distribution."""

        return False

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
        selected_loss_scales: dict[str, float] | None = None,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        is_fastwam = model_type is SupportedModel.FASTWAM_ADAPTIVE
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch.get("prev_logprobs")
        if prev_logprobs is None and not self._allows_absent_action_logprobs():
            raise KeyError("PPO microbatch requires prev_logprobs.")
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if model_type in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif model_type in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        if is_fastwam:
            route_info = micro_batch.get("route_info")
            emitted_gate = micro_batch.get("emitted_gate")
            if route_info is None or emitted_gate is None:
                raise KeyError("FastWAM replay requires route_info and emitted_gate.")
            if forward_inputs is None:
                raise KeyError("FastWAM replay requires forward_inputs.")
            if "gate_kv_denoise_timesteps" in forward_inputs:
                if emitted_gate.kv_metadata is None:
                    raise ValueError("Stored Gate K/V replay requires K/V metadata.")
                forward_inputs = dict(forward_inputs)
                forward_inputs["gate_kv_layer_indices"] = torch.tensor(
                    emitted_gate.kv_metadata.layer_indices,
                    dtype=torch.long,
                    device=self.device,
                )
            kwargs.update(
                {
                    "route_info": route_info,
                    "emitted_gate": emitted_gate,
                    "compute_base_logprobs": bool(
                        (
                            self.cfg.algorithm.get("regularization", {})
                            .get("base_uncond_kl", {})
                            .get("enabled", False)
                        )
                        or (
                            self.cfg.algorithm.get("regularization", {})
                            .get("base_uncond_kl", {})
                            .get("log_metric", False)
                        )
                    ),
                }
            )

        compute_values = self.cfg.algorithm.adv_type == "gae"
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=(
                    is_fastwam or self.cfg.algorithm.get("entropy_bonus", 0.0) > 0
                ),
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if model_type in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        if is_fastwam:
            loss, metrics_data = self._compute_fastwam_loss(
                micro_batch=micro_batch,
                output_dict=output_dict,
                selected_loss_scales=selected_loss_scales,
            )
            normalization_statistics = getattr(
                self,
                "_fastwam_advantage_normalization_statistics",
                None,
            )
            if normalization_statistics is not None:
                if not isinstance(normalization_statistics, dict) or (
                    "floor_hit_fraction" not in normalization_statistics
                ):
                    raise RuntimeError(
                        "FastWAM training is missing advantage-normalization telemetry."
                    )
                metrics_data["normalize/floor_hit_fraction"] = float(
                    normalization_statistics["floor_hit_fraction"]
                )
        else:
            loss_kwargs = {
                "loss_type": self.cfg.algorithm.loss_type,
                "logprob_type": self.cfg.algorithm.logprob_type,
                "reward_type": self.cfg.algorithm.reward_type,
                "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                "logprobs": output_dict["logprobs"],
                "values": output_dict.get("values", None),
                "old_logprobs": prev_logprobs,
                "advantages": advantages,
                "returns": returns,
                "prev_values": prev_values,
                "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                "value_clip": self.cfg.algorithm.get("value_clip", None),
                "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                "loss_mask": loss_mask,
                "loss_mask_sum": loss_mask_sum,
                "max_episode_steps": self.cfg.env.train.max_episode_steps,
                "task_type": self.cfg.runner.task_type,
                "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
            }

            if model_type in [
                SupportedModel.GR00T_N1D6,
                SupportedModel.GR00T_N1D7,
            ]:
                loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get(
                    "clip_ratio_c", 3.0
                )
                if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                    loss_kwargs["clip_log_ratio_min"] = (
                        self.cfg.algorithm.clip_log_ratio_min
                    )
                if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                    loss_kwargs["clip_log_ratio_max"] = (
                        self.cfg.algorithm.clip_log_ratio_max
                    )

            loss, metrics_data = policy_loss(**loss_kwargs)
            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if (
                self.cfg.algorithm.get("entropy_bonus", 0.0) > 0
                and not loss_kwargs["critic_warmup"]
            ):
                entropy = output_dict["entropy"]
                entropy = reshape_entropy(
                    entropy,
                    entropy_type=self.cfg.algorithm.entropy_type,
                    action_dim=self.cfg.actor.model.get("action_dim", 7),
                    batch_size=output_dict["logprobs"].shape[0],
                )
                entropy_loss = masked_mean(entropy, mask=loss_mask)
                loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
            metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()
        if is_fastwam:
            metrics_data["fastwam/fsdp_view_restore_handles"] = float(
                self._restore_fastwam_fsdp_parameter_views_after_backward()
            )

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def set_fastwam_idm_cost(
        self,
        idm_cost: float,
        runner_step: int,
    ) -> dict[str, float]:
        """Publish an IDM cost while retaining the configured UNCOND cost."""

        branch_cost = self.cfg.algorithm.get("fixed_branch_cost", {})
        published = self.set_fastwam_branch_costs(
            idm_cost,
            float(branch_cost.get("uncond_cost", 0.0)),
            runner_step,
        )
        return {
            "runner_step": published["runner_step"],
            "idm_cost": published["idm_cost"],
        }

    def set_fastwam_branch_costs(
        self,
        idm_cost: float,
        uncond_cost: float,
        runner_step: int,
    ) -> dict[str, float]:
        """Publish lagged, non-negative IDM and UNCOND costs for one step."""

        if (
            SupportedModel(self.cfg.actor.model.model_type)
            is not SupportedModel.FASTWAM_ADAPTIVE
        ):
            raise ValueError("Runtime IDM cost is specific to fastwam_adaptive.")
        branch_cost = self.cfg.algorithm.get("fixed_branch_cost", {})
        runtime_control_enabled = branch_cost.get("controller") is not None or bool(
            branch_cost.get("fair_cost", {}).get("enabled", False)
        )
        if not bool(branch_cost.get("enabled", False)) or not runtime_control_enabled:
            raise ValueError(
                "Runtime IDM cost requires enabled FastWAM IDM cost control."
            )
        if int(runner_step) != int(self.version):
            raise ValueError(
                "Runtime IDM cost step does not match the actor version: "
                f"{runner_step} != {self.version}."
            )
        idm_cost = float(idm_cost)
        uncond_cost = float(uncond_cost)
        if any(
            not math.isfinite(cost) or cost < 0.0 for cost in (idm_cost, uncond_cost)
        ):
            raise ValueError("Runtime branch costs must be finite and non-negative.")
        self._fastwam_runtime_idm_cost = idm_cost
        self._fastwam_runtime_uncond_cost = uncond_cost
        self._fastwam_runtime_branch_cost_step = int(runner_step)
        self._fastwam_runtime_idm_cost_step = int(runner_step)
        return {
            "runner_step": float(runner_step),
            "idm_cost": idm_cost,
            "uncond_cost": uncond_cost,
        }

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = self._optimizer_metrics(grad_norm, lr_list)
        append_to_dict(metrics, metric_data)
