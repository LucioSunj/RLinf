# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synchronous single-owner FR3 training using the shared FastWAM objectives.

Only bridge calls touch hardware. A policy version stays fixed until every
physical episode in the update is sealed; replay and optimization then run on
the same resident policy. Resume restores numerical state and starts a new
operator-confirmed physical episode.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from fastwam.adapters import sha256_file
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.fastwam_dual_ppo import (
    compute_fastwam_dual_ppo_loss,
    finalize_fastwam_weighted_metrics,
    pop_fastwam_weighted_metric_sums,
)
from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.algorithms.utils import safe_normalize
from rlinf.config_contracts import (
    build_fastwam_checkpoint_contract,
    validate_fastwam_training_checkpoint_contract,
)
from rlinf.envs.real.fr3_polymetis.env import Fr3PolymetisEnv
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.nested_dict_process import map_nested_tensors
from rlinf.utils.utils import get_rng_state, set_rng_state

from .contracts import ChunkRouteRecord, WAMRoute
from .critic import CriticKind, critic_parent_checkpoint_sha256
from .fr3_config import validate_fr3_training_config
from .fr3_policy import build_fr3_mixed_model
from .fr3_runtime import fr3_observation_batch
from .online_idm_bc.actor import assemble_online_idm_bc_loss
from .online_idm_bc.config import (
    ONLINE_IDM_BC_FLOW_VALID,
    ONLINE_IDM_BC_TEACHER_SECONDS,
)
from .optimizer import partition_fastwam_trainable_parameters
from .route_neutral_online.actor import base_entropy_loss_correction
from .route_neutral_online.task_budget import (
    TaskBudgetController,
    apply_task_chunk_costs,
    validate_task_cost_metrics,
)


def _cpu(value: Any) -> Any:
    return map_nested_tensors(copy.deepcopy(value), lambda t: t.detach().cpu().clone())


class Fr3SynchronousRunner:
    """One physical environment, one immutable rollout version, one optimizer."""

    def __init__(self, cfg: Any, *, policy: Any = None, env: Any = None) -> None:
        self.cfg = cfg
        validate_fr3_training_config(cfg)
        self.action_horizon = int(cfg.actor.model.runtime.generation_horizon)
        self.device = torch.device(str(cfg.fr3.device))
        if self.device.type != "cpu":
            if (
                Worker.torch_platform is None
                or Worker.torch_device_type != self.device.type
            ):
                raise ValueError(
                    "FR3 configured device differs from the available accelerator platform."
                )
            Worker.torch_platform.set_device(self.device)
        self.step = self.optimizer_steps = self.episodes = 0
        self._collecting = False
        self.output = Path(cfg.fr3.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        seed = int(cfg.actor.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            str(cfg.actor.model.precision)
        ]
        self.policy = (
            policy
            if policy is not None
            else build_fr3_mixed_model(cfg.actor.model, dtype)
        ).to(self.device)
        self.policy.train()
        self.env = env if env is not None else Fr3PolymetisEnv(cfg.env.train)
        self.groups = partition_fastwam_trainable_parameters(
            self.policy.named_parameters()
        )
        optim = cfg.actor.optim
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "name": name,
                    "params": params,
                    "lr": float(
                        optim[
                            {
                                "gate": "gate_lr",
                                "uncond_lora": "lora_lr",
                                "value_head": "value_lr",
                            }[name]
                        ]
                    ),
                }
                for name, params in self.groups.items()
            ],
            betas=(float(optim.adam_beta1), float(optim.adam_beta2)),
            eps=float(optim.adam_eps),
            weight_decay=float(optim.weight_decay),
        )
        # The explicit constant schedule retains the source fixed learning rates.
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _: 1.0
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.controller = TaskBudgetController(
            OmegaConf.to_container(
                cfg.algorithm.fixed_branch_cost.controller, resolve=True
            ),
            OmegaConf.to_container(
                cfg.algorithm.fixed_branch_cost.task_budget, resolve=True
            ),
        )
        self.fr3_contract = {
            key: cfg.fr3[key]
            for key in (
                "task_name",
                "task_id",
                "prompt",
                "parent_checkpoint_sha256",
                "statistics_sha256",
                "action_dt",
            )
        } | {"action_horizon": self.action_horizon, "action_dim": 8}
        if sha256_file(cfg.fr3.statistics) != self.fr3_contract["statistics_sha256"]:
            raise ValueError(
                "FR3 normalization statistics differ from the confirmed contract."
            )
        self.contract = build_fastwam_checkpoint_contract(cfg, world_size=1)
        self.contract["fr3"] = self.fr3_contract | {
            key: cfg.fr3[key]
            for key in (
                "episodes_per_update",
                "max_episode_actions",
                "max_episode_seconds",
            )
        }
        self.bc_bootstrap = None
        if cfg.runner.get("resume_dir"):
            self.load_checkpoint(cfg.runner.resume_dir)
        elif policy is None:
            self.bootstrap_bc()
        self.policy.set_global_step(self.step)

    def bootstrap_bc(self) -> None:
        """Use the native strict same-parent dual-adapter bootstrap validator."""
        from .execution.actor import EmbodiedFSDPActor

        sidecar = torch.load(
            self.cfg.runner.bootstrap_uncond_lora_sidecar,
            map_location="cpu",
            weights_only=True,
        )
        for branch in ("action", "video"):
            extra = sidecar[branch]["metadata"].get("extra", {})
            # Earlier H16 sidecars predate the explicit horizon metadata.
            if extra.get("action_horizon", 16) != self.action_horizon:
                raise ValueError(
                    "FR3 BC sidecar action horizon differs from the task contract."
                )
            if extra.get("statistics_sha256") != self.fr3_contract["statistics_sha256"]:
                raise ValueError(
                    "FR3 BC sidecar normalization statistics differ from its task contract."
                )

        # The native loader only needs these policy/step fields and deliberately
        # contains no FSDP or distributed operations. Share its provenance checks.
        class Owner:
            pass

        owner = Owner()
        owner.cfg, owner.version, owner.optimizer_steps = (
            self.cfg,
            self.step,
            self.optimizer_steps,
        )
        owner._fastwam_policy_module = lambda: self.policy
        self.bc_bootstrap = EmbodiedFSDPActor.bootstrap_fastwam_uncond_lora(
            owner,
            self.cfg.runner.bootstrap_uncond_lora_sidecar,
            self.cfg.runner.bootstrap_uncond_lora_sidecar_sha256,
        )

    def _observation(self, raw: dict, *, reset: bool) -> dict:
        obs = fr3_observation_batch(raw)
        obs["_fastwam_env_ids"] = torch.tensor([0])
        obs["_fastwam_reset_mask"] = torch.tensor([reset])
        return obs

    @staticmethod
    def _seal_previous(rows: list[dict], reward: float, info: dict) -> None:
        if rows:
            rows[-1]["reward_env"] += float(reward)
            rows[-1]["episode_seal"] = copy.deepcopy(info)

    @torch.no_grad()
    def _bootstrap_value(self, raw: dict) -> torch.Tensor:
        obs = self._observation(raw, reset=False)
        critic = self.policy._require_critic()
        if self.policy._critic_kind() is CriticKind.FASTWAM_CURRENT_FRAME_VALUE:
            value = critic.value_from_features(
                self.policy.runtime.critic_features(env_obs=obs)
            )
        else:
            value = critic.predict_value_batch(
                self.policy.runtime.critic_observation(env_obs=obs)
            )
        return value.detach().cpu().float().reshape(1)

    def _record_receipt(
        self, sample: dict, actions: torch.Tensor, reward: float, info: dict, raw: dict
    ) -> dict:
        mask = torch.from_numpy(
            np.array(info["execution_mask"], dtype=bool, copy=True)
        ).reshape(1, self.action_horizon)
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("FR3 execution receipt must be one contiguous prefix.")
        if (
            info["policy_version"] != self.step
            or info["episode_id"] != self.env.episode_id
        ):
            raise ValueError(
                "FR3 execution receipt differs from sealed policy/episode identity."
            )
        route, gate = sample["route_info"], sample["emitted_gate"]
        if (
            int(route.chunk_ids.item()) != info["chunk_id"]
            or int(route.actor_versions.item()) != self.step
        ):
            raise ValueError("FR3 policy route and physical receipt identity differ.")
        valid = bool(info["chunk_valid"]) and bool(mask.any())
        eligible = (
            valid
            and bool(info["gate_valid"])
            and bool(gate.valid.item())
            and not bool(route.route_was_forced.item())
        )
        row = _cpu(sample)
        row.update(
            observation=_cpu(raw),
            requested_actions=actions.detach().cpu().clone(),
            receipt=copy.deepcopy(info),
            reward_env=float(reward),
            flow_execution_mask=mask,
            flow_valid_mask=torch.tensor([valid]),
            gate_valid_mask=torch.tensor([eligible]),
        )
        row["forward_inputs"]["fr3_execution_mask"] = mask.clone()
        row["forward_inputs"][ONLINE_IDM_BC_FLOW_VALID] = torch.tensor([valid])
        return row

    def collect_episode(self, decision: dict) -> list[dict]:
        """Collect one sealed episode; never resume or retry uncertain actions."""
        status = self.env.status()
        max_actions = self.cfg.fr3.max_episode_actions
        max_seconds = self.cfg.fr3.max_episode_seconds
        for key, expected in {
            "action_dt": float(self.cfg.fr3.action_dt),
            "execution_horizon": self.action_horizon,
            "max_episode_actions": max_actions,
            "max_episode_seconds": max_seconds,
        }.items():
            if key not in status or status[key] != expected:
                raise ValueError(
                    f"FR3 bridge {key} differs from the training contract."
                )
        while not status["scene_ready"]:
            time.sleep(float(self.cfg.fr3.poll_interval_s))
            status = self.env.status()
        self.policy.runtime.clear_session()
        raw, _ = self.env.reset(
            options={"policy_version": self.step, "episode_id": uuid.uuid4().hex}
        )
        self._collecting = True
        rows: list[dict] = []
        executed, start, reset = 0, time.monotonic(), True
        bootstrap = torch.zeros(1)
        allow_bootstrap = False
        inference_seconds = execution_seconds = teacher_seconds = 0.0
        ending_info = {}
        try:
            while True:
                status = self.env.status()
                if not status.get("stop_reason") and (
                    (max_actions is not None and executed >= max_actions)
                    or (
                        max_seconds is not None
                        and time.monotonic() - start >= max_seconds
                    )
                ):
                    self.env.outcome("time_limit")
                    status = self.env.status()
                if status.get("stop_reason"):
                    final, reward, _, _, info = self.env.seal()
                    ending_info = info
                    self._seal_previous(rows, reward, info)
                    allow_bootstrap = bool(
                        info["bootstrap_allowed"] and info["final_observation_valid"]
                    )
                    if allow_bootstrap:
                        bootstrap = self._bootstrap_value(final)
                    break
                started_inference = time.perf_counter()
                with torch.no_grad():
                    actions, sample = self.policy.predict_action_batch(
                        self._observation(raw, reset=reset),
                        mode="train",
                        compute_values=True,
                    )
                inference_seconds += time.perf_counter() - started_inference
                teacher_seconds += float(
                    torch.as_tensor(
                        sample["forward_inputs"].get(ONLINE_IDM_BC_TEACHER_SECONDS, 0)
                    ).sum()
                )
                reset = False
                route, gate = sample["route_info"], sample["emitted_gate"]
                # An operator outcome or bridge deadline can arrive during model
                # inference. Seal it before submitting any additional target.
                if self.env.status().get("stop_reason"):
                    final, reward, _, _, info = self.env.seal()
                    ending_info = info
                    self._seal_previous(rows, reward, info)
                    allow_bootstrap = bool(
                        info["bootstrap_allowed"] and info["final_observation_valid"]
                    )
                    if allow_bootstrap:
                        bootstrap = self._bootstrap_value(final)
                    break
                started_execution = time.perf_counter()
                try:
                    final, reward, terminated, truncated, info = self.env.step_chunk(
                        actions,
                        policy_version=self.step,
                        chunk_id=int(route.chunk_ids.item()),
                        route=int(route.route_used.item()),
                        eligible=bool(gate.valid.item()),
                        forced=bool(route.route_was_forced.item()),
                    )
                except Exception as error:
                    execution_seconds += time.perf_counter() - started_execution
                    # A rejected request can race an idle operator outcome. Only
                    # an unconsumed chunk identity proves no unknown action ran.
                    try:
                        status = self.env.status()
                        if status.get("stop_reason") and status.get(
                            "next_chunk_id"
                        ) == int(route.chunk_ids.item()):
                            final, reward, _, _, info = self.env.seal()
                            ending_info = info
                            self._seal_previous(rows, reward, info)
                            allow_bootstrap = bool(
                                info["bootstrap_allowed"]
                                and info["final_observation_valid"]
                            )
                            if allow_bootstrap:
                                bootstrap = self._bootstrap_value(final)
                            break
                    except Exception:
                        pass
                    # Previous complete rows form a trusted fragment ending at the
                    # last measured pre-action state, never at a post-reset state.
                    bootstrap = sample["prev_values"].detach().cpu().float().reshape(1)
                    allow_bootstrap = bool(rows)
                    ending_info = {
                        "stop_reason": "technical_interrupt",
                        "terminated": False,
                        "truncated": True,
                    }
                    self._event(
                        {
                            "event": "technical_interrupt",
                            "episode_id": self.env.episode_id,
                            "error": str(error),
                        }
                    )
                    break
                execution_seconds += time.perf_counter() - started_execution
                ending_info = info
                row = self._record_receipt(sample, actions, reward, info, raw)
                executed += int(row["flow_execution_mask"].sum())
                if bool(row["flow_valid_mask"].item()):
                    rows.append(row)
                elif terminated or truncated:
                    self._seal_previous(rows, reward, info)
                    allow_bootstrap = bool(
                        info["bootstrap_allowed"] and info["final_observation_valid"]
                    )
                    if allow_bootstrap:
                        bootstrap = self._bootstrap_value(final)
                    break
                else:
                    # A rejected nonterminal transition cannot connect two GAE
                    # fragments, even if the bridge has not ended the episode.
                    bootstrap = sample["prev_values"].detach().cpu().float().reshape(1)
                    allow_bootstrap = bool(rows)
                    ending_info = {
                        "stop_reason": "rejected_transition",
                        "terminated": False,
                        "truncated": True,
                    }
                    self.env.stop()
                    break
                raw = final
                if terminated or truncated:
                    allow_bootstrap = bool(
                        info["bootstrap_allowed"] and info["final_observation_valid"]
                    )
                    if allow_bootstrap:
                        bootstrap = self._bootstrap_value(final)
                    break
            self.episodes += 1
            if rows:
                self._finish_fragment(rows, decision, bootstrap, allow_bootstrap)
            self._event(
                {
                    "event": "episode_sealed",
                    "episode_id": self.env.episode_id,
                    "policy_version": self.step,
                    "valid_chunks": len(rows),
                    "executed_actions": executed,
                    "bootstrap_allowed": allow_bootstrap,
                    "stop_reason": ending_info.get("stop_reason"),
                    "terminated": bool(ending_info.get("terminated", False)),
                    "truncated": bool(ending_info.get("truncated", False)),
                    "inference_wall_seconds": inference_seconds,
                    "execution_rpc_wall_seconds": execution_seconds,
                    "training_teacher_wall_seconds": teacher_seconds,
                }
            )
            return rows
        except BaseException:
            self.env.stop()
            raise
        finally:
            self._collecting = False

    def _finish_fragment(
        self, rows: list[dict], decision: dict, bootstrap: torch.Tensor, allowed: bool
    ) -> None:
        n = len(rows)
        routes = ChunkRouteRecord.stack([row["route_info"] for row in rows], dim=0)
        charge = torch.tensor([[bool(row["gate_valid_mask"].item())] for row in rows])
        cost_owner = SimpleNamespace(
            version=self.step,
            _fastwam_task_cost_decision={
                "runner_step": self.step,
                "tasks": {str(self.cfg.fr3.task_id): decision},
            },
            rollout_batch={
                "route_info": routes,
                "chunk_valid": torch.ones((n, 1), dtype=torch.bool),
                "forward_inputs": {
                    "multitask_task_id": torch.full(
                        (n, 1), int(self.cfg.fr3.task_id), dtype=torch.long
                    )
                },
            },
        )
        costs, audit = apply_task_chunk_costs(
            cost_owner, torch.tensor([[[row["reward_env"]]] for row in rows]), charge
        )
        validate_task_cost_metrics([audit.to_metrics()])
        rows[0]["task_cost_audit"] = audit.to_artifact()
        self._event(
            {
                "event": "task_cost_audit",
                "episode_id": self.env.episode_id,
                "audit": audit.to_artifact(),
            }
        )
        values = torch.cat(
            [row["prev_values"].float().reshape(1, 1) for row in rows]
            + [bootstrap.reshape(1, 1)]
        )
        dones = torch.zeros((n + 1, 1), dtype=torch.bool)
        dones[-1] = not allowed
        advantages, returns = compute_gae_advantages_and_returns(
            rewards=costs.rewards[..., 0],
            values=values,
            dones=dones,
            gamma=float(self.cfg.algorithm.gamma),
            gae_lambda=float(self.cfg.algorithm.gae_lambda),
            normalize_advantages=False,
        )
        for index, row in enumerate(rows):
            row.update(
                advantages=advantages[index],
                returns=returns[index : index + 1],
                reward_train=float(costs.rewards[index].item()),
                branch_cost=float(costs.costs[index].item()),
            )

    def _feedback(self, rows: list[dict]) -> dict:
        prefix = f"task/{self.cfg.fr3.task_id}"
        eligible = [r for r in rows if bool(r["gate_valid_mask"].item())]

        def is_idm(row):
            return int(row["route_info"].route_used.item()) == int(WAMRoute.IDM)

        return {
            f"{prefix}/valid_chunks": len(rows),
            f"{prefix}/gate_chunks": len(eligible),
            f"{prefix}/idm_chunks": sum(is_idm(r) for r in rows),
            f"{prefix}/eligible_idm_chunks": sum(is_idm(r) for r in eligible),
            f"{prefix}/behavior_probability_sum": sum(
                float(r["emitted_gate"].behavior_probability.item()) for r in eligible
            ),
            f"{prefix}/forced_chunks": sum(
                bool(r["route_info"].route_was_forced.item()) for r in rows
            ),
        }

    def update(self, rows: list[dict]) -> dict[str, float]:
        """Replay every valid row with globally selected Gate/Flow normalization."""
        if self._collecting or not rows:
            raise RuntimeError("FR3 update requires a nonempty sealed rollout.")
        raw_advantages = torch.stack([row["advantages"] for row in rows])
        advantages = (
            safe_normalize(
                raw_advantages,
                loss_mask=torch.ones_like(raw_advantages, dtype=torch.bool),
            )
            if len(rows) > 1
            else torch.zeros_like(raw_advantages)
        )
        ng = sum(bool(row["gate_valid_mask"].item()) for row in rows)
        nf = sum(
            int(row["route_info"].route_used.item()) == int(WAMRoute.UNCOND)
            for row in rows
        )
        gs, fs = 1.0 / max(ng, 1), 1.0 / max(nf, 1)
        warmup = self.step < int(self.cfg.fr3.warmup_updates)
        metrics: dict[str, float] = {
            "warmup": float(warmup),
            "valid_chunks": len(rows),
            "gate_chunks": ng,
            "uncond_chunks": nf,
        }
        shared_metrics: dict[str, list[float]] = {}
        for _ in range(int(self.cfg.algorithm.update_epoch)):
            self.optimizer.zero_grad(set_to_none=True)
            for index, stored in enumerate(rows):
                row = map_nested_tensors(stored, lambda t: t.to(self.device))
                out = self.policy.default_forward(
                    forward_inputs=row["forward_inputs"],
                    route_info=row["route_info"],
                    emitted_gate=row["emitted_gate"],
                )
                critic_cfg = self.cfg.algorithm.critic_loss
                value_loss, _ = compute_ppo_critic_loss(
                    values=out["values"].float(),
                    returns=row["returns"],
                    prev_values=row["prev_values"],
                    value_clip=float(critic_cfg.value_clip),
                    huber_delta=float(critic_cfg.huber_delta),
                )
                loss = float(critic_cfg.loss_weight) * value_loss / len(rows)
                if not warmup:
                    gate, flow = (
                        self.cfg.algorithm.gate_ppo,
                        self.cfg.algorithm.uncond_flow_ppo,
                    )
                    adv = advantages[index].to(self.device)
                    policy_loss, policy_metrics = compute_fastwam_dual_ppo_loss(
                        gate_logprobs=out["gate_logprobs"].float(),
                        gate_old_logprobs=row["emitted_gate"].old_logprob.float(),
                        gate_advantages=adv,
                        gate_valid_mask=row["gate_valid_mask"],
                        gate_clip_ratio_low=float(gate.clip_ratio_low),
                        gate_clip_ratio_high=float(gate.clip_ratio_high),
                        gate_base_probabilities=out["gate_base_probabilities"].float(),
                        gate_behavior_probabilities=out[
                            "gate_behavior_probabilities"
                        ].float(),
                        gate_entropy_coefficient=float(gate.entropy_coefficient),
                        gate_loss_coefficient=float(gate.loss_weight),
                        flow_logprobs=out["flow_logprobs"].float(),
                        flow_old_logprobs=row["prev_logprobs"].float(),
                        flow_advantages=adv,
                        route_used=row["route_info"].route_used,
                        flow_valid_mask=row["flow_valid_mask"],
                        flow_execution_mask=row["flow_execution_mask"],
                        flow_clip_ratio_low=float(flow.clip_ratio_low),
                        flow_clip_ratio_high=float(flow.clip_ratio_high),
                        flow_entropy=out["flow_entropy"],
                        flow_entropy_coefficient=float(flow.entropy_coefficient),
                        flow_loss_coefficient=float(flow.loss_weight),
                        gate_selected_loss_scale=gs,
                        flow_selected_loss_scale=fs,
                    )
                    for key, value in policy_metrics.items():
                        shared_metrics.setdefault(key, []).append(float(value))
                    policy_loss = policy_loss + base_entropy_loss_correction(
                        base=out["gate_base_probabilities"],
                        behavior=out["gate_behavior_probabilities"],
                        valid=row["gate_valid_mask"],
                        coefficient=float(gate.entropy_coefficient)
                        * float(gate.loss_weight),
                        selected_loss_scale=gs,
                    )
                    policy_loss, bc_metrics = assemble_online_idm_bc_loss(
                        current_loss=policy_loss,
                        output_dict=out,
                        config=self.policy.online_idm_bc_config,
                        selected_loss_scale=fs if nf else 0.0,
                        metric_scale_numerator=1.0,
                        flow_metric_loss=policy_metrics.get(
                            "uncond_flow/policy_loss", 0.0
                        ),
                    )
                    metrics["online_idm_bc/weighted_loss"] = metrics.get(
                        "online_idm_bc/weighted_loss", 0.0
                    ) + float(bc_metrics["online_idm_bc/weighted_loss"])
                    loss = loss + policy_loss
                metrics["loss"] = metrics.get("loss", 0.0) + float(loss.detach())
                loss.backward()
            if warmup:
                for name in ("gate", "uncond_lora"):
                    for parameter in self.groups[name]:
                        parameter.grad = None
            grad = torch.nn.utils.clip_grad_norm_(
                [p for g in self.groups.values() for p in g],
                float(self.cfg.actor.optim.clip_grad),
            )
            if not torch.isfinite(grad):
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("FR3 nonfinite gradient; optimizer update rejected.")
            metrics["grad_norm"] = float(grad)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer_steps += 1
        sums, maxima = pop_fastwam_weighted_metric_sums(shared_metrics)
        metrics.update(finalize_fastwam_weighted_metrics(sums))
        metrics.update(maxima)
        return metrics

    def run_update(self) -> dict[str, float]:
        self.policy.set_global_step(self.step)
        decision = self.controller.decisions(self.step)
        rows = []
        for _ in range(int(self.cfg.fr3.episodes_per_update)):
            rows.extend(
                self.collect_episode(decision["tasks"][str(self.cfg.fr3.task_id)])
            )
        observation = self.controller.observe(self.step, self._feedback(rows))
        if not rows:
            raise RuntimeError(
                "FR3 collected no valid transitions; no optimizer update or checkpoint produced."
            )
        metrics = self.update(rows)
        self.step += 1
        self.policy.set_global_step(self.step)
        self._event(
            {
                "event": "update",
                "step": self.step,
                "metrics": metrics,
                "budget": observation,
            }
        )
        torch.save(_cpu(rows), self.output / f"rollout_{self.step:06d}.pt")
        if self.step % int(self.cfg.fr3.checkpoint_every) == 0:
            self.save_checkpoint()
        return metrics

    def _event(self, event: dict) -> None:
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def save_checkpoint(self) -> Path:
        if self._collecting or self.controller.observed_runner_steps != self.step:
            raise RuntimeError(
                "FR3 checkpoints require a sealed rollout and consumed controller feedback."
            )
        target = self.output / "checkpoints" / f"global_step_{self.step}" / "rank_0.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "fastwam-adaptive-rl-checkpoint-v1",
            "step": self.step,
            "optimizer_steps": self.optimizer_steps,
            "parent_checkpoint_sha256": self.fr3_contract["parent_checkpoint_sha256"],
            "critic_parent_checkpoint_sha256": critic_parent_checkpoint_sha256(
                self.cfg.actor.model.critic
            ),
            "contract": self.contract,
            "fr3_contract": self.fr3_contract,
            "policy": self.policy.trainable_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "rng": get_rng_state(),
            "fr3_runtime": {
                "controller": self.controller.state_dict(),
                "episodes": self.episodes,
            },
        }
        if self.bc_bootstrap is not None:
            payload["bc_bootstrap"] = self.bc_bootstrap
        temporary = target.with_suffix(".tmp")
        torch.save(_cpu(payload), temporary)
        os.replace(temporary, target)
        OmegaConf.save(self.cfg, target.parent / "training_config.yaml", resolve=True)
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path = path / "rank_0.pt" if path.is_dir() else path
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema") != "fastwam-adaptive-rl-checkpoint-v1"
            or checkpoint.get("fr3_contract") != self.fr3_contract
        ):
            raise ValueError("FR3 native training checkpoint task contract differs.")
        if (
            checkpoint.get("parent_checkpoint_sha256")
            != self.fr3_contract["parent_checkpoint_sha256"]
            or checkpoint.get("critic_parent_checkpoint_sha256")
            != critic_parent_checkpoint_sha256(self.cfg.actor.model.critic)
            or checkpoint["policy"]["actor_version"] != checkpoint["step"]
        ):
            raise ValueError("FR3 checkpoint parent, critic or policy version differs.")
        validate_fastwam_training_checkpoint_contract(
            checkpoint["contract"],
            self.contract,
            allow_n4_to_three_rollout_expansion=False,
            owner="FR3 runner",
        )
        self.policy.load_trainable_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        self.controller.load_state_dict(checkpoint["fr3_runtime"]["controller"])
        self.step, self.optimizer_steps = (
            int(checkpoint["step"]),
            int(checkpoint["optimizer_steps"]),
        )
        self.episodes = int(checkpoint["fr3_runtime"]["episodes"])
        if self.controller.observed_runner_steps != self.step:
            raise ValueError("FR3 checkpoint controller/update counters differ.")
        self.bc_bootstrap = checkpoint.get("bc_bootstrap")
        self.policy.runtime.clear_session()
        self.policy._replay_text_ids.clear()
        self.policy._inactive_rollout_template = None
        self.policy.set_global_step(self.step)
        set_rng_state(checkpoint["rng"])

    def run(self) -> None:
        try:
            if self.step == 0:
                self.save_checkpoint()
            while self.step < int(self.cfg.fr3.updates):
                self.run_update()
            self.save_checkpoint()
        finally:
            self.env.close()
