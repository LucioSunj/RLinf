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

"""WebSocket reset/decide/commit protocol and four-episode training barrier."""

from __future__ import annotations

import json
from pathlib import Path
from time import time

import torch

from .contracts import Route
from .data import TASKS
from .evaluation import evaluation_route


def load_rl_weights(policy, path, task):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (
        ("model_type", "lingbot_va_route_neutral"),
        ("stage", "rl"),
        ("task", task),
        ("parent_path", policy.parent_path),
        ("routing_config", policy.config.to_dict()),
        ("normalizer", policy.runtime.normalizer.to_dict()),
    ):
        if payload[key] != expected:
            raise ValueError(f"Evaluation checkpoint has a different {key}.")
    if payload["step"] != 30:
        raise ValueError(
            "The six-method main evaluation uses the completed step30 model."
        )
    policy.gate.load_state_dict(payload["gate"], strict=True)
    policy.core.adapters.load_state_dict(payload["lora"], strict=True)
    policy.actor_version = payload["actor_version"]
    return {k: v.detach().cpu().clone() for k, v in payload["lora"].items()}


class PolicyService:
    def __init__(
        self,
        policy,
        output,
        *,
        task,
        trainer=None,
        schedule=None,
        final_lora=None,
        metric_logger=None,
    ):
        self.policy, self.trainer, self.task = policy, trainer, task
        self.metric_logger = metric_logger
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.episodes = [], []
        self.trial, self.chunk_index, self.executed_steps = None, 0, 0
        self.schedule = (
            {}
            if schedule is None
            else {r["trial_id"]: r for r in schedule["trials"] if r["task"] == task}
        )
        self.bc_lora = {
            k: v.detach().cpu().clone()
            for k, v in policy.core.adapters.state_dict().items()
        }
        self.final_lora = final_lora
        self.episode_id = 0
        self.episode_open = False

    def status(self):
        return {
            "task": self.task,
            "step": self.policy.actor_version,
            "training": self.trainer is not None,
            "warmup": self.trainer.warmup if self.trainer else False,
            "episodes_in_round": len(self.episodes),
            "complete": bool(
                self.trainer and self.trainer.step >= self.trainer.config.updates
            ),
            "max_steps": TASKS[self.task]["max_steps"],
        }

    def infer(self, request):
        operation = request["operation"]
        if operation == "status":
            return self.status()
        if operation == "reset":
            if self.episode_open:
                raise ValueError(
                    "Finish or abort the current episode before resetting."
                )
            if (
                self.trainer is not None
                and self.trainer.step >= self.trainer.config.updates
            ):
                raise ValueError("All 30 training updates are complete.")
            self.trial = self.schedule[request["trial_id"]] if self.schedule else None
            if self.trial:
                if (
                    self.output
                    / "episodes"
                    / f"{self.trial['scene_id']}_{self.trial['method']}.json"
                ).exists():
                    raise ValueError(
                        "This matched physical evaluation trial already has a result."
                    )
                source = (
                    self.bc_lora
                    if self.trial["method"] == "always_uncond_bc"
                    else self.final_lora
                )
                if source is None:
                    raise ValueError(
                        "Load both offline BC and final RL adapters before main evaluation."
                    )
                self.policy.core.adapters.load_state_dict(source, strict=True)
                self.policy.rng.manual_seed(self.trial["seed"])
            self.episode_id += 1
            self.episode, self.chunk_index, self.executed_steps = [], 0, 0
            self.client_started = request["episode_started"]
            self.reset_metadata = request["reset_metadata"]
            self.episode_open = True
            self.policy.reset_episode(
                TASKS[self.task]["instruction"], request["observation"]
            )
            return self.status()
        if operation == "decide":
            if not self.episode_open:
                raise ValueError("Reset a physical episode before requesting a chunk.")
            kwargs = {
                "training": self.trainer is not None,
                "warmup": self.trainer.warmup if self.trainer else False,
            }
            if self.trial:
                kwargs.update(
                    evaluation_route(
                        self.trial["method"],
                        self.chunk_index,
                        self.trial["routing_rate"],
                    )
                )
            elif self.trainer is None:
                kwargs.update(
                    route=Route(request.get("route", "idm")),
                    gate_framework=request.get("gate_framework", True),
                )
            decision = self.policy.decide(request["observation"], **kwargs)
            self.gate_framework = kwargs.get("gate_framework", True)
            self.observation_time = request["observation"]["timestamp"]
            return {
                "actions": decision.sample.actions.numpy(),
                "route": decision.sample.route.value,
                "probability": decision.probability,
                "actor_version": decision.actor_version,
                "timings": decision.sample.timings,
                "inference_finished": time(),
            }
        if operation == "abort":
            # Explicit operator abort discards incomplete sampling; checkpoint
            # progress is unchanged and the next command must be a physical reset.
            self.episode, self.episode_open = [], False
            self.policy.pending_decision = None
            self.policy.runtime.pending, self.policy.runtime.active = None, False
            with (self.output / "aborts.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": self.policy.actor_version,
                            "time": time(),
                            "reason": request["reason"],
                        }
                    )
                    + "\n"
                )
            return self.status()
        if operation != "commit":
            raise ValueError(f"Unknown operation {operation}.")
        feedback = request["feedback"]
        decision = self.policy.commit_execution(
            feedback["observed_frames"],
            feedback["executed_actions"],
            terminated=feedback["terminated"],
            success=feedback["success"],
            autonomous=feedback["autonomous"],
        )
        self.executed_steps += decision.executed_steps
        if (
            decision.terminal
            and not decision.executed_steps
            and decision.reward
            and self.episode
        ):
            # A human may mark the previous execution successful during inference
            # wait. Credit its last actual action, not this unexecuted proposal.
            self.episode[-1].reward = 1.0
            self.episode[-1].terminal = True
        self.episode.append(decision)
        row = {
            "episode_id": self.episode_id,
            "trial_id": self.trial["trial_id"] if self.trial else None,
            "task": self.task,
            "actor_version": decision.actor_version,
            "chunk": self.chunk_index,
            "route": decision.sample.route.value,
            "probability": decision.probability,
            "epsilon": decision.epsilon,
            "reward": decision.reward,
            "autonomous": decision.autonomous,
            "reason": feedback["reason"],
            "stage_progress": feedback["stage_progress"],
            "executed_steps": decision.executed_steps,
            "model_actions": decision.sample.actions.tolist(),
            "submitted_actions": feedback["executed_actions"].tolist(),
            "submitted_commands": feedback["submitted_commands"].tolist(),
            "observation_timestamp": self.observation_time,
            "action_timestamps": feedback["action_timestamps"],
            "observation_timestamps": feedback["observation_timestamps"],
            "observation_to_action_s": feedback["action_timestamps"][0]
            - self.observation_time
            if feedback["action_timestamps"]
            else None,
            "timings": decision.sample.timings,
            "noise": vars(decision.sample.noise),
            "timing_mode": "gate_framework" if self.gate_framework else "standalone",
        }
        with (self.output / "chunks.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        self.chunk_index += 1
        result = self.status()
        if not decision.terminal:
            return result
        self.episode_open = False
        summary = {
            "episode_id": self.episode_id,
            "task": self.task,
            "trial_id": self.trial["trial_id"] if self.trial else None,
            "success": bool(decision.reward),
            "stage_progress": feedback["stage_progress"],
            "chunks": len(self.episode),
            "idm_chunks": sum(d.sample.route is Route.IDM for d in self.episode),
            "autonomous": all(d.autonomous for d in self.episode),
            "executed_steps": self.executed_steps,
            "completion_s": request["episode_finished"] - self.client_started,
            "reset": self.reset_metadata,
        }
        directory = self.output / "episodes"
        directory.mkdir(exist_ok=True)
        identity = (
            f"{self.trial['scene_id']}_{self.trial['method']}"
            if self.trial
            else f"step_{self.policy.actor_version}_episode_{self.episode_id}"
        )
        (directory / f"{identity}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        if self.trainer is not None:
            self.episodes.append(self.episode)
            if len(self.episodes) == self.trainer.config.episodes_per_update:
                audit_path = (
                    self.output / "rollouts" / f"update_{self.trainer.step + 1}.pt"
                )
                audit_path.parent.mkdir(exist_ok=True)
                torch.save(self.episodes, audit_path)
                metrics = self.trainer.update(self.episodes)
                self.episodes = []
                with (self.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                if self.metric_logger is not None:
                    self.metric_logger.log(
                        {f"train/{k}": v for k, v in metrics.items()},
                        step=self.trainer.step,
                    )
                if self.trainer.step % self.trainer.config.save_interval == 0:
                    self.trainer.save(
                        self.output / "checkpoints" / f"global_step_{self.trainer.step}"
                    )
                result = {**self.status(), "update_metrics": metrics}
        result["episode_result"] = summary
        return result
