# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""One synchronous physical executor, with explicit manual episode boundaries."""

from __future__ import annotations

from copy import deepcopy

from .action_codec import ActionCodec
from .contracts import (
    ActionProposal,
    EpisodeOutcome,
    ExecutionReceipt,
    ObservationSnapshot,
    RealActionProtocol,
    RobotBackend,
    StepReceipt,
)
from .observation import ObservationAdapter


class SynchronousRobotEnv:
    """Send one primitive at a time and stop before the next after any terminal event.

    This is a command submission boundary, not a hardware emergency stop. The
    driver must document blocking calls, acknowledgement and cancellation.
    """

    num_envs = 1

    def __init__(
        self,
        backend: RobotBackend,
        observation: ObservationAdapter,
        codec: ActionCodec,
        protocol: RealActionProtocol,
        *,
        run_id: str,
        calibration_id: str,
        allow_motion: bool = False,
        episode_timeout_seconds: float = 120.0,
    ) -> None:
        if episode_timeout_seconds <= 0:
            raise ValueError("Episode timeout must be positive.")
        self.backend = backend
        self.observation = observation
        self.codec = codec
        self.protocol = protocol
        self.run_id = run_id
        self.calibration_id = calibration_id
        self.allow_motion = allow_motion
        self.episode_timeout_seconds = episode_timeout_seconds
        self.identity: dict | None = None
        self.outcome: EpisodeOutcome | None = None
        self.stopped_reason: str | None = None
        self._used_episode_ids: set[str] = set()
        self._proposals: set[str] = set()
        self._start_time = 0.0

    def begin_episode(
        self,
        *,
        episode_id: str,
        task_id: str,
        instruction: str,
        layout_id: str,
        operator_ready: bool,
    ) -> None:
        """Record the operator's completed reset; never call reset/go_to_rest."""
        if not operator_ready:
            raise ValueError("An explicit operator-ready event is required.")
        if self.identity is not None:
            raise RuntimeError("Finish the previous attempt before beginning another.")
        if episode_id in self._used_episode_ids:
            raise ValueError("Episode IDs cannot be reused, including on resume.")
        self._used_episode_ids.add(episode_id)
        self.identity = {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "task_id": task_id,
            "instruction": instruction,
            "layout_id": layout_id,
            "calibration_id": self.calibration_id,
        }
        self.observation.reset()
        self.outcome = None
        self.stopped_reason = None
        self.backend.begin_episode(episode_id, layout_id)
        self._start_time = self.backend.clock.now()

    def observe(self, chunk_id: int) -> ObservationSnapshot:
        if self.identity is None or self.stopped_reason is not None:
            raise RuntimeError("Observation requires a started, unstopped episode.")
        snapshot = self.backend.observe(**self.identity, chunk_id=chunk_id)
        self.observation.validate_freshness(snapshot, self.backend.clock.now())
        return deepcopy(snapshot)

    def request_stop(self, reason: str) -> None:
        """Latch stop immediately; scoring remains a separate episode-bound event."""
        if not reason:
            raise ValueError("A stop reason is required.")
        if self.stopped_reason is None:
            self.stopped_reason = reason

    def confirm_outcome(self, outcome: EpisodeOutcome) -> None:
        if self.identity is None or outcome.episode_id != self.identity["episode_id"]:
            raise ValueError("Outcome belongs to a different episode.")
        if self.outcome is not None:
            raise ValueError("An attempt accepts exactly one terminal score.")
        self.outcome = outcome
        self.request_stop(outcome.kind)

    def _poll(self) -> None:
        if self.outcome is not None or self.stopped_reason is not None:
            return
        episode_id = self.identity["episode_id"]
        outcome = self.backend.poll_outcome(episode_id)
        if (
            outcome is None
            and self.backend.clock.now() - self._start_time
            >= self.episode_timeout_seconds
        ):
            outcome = EpisodeOutcome(episode_id, "timeout")
        if outcome is not None:
            self.confirm_outcome(outcome)

    def execute_prefix(self, proposal: ActionProposal) -> ExecutionReceipt:
        """Retain unmodified replay while recording limited commands and known sends."""
        if (
            self.identity is None
            or proposal.snapshot.episode_id != self.identity["episode_id"]
        ):
            raise ValueError("Proposal does not belong to the active episode.")
        if proposal.proposal_id in self._proposals:
            raise RuntimeError("A proposal cannot be retried or executed twice.")
        self._proposals.add(proposal.proposal_id)
        if not self.backend.is_mock and not self.allow_motion:
            raise PermissionError("Live motion is disabled; no command was submitted.")
        expected = (self.protocol.generation_horizon, 7)
        if (
            proposal.canonical_actions.shape != expected
            or proposal.normalized_actions.shape != expected
        ):
            raise ValueError(
                f"The proposal must retain complete generated actions {expected}."
            )
        clock = self.backend.clock
        started = clock.now()
        steps = []
        final_state = None
        try:
            self.observation.validate_freshness(proposal.snapshot, started)
            for index, action in enumerate(
                proposal.canonical_actions[: self.protocol.execution_horizon]
            ):
                self._poll()
                if self.stopped_reason is not None:
                    break
                anchor_state = self.backend.read_state()
                command = self.codec.command(
                    action, self.observation.canonical_pose(anchor_state)
                )
                command = self.backend.limit_command(command)
                self._poll()
                if self.stopped_reason is not None:
                    break
                sent = clock.now()
                acknowledged = False
                try:
                    acknowledged = self.backend.send(command)
                    if acknowledged:
                        final_state = deepcopy(self.backend.read_state())
                finally:
                    steps.append(
                        StepReceipt(
                            index,
                            command,
                            sent,
                            clock.now(),
                            bool(acknowledged),
                            final_state if acknowledged else None,
                        )
                    )
                if not acknowledged:
                    raise ConnectionError(
                        "Driver cannot establish whether the command executed."
                    )
                self._poll()
                if self.stopped_reason is not None:
                    break
                clock.sleep(
                    max(0.0, self.protocol.sample_period - (clock.now() - sent))
                )
        except (ConnectionError, TimeoutError, ValueError) as error:
            if self.outcome is None:
                self.confirm_outcome(
                    EpisodeOutcome(
                        proposal.snapshot.episode_id, "censored", detail=str(error)
                    )
                )
        return ExecutionReceipt(
            proposal.proposal_id,
            tuple(steps),
            self.protocol.execution_horizon,
            self.outcome,
            self.stopped_reason,
            started,
            clock.now(),
            final_state,
        )

    def finish_episode(self) -> EpisodeOutcome:
        """Close an already stopped/scored attempt without observing a reset state."""
        if self.outcome is None:
            raise RuntimeError(
                "Stop and confirm the outcome before finishing the episode."
            )
        outcome = self.outcome
        self.identity = None
        return outcome
