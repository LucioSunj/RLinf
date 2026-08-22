# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastwam.causal_prediction import (
    CausalComputeMode,
    CausalControlKind,
    CausalDomain,
    CausalGateFeatureRecordV1,
    CausalInterventionSpecV2,
    CausalSamplingStratum,
    CausalStateIdentityV2,
    build_gate_training_example,
)
from scripts.causal_prediction.analysis_v2 import analyze_v2

from rlinf.envs.action_contract import (
    NORMALIZED_ACTION_STAGE,
    SUBMITTED_LIBERO_ACTION_STAGE,
    ActionExecutionTrace,
    ActionStageStatistics,
)
from rlinf.envs.libero.action_contract import LiberoActionContract
from rlinf.envs.libero.causal_collection import (
    CausalSourceCollectorV2,
    FutureLatentEntryV1,
    SourceChunkTraceV2,
    SourceEpisodeTraceV2,
    assign_future_donors,
    select_source_chunk,
)
from rlinf.envs.libero.causal_snapshot import CausalSnapshotV1, CausalSnapshotV2
from rlinf.models.embodiment.wam_policy.causal_evaluation_v2 import (
    SameChunkCausalEvaluationRunnerV2,
    SameChunkRouteDecisionV2,
)
from rlinf.models.embodiment.wam_policy.causal_experiment_v2 import (
    BranchObservationV2,
    PairedCausalForkRunnerV2,
)
from rlinf.models.embodiment.wam_policy.causal_routing_v2 import (
    EpisodeComputeBudgetV2,
)
from rlinf.models.embodiment.wam_policy.causal_runtime import (
    CausalChunkSample,
    CausalConditionContract,
)


def _contract() -> LiberoActionContract:
    return LiberoActionContract(
        low=(-1.0,) * 7,
        high=(1.0,) * 7,
        dimension_names=("x", "y", "z", "rx", "ry", "rz", "gripper"),
        gripper_dimension_index=6,
        outer_environment_classes=("unit.Outer",),
        underlying_environment_classes=("unit.Task",),
        robot_class="unit.SingleArm",
        robot_model="Panda",
        controller_class="unit.OSC",
        controller_name="OSC_POSE",
        controller_input_low=(-1.0,) * 6,
        controller_input_high=(1.0,) * 6,
        controller_output_low=(-0.05,) * 6,
        controller_output_high=(0.05,) * 6,
        gripper_class="unit.Gripper",
        gripper_dof=1,
        gripper_speed=0.01,
        control_frequency_hz=20,
        environment_horizon=10,
        dependency_versions=(),
    )


def _identity(trace: SourceChunkTraceV2) -> CausalStateIdentityV2:
    return CausalStateIdentityV2(
        domain=CausalDomain.CLEAN,
        suite="spatial",
        local_task_id=3,
        global_task_uid="spatial:3",
        task_name="synthetic",
        clean_base_task_uid="spatial:3",
        trial_id=2,
        reset_id=152,
        source_episode_id="episode-2",
        chunk_index=trace.chunk_index,
        policy_seed=42,
        model_seed=42,
    )


def _trace(index: int) -> SourceChunkTraceV2:
    components = {
        "gripper_transition": float(index == 1),
        "action_curvature": float(index),
        "contact_proximity": float(index) / 2,
        "predicate_transition": float(index == 2),
        "action_precision": float(2 - index),
    }
    return SourceChunkTraceV2(
        chunk_index=index,
        elapsed_steps=index * 10,
        eligible=True,
        terminated=False,
        remaining_action_capacity=20,
        raw_observation={"pixels": np.full((2, 2), index, dtype=np.uint8)},
        submitted_actions=((float(index),) * 7,),
        predicate_vector=(index == 2,),
        predicate_progress=float(index == 2),
        criticality_components=components,
        contact_active=index == 1,
        previous_contact_active=False,
        gripper_closing=True,
        nearest_task_object_distance_m=0.04,
        predicate_changed=index == 2,
        lightweight_summary={"index": index},
    )


def _features():
    return {
        "current_video_kv": torch.zeros(30, 4),
        "current_video_mask": torch.ones(30, dtype=torch.bool),
        "language": torch.zeros(2, 4),
        "language_mask": torch.ones(2, dtype=torch.bool),
        "proprio": torch.zeros(8),
        "history": torch.zeros(4, 3),
        "history_mask": torch.zeros(4, dtype=torch.bool),
        "action_proposal": torch.zeros(7),
        "remaining_budget": torch.ones(1),
        "previous_mode": torch.tensor([1.0, 0.0, 0.0]),
        "steps_to_go": torch.ones(1),
    }


class _Env:
    num_envs = 1
    action_contract = _contract()

    def restore_causal_snapshot(self, snapshot):
        del snapshot
        self.task_ids = np.array([3])
        self.trial_ids = np.array([2])
        self.reset_state_ids = np.array([152])
        self.success_once = np.array([False])
        self.returns = np.array([0.0])
        self.elapsed_steps = np.array([0])
        self.success_episode_len = np.array([0])
        self.done = np.array([False])
        return {"task_descriptions": ["synthetic"]}

    def chunk_step_with_action_trace(self, actions, live_contract):
        submitted = ActionStageStatistics.from_values(
            stage=SUBMITTED_LIBERO_ACTION_STAGE,
            values=actions,
            low=live_contract.low,
            high=live_contract.high,
            gripper_dimension_index=live_contract.gripper_dimension_index,
            action_contract_sha256=live_contract.canonical_sha256,
        )
        self.elapsed_steps += actions.shape[1]
        if bool((actions[..., 0] > 0).any()):
            self.success_once[:] = True
            self.returns[:] = 1.0
            self.success_episode_len[:] = self.elapsed_steps
        return ([{"task_descriptions": ["synthetic"]}], None), submitted


class _Runtime:
    action_protocol = SimpleNamespace(max_episode_steps=10)

    def sample_causal_intervention(self, *, env_obs, spec, **kwargs):
        del env_obs, kwargs
        value = 1.0 if spec.mode is CausalComputeMode.C2_FULL else 0.0
        actions = torch.full((1, 10, 7), value)
        trace = ActionExecutionTrace(
            (
                ActionStageStatistics.from_values(
                    stage=NORMALIZED_ACTION_STAGE,
                    values=actions,
                    low=(-1.0,) * 7,
                    high=(1.0,) * 7,
                    gripper_dimension_index=6,
                    action_contract_sha256=_contract().canonical_sha256,
                ),
            )
        )
        future = int(spec.mode is CausalComputeMode.C2_FULL)
        return CausalChunkSample(
            mode=spec.mode,
            actions=actions,
            normalized_actions=actions,
            action_execution_trace=trace,
            condition_contract=CausalConditionContract(
                mode=spec.mode,
                logical_input_frames=9,
                logical_future_frames=8,
                current_frame_video_tokens=1,
                physical_video_tokens=1 + future,
                physical_future_tokens=future,
                control=spec.control,
            ),
            video_denoise_calls=10 * future,
            latency_ms={
                "video_denoise": 2.0 * future,
                "action_dit": 1.0,
                "critical_path": 1.0 + 2.0 * future,
            },
            control=spec.control,
            proposal_seeds=(spec.action_seed,),
            action_denoise_calls=10,
        )


def test_source_to_paired_label_to_analysis_synthetic_closed_loop() -> None:
    traces = tuple(_trace(index) for index in range(3))

    def replay_to_snapshot(trace, identity, sampling):
        runtime = CausalSnapshotV1(
            snapshot_id=identity.snapshot_id,
            worker_state={"schema": "causal-snapshot-v1"},
            wrapper_state={},
            current_raw_observation=trace.raw_observation,
            recent_history=(),
            policy_runtime_state={},
            driver_rng_state={},
            source_policy=sampling.source_policy,
            previous_mode=None,
            chunk_index=trace.chunk_index,
            remaining_budget=1.0,
        )
        snapshot = CausalSnapshotV2(
            runtime_snapshot=runtime,
            identity=identity,
            sampling=sampling,
            source_route="always_c0",
            previous_mode=None,
            remaining_budget=1.0,
            predicate_before=trace.predicate_vector,
            source_trace_summary=trace.lightweight_summary,
            parent_checkpoint_identity="parent",
            statistics_identity="statistics",
        )
        replay = {
            "raw_observation": trace.raw_observation,
            "submitted_actions": trace.submitted_actions,
            "predicate_vector": trace.predicate_vector,
        }
        return snapshot, replay

    collector = CausalSourceCollectorV2(
        run_source_episode=lambda policy: SourceEpisodeTraceV2(
            source_policy=policy,
            final_success=False,
            traces=traces,
        ),
        replay_to_snapshot=replay_to_snapshot,
        extract_gate_features=lambda _snapshot: _features(),
    )
    selected = collector.collect_one(
        source_policy="always_c0",
        stratum=CausalSamplingStratum.UNIFORM,
        selection_seed=42,
        build_identity=_identity,
    )
    env = _Env()
    runner = PairedCausalForkRunnerV2(
        env=env,
        runtime=_Runtime(),
        restore_policy_runtime=lambda _state: None,
        restore_history=lambda _history: None,
        observe_branch=lambda current_env: BranchObservationV2(
            predicate_vector=(bool(current_env.success_once[0]),),
            predicate_progress=float(current_env.success_once[0]),
            contact_events={"target": int(current_env.success_once[0])},
        ),
    )
    specs = []
    for replicate in range(2):
        seed = 42 + 101 * replicate
        specs.extend(
            (
                CausalInterventionSpecV2(
                    mode=CausalComputeMode.C0_CURRENT,
                    control=CausalControlKind.STANDARD,
                    treatment_chunks=1,
                    continuation_mode=CausalComputeMode.C2_FULL,
                    replicate=replicate,
                    action_seed=seed,
                    video_seed=None,
                ),
                CausalInterventionSpecV2(
                    mode=CausalComputeMode.C2_FULL,
                    control=CausalControlKind.STANDARD,
                    treatment_chunks=1,
                    continuation_mode=CausalComputeMode.C2_FULL,
                    replicate=replicate,
                    action_seed=seed,
                    video_seed=100 + replicate,
                ),
            )
        )
    records = runner.run_snapshot(
        snapshot=selected.snapshot,
        state_index=0,
        specs=specs,
    )
    assert len(records) == 4
    feature = CausalGateFeatureRecordV1(
        state=selected.snapshot.identity,
        tensor_shard="gate_features/shard-000.pt",
        tensor_row=0,
        proposal_variant="one_proposal",
        feature_names=(
            "current_video_kv",
            "language",
            "proprio",
            "history",
            "action_proposal",
            "remaining_budget",
            "previous_mode",
            "steps_to_go",
        ),
    )
    example = build_gate_training_example(
        feature=feature,
        records=records,
        modes=(CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL),
        fold=0,
        split="test",
    )
    assert example.empirical_uplift["c2_full"] == 1.0
    report = analyze_v2(
        [record.to_artifact() for record in records],
        population_size=3,
        bootstrap_resamples=3,
    )
    assert report["record_audit"]["status"] == "PASS"
    assert report["self_normalized_ipw_ate"] == 1.0


def test_sampling_probability_phase_and_future_donors_are_frozen() -> None:
    selected, metadata = select_source_chunk(
        tuple(_trace(index) for index in range(3)),
        stratum="uniform",
        seed=2,
        source_policy="always_c0",
        source_final_success=False,
    )
    assert metadata.joint_inclusion_probability == 0.5 / 3
    assert metadata.phase.value in {
        "pre_contact",
        "contact",
        "subgoal_transition",
    }
    entries = [
        FutureLatentEntryV1(
            state_id=f"state-{index}",
            suite="spatial",
            global_task_uid="spatial:3",
            source_episode_id="episode-a" if index != 1 else "episode-b",
            chunk_index=index * 2,
            criticality_bin=0,
            instruction_id="instruction-a" if index != 2 else "instruction-b",
            tensor_shard="future.pt",
            tensor_row=index,
        )
        for index in range(3)
    ]
    donors = assign_future_donors(entries)
    assert "state-0" in donors
    assert selected.chunk_index in {0, 1, 2}

    values = _trace(0).__dict__.copy()
    values["remaining_action_capacity"] = 9
    with pytest.raises(ValueError, match="complete treatment block"):
        SourceChunkTraceV2(**values)


def test_same_chunk_budget_debits_overhead_and_falls_back_to_fastest() -> None:
    from fastwam.causal_prediction import UpliftGateOutput

    output = UpliftGateOutput(
        modes=(CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL),
        q_values=torch.tensor([[0.1, 0.9]]),
        uplift=torch.tensor([[0.0, 0.8]]),
        uncertainty=torch.zeros(1, 2),
        normalized_cost=torch.tensor([[0.0, 1.0]]),
    )
    budget = EpisodeComputeBudgetV2(total_cost=1.0, remaining_cost=1.0)
    budget.debit_overhead(proposal_cost=0.2, gate_cost=0.1)
    assert budget.select(output, beta=0.0, cost_weight=0.0) == 0
    assert budget.remaining_cost == pytest.approx(0.7)


def test_synthetic_episode_routes_chunk_zero_and_hard_budget_falls_back() -> None:
    env = _Env()
    env.restore_causal_snapshot(None)
    runtime = _Runtime()
    runner = SameChunkCausalEvaluationRunnerV2(
        env=env,
        runtime=runtime,
        action_contract=_contract(),
    )
    seen_contexts = []

    def decide(context):
        seen_contexts.append(context)
        return SameChunkRouteDecisionV2(
            desired_mode=CausalComputeMode.C2_FULL,
            mode_costs={
                CausalComputeMode.C0_CURRENT: 0.0,
                CausalComputeMode.C2_FULL: 1.0,
            },
            proposal_cost=0.2,
            gate_cost=0.1,
            proposal_calls=1,
            gate_calls=1,
        )

    result = runner.run_episode(
        initial_observation={"task_descriptions": ["synthetic"]},
        route_decision=decide,
        seed_schedule=lambda chunk: (42 + chunk, 100 + chunk),
        budget=EpisodeComputeBudgetV2(total_cost=0.5, remaining_cost=0.5),
    )
    assert len(seen_contexts) == len(result.chunks) == 1
    assert seen_contexts[0].chunk_index == 0
    assert seen_contexts[0].no_history
    assert seen_contexts[0].no_previous_route
    assert result.chunks[0].desired_mode is CausalComputeMode.C2_FULL
    assert result.chunks[0].executed_mode is CausalComputeMode.C0_CURRENT
    assert result.prediction_calls == 0
    assert result.proposal_calls == result.gate_calls == 1
    assert result.submitted_action_count == 10
    assert result.budget_remaining == pytest.approx(0.2)
