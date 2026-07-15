"""CPU-only matched-budget selector, trace, and GRPO diagnostic tests."""

from __future__ import annotations

import json

import pytest
import torch

from _gate_test_imports import load_gate_modules


mods = load_gate_modules()
selectors = mods.selectors
diagnostics = mods.diagnostics
trace = mods.trace


def _context(*, uid="episode-a", slots=(0,), phase=None, reliable=True):
    batch = len(slots)
    return {
        "episode_uid": [uid] * batch,
        "decision_index": torch.tensor(slots),
        "task_description": ["pick object"] * batch,
        "task_suite_name": ["libero_10"] * batch,
        "task_id": torch.zeros(batch, dtype=torch.long),
        "factor": ["camera"] * batch,
        "level": ["L3"] * batch,
        "reset_state_id": torch.full((batch,), 17),
        "env_seed": torch.full((batch,), 123),
        "episode_manifest_sha256": ["e" * 64] * batch,
        "phase": [phase or "approach"] * batch,
        "phase_reliable": torch.full((batch,), reliable, dtype=torch.bool),
    }


def _select(selector, context):
    return selector.select(torch.tensor([[1.0, 0.0]] * len(context["episode_uid"])), context)


def test_learned_and_forced_selectors_preserve_binary_contract():
    learned = selectors.build_eval_mode_selector(
        {"kind": "learned", "max_decisions": 70}
    )
    result = learned.select(torch.tensor([[0.0, 2.0], [3.0, 0.0]]))
    assert result.modes.tolist() == [1, 0]
    assert result.reserved_modes is None

    forced = selectors.build_eval_mode_selector(
        {"kind": "forced", "mode": 1, "max_decisions": 70}
    )
    result = _select(forced, _context(slots=(0, 12)))
    assert result.modes.tolist() == [1, 1]
    assert result.reserved_modes.shape == (2, 70)
    assert result.reserved_idm_count.tolist() == [70, 70]


def test_random_k_reserves_exactly_k_of_70_and_is_order_independent():
    schedule = selectors.build_random_k_schedule(
        episode_uid="episode-a", max_decisions=70, k=18, seed=9
    )
    assert len(schedule) == 70 and sum(schedule) == 18
    assert schedule == selectors.build_random_k_schedule(
        episode_uid="episode-a", max_decisions=70, k=18, seed=9
    )
    assert schedule != selectors.build_random_k_schedule(
        episode_uid="episode-b", max_decisions=70, k=18, seed=9
    )
    for k in (0, 70):
        assert sum(
            selectors.build_random_k_schedule(
                episode_uid="x", max_decisions=70, k=k, seed=0
            )
        ) == k


def test_random_k_selection_uses_registered_slot_not_batch_rng():
    selector = selectors.RandomKSelector(max_decisions=70, k=11, seed=4)
    context = _context(slots=(3, 19, 68))
    result = _select(selector, context)
    expected = selector.schedule_for("episode-a")
    assert result.modes.tolist() == [expected[3], expected[19], expected[68]]
    assert result.reserved_modes[0].sum().item() == 11


def test_periodic_k_has_exact_budget_and_balanced_gaps():
    schedule = selectors.build_periodic_k_schedule(max_decisions=70, k=9)
    indices = [index for index, mode in enumerate(schedule) if mode]
    assert len(indices) == 9
    gaps = [right - left for left, right in zip(indices, indices[1:])]
    assert max(gaps) - min(gaps) <= 1


def test_episode_mixture_is_constant_and_bernoulli_is_preregistered():
    mixture = selectors.EpisodeMixtureSelector(
        max_decisions=70, p_idm=0.5, seed=2
    )
    mixed = mixture.schedule_for("episode-a")
    assert len(set(mixed)) == 1
    bernoulli = selectors.BernoulliSelector(
        max_decisions=70, p_idm=0.5, seed=2
    )
    schedule = bernoulli.schedule_for("episode-a")
    assert schedule == bernoulli.schedule_for("episode-a")
    assert len(schedule) == 70


def test_phase_heuristic_requires_reliable_pretreatment_phase():
    selector = selectors.PhaseHeuristicSelector(
        max_decisions=70,
        idm_phases=["contact_alignment"],
        seed=0,
    )
    result = _select(
        selector,
        {
            **_context(slots=(2, 3)),
            "phase": ["approach", "contact_alignment"],
        },
    )
    assert result.modes.tolist() == [0, 1]
    with pytest.raises(ValueError, match="phase_reliable"):
        _select(selector, _context(slots=(0,), reliable=False))


def test_manifest_selector_validates_length_identity_and_hashes(tmp_path):
    manifest = tmp_path / "modes.json"
    payload = {
        "version": 1,
        "max_decisions": 70,
        "provenance": {
            "checkpoint_sha256": "c" * 64,
            "episode_manifest_sha256": "e" * 64,
        },
        "episodes": {"episode-a": {"reserved_modes": [0] * 69 + [1]}},
    }
    manifest.write_text(json.dumps(payload))
    selector = selectors.ManifestSelector(
        manifest_path=manifest,
        expected_checkpoint_sha256="c" * 64,
        expected_episode_manifest_sha256="e" * 64,
        max_decisions=70,
    )
    assert selector.schedule_for("episode-a")[-1] == 1
    assert selector.manifest_sha256 == selectors.sha256_file(manifest)
    assert _select(selector, _context(slots=(69,))).modes.item() == 1
    bad_context = {**_context(slots=(0,)), "episode_manifest_sha256": ["bad"]}
    with pytest.raises(ValueError, match="runtime episode manifest SHA"):
        _select(selector, bad_context)
    with pytest.raises(KeyError, match="missing"):
        selector.schedule_for("other")

    payload["episodes"]["episode-a"]["reserved_modes"] = [0] * 69
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exactly 70"):
        selectors.ManifestSelector(manifest_path=manifest, max_decisions=70)


def _reference_records():
    records = []
    specs = (
        ("episode-a", "pick", "camera", 10),
        ("episode-b", "pick", "camera", 20),
        ("episode-c", "place", "layout", 5),
    )
    for uid, task, factor, usage in specs:
        records.append(
            {
                "schema_version": 2,
                "method": "learned",
                "episode_uid": uid,
                "task": task,
                "factor": factor,
                "max_decisions": 70,
                "reference_modes": [1] * usage + [0] * (70 - usage),
                "reference_phase": ["approach"] * 35
                + ["contact_alignment"] * 35,
                "reference_phase_reliable": [True] * 70,
                "wam_checkpoint_sha256": "w" * 64,
                "gate_checkpoint_sha256": "g" * 64,
                "episode_manifest_sha256": "e" * 64,
            }
        )
    return records


def _reference_manifest(method, records=None):
    records = _reference_records() if records is None else records
    return selectors.make_reference_matched_mode_manifest(
        records=records,
        method=method,
        episode_uids=["episode-a", "episode-b", "episode-c"],
        checkpoint_sha256="w" * 64,
        episode_manifest_sha256="e" * 64,
        reference_trace_sha256="r" * 64,
        seed=9,
        max_decisions=70,
    )


def test_reference_random_k_preserves_each_episode_quota_and_order_independence():
    forward = _reference_manifest("reference_random_k")
    reverse = _reference_manifest(
        "reference_random_k", list(reversed(_reference_records()))
    )
    assert forward == reverse
    for uid, expected in (
        ("episode-a", 10),
        ("episode-b", 20),
        ("episode-c", 5),
    ):
        modes = forward["episodes"][uid]["reserved_modes"]
        assert len(modes) == 70 and sum(modes) == expected
    assert forward["provenance"]["determinism"].startswith("sha256")


def test_reference_task_factor_and_phase_matching_conserve_cell_quotas():
    task_factor = _reference_manifest("reference_task_factor")
    assert sum(
        task_factor["episodes"][uid]["reserved_idm_calls"]
        for uid in ("episode-a", "episode-b")
    ) == 30
    assert task_factor["episodes"]["episode-c"]["reserved_idm_calls"] == 5
    assert all(
        quota["reference_idm_calls"] == quota["reserved_idm_calls"]
        for quota in task_factor["provenance"]["quota_conservation"]
    )

    phase = _reference_manifest("reference_phase")
    provenance = phase["provenance"]
    assert provenance["reference_phase_matching"] is True
    assert "not_strict_post_branch" in provenance["reference_phase_semantics"]
    assert provenance["total_reference_idm_calls"] == 35
    assert provenance["total_reserved_idm_calls"] == 35
    assert all(
        quota["reference_idm_calls"] == quota["reserved_idm_calls"]
        for quota in provenance["quota_conservation"]
    )


def test_reference_manifest_fails_closed_on_identity_slots_and_phase_quality():
    records = _reference_records()
    records[0] = {**records[0], "wam_checkpoint_sha256": "bad"}
    with pytest.raises(ValueError, match="checkpoint SHA"):
        _reference_manifest("reference_random_k", records)

    records = _reference_records()
    records[0] = {**records[0], "reference_modes": [0] * 69}
    with pytest.raises(ValueError, match="exactly 70"):
        _reference_manifest("reference_random_k", records)

    records = _reference_records()
    records[0] = {
        **records[0],
        "reference_phase": ["unknown"] + records[0]["reference_phase"][1:],
        "reference_phase_reliable": [False] + [True] * 69,
    }
    phase_manifest = _reference_manifest("reference_phase", records)
    assert phase_manifest["provenance"]["reference_phase_unreliable_slots"] == 1
    assert any(
        quota["cell"][-1] == "UNKNOWN"
        for quota in phase_manifest["provenance"]["quota_conservation"]
    )

    with pytest.raises(ValueError, match="episode set"):
        selectors.make_reference_matched_mode_manifest(
            records=_reference_records(),
            method="reference_random_k",
            episode_uids=["episode-a", "episode-b"],
            checkpoint_sha256="w" * 64,
            episode_manifest_sha256="e" * 64,
            reference_trace_sha256="r" * 64,
            seed=0,
            max_decisions=70,
        )


def test_reference_manifest_rejects_empty_identity_and_quota_mismatch():
    with pytest.raises(ValueError, match="unique, non-empty"):
        selectors.make_reference_matched_mode_manifest(
            records=_reference_records(),
            method="reference_random_k",
            episode_uids=[""],
            checkpoint_sha256="w" * 64,
            episode_manifest_sha256="e" * 64,
            reference_trace_sha256="r" * 64,
            seed=0,
        )

    records = _reference_records()
    records[0] = {**records[0], "reference_idm_calls": 11}
    with pytest.raises(ValueError, match="declared IDM quota"):
        _reference_manifest("reference_random_k", records)


def test_reference_phase_manifest_rejects_non_boolean_reliability():
    records = _reference_records()
    records[0] = {
        **records[0],
        "reference_phase_reliable": ["true"] * 70,
    }
    with pytest.raises(ValueError, match="reliability must be boolean"):
        _reference_manifest("reference_phase", records)


def test_reference_jsonl_loader_and_manifest_selector_keep_provenance(tmp_path):
    reference_path = tmp_path / "reference.jsonl"
    reference_path.write_text(
        "\n".join(json.dumps(record) for record in _reference_records()) + "\n"
    )
    assert selectors.load_canonical_reference_trace(reference_path) == (
        _reference_records()
    )
    payload = _reference_manifest("reference_phase")
    manifest = tmp_path / "reference_modes.json"
    selectors.write_json_atomic(manifest, payload)
    loaded = selectors.ManifestSelector(
        manifest_path=manifest,
        expected_checkpoint_sha256="w" * 64,
        expected_episode_manifest_sha256="e" * 64,
        max_decisions=70,
    )
    provenance = loaded.provenance()
    assert provenance["mode_manifest_sha256"] == selectors.sha256_file(manifest)
    assert provenance["manifest_provenance"]["reference_phase_matching"] is True


def test_selector_rejects_out_of_range_decision_slot():
    selector = selectors.RandomKSelector(max_decisions=70, k=10, seed=0)
    with pytest.raises(ValueError, match="decision_index"):
        _select(selector, _context(slots=(70,)))


def test_grpo_group_diagnostics_identify_zero_signal_groups():
    metrics = diagnostics.compute_grpo_group_diagnostics(
        rewards=torch.tensor([[[0.0], [0.0], [0.0], [2.0]]]),
        dones=torch.zeros(2, 4, 1, dtype=torch.bool),
        advantages=torch.tensor([[[0.0], [0.0], [-1.0], [1.0]]]),
        loss_mask=torch.ones(1, 4, 1, dtype=torch.bool),
        group_size=2,
    )
    assert metrics["gate/group_return_variance"] == pytest.approx(0.5)
    assert metrics["gate/nonzero_return_variance_fraction"] == pytest.approx(0.5)
    assert metrics["gate/nonzero_return_variance_group_count"] == 1.0
    assert metrics["gate/zero_advantage_group_fraction"] == pytest.approx(0.5)
    assert metrics["gate/zero_advantage_group_count"] == 1.0
    assert metrics["gate/group_return_variance_sum"] == pytest.approx(1.0)
    assert metrics["gate/effective_sample_fraction"] == pytest.approx(0.5)
    assert metrics["gate/effective_group_count"] == pytest.approx(1.0)
    assert metrics["gate/effective_sample_count"] == pytest.approx(2.0)


def _rank_diagnostics(
    *, groups, nonzero, zero_advantage, variance, effective_groups, effective_samples
):
    samples = groups * 8
    return {
        "gate/group_count": float(groups),
        "gate/nonzero_return_variance_fraction": float(nonzero),
        "gate/nonzero_return_variance_group_count": float(groups * nonzero),
        "gate/zero_return_variance_fraction": float(1.0 - nonzero),
        "gate/zero_advantage_group_fraction": float(zero_advantage),
        "gate/zero_advantage_group_count": float(groups * zero_advantage),
        "gate/group_return_variance": float(variance),
        "gate/group_return_variance_sum": float(groups * variance),
        "gate/effective_group_count": float(effective_groups),
        "gate/effective_sample_count": float(effective_samples),
        "gate/effective_sample_fraction": float(effective_samples / samples),
    }


def test_cumulative_grpo_diagnostics_weight_ranks_and_updates_exactly():
    payload = diagnostics.new_gate_diagnostics_state(
        seed=7, target_idm_usage=0.5
    )
    diagnostics.accumulate_grpo_gate_diagnostics(
        payload,
        step=1,
        group_size=8,
        rank_metrics=[
            _rank_diagnostics(
                groups=2,
                nonzero=0.5,
                zero_advantage=0.5,
                variance=0.25,
                effective_groups=1,
                effective_samples=10,
            ),
            _rank_diagnostics(
                groups=1,
                nonzero=1.0,
                zero_advantage=0.0,
                variance=1.0,
                effective_groups=1,
                effective_samples=8,
            ),
        ],
    )
    assert payload["diagnostic_updates"] == 1
    assert payload["diagnostic_rank_batches"] == 2
    assert payload["group_count"] == 3
    assert payload["nonzero_return_variance_fraction"] == pytest.approx(2 / 3)
    assert payload["zero_advantage_group_fraction"] == pytest.approx(1 / 3)
    assert payload["group_return_variance"] == pytest.approx(0.5)
    assert payload["effective_group_count"] == 2
    assert payload["effective_sample_fraction"] == pytest.approx(18 / 24)

    diagnostics.accumulate_grpo_gate_diagnostics(
        payload,
        step=2,
        group_size=8,
        rank_metrics=[
            _rank_diagnostics(
                groups=3,
                nonzero=0.0,
                zero_advantage=1.0,
                variance=0.0,
                effective_groups=0,
                effective_samples=0,
            )
        ],
    )
    assert payload["diagnostic_updates"] == 2
    assert payload["diagnostic_rank_batches"] == 3
    assert payload["group_count"] == 6
    assert payload["nonzero_return_variance_fraction"] == pytest.approx(1 / 3)
    assert payload["zero_advantage_group_fraction"] == pytest.approx(4 / 6)
    assert payload["effective_group_count"] == 2
    assert payload["effective_sample_fraction"] == pytest.approx(18 / 48)


def test_gate_diagnostics_and_collapse_state_resume_as_one_contract():
    payload = diagnostics.new_gate_diagnostics_state(
        seed=11, target_idm_usage=0.5, evidence_run_id="a" * 64
    )
    metric = _rank_diagnostics(
        groups=2,
        nonzero=0.5,
        zero_advantage=0.5,
        variance=0.25,
        effective_groups=1,
        effective_samples=8,
    )
    for step in range(1, 4):
        diagnostics.accumulate_grpo_gate_diagnostics(
            payload, step=step, group_size=8, rank_metrics=[metric]
        )
    tracker = diagnostics.BudgetCollapseTracker(target_idm_usage=0.5)
    for usage in (0.98, 0.99, 0.97):
        diagnostics.update_gate_eval_diagnostics(
            payload, idm_usage=usage, tracker=tracker
        )
    assert payload["collapsed"] and payload["collapse_consecutive"] == 3

    restored_tracker = diagnostics.BudgetCollapseTracker(target_idm_usage=0.5)
    restored_tracker.load_state_dict(json.loads(json.dumps(tracker.state_dict())))
    restored = diagnostics.validate_gate_diagnostics_state(
        json.loads(json.dumps(payload)),
        seed=11,
        target_idm_usage=0.5,
        evidence_run_id="a" * 64,
        step=3,
        group_size=8,
        tracker=restored_tracker,
    )
    assert restored["diagnostic_updates"] == 3
    assert restored["ever_collapsed"] is True

    with pytest.raises(ValueError, match="evidence run ID"):
        diagnostics.validate_gate_diagnostics_state(
            json.loads(json.dumps(payload)),
            seed=11,
            target_idm_usage=0.5,
            evidence_run_id="b" * 64,
            step=3,
            group_size=8,
            tracker=restored_tracker,
        )

    corrupted = {**restored, "cumulative_effective_sample_count": 999}
    with pytest.raises(ValueError, match="effective samples exceed"):
        diagnostics.validate_gate_diagnostics_state(
            corrupted,
            seed=11,
            target_idm_usage=0.5,
            evidence_run_id="a" * 64,
            step=3,
            group_size=8,
            tracker=restored_tracker,
        )
    with pytest.raises(ValueError, match="not boolean"):
        diagnostics.validate_gate_diagnostics_state(
            {**restored, "schema_version": True},
            seed=11,
            target_idm_usage=0.5,
            evidence_run_id="a" * 64,
            step=3,
            group_size=8,
            tracker=restored_tracker,
        )
    with pytest.raises(ValueError, match="JSON booleans"):
        diagnostics.validate_gate_diagnostics_state(
            {**restored, "collapsed": "true"},
            seed=11,
            target_idm_usage=0.5,
            evidence_run_id="a" * 64,
            step=3,
            group_size=8,
            tracker=restored_tracker,
        )


def test_gate_diagnostics_json_is_atomic_and_machine_readable(tmp_path):
    path = tmp_path / "gate_diagnostics.json"
    payload = {
        "schema_version": 1,
        "seed": 42,
        "step": 7,
        "nonzero_return_variance_fraction": 0.75,
        "zero_advantage_group_fraction": 0.25,
        "effective_group_count": 3.0,
        "effective_sample_count": 21.0,
        "target_usage_error": 0.1,
        "collapsed": False,
        "ever_collapsed": True,
    }
    diagnostics.write_gate_diagnostics(path, payload)
    assert json.loads(path.read_text()) == payload
    diagnostics.write_gate_diagnostics(path, {**payload, "step": 8})
    assert json.loads(path.read_text())["step"] == 8
    assert not (tmp_path / ".gate_diagnostics.json.tmp").exists()


def test_budget_collapse_requires_three_consecutive_extreme_evals_and_resumes():
    tracker = diagnostics.BudgetCollapseTracker(target_idm_usage=0.5)
    assert tracker.update(0.98)["gate/collapsed"] == 0.0
    assert tracker.update(0.99)["gate/collapsed"] == 0.0
    assert tracker.update(0.97)["gate/collapsed"] == 1.0
    state = tracker.state_dict()
    restored = diagnostics.BudgetCollapseTracker(target_idm_usage=0.5)
    restored.load_state_dict(state)
    assert restored.ever_collapsed and restored.consecutive == 3
    recovered = restored.update(0.4)
    assert recovered["gate/collapsed"] == 0.0
    assert recovered["gate/ever_collapsed"] == 1.0

    corrupted = state | {"state": {**state["state"], "consecutive": 2}}
    with pytest.raises(ValueError, match="collapse flag disagrees"):
        diagnostics.BudgetCollapseTracker(
            target_idm_usage=0.5
        ).load_state_dict(corrupted)


def test_canonical_trace_merges_success_and_70_slot_schedule(tmp_path):
    context = _context(slots=(0,))
    schedule = torch.tensor([[0] * 69 + [1]])
    rollout = trace.RolloutGateTraceBuilder(
        method="random_k",
        max_decisions=70,
        selector_provenance={
            "method": "random_k",
            "mode_manifest_sha256": "m" * 64,
        },
        gate_checkpoint_sha256="g" * 64,
        wam_checkpoint_sha256="w" * 64,
    )
    rollout.add_batch(
        context=context,
        modes=torch.tensor([0]),
        costs=torch.tensor([0.2]),
        active_mask=torch.tensor([True]),
        reserved_modes=schedule,
    )
    env = trace.EnvGateTraceBuilder(max_decisions=70)
    env.register_batch(context)
    env.update_after_step(
        context_before_action=context,
        success_once=torch.tensor([True]),
        active_before_action=torch.tensor([True]),
    )
    merged = trace.merge_gate_eval_traces(
        env_records=env.records(),
        rollout_records=rollout.records(),
        expected_max_decisions=70,
    )
    record = merged[0]
    assert record["method"] == "random_k"
    assert record["success"] is True and record["success_slot"] == 0
    assert len(record["reserved_modes"]) == 70
    assert record["actual_modes_before_success"] == [0]
    assert record["actual_cost_before_success"] == pytest.approx([0.2])
    assert record["gate_checkpoint_sha256"] == "g" * 64
    output = tmp_path / "trace.jsonl"
    trace.write_gate_eval_jsonl(output, merged)
    assert json.loads(output.read_text().strip())["episode_uid"] == "episode-a"


def test_learned_trace_keeps_full_reference_but_absorbs_actual_compute():
    rollout = trace.RolloutGateTraceBuilder(
        method="learned",
        max_decisions=70,
        selector_provenance={"method": "learned"},
        gate_checkpoint_sha256="g" * 64,
        wam_checkpoint_sha256="w" * 64,
    )
    env = trace.EnvGateTraceBuilder(max_decisions=70)
    first = _context(slots=(0,))
    env.register_batch(first)
    for slot in range(70):
        context = _context(slots=(slot,))
        rollout.add_batch(
            context=context,
            modes=torch.tensor([slot % 2]),
            costs=torch.tensor([1.0 + slot]),
            active_mask=torch.tensor([slot == 0]),
            reserved_modes=None,
        )
    env.update_after_step(
        context_before_action=first,
        success_once=torch.tensor([True]),
        active_before_action=torch.tensor([True]),
    )
    merged = trace.merge_gate_eval_traces(
        env_records=env.records(),
        rollout_records=rollout.records(),
        expected_max_decisions=70,
    )[0]
    assert merged["schema_version"] == 2
    assert merged["reference_modes"] == [slot % 2 for slot in range(70)]
    assert merged["reference_idm_calls"] == 35
    assert merged["reference_reserved_idm_calls"] == 35
    assert merged["reference_total_cost"] == pytest.approx(sum(range(1, 71)))
    assert merged["reserved_modes"] == [None] * 70
    assert merged["actual_slots_before_success"] == [0]
    assert merged["actual_total_cost"] == pytest.approx(1.0)


def test_learned_trace_fails_if_post_success_reference_inference_stops():
    rollout = trace.RolloutGateTraceBuilder(
        method="learned",
        max_decisions=70,
        selector_provenance={"method": "learned"},
        gate_checkpoint_sha256="g" * 64,
        wam_checkpoint_sha256="w" * 64,
    )
    rollout.add_batch(
        context=_context(slots=(0,)),
        modes=torch.tensor([1]),
        costs=torch.tensor([1.0]),
        active_mask=torch.tensor([True]),
        reserved_modes=None,
    )
    with pytest.raises(ValueError, match="full fixed horizon"):
        rollout.records()
