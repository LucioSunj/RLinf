#!/usr/bin/env bash
# SETTING: Full frozen Plus-Full evaluation of a preregistered 4-6 point compute sweep across five Gate training seeds.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR + independently trained G-action/G-uplift/GRPO checkpoints listed in GATE_SWEEP_SPEC.
# SCIENTIFIC GOAL: Measure learned success-compute points and emit complete 70-slot reference traces for matched-budget controls.
# ACCEPTANCE: Complete >=4-budget x >=5-seed grid with monotone distinct lambda coordinates, validation usage within <=0.10 of target, adjacent mean usage separated by >=0.03, and complete Plus-Full traces.
# REQUIRED INPUTS: v3 GATE_SWEEP_SPEC listing immutable E5 checkpoint evidence registrations, shared WAM artifacts, PLUS_FULL_MANIFEST, GATE_PHASE_FN, and pinned Plus checkout.
# OUTPUTS: per-suite runs strictly merged into one logical Plus-Full trace per budget-seed, learned_trace_index.json, and decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
require_env GATE_SWEEP_SPEC
require_env PLUS_FULL_MANIFEST
require_env GATE_PHASE_FN
require_file "${GATE_SWEEP_SPEC}"
require_file "${PLUS_FULL_MANIFEST}"
configure_plus_runtime "${PLUS_FULL_MANIFEST}"
validate_plus_manifest "${PLUS_FULL_MANIFEST}"
gate_wam_overrides

CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
MIN_BUDGETS=${MIN_BUDGETS:-4}
MAX_BUDGETS=${MAX_BUDGETS:-6}
MIN_SEEDS=${MIN_SEEDS:-5}
MIN_ACTUAL_SEPARATION=${MIN_ACTUAL_SEPARATION:-0.03}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E6_learned_sweep/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
PLAN_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e6_sweep.XXXXXX")
SUITE_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e6_suites.XXXXXX")
trap 'rm -f "${PLAN_TSV}" "${SUITE_TSV}"' EXIT
build_plus_suite_plan "${PLUS_FULL_MANIFEST}" \
    "${RUN_DIR}/plus_suite_manifests" "${SUITE_TSV}"
VALIDATE_CMD=(
    python "${TRACE_INDEX_TOOL}" validate-sweep
    --spec "${GATE_SWEEP_SPEC}"
    --out-tsv "${PLAN_TSV}"
    --min-budgets "${MIN_BUDGETS}"
    --max-budgets "${MAX_BUDGETS}"
    --min-seeds "${MIN_SEEDS}"
    --min-actual-separation "${MIN_ACTUAL_SEPARATION}"
)
run_planning_command "${VALIDATE_CMD[@]}"

INDEX="${RUN_DIR}/learned_trace_index.json"
while IFS=$'\t' read -r checkpoint budget gate_seed stable lambda_cost \
    usage_tolerance validation_idm_usage checkpoint_sha checkpoint_step \
    diagnostics_sha e5_decision_sha e5_evidence_sha \
    training_manifest_sha e5_evidence_path diagnostics_path e5_decision_path \
    training_manifest_path; do
    [[ -n "${checkpoint}" ]] || continue
    require_file "${checkpoint}"
    require_file "${checkpoint}.meta.json"
    POINT_DIR="${RUN_DIR}/budget_${budget}/seed_${gate_seed}"
    prepare_run_dir "${POINT_DIR}"
    TRACE="${POINT_DIR}/learned_trace.jsonl"
    SUITE_TRACE_BINDINGS=()
    while IFS=$'\t' read -r task_suite suite_manifest suite_episodes _logical_sha; do
        [[ -n "${task_suite}" ]] || continue
        suite_slug=$(basename "${suite_manifest}" .json)
        SUITE_DIR="${POINT_DIR}/suites/${suite_slug}"
        SUITE_TRACE="${SUITE_DIR}/learned_trace.jsonl"
        prepare_run_dir "${SUITE_DIR}"
        OVERRIDES=(
            "${GATE_WAM_OVERRIDES[@]}"
            "runner.only_eval=true"
            "runner.ckpt_path=${checkpoint}"
            "runner.logger.log_path=${SUITE_DIR}"
            "runner.logger.experiment_name=e6_learned_b${budget}_seed${gate_seed}_${suite_slug}"
            "actor.seed=${gate_seed}"
            "env.eval.seed=${gate_seed}"
            "env.eval.task_suite_name=${task_suite}"
            "env.eval.episode_manifest_path=${suite_manifest}"
            "env.eval.gate_phase_fn=${GATE_PHASE_FN}"
            "env.eval.total_num_envs=${suite_episodes}"
            "actor.model.gate.bc_init_path=${checkpoint}"
            "actor.model.gate.kl_prior.enabled=false"
            "actor.model.gate.eval_policy.kind=learned"
            "actor.model.gate.eval_policy.seed=${gate_seed}"
            "actor.model.gate.eval_policy.trace_path=${SUITE_TRACE}"
            "actor.model.gate.eval_control.kind=null"
            "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
        )
        SUITE_SCOPE_START=${#RUN_COMMAND_LOG[@]}
        resolve_rlinf_gate_config "${CONFIG_NAME}" \
            "${SUITE_DIR}/resolved_config.yaml" "${OVERRIDES[@]}"
        RUN_ARTIFACTS=(
            "${GATE_SWEEP_SPEC}" "${checkpoint}" "${checkpoint}.meta.json"
            "${SHARED_CKPT}" "${DATASET_STATS}" "${COST_PROFILE}"
            "${PLUS_FULL_MANIFEST}" "${suite_manifest}"
            "${e5_evidence_path}"
            "${diagnostics_path}" "${e5_decision_path}"
            "${training_manifest_path}"
        )
        CMD=(bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" -- "${OVERRIDES[@]}")
        run_command "${CMD[@]}"
        write_scoped_run_manifest \
            "${SUITE_DIR}/run_manifest.json" "${SUITE_SCOPE_START}"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_file "${SUITE_TRACE}"
        fi
        SUITE_TRACE_BINDINGS+=("${task_suite}=${SUITE_TRACE}")
    done < "${SUITE_TSV}"
    merge_plus_suite_traces "${PLUS_FULL_MANIFEST}" "${TRACE}" \
        "${SUITE_TRACE_BINDINGS[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        require_file "${TRACE}"
    fi
    run_command python "${TRACE_INDEX_TOOL}" add --out "${INDEX}" \
        --path "${TRACE}" --role learned --budget "${budget}" \
        --gate-seed "${gate_seed}" --stable "${stable}" \
        --lambda-cost "${lambda_cost}" \
        --usage-tolerance "${usage_tolerance}" \
        --validation-idm-usage "${validation_idm_usage}" \
        --checkpoint-sha256 "${checkpoint_sha}" \
        --checkpoint-step "${checkpoint_step}" \
        --diagnostics-sha256 "${diagnostics_sha}" \
        --e5-decision-sha256 "${e5_decision_sha}" \
        --e5-evidence-sha256 "${e5_evidence_sha}" \
        --e5-evidence-path "${e5_evidence_path}" \
        --run-manifest-sha256 "${training_manifest_sha}"
done < "${PLAN_TSV}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${INDEX}"
fi
run_command python "${TRACE_INDEX_TOOL}" validate --index "${INDEX}"
run_command python "${DECISION_TOOL}" contract --check e6_learned_sweep_complete \
    --evidence "${INDEX}" --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
