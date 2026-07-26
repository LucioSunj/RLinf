#!/usr/bin/env bash
# SETTING: Suite-partitioned, logically merged Plus-Full forced endpoints plus exact-episode, task/factor and learned-reference-phase Random-K schedules.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR; EVAL_GATE_CKPT only supplies a contract-compatible Gate shell for fixed selectors.
# SCIENTIFIC GOAL: Build strict equal-budget controls that separate state-dependent allocation from endpoints and random compute placement.
# ACCEPTANCE: Forced U/I and all three matched-random roles cover identical Plus-Full episode IDs; each schedule conserves its registered quota.
# REQUIRED INPUTS: EVAL_GATE_CKPT, LEARNED_TRACE_INDEX, E6_LEARNED_DECISION, E5_DECISION, shared WAM artifacts, PLUS_FULL_MANIFEST and GATE_PHASE_FN.
# OUTPUTS: per-suite runs merged into forced/random logical traces, schedules, baseline_trace_index.json, configs, manifests and decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
for name in EVAL_GATE_CKPT LEARNED_TRACE_INDEX E6_LEARNED_DECISION E5_DECISION PLUS_FULL_MANIFEST; do
    require_env "${name}"
done
require_env GATE_PHASE_FN
require_file "${EVAL_GATE_CKPT}"
require_file "${EVAL_GATE_CKPT}.meta.json"
require_file "${LEARNED_TRACE_INDEX}"
require_file "${PLUS_FULL_MANIFEST}"
require_passed_decision "${E5_DECISION}"
require_passed_decision "${E6_LEARNED_DECISION}"
configure_plus_runtime "${PLUS_FULL_MANIFEST}"
validate_plus_manifest "${PLUS_FULL_MANIFEST}"
gate_wam_overrides

CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E6_forced_random/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
INDEX="${RUN_DIR}/baseline_trace_index.json"
LEARNED_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e6_learned.XXXXXX")
SUITE_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e6_suites.XXXXXX")
trap 'rm -f "${LEARNED_TSV}" "${SUITE_TSV}"' EXIT
build_plus_suite_plan "${PLUS_FULL_MANIFEST}" \
    "${RUN_DIR}/plus_suite_manifests" "${SUITE_TSV}"
EXPORT_CMD=(python "${TRACE_INDEX_TOOL}" export --index "${LEARNED_TRACE_INDEX}" --role learned --out-tsv "${LEARNED_TSV}")
run_planning_command "${EXPORT_CMD[@]}"

run_fixed_eval() {
    local label=$1
    local role=$2
    local trace=$3
    local selector_kind=$4
    local mode_manifest=${5:-}
    local run_dir
    run_dir=$(dirname "${trace}")
    prepare_run_dir "${run_dir}"
    local suite_bindings=()
    local task_suite suite_manifest suite_episodes suite_slug suite_dir suite_trace
    while IFS=$'\t' read -r task_suite suite_manifest suite_episodes _logical_sha; do
        [[ -n "${task_suite}" ]] || continue
        suite_slug=$(basename "${suite_manifest}" .json)
        suite_dir="${run_dir}/suites/${suite_slug}"
        suite_trace="${suite_dir}/trace.jsonl"
        prepare_run_dir "${suite_dir}"
        local overrides=(
            "${GATE_WAM_OVERRIDES[@]}"
            "runner.only_eval=true"
            "runner.ckpt_path=${EVAL_GATE_CKPT}"
            "runner.logger.log_path=${suite_dir}"
            "runner.logger.experiment_name=${label}_${suite_slug}"
            "env.eval.task_suite_name=${task_suite}"
            "env.eval.episode_manifest_path=${suite_manifest}"
            "env.eval.gate_phase_fn=${GATE_PHASE_FN}"
            "env.eval.total_num_envs=${suite_episodes}"
            "actor.model.gate.bc_init_path=${EVAL_GATE_CKPT}"
            "actor.model.gate.kl_prior.enabled=false"
            "actor.model.gate.eval_policy.kind=${selector_kind}"
            "actor.model.gate.eval_policy.trace_path=${suite_trace}"
            "actor.model.gate.eval_control.kind=null"
        )
        if [[ "${selector_kind}" == "forced" ]]; then
            if [[ "${role}" == "forced_uncond" ]]; then
                overrides+=("actor.model.gate.eval_policy.mode=0")
            else
                overrides+=("actor.model.gate.eval_policy.mode=1")
            fi
        else
            overrides+=("actor.model.gate.eval_policy.manifest_path=${mode_manifest}")
        fi
        if [[ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]]; then
            overrides+=("${EXTRA_OVERRIDES[@]}")
        fi
        local suite_scope_start=${#RUN_COMMAND_LOG[@]}
        resolve_rlinf_gate_config "${CONFIG_NAME}" \
            "${suite_dir}/resolved_config.yaml" "${overrides[@]}"
        RUN_ARTIFACTS=(
            "${EVAL_GATE_CKPT}" "${EVAL_GATE_CKPT}.meta.json" "${SHARED_CKPT}"
            "${DATASET_STATS}" "${COST_PROFILE}" "${PLUS_FULL_MANIFEST}"
            "${suite_manifest}" "${E5_DECISION}" "${E6_LEARNED_DECISION}"
            "${LEARNED_TRACE_INDEX}"
        )
        if [[ -n "${mode_manifest}" ]]; then
            RUN_ARTIFACTS+=("${mode_manifest}")
        fi
        local cmd=(bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" -- "${overrides[@]}")
        run_command "${cmd[@]}"
        write_scoped_run_manifest \
            "${suite_dir}/run_manifest.json" "${suite_scope_start}"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_file "${suite_trace}"
        fi
        suite_bindings+=("${task_suite}=${suite_trace}")
    done < "${SUITE_TSV}"
    merge_plus_suite_traces "${PLUS_FULL_MANIFEST}" "${trace}" \
        "${suite_bindings[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        require_file "${trace}"
    fi
}

UNCOND_TRACE="${RUN_DIR}/forced_uncond/trace.jsonl"
IDM_TRACE="${RUN_DIR}/forced_idm/trace.jsonl"
run_fixed_eval e6_forced_uncond forced_uncond "${UNCOND_TRACE}" forced
run_command python "${TRACE_INDEX_TOOL}" add --out "${INDEX}" --path "${UNCOND_TRACE}" --role forced_uncond
run_fixed_eval e6_forced_idm forced_idm "${IDM_TRACE}" forced
run_command python "${TRACE_INDEX_TOOL}" add --out "${INDEX}" --path "${IDM_TRACE}" --role forced_idm

while IFS=$'\t' read -r reference_trace _role budget gate_seed _stable; do
    [[ -n "${reference_trace}" ]] || continue
    reference_trace_sha=$(file_sha256 "${reference_trace}")
    for matched in reference_random_k reference_task_factor reference_phase; do
        case "${matched}" in
            reference_random_k) analysis_role=episode_random_k ;;
            reference_task_factor) analysis_role=task_random_k ;;
            reference_phase) analysis_role=reference_phase_random_k ;;
        esac
        POINT_DIR="${RUN_DIR}/budget_${budget}/seed_${gate_seed}/${matched}"
        prepare_run_dir "${POINT_DIR}"
        MODE_MANIFEST="${POINT_DIR}/mode_manifest.json"
        TRACE="${POINT_DIR}/trace.jsonl"
        # --final is the call-site half of the two-key test-split lock; the
        # operator must additionally export STAGE2_FINAL_EVAL=1 (never set here).
        BUILD_CMD=(
            python examples/embodiment/build_gate_mode_manifest.py
            --final
            --episode-manifest "${PLUS_FULL_MANIFEST}"
            --checkpoint "${SHARED_CKPT}"
            --reference-trace "${reference_trace}"
            --kind "${matched}"
            --max-decisions 70
            --seed "${gate_seed}"
            --out "${MODE_MANIFEST}"
        )
        run_command "${BUILD_CMD[@]}"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_file "${MODE_MANIFEST}"
        fi
        run_fixed_eval "e6_${matched}_b${budget}_seed${gate_seed}" \
            "${analysis_role}" "${TRACE}" manifest "${MODE_MANIFEST}"
        run_command python "${TRACE_INDEX_TOOL}" add --out "${INDEX}" \
            --path "${TRACE}" --role "${analysis_role}" --budget "${budget}" \
            --gate-seed "${gate_seed}" \
            --reference-trace-sha256 "${reference_trace_sha}"
    done
done < "${LEARNED_TSV}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${INDEX}"
fi
run_command python "${TRACE_INDEX_TOOL}" validate --index "${INDEX}"
run_command python "${DECISION_TOOL}" contract --check e6_forced_and_matched_random_complete \
    --evidence "${INDEX}" --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
