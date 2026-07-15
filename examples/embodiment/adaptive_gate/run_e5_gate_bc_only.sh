#!/usr/bin/env bash
# SETTING: Frozen Plus-Full evaluation of the cross-fitted G-uplift Gate with no online policy-gradient update.
# MODEL/CHECKPOINT LINEAGE: final S-DR frozen WAM + independent G-uplift checkpoint; this is BC-only, not GRPO.
# SCIENTIFIC GOAL: Measure how far direct counterfactual supervision travels before task-reward fine-tuning.
# ACCEPTANCE: Strict WAM/paired/feature sidecar binding passes and one canonical 70-slot reference trace is complete.
# REQUIRED INPUTS: UPLIFT_CKPT, shared WAM artifacts, PLUS_FULL_MANIFEST, E4_DECISION and pinned Plus checkout.
# OUTPUTS: per-suite runs strictly merged into one logical Plus-Full reference trace, configs, manifests and decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
require_env UPLIFT_CKPT
require_env PLUS_FULL_MANIFEST
require_env E4_DECISION
require_file "${UPLIFT_CKPT}"
require_file "${UPLIFT_CKPT}.meta.json"
require_file "${PLUS_FULL_MANIFEST}"
require_passed_decision "${E4_DECISION}"
configure_plus_runtime "${PLUS_FULL_MANIFEST}"
validate_plus_manifest "${PLUS_FULL_MANIFEST}"
gate_wam_overrides

CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E5_uplift_bc_only/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
TRACE="${RUN_DIR}/reference_trace.jsonl"
SUITE_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e5_bc_suites.XXXXXX")
trap 'rm -f "${SUITE_TSV}"' EXIT
build_plus_suite_plan "${PLUS_FULL_MANIFEST}" \
    "${RUN_DIR}/plus_suite_manifests" "${SUITE_TSV}"
SUITE_TRACE_BINDINGS=()
while IFS=$'\t' read -r task_suite suite_manifest suite_episodes _logical_sha; do
    [[ -n "${task_suite}" ]] || continue
    suite_slug=$(basename "${suite_manifest}" .json)
    SUITE_DIR="${RUN_DIR}/suites/${suite_slug}"
    SUITE_TRACE="${SUITE_DIR}/reference_trace.jsonl"
    prepare_run_dir "${SUITE_DIR}"
    OVERRIDES=(
        "${GATE_WAM_OVERRIDES[@]}"
        "runner.only_eval=true"
        "runner.ckpt_path=${UPLIFT_CKPT}"
        "runner.logger.log_path=${SUITE_DIR}"
        "runner.logger.experiment_name=e5_uplift_bc_${suite_slug}"
        "actor.model.gate.bc_init_path=${UPLIFT_CKPT}"
        "actor.model.gate.eval_policy.kind=learned"
        "actor.model.gate.eval_policy.trace_path=${SUITE_TRACE}"
        "actor.model.gate.eval_control.kind=null"
        "env.eval.task_suite_name=${task_suite}"
        "env.eval.episode_manifest_path=${suite_manifest}"
        "env.eval.total_num_envs=${suite_episodes}"
        "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
    )
    SUITE_SCOPE_START=${#RUN_COMMAND_LOG[@]}
    resolve_rlinf_gate_config "${CONFIG_NAME}" \
        "${SUITE_DIR}/resolved_config.yaml" "${OVERRIDES[@]}"
    RUN_ARTIFACTS=(
        "${UPLIFT_CKPT}" "${UPLIFT_CKPT}.meta.json" "${SHARED_CKPT}"
        "${DATASET_STATS}" "${COST_PROFILE}" "${PLUS_FULL_MANIFEST}"
        "${suite_manifest}" "${E4_DECISION}"
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
run_command python "${DECISION_TOOL}" p0 --check uplift_bc_only_trace_complete \
    --evidence "${TRACE}" --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
