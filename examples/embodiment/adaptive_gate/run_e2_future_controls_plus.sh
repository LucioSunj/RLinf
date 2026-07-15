#!/usr/bin/env bash
# SETTING: Frozen Plus-Full episodes, identical reset/seed order, five runtime interventions on one shared WAM.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR only; valid, masked, repeated, shuffled and extra-compute paths share weights.
# SCIENTIFIC GOAL: Separate useful future semantics from conditioning collapse and generic additional action compute.
# ACCEPTANCE: NoRead/ExtraCompute are within 5% latency; valid IDM beats NoRead, RepeatCurrent and Shuffled with positive CIs.
# REQUIRED INPUTS: shared WAM artifacts, CONTROL_PROFILE, PLUS_FULL_MANIFEST, E1_DECISION, phase callback and target GPU.
# OUTPUTS: five suite-merged canonical paired traces, donor bank, control_trials.jsonl, configs, manifests and E2 decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
require_env CONTROL_PROFILE
require_env PLUS_FULL_MANIFEST
require_env E1_DECISION
require_env GATE_PHASE_FN
require_file "${CONTROL_PROFILE}"
require_file "${PLUS_FULL_MANIFEST}"
require_passed_decision "${E1_DECISION}"
configure_plus_runtime "${PLUS_FULL_MANIFEST}"
validate_plus_manifest "${PLUS_FULL_MANIFEST}"
gate_wam_overrides

FASTWAM_ROOT=${FASTWAM_ROOT:-"${WORKSPACE_ROOT}/FastWAM"}
require_dir "${FASTWAM_ROOT}"
CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
WAM_SEED=${WAM_SEED:-0}
DONOR_SEED=${DONOR_SEED:-0}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}
SEED=${SEED:-0}
DELTA_REF=${DELTA_REF:-$(python -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload["metrics"]["delta_ref"]
print(value["point"] if isinstance(value, dict) else value)
' "${E1_DECISION}")}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E2_future_controls/${RUN_ID}"
DONOR_DIR="${RUN_DIR}/valid_idm/donors"
DONOR_BANK="${RUN_DIR}/shuffled_future_bank.pt"
prepare_run_dir "${RUN_DIR}"
SUITE_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e2_suites.XXXXXX")
trap 'rm -f "${SUITE_TSV}"' EXIT
build_plus_suite_plan "${PLUS_FULL_MANIFEST}" \
    "${RUN_DIR}/plus_suite_manifests" "${SUITE_TSV}"

CONTROL_TRACES=()
for control in valid_idm no_read repeat_current shuffled extra_compute; do
    CONTROL_DIR="${RUN_DIR}/${control}"
    TRACE="${CONTROL_DIR}/trace.jsonl"
    prepare_run_dir "${CONTROL_DIR}"
    CAPTURE_DIR=null
    BANK_PATH=null
    if [[ "${control}" == "valid_idm" ]]; then
        CAPTURE_DIR="${DONOR_DIR}"
        prepare_run_dir "${DONOR_DIR}"
    elif [[ "${control}" == "shuffled" ]]; then
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_file "${DONOR_BANK}"
        fi
        BANK_PATH="${DONOR_BANK}"
    fi
    SUITE_TRACE_BINDINGS=()
    while IFS=$'\t' read -r task_suite suite_manifest suite_episodes _logical_sha; do
        [[ -n "${task_suite}" ]] || continue
        suite_slug=$(basename "${suite_manifest}" .json)
        SUITE_DIR="${CONTROL_DIR}/suites/${suite_slug}"
        SUITE_TRACE="${SUITE_DIR}/trace.jsonl"
        prepare_run_dir "${SUITE_DIR}"
        OVERRIDES=(
            "${GATE_WAM_OVERRIDES[@]}"
            "runner.only_eval=true"
            "runner.ckpt_path=null"
            "runner.logger.log_path=${SUITE_DIR}"
            "runner.logger.experiment_name=${control}_${suite_slug}"
            "actor.model.gate.eval_policy.kind=learned"
            "actor.model.gate.eval_policy.trace_path=${SUITE_TRACE}"
            "actor.model.gate.eval_control.kind=${control}"
            "actor.model.gate.eval_control.profile_path=${CONTROL_PROFILE}"
            "actor.model.gate.eval_control.cost_metric=latency_ms"
            "actor.model.gate.eval_control.require_compute_matched=true"
            "actor.model.gate.eval_control.wam_seed=${WAM_SEED}"
            "actor.model.gate.eval_control.donor_seed=${DONOR_SEED}"
            "actor.model.gate.eval_control.donor_bank_path=${BANK_PATH}"
            "actor.model.gate.eval_control.expected_donor_wam_seed=${WAM_SEED}"
            "actor.model.gate.eval_control.capture_donor_dir=${CAPTURE_DIR}"
            "actor.model.gate.eval_control.capture_overwrite=false"
            "env.eval.task_suite_name=${task_suite}"
            "env.eval.episode_manifest_path=${suite_manifest}"
            "env.eval.test_episode_manifest_path=null"
            "env.eval.gate_phase_fn=${GATE_PHASE_FN}"
            "env.eval.total_num_envs=${suite_episodes}"
            "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
        )
        SUITE_SCOPE_START=${#RUN_COMMAND_LOG[@]}
        resolve_rlinf_gate_config "${CONFIG_NAME}" \
            "${SUITE_DIR}/resolved_config.yaml" "${OVERRIDES[@]}"
        RUN_ARTIFACTS=(
            "${SHARED_CKPT}" "${DATASET_STATS}" "${COST_PROFILE}"
            "${CONTROL_PROFILE}" "${PLUS_FULL_MANIFEST}" "${suite_manifest}"
            "${E1_DECISION}"
        )
        if [[ "${control}" == "shuffled" ]]; then
            RUN_ARTIFACTS+=("${DONOR_BANK}")
        fi
        CMD=(
            bash examples/embodiment/run_embodiment.sh
            "${CONFIG_NAME}" -- "${OVERRIDES[@]}"
        )
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
    CONTROL_TRACES+=("${TRACE}")

    if [[ "${control}" == "valid_idm" ]]; then
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_glob "${DONOR_DIR}/donor_*.pt"
        fi
        BANK_CMD=(
            env "PYTHONPATH=${FASTWAM_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
            python "${FASTWAM_ROOT}/scripts/build_shuffled_future_bank.py"
            --inputs "${DONOR_DIR}/donor_*.pt"
            --profile "${CONTROL_PROFILE}"
            --shared-ckpt "${SHARED_CKPT}"
            --dataset-stats "${DATASET_STATS}"
            --out "${DONOR_BANK}"
        )
        BANK_SCOPE_START=${#RUN_COMMAND_LOG[@]}
        RUN_ARTIFACTS=(
            "${SHARED_CKPT}" "${DATASET_STATS}" "${CONTROL_PROFILE}"
            "${PLUS_FULL_MANIFEST}"
        )
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            add_glob_artifacts "${DONOR_DIR}/donor_*.pt"
        fi
        run_command "${BANK_CMD[@]}"
        write_scoped_run_manifest \
            "${RUN_DIR}/run_manifest_donor_bank.json" "${BANK_SCOPE_START}"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            require_file "${DONOR_BANK}"
        fi
    fi
done

MERGED="${RUN_DIR}/control_trials.jsonl"
MERGE_CMD=(python "${MERGE_JSONL_TOOL}" --out "${MERGED}")
for trace in "${CONTROL_TRACES[@]}"; do
    MERGE_CMD+=(--input "${trace}")
done
run_command "${MERGE_CMD[@]}"
DECISION_CMD=(
    python "${DECISION_TOOL}" e2
    --trials "${MERGED}"
    --profile "${CONTROL_PROFILE}"
    --delta-ref "${DELTA_REF}"
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
    --seed "${SEED}"
    --out "${RUN_DIR}/decision.json"
)
RUN_ARTIFACTS=(
    "${SHARED_CKPT}" "${DATASET_STATS}" "${CONTROL_PROFILE}"
    "${PLUS_FULL_MANIFEST}" "${E1_DECISION}"
)
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS+=("${DONOR_BANK}" "${MERGED}")
fi
freeze_cli_config "${RUN_DIR}/decision_config.json" \
    "stage=E2" "delta_ref=${DELTA_REF}" \
    "bootstrap_samples=${BOOTSTRAP_SAMPLES}" "seed=${SEED}"
run_command "${DECISION_CMD[@]}"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
