#!/usr/bin/env bash
# SETTING: Balanced train-support LIBERO-Plus trajectories, cloned only at action-chunk boundaries with fixed continuations.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR branches share WAM/action seeds; collection never reads Plus-Full outcomes.
# SCIENTIFIC GOAL: Measure decision-level UNCOND-versus-IDM treatment outcomes without task/reset or simulator-state confounding.
# ACCEPTANCE: Snapshot restore is pre-approved and paired-v1 validates identities, tensors, splits, provenance and finite outcomes.
# REQUIRED INPUTS: shared WAM artifacts, PAIRED_TRAIN_MANIFEST, disjoint PLUS_FULL_MANIFEST, callbacks and snapshot decision.
# OUTPUTS: per-suite physical paired-v1 datasets, one strict logical merged paired-v1 with global folds, validation and decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
require_env PAIRED_TRAIN_MANIFEST
require_env PLUS_FULL_MANIFEST
require_env E3_SNAPSHOT_DECISION
require_env PROGRESS_FN
require_env GATE_PHASE_FN
require_file "${PAIRED_TRAIN_MANIFEST}"
require_file "${PLUS_FULL_MANIFEST}"
require_passed_decision "${E3_SNAPSHOT_DECISION}"
configure_plus_runtime "${PAIRED_TRAIN_MANIFEST}"
validate_disjoint_plus_manifests \
    "${PAIRED_TRAIN_MANIFEST}" "${PLUS_FULL_MANIFEST}"
gate_wam_overrides

CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
COLLECTOR_SEED=${COLLECTOR_SEED:-0}
MAX_REFERENCE_DECISIONS=${MAX_REFERENCE_DECISIONS:-70}
MAX_BRANCH_DECISIONS=${MAX_BRANCH_DECISIONS:-70}
SENSITIVITY_FRACTION=${SENSITIVITY_FRACTION:-0.2}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_seed${COLLECTOR_SEED}}
RUN_DIR="${EXPERIMENT_ROOT}/E3_paired_collection/${RUN_ID}"
PAIRED_OUT=${PAIRED_OUT:-"${RUN_DIR}/paired_v1"}
prepare_run_dir "${RUN_DIR}"
if [[ "${DRY_RUN}" -eq 0 && -e "${PAIRED_OUT}" ]]; then
    die "logical PAIRED_OUT already exists and will not be overwritten: ${PAIRED_OUT}"
fi
SUITE_TSV=$(mktemp "${TMPDIR:-/tmp}/adaptive_gate_e3_suites.XXXXXX")
trap 'rm -f "${SUITE_TSV}"' EXIT
build_plus_suite_plan "${PAIRED_TRAIN_MANIFEST}" \
    "${RUN_DIR}/plus_suite_manifests" "${SUITE_TSV}"
if [[ -n "${NUM_EPISODES:-}" ]]; then
    die "NUM_EPISODES is forbidden for exact logical collection; freeze a separate complete pilot manifest instead"
fi

SUITE_PAIRED_BINDINGS=()
while IFS=$'\t' read -r task_suite suite_manifest _suite_episodes _logical_sha; do
    [[ -n "${task_suite}" ]] || continue
    suite_slug=$(basename "${suite_manifest}" .json)
    SUITE_DIR="${RUN_DIR}/suites/${suite_slug}"
    SUITE_PAIRED="${SUITE_DIR}/paired_v1"
    SUITE_SNAPSHOTS="${SUITE_PAIRED}/snapshots"
    prepare_run_dir "${SUITE_DIR}"
    CONFIG_OVERRIDES=(
        "${GATE_WAM_OVERRIDES[@]}"
        "env.eval.task_suite_name=${task_suite}"
        "env.eval.episode_manifest_path=${suite_manifest}"
        "env.eval.test_episode_manifest_path=${PLUS_FULL_MANIFEST}"
        "env.eval.gate_phase_fn=${GATE_PHASE_FN}"
        "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
    )
    SUITE_SCOPE_START=${#RUN_COMMAND_LOG[@]}
    resolve_rlinf_gate_config \
        "${CONFIG_NAME}" "${SUITE_DIR}/resolved_config.yaml" \
        "${CONFIG_OVERRIDES[@]}"
    CMD=(
        python examples/embodiment/collect_gate_paired_states.py
        --episode-manifest "${suite_manifest}"
        --heldout-test-manifest "${PLUS_FULL_MANIFEST}"
        --out "${SUITE_PAIRED}"
        --snapshot-dir "${SUITE_SNAPSHOTS}"
        --collector-seed "${COLLECTOR_SEED}"
        --max-reference-decisions "${MAX_REFERENCE_DECISIONS}"
        --max-branch-decisions "${MAX_BRANCH_DECISIONS}"
        --sensitivity-fraction "${SENSITIVITY_FRACTION}"
        --rlinf-config-dir "${PROJECT_REPO_ROOT}/examples/embodiment/config"
        --rlinf-config-name "${CONFIG_NAME}"
        --progress-fn "${PROGRESS_FN}"
    )
    for override in "${CONFIG_OVERRIDES[@]}"; do
        CMD+=(--config-override "${override}")
    done
    RUN_ARTIFACTS=(
        "${SHARED_CKPT}" "${DATASET_STATS}" "${COST_PROFILE}"
        "${PAIRED_TRAIN_MANIFEST}" "${suite_manifest}"
        "${PLUS_FULL_MANIFEST}" "${E3_SNAPSHOT_DECISION}"
    )
    run_command "${CMD[@]}"
    write_scoped_run_manifest \
        "${SUITE_DIR}/run_manifest.json" "${SUITE_SCOPE_START}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        require_file "${SUITE_PAIRED}/metadata.json"
        require_file "${SUITE_PAIRED}/splits.json"
    fi
    SUITE_PAIRED_BINDINGS+=("${task_suite}=${SUITE_PAIRED}")
done < "${SUITE_TSV}"

MERGE_CMD=(
    python examples/embodiment/merge_gate_paired_data.py
    --episode-manifest "${PAIRED_TRAIN_MANIFEST}"
    --out "${PAIRED_OUT}"
    --summary-out "${RUN_DIR}/paired_merge_summary.json"
)
for binding in "${SUITE_PAIRED_BINDINGS[@]}"; do
    MERGE_CMD+=(--suite-paired "${binding}")
done
RUN_ARTIFACTS=(
    "${SHARED_CKPT}" "${DATASET_STATS}" "${COST_PROFILE}"
    "${PAIRED_TRAIN_MANIFEST}" "${PLUS_FULL_MANIFEST}"
    "${E3_SNAPSHOT_DECISION}"
)
if [[ "${DRY_RUN}" -eq 0 ]]; then
    for binding in "${SUITE_PAIRED_BINDINGS[@]}"; do
        suite_paired=${binding#*=}
        RUN_ARTIFACTS+=("${suite_paired}/metadata.json" "${suite_paired}/splits.json")
        add_glob_artifacts "${suite_paired}/tensors/*.pt"
    done
fi
run_command "${MERGE_CMD[@]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${PAIRED_OUT}/states.jsonl"
    require_file "${PAIRED_OUT}/outcomes.jsonl"
    require_file "${PAIRED_OUT}/splits.json"
    require_file "${PAIRED_OUT}/metadata.json"
    require_glob "${PAIRED_OUT}/tensors/*.pt"
fi
VALIDATE_CMD=(
    python examples/embodiment/validate_gate_paired_data.py
    --paired "${PAIRED_OUT}"
    --summary-out "${RUN_DIR}/paired_validation.json"
)
run_command "${VALIDATE_CMD[@]}"
run_command python "${DECISION_TOOL}" contract \
    --check paired_v1_collection_contract \
    --evidence "${RUN_DIR}/paired_validation.json" \
    --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
