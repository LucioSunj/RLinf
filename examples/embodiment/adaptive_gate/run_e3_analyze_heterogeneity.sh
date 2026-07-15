#!/usr/bin/env bash
# SETTING: Offline task/trajectory-clustered analysis of immutable paired-v1 success and one-/three-chunk progress outcomes.
# MODEL/CHECKPOINT LINEAGE: no model training; analysis consumes only paired branches from the final frozen S-DR collector.
# SCIENTIFIC GOAL: Test whether IDM benefit varies materially across decisions and whether paired top-K beats Random-K.
# ACCEPTANCE: >=20% helpful, >=30% neutral/harmful, and material positive top-K gains at two registered budgets.
# REQUIRED INPUTS: validated logical PAIRED_DATASET and a passed E3 collection-contract decision; no Plus-Full evaluation outcomes.
# OUTPUTS: success/progress metrics, normalized paired outcomes, validation, run manifests and E3 decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env PAIRED_DATASET
require_env E3_COLLECT_DECISION
require_dir "${PAIRED_DATASET}"
require_file "${PAIRED_DATASET}/states.jsonl"
require_file "${PAIRED_DATASET}/outcomes.jsonl"
require_file "${PAIRED_DATASET}/splits.json"
require_file "${PAIRED_DATASET}/metadata.json"
require_glob "${PAIRED_DATASET}/tensors/*.pt"
require_passed_decision "${E3_COLLECT_DECISION}"

ANALYZE_RESULTS_TOOL="${WORKSPACE_ROOT}/scripts/adaptive_gate/analyze_results.py"
require_file "${ANALYZE_RESULTS_TOOL}"
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}
MATERIAL_EFFECT=${MATERIAL_EFFECT:-0.03}
SEED=${SEED:-0}
BUDGETS=${BUDGETS:-"0.25 0.5 0.75"}
read -r -a BUDGET_VALUES <<< "${BUDGETS}"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_seed${SEED}}
RUN_DIR="${EXPERIMENT_ROOT}/E3_heterogeneity/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"

RUN_ARTIFACTS=(
    "${E3_COLLECT_DECISION}"
    "${PAIRED_DATASET}/states.jsonl"
    "${PAIRED_DATASET}/outcomes.jsonl"
    "${PAIRED_DATASET}/splits.json"
    "${PAIRED_DATASET}/metadata.json"
)
add_glob_artifacts "${PAIRED_DATASET}/tensors/*.pt"
run_command python examples/embodiment/validate_gate_paired_data.py \
    --paired "${PAIRED_DATASET}" \
    --summary-out "${RUN_DIR}/paired_validation.json"
RUN_ARTIFACTS+=("${RUN_DIR}/paired_validation.json")
freeze_cli_config "${RUN_DIR}/resolved_config.json" \
    "stage=E3" "paired=${PAIRED_DATASET}" \
    "bootstrap_samples=${BOOTSTRAP_SAMPLES}" \
    "material_effect=${MATERIAL_EFFECT}" "budgets=${BUDGETS}" "seed=${SEED}"

for outcome in success progress_1 progress_3; do
    METRICS="${RUN_DIR}/${outcome}_metrics.json"
    OUTCOMES="${RUN_DIR}/${outcome}_outcomes.jsonl"
    ANALYZE_CMD=(
        python "${ANALYZE_RESULTS_TOOL}" e3
        --paired "${PAIRED_DATASET}"
        --outcome "${outcome}"
        --budgets "${BUDGET_VALUES[@]}"
        --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
        --material-effect "${MATERIAL_EFFECT}"
        --seed "${SEED}"
        --normalized-outcomes "${OUTCOMES}"
        --out "${METRICS}"
        "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
    )
    ANALYZE_SCOPE_START=${#RUN_COMMAND_LOG[@]}
    run_command "${ANALYZE_CMD[@]}"
    write_scoped_run_manifest \
        "${RUN_DIR}/run_manifest_${outcome}.json" "${ANALYZE_SCOPE_START}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        require_file "${METRICS}"
        require_file "${OUTCOMES}"
    fi
done

DECISION_CMD=(
    python "${DECISION_TOOL}" e3
    --metrics "${RUN_DIR}/success_metrics.json"
    --material-effect "${MATERIAL_EFFECT}"
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
    --seed "${SEED}"
    --out "${RUN_DIR}/decision.json"
)
run_command "${DECISION_CMD[@]}"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
