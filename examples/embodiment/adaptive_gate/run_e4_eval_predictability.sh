#!/usr/bin/env bash
# SETTING: Trajectory-held-out OOF evaluation on paired-v1 with identical folds for benefit and difficulty scorers.
# MODEL/CHECKPOINT LINEAGE: independent G-uplift and analysis-only G-difficulty models trained from the same frozen S-DR pairs.
# SCIENTIFIC GOAL: Test pre-choice benefit predictability beyond generic UNCOND failure/difficulty and random allocation.
# ACCEPTANCE: AUROC>=0.65 with CI_low>0.55, AUPRC>=prevalence+0.10, and top-K wins at two budgets.
# REQUIRED INPUTS: logical PAIRED_DATASET, UPLIFT_CKPT, DIFFICULTY_CKPT and passed E4 training-contract decision.
# OUTPUTS: provenance-checked predictability_metrics.json, validation, resolved config, manifests and E4 decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env PAIRED_DATASET
require_env UPLIFT_CKPT
require_env DIFFICULTY_CKPT
require_env E4_TRAIN_DECISION
require_dir "${PAIRED_DATASET}"
require_file "${PAIRED_DATASET}/states.jsonl"
require_file "${PAIRED_DATASET}/outcomes.jsonl"
require_file "${PAIRED_DATASET}/splits.json"
require_file "${PAIRED_DATASET}/metadata.json"
require_glob "${PAIRED_DATASET}/tensors/*.pt"
require_file "${UPLIFT_CKPT}"
require_file "${UPLIFT_CKPT}.meta.json"
require_file "${UPLIFT_CKPT}.oof.pt"
require_file "${DIFFICULTY_CKPT}"
require_passed_decision "${E4_TRAIN_DECISION}"

ANALYZE_RESULTS_TOOL="${WORKSPACE_ROOT}/scripts/adaptive_gate/analyze_results.py"
require_file "${ANALYZE_RESULTS_TOOL}"
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}
MATERIAL_EFFECT=${MATERIAL_EFFECT:-0.03}
DIFFICULTY_BINS=${DIFFICULTY_BINS:-5}
CALIBRATION_BINS=${CALIBRATION_BINS:-10}
SEED=${SEED:-0}
BUDGETS=${BUDGETS:-"0.25 0.5 0.75"}
read -r -a BUDGET_VALUES <<< "${BUDGETS}"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_seed${SEED}}
RUN_DIR="${EXPERIMENT_ROOT}/E4_predictability/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"

RUN_ARTIFACTS=(
    "${E4_TRAIN_DECISION}"
    "${UPLIFT_CKPT}" "${UPLIFT_CKPT}.meta.json" "${UPLIFT_CKPT}.oof.pt"
    "${DIFFICULTY_CKPT}"
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
    "stage=E4" "paired=${PAIRED_DATASET}" \
    "uplift=${UPLIFT_CKPT}" "difficulty=${DIFFICULTY_CKPT}" \
    "bootstrap_samples=${BOOTSTRAP_SAMPLES}" \
    "material_effect=${MATERIAL_EFFECT}" "difficulty_bins=${DIFFICULTY_BINS}" \
    "calibration_bins=${CALIBRATION_BINS}" "budgets=${BUDGETS}" "seed=${SEED}"

METRICS="${RUN_DIR}/predictability_metrics.json"
ANALYZE_CMD=(
    python "${ANALYZE_RESULTS_TOOL}" e4
    --paired "${PAIRED_DATASET}"
    --sidecar "${UPLIFT_CKPT}.meta.json"
    --oof "${UPLIFT_CKPT}.oof.pt"
    --difficulty-oof "${DIFFICULTY_CKPT}"
    --difficulty-bins "${DIFFICULTY_BINS}"
    --calibration-bins "${CALIBRATION_BINS}"
    --budgets "${BUDGET_VALUES[@]}"
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
    --material-effect "${MATERIAL_EFFECT}"
    --seed "${SEED}"
    --out "${METRICS}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
ANALYZE_SCOPE_START=${#RUN_COMMAND_LOG[@]}
run_command "${ANALYZE_CMD[@]}"
write_scoped_run_manifest \
    "${RUN_DIR}/run_manifest_analysis.json" "${ANALYZE_SCOPE_START}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${METRICS}"
fi
DECISION_CMD=(
    python "${DECISION_TOOL}" e4
    --metrics "${METRICS}"
    --out "${RUN_DIR}/decision.json"
)
run_command "${DECISION_CMD[@]}"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
