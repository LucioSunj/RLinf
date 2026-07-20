#!/usr/bin/env bash
# SETTING: Five-fold trajectory-grouped cross-fitting on immutable paired-v1 decision states.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR paired outcomes -> independent G-uplift Gate; no G-action initialization.
# SCIENTIFIC GOAL: Learn IDM helpful-discordance and an independent, equally featured UNCOND-difficulty comparator.
# ACCEPTANCE: Both scorers use the frozen trajectory folds; uplift emits a strict sidecar and both emit finite OOF scores.
# REQUIRED INPUTS: validated logical PAIRED_DATASET directory and E3_DECISION; its contract binds parent manifest/WAM/stats/solver/horizon provenance.
# OUTPUTS: deployable gate_uplift.pt, analysis-only gate_difficulty.pt, configs, manifests and decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env PAIRED_DATASET
require_env E3_DECISION
require_dir "${PAIRED_DATASET}"
require_file "${PAIRED_DATASET}/states.jsonl"
require_file "${PAIRED_DATASET}/outcomes.jsonl"
require_file "${PAIRED_DATASET}/splits.json"
require_file "${PAIRED_DATASET}/metadata.json"
require_glob "${PAIRED_DATASET}/tensors/*.pt"
require_passed_decision "${E3_DECISION}"

FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-3e-4}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_seed${SEED}}
RUN_DIR="${EXPERIMENT_ROOT}/E4_uplift_train/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
OUT="${RUN_DIR}/gate_uplift.pt"
DIFFICULTY_OUT="${RUN_DIR}/gate_difficulty.pt"
RUN_ARTIFACTS=(
    "${E3_DECISION}"
    "${PAIRED_DATASET}/states.jsonl"
    "${PAIRED_DATASET}/outcomes.jsonl"
    "${PAIRED_DATASET}/splits.json"
    "${PAIRED_DATASET}/metadata.json"
)
add_glob_artifacts "${PAIRED_DATASET}/tensors/*.pt"
run_command python examples/embodiment/validate_gate_paired_data.py \
    --paired "${PAIRED_DATASET}" --summary-out "${RUN_DIR}/paired_validation.json"
RUN_ARTIFACTS+=("${RUN_DIR}/paired_validation.json")
freeze_cli_config "${RUN_DIR}/resolved_config.json" \
    "kind=gate_uplift" "paired=${PAIRED_DATASET}" "folds=${FOLDS}" \
    "epochs=${EPOCHS}" "batch_size=${BATCH_SIZE}" "lr=${LR}" \
    "seed=${SEED}" "device=${DEVICE}" \
    "difficulty_control=${DIFFICULTY_OUT}"

UPLIFT_CMD=(
    python examples/embodiment/train_gate_benefit.py
    --paired "${PAIRED_DATASET}"
    --out "${OUT}"
    --target helpful
    --enabled-features world proprio text
    --folds "${FOLDS}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${DEVICE}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
DIFFICULTY_CMD=(
    python examples/embodiment/train_gate_benefit.py
    --paired "${PAIRED_DATASET}"
    --out "${DIFFICULTY_OUT}"
    --target difficulty
    --enabled-features world proprio text
    --folds "${FOLDS}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${DEVICE}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
UPLIFT_SCOPE_START=${#RUN_COMMAND_LOG[@]}
run_command "${UPLIFT_CMD[@]}"
write_scoped_run_manifest \
    "${RUN_DIR}/run_manifest_uplift.json" "${UPLIFT_SCOPE_START}"
DIFFICULTY_SCOPE_START=${#RUN_COMMAND_LOG[@]}
run_command "${DIFFICULTY_CMD[@]}"
write_scoped_run_manifest \
    "${RUN_DIR}/run_manifest_difficulty.json" "${DIFFICULTY_SCOPE_START}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${OUT}"
    require_file "${OUT}.meta.json"
    require_file "${OUT}.oof.pt"
    require_file "${DIFFICULTY_OUT}"
fi
run_command python "${DECISION_TOOL}" contract \
    --check gate_uplift_and_difficulty_crossfit_contract \
    --evidence "${OUT}.meta.json" --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
