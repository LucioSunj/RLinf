#!/usr/bin/env bash
# SETTING: Offline supervised Gate fit on WAM-Train action-agreement proxy labels.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR -> G-action labels -> independent G-action Gate checkpoint.
# SCIENTIFIC GOAL: Establish the Recommended low-cost proxy baseline without claiming closed-loop uplift supervision.
# ACCEPTANCE: Strict shard contract loads, finite train/validation metrics, and checkpoint plus sidecar are written.
# REQUIRED INPUTS: ACTION_LABELS glob and E1_DECISION; architecture must match the online binary Gate.
# OUTPUTS: gate_action_bc.pt, gate_action_bc.pt.meta.json, resolved_config.json, decision.json and run_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env ACTION_LABELS
require_env E1_DECISION
require_glob "${ACTION_LABELS}"
require_passed_decision "${E1_DECISION}"

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-3e-4}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_seed${SEED}}
RUN_DIR="${EXPERIMENT_ROOT}/G_action_bc/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
OUT="${RUN_DIR}/gate_action_bc.pt"
RUN_ARTIFACTS=("${E1_DECISION}")
add_glob_artifacts "${ACTION_LABELS}"
freeze_cli_config "${RUN_DIR}/resolved_config.json" \
    "kind=gate_action_bc" "labels=${ACTION_LABELS}" "epochs=${EPOCHS}" \
    "batch_size=${BATCH_SIZE}" "lr=${LR}" "seed=${SEED}" "device=${DEVICE}"

CMD=(
    python examples/embodiment/train_gate_bc.py
    --labels "${ACTION_LABELS}"
    --out "${OUT}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --seed "${SEED}"
    --device "${DEVICE}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
run_command "${CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${OUT}"
    require_file "${OUT}.meta.json"
fi
run_command python "${DECISION_TOOL}" contract --check g_action_bc_contract \
    --evidence "${OUT}.meta.json" --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
