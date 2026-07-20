#!/usr/bin/env bash
# SETTING: One real held-out-training LIBERO-Plus episode at an action-chunk boundary.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR only; Gate weights are irrelevant to forced snapshot smoke.
# SCIENTIFIC GOAL: Verify exact mid-episode simulator/controller/RNG restore before collecting counterfactual treatments.
# ACCEPTANCE: Restored RGB is byte-identical, proprio/state atol<=1e-6, and immutable episode counters/identity match.
# REQUIRED INPUTS: shared WAM artifacts, PAIRED_TRAIN_MANIFEST, PLUS_FULL_MANIFEST and E2_DECISION.
# OUTPUTS: smoke result, resolved RLinf config, decision.json and run_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
require_env PAIRED_TRAIN_MANIFEST
require_env PLUS_FULL_MANIFEST
require_env E2_DECISION
require_file "${PAIRED_TRAIN_MANIFEST}"
require_file "${PLUS_FULL_MANIFEST}"
require_passed_decision "${E2_DECISION}"
configure_plus_runtime "${PAIRED_TRAIN_MANIFEST}"
validate_disjoint_plus_manifests "${PAIRED_TRAIN_MANIFEST}" "${PLUS_FULL_MANIFEST}"
gate_wam_overrides

CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
PROGRESS_FN=${PROGRESS_FN:-rlinf.models.embodiment.gate_policy.libero_progress:success_only_progress}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E3_snapshot_smoke/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
CONFIG_OVERRIDES=(
    "${GATE_WAM_OVERRIDES[@]}"
    "env.eval.episode_manifest_path=${PAIRED_TRAIN_MANIFEST}"
    "env.eval.test_episode_manifest_path=${PLUS_FULL_MANIFEST}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
resolve_rlinf_gate_config "${CONFIG_NAME}" "${RUN_DIR}/resolved_config.yaml" \
    "${CONFIG_OVERRIDES[@]}"
RUN_ARTIFACTS=(
    "${SHARED_CKPT}" "${DATASET_STATS}" "${COST_PROFILE}"
    "${PAIRED_TRAIN_MANIFEST}" "${PLUS_FULL_MANIFEST}" "${E2_DECISION}"
)
CMD=(
    python examples/embodiment/smoke_libero_gate_snapshot.py
    --episode-manifest "${PAIRED_TRAIN_MANIFEST}"
    --heldout-test-manifest "${PLUS_FULL_MANIFEST}"
    --rlinf-config-dir "${PROJECT_REPO_ROOT}/examples/embodiment/config"
    --rlinf-config-name "${CONFIG_NAME}"
    --progress-fn "${PROGRESS_FN}"
)
for override in "${CONFIG_OVERRIDES[@]}"; do
    CMD+=(--config-override "${override}")
done
run_command "${CMD[@]}"
run_command python "${DECISION_TOOL}" contract --check libero_plus_snapshot_roundtrip \
    --evidence "${PAIRED_TRAIN_MANIFEST}" --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
