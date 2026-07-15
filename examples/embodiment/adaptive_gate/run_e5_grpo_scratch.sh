#!/usr/bin/env bash
# SETTING: Five independent zero-supervision GRPO runs on held-out LIBERO-Plus Gate-Train with fixed interior compute pressure.
# MODEL/CHECKPOINT LINEAGE: final frozen S-DR -> randomly initialized G-GRPO-scratch Gate for each seed; no BC/uplift weights.
# SCIENTIFIC GOAL: Test whether sparse task reward minus compute cost can discover useful decision-level allocation from scratch.
# ACCEPTANCE: E5 health passes and final validation IDM usage is within BUDGET_USAGE_TOLERANCE<=0.10 of TARGET_IDM_USAGE; E6 additionally requires distinct monotone lambda/actual-usage coordinates.
# REQUIRED INPUTS: shared WAM artifacts, disjoint GATE_TRAIN_MANIFEST/GATE_VAL_MANIFEST/PLUS_FULL_MANIFEST, and PASS E4_DECISION.
# OUTPUTS: one RLinf run, checkpoints, exact checkpoint diagnostics/run_manifest per seed, aggregate E5 decision, and immutable checkpoint evidence registrations.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_shared_gate_inputs
require_env E4_DECISION
require_passed_decision "${E4_DECISION}"
validate_gate_training_manifests
configure_plus_runtime "${GATE_TRAIN_MANIFEST}"
gate_wam_overrides
gate_manifest_overrides

CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_gate}
SEEDS=${SEEDS:-"0 1 2 3 4"}
MAX_STEPS=${MAX_STEPS:-1000}
VAL_INTERVAL=${VAL_INTERVAL:-40}
SAVE_INTERVAL=${SAVE_INTERVAL:-40}
LAMBDA_COST=${LAMBDA_COST:-0.05}
TARGET_IDM_USAGE=${TARGET_IDM_USAGE:-0.5}
BUDGET_USAGE_TOLERANCE=${BUDGET_USAGE_TOLERANCE:-0.10}
python -c 'import math,sys
target,lam,tol=map(float,sys.argv[1:])
assert 0 < target < 1, "TARGET_IDM_USAGE must be interior"
assert math.isfinite(lam) and lam >= 0, "LAMBDA_COST must be finite and non-negative"
assert 0 < tol <= 0.10, "BUDGET_USAGE_TOLERANCE must be in (0,0.10]"' \
    "${TARGET_IDM_USAGE}" "${LAMBDA_COST}" "${BUDGET_USAGE_TOLERANCE}"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E5_grpo_scratch/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
IFS=' ' read -r -a SEED_ARRAY <<< "${SEEDS}"
[[ "${#SEED_ARRAY[@]}" -eq 5 ]] || die "E5 requires exactly five independent seeds"
SEEN_SEEDS=" "
for seed in "${SEED_ARRAY[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || die "E5 seeds must be non-negative integers: ${seed}"
    case "${SEEN_SEEDS}" in
        *" ${seed} "*) die "E5 seeds must be unique: ${seed}" ;;
    esac
    SEEN_SEEDS="${SEEN_SEEDS}${seed} "
done
METRICS=()
CHECKPOINTS=()
CHECKPOINT_DIAGNOSTICS=()
RUN_MANIFESTS=()

for seed in "${SEED_ARRAY[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || die "invalid integer seed: ${seed}"
    SEED_DIR="${RUN_DIR}/seed_${seed}"
    EXPERIMENT_NAME="e5_grpo_scratch_seed_${seed}"
    prepare_run_dir "${SEED_DIR}"
    DIAGNOSTICS="${SEED_DIR}/${EXPERIMENT_NAME}/gate_diagnostics.json"
    EVIDENCE_RUN_ID=$(python -c \
        'import hashlib,sys; print(hashlib.sha256("|".join(sys.argv[1:]).encode()).hexdigest())' \
        "${SEED_DIR}" "${EXPERIMENT_NAME}" "${seed}" "${TARGET_IDM_USAGE}" \
        "${LAMBDA_COST}" "${BUDGET_USAGE_TOLERANCE}")
    OVERRIDES=(
        "${GATE_WAM_OVERRIDES[@]}"
        "${GATE_MANIFEST_OVERRIDES[@]}"
        "runner.logger.log_path=${SEED_DIR}"
        "runner.logger.experiment_name=${EXPERIMENT_NAME}"
        "runner.max_steps=${MAX_STEPS}"
        "runner.val_check_interval=${VAL_INTERVAL}"
        "runner.save_interval=${SAVE_INTERVAL}"
        "runner.ckpt_path=null"
        "actor.seed=${seed}"
        "env.train.seed=${seed}"
        "env.eval.seed=${seed}"
        "gate_reward.lambda_cost=${LAMBDA_COST}"
        "gate_diagnostics.collapse.enabled=true"
        "gate_diagnostics.evidence_run_id=${EVIDENCE_RUN_ID}"
        "gate_diagnostics.collapse.target_idm_usage=${TARGET_IDM_USAGE}"
        "actor.model.gate.bc_init_path=null"
        "actor.model.gate.kl_prior.enabled=false"
        "actor.model.gate.kl_prior.path=null"
        "actor.model.gate.kl_prior.beta=0.0"
        "actor.model.gate.kl_prior.beta_end=0.0"
        "actor.model.gate.eval_policy.kind=learned"
        "actor.model.gate.eval_policy.trace_path=null"
        "actor.model.gate.eval_control.kind=null"
        "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
    )
    SEED_SCOPE_START=${#RUN_COMMAND_LOG[@]}
    resolve_rlinf_gate_config "${CONFIG_NAME}" "${SEED_DIR}/resolved_config.yaml" \
        "${OVERRIDES[@]}"
    RUN_ARTIFACTS=(
        "${SHARED_CKPT}" "${DATASET_STATS}" "${COST_PROFILE}"
        "${GATE_TRAIN_MANIFEST}" "${GATE_VAL_MANIFEST}"
        "${PLUS_FULL_MANIFEST}" "${E4_DECISION}"
    )
    CMD=(bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" -- "${OVERRIDES[@]}")
    run_command "${CMD[@]}"
    write_scoped_run_manifest \
        "${SEED_DIR}/run_manifest.json" "${SEED_SCOPE_START}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        require_file "${DIAGNOSTICS}"
        CHECKPOINT_STEP=$(python -c \
            'import json,sys; print(int(json.load(open(sys.argv[1]))["step"]))' \
            "${DIAGNOSTICS}")
        CHECKPOINT_ROOT="${SEED_DIR}/${EXPERIMENT_NAME}/checkpoints/global_step_${CHECKPOINT_STEP}"
        CHECKPOINT="${CHECKPOINT_ROOT}/actor/model_state_dict/full_weights.pt"
        CHECKPOINT_DIAGNOSTIC="${CHECKPOINT_ROOT}/gate_diagnostics.json"
        require_file "${CHECKPOINT}"
        require_file "${CHECKPOINT}.meta.json"
        require_file "${CHECKPOINT_DIAGNOSTIC}"
    else
        CHECKPOINT_ROOT="${SEED_DIR}/${EXPERIMENT_NAME}/checkpoints/global_step_<decision-step>"
        CHECKPOINT="${CHECKPOINT_ROOT}/actor/model_state_dict/full_weights.pt"
        CHECKPOINT_DIAGNOSTIC="${DIAGNOSTICS}"
    fi
    METRICS+=("${CHECKPOINT_DIAGNOSTIC}")
    CHECKPOINTS+=("${CHECKPOINT}")
    CHECKPOINT_DIAGNOSTICS+=("${CHECKPOINT_DIAGNOSTIC}")
    RUN_MANIFESTS+=("${SEED_DIR}/run_manifest.json")
done

DECISION_CMD=(python "${DECISION_TOOL}" e5 --out "${RUN_DIR}/decision.json" --metrics)
DECISION_CMD+=("${METRICS[@]}")
run_command "${DECISION_CMD[@]}"

REGISTER_CHECKPOINTS=0
if [[ "${DRY_RUN}" -eq 1 ]]; then
    REGISTER_CHECKPOINTS=1
elif [[ "$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "${RUN_DIR}/decision.json")" == "PASS" ]]; then
    REGISTER_CHECKPOINTS=1
fi
if [[ "${REGISTER_CHECKPOINTS}" -eq 1 ]]; then
    for index in "${!SEED_ARRAY[@]}"; do
        seed=${SEED_ARRAY[$index]}
        EVIDENCE="${RUN_DIR}/seed_${seed}/checkpoint_evidence.json"
        run_command python "${TRACE_INDEX_TOOL}" register-e5-checkpoint \
            --checkpoint "${CHECKPOINTS[$index]}" \
            --diagnostics "${CHECKPOINT_DIAGNOSTICS[$index]}" \
            --run-manifest "${RUN_MANIFESTS[$index]}" \
            --e5-decision "${RUN_DIR}/decision.json" \
            --gate-seed "${seed}" --target-budget "${TARGET_IDM_USAGE}" \
            --lambda-cost "${LAMBDA_COST}" \
            --usage-tolerance "${BUDGET_USAGE_TOLERANCE}" \
            --out "${EVIDENCE}"
    done
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
