#!/usr/bin/env bash
PROJECT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${PROJECT_REPO_ROOT}/../scripts/adaptive_gate/_common.sh"
cd "${PROJECT_REPO_ROOT}"
export EMBODIED_PATH="${PROJECT_REPO_ROOT}/examples/embodiment"

configure_plus_runtime() {
    local frozen_manifest=${1:-${LIBERO_PLUS_MANIFEST:-}}
    require_env LIBERO_PLUS_ROOT
    require_env LIBERO_PLUS_COMMIT
    [[ -n "${frozen_manifest}" ]] || die \
        "configure_plus_runtime requires the exact frozen manifest path"
    require_file "${frozen_manifest}"
    export LIBERO_TYPE=plus
    export LIBERO_PLUS_ROOT
    export LIBERO_PLUS_COMMIT
    export LIBERO_PLUS_MANIFEST="${frozen_manifest}"
    unset LIBERO_SUFFIX LIBERO_PERTURBATION
    export PYTHONPATH="${LIBERO_PLUS_ROOT}:${PROJECT_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

require_shared_gate_inputs() {
    require_env SHARED_CKPT
    require_env DATASET_STATS
    require_env COST_PROFILE
    require_env S_DR_SELECTION
    require_file "${SHARED_CKPT}"
    require_file "${DATASET_STATS}"
    require_file "${COST_PROFILE}"
    require_file "${S_DR_SELECTION}"
    export FASTWAM_CONFIGS=${FASTWAM_CONFIGS:-"${WORKSPACE_ROOT}/FastWAM/configs"}
    require_dir "${FASTWAM_CONFIGS}"
    run_command python "${WORKSPACE_ROOT}/FastWAM/scripts/validate_sdr_checkpoint.py" \
        check-selection --selection "${S_DR_SELECTION}" --checkpoint "${SHARED_CKPT}"
}

gate_wam_overrides() {
    GATE_WAM_OVERRIDES=(
        "actor.model.wam.configs_dir=${FASTWAM_CONFIGS}"
        "actor.model.wam.ckpt=${SHARED_CKPT}"
        "actor.model.wam.dataset_stats_path=${DATASET_STATS}"
        "actor.model.wam.cost_table_path=${COST_PROFILE}"
        "actor.model.wam.inference_steps=${INFERENCE_STEPS:-20}"
        "actor.model.wam.sigma_shift=${SIGMA_SHIFT:-null}"
    )
}

resolve_rlinf_gate_config() {
    local config_name=$1
    local out=$2
    shift 2
    resolve_hydra_config "${PROJECT_REPO_ROOT}/examples/embodiment/config" \
        "${config_name}" "${out}" "$@"
}

run_rlinf_gate() {
    local config_name=$1
    shift
    local cmd=(
        bash examples/embodiment/run_embodiment.sh "${config_name}" -- "$@"
    )
    run_command "${cmd[@]}"
}

manifest_episode_count() {
    python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(len(p["episodes"]))' "$1"
}

build_plus_suite_plan() {
    local manifest=$1
    local out_dir=$2
    local out_tsv=$3
    local cmd=(
        python "${PLUS_SUITE_TOOL}" partition
        --manifest "${manifest}"
        --out-dir "${out_dir}"
        --out-tsv "${out_tsv}"
    )
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        cmd+=(--materialize)
    fi
    # Planning must run even for --dry-run so every emitted per-suite command
    # is concrete. In dry-run mode only the temporary TSV is written.
    run_planning_command "${cmd[@]}"
}

merge_plus_suite_traces() {
    local manifest=$1
    local out=$2
    shift 2
    local cmd=(
        python "${PLUS_SUITE_TOOL}" merge-traces
        --manifest "${manifest}"
        --out "${out}"
    )
    local binding
    for binding in "$@"; do
        cmd+=(--suite-trace "${binding}")
    done
    run_command "${cmd[@]}"
}

validate_gate_training_manifests() {
    require_env GATE_TRAIN_MANIFEST
    require_env GATE_VAL_MANIFEST
    require_env PLUS_FULL_MANIFEST
    require_file "${GATE_TRAIN_MANIFEST}"
    require_file "${GATE_VAL_MANIFEST}"
    require_file "${PLUS_FULL_MANIFEST}"
    validate_disjoint_plus_manifests "${GATE_TRAIN_MANIFEST}" "${PLUS_FULL_MANIFEST}"
    validate_disjoint_plus_manifests "${GATE_VAL_MANIFEST}" "${PLUS_FULL_MANIFEST}"
    validate_disjoint_plus_manifests "${GATE_TRAIN_MANIFEST}" "${GATE_VAL_MANIFEST}"
}

gate_manifest_overrides() {
    local val_count
    val_count=$(manifest_episode_count "${GATE_VAL_MANIFEST}")
    GATE_MANIFEST_OVERRIDES=(
        "env.train.episode_manifest_path=${GATE_TRAIN_MANIFEST}"
        "env.train.test_episode_manifest_path=${PLUS_FULL_MANIFEST}"
        "env.eval.episode_manifest_path=${GATE_VAL_MANIFEST}"
        "env.eval.test_episode_manifest_path=${PLUS_FULL_MANIFEST}"
        "env.eval.total_num_envs=${val_count}"
    )
}
