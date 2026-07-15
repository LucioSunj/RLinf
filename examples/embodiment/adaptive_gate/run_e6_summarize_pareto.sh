#!/usr/bin/env bash
# SETTING: Paired task/factor/level bootstrap over canonical Plus-Full forced, learned, and matched-random traces.
# MODEL/CHECKPOINT LINEAGE: Read-only analysis of one frozen S-DR lineage and its independently trained Gate sweep.
# SCIENTIFIC GOAL: Decide whether learned allocation improves the success-compute frontier beyond matched random placement.
# ACCEPTANCE: >=2 distinct actual-compute-matched interior budgets beat task/reference-phase Random-K, frontier-area CI low>0, and >=4 stable seeds; x<=0.5,r>=0.8 is strong.
# REQUIRED INPUTS: LEARNED_TRACE_INDEX, BASELINE_TRACE_INDEX, PASS E5/E6 learned/baseline decisions.
# OUTPUTS: merged_trace_index.json, pareto_metrics.json, preregistered E6 decision.json and run_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
for name in LEARNED_TRACE_INDEX BASELINE_TRACE_INDEX E5_DECISION E6_LEARNED_DECISION E6_BASELINE_DECISION; do
    require_env "${name}"
    require_file "${!name}"
done
require_passed_decision "${E5_DECISION}"
require_passed_decision "${E6_LEARNED_DECISION}"
require_passed_decision "${E6_BASELINE_DECISION}"

BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}
ANALYSIS_SEED=${ANALYSIS_SEED:-0}
MATERIAL_EFFECT=${MATERIAL_EFFECT:-0.03}
BUDGET_TOLERANCE=${BUDGET_TOLERANCE:-0.10}
MIN_ACTUAL_BUDGET_SEPARATION=${MIN_ACTUAL_BUDGET_SEPARATION:-0.03}
ACTUAL_COMPUTE_MATCH_TOLERANCE=${ACTUAL_COMPUTE_MATCH_TOLERANCE:-0.10}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E6_pareto/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
INDEX="${RUN_DIR}/merged_trace_index.json"
METRICS="${RUN_DIR}/pareto_metrics.json"
DECISION="${RUN_DIR}/decision.json"

freeze_cli_config "${RUN_DIR}/resolved_config.json" \
    "stage=E6" "learned_trace_index=${LEARNED_TRACE_INDEX}" \
    "baseline_trace_index=${BASELINE_TRACE_INDEX}" \
    "bootstrap_samples=${BOOTSTRAP_SAMPLES}" "analysis_seed=${ANALYSIS_SEED}" \
    "material_effect=${MATERIAL_EFFECT}" "budget_tolerance=${BUDGET_TOLERANCE}" \
    "min_actual_budget_separation=${MIN_ACTUAL_BUDGET_SEPARATION}" \
    "actual_compute_match_tolerance=${ACTUAL_COMPUTE_MATCH_TOLERANCE}" \
    "expected_max_decisions=70"
run_command python "${TRACE_INDEX_TOOL}" merge --out "${INDEX}" \
    --index "${LEARNED_TRACE_INDEX}" --index "${BASELINE_TRACE_INDEX}"
ANALYZE_CMD=(
    python "${ANALYZE_RESULTS_TOOL}" e6
    --index "${INDEX}"
    --expected-max-decisions 70
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
    --seed "${ANALYSIS_SEED}"
    --material-effect "${MATERIAL_EFFECT}"
    --budget-tolerance "${BUDGET_TOLERANCE}"
    --min-actual-budget-separation "${MIN_ACTUAL_BUDGET_SEPARATION}"
    --actual-compute-match-tolerance "${ACTUAL_COMPUTE_MATCH_TOLERANCE}"
    --out "${METRICS}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
run_command "${ANALYZE_CMD[@]}"
run_command python "${DECISION_TOOL}" e6 --metrics "${METRICS}" --out "${DECISION}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${INDEX}"
    require_file "${METRICS}"
    require_file "${DECISION}"
fi
RUN_ARTIFACTS=(
    "${LEARNED_TRACE_INDEX}" "${BASELINE_TRACE_INDEX}" "${E5_DECISION}"
    "${E6_LEARNED_DECISION}" "${E6_BASELINE_DECISION}"
)
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS+=("${INDEX}" "${METRICS}" "${DECISION}")
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
