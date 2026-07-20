"""Adaptive-gate launcher decision compatibility checks."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_ROOT = ROOT / "examples" / "embodiment" / "adaptive_gate"
CONTRACT_LAUNCHERS = (
    "run_e3_collect_paired_states.sh",
    "run_e3_snapshot_smoke.sh",
    "run_e4_train_uplift.sh",
    "run_e5_gate_bc_only.sh",
    "run_e6_forced_and_random.sh",
    "run_e6_learned_sweep.sh",
    "run_g_action_bc.sh",
)


@pytest.mark.parametrize("name", CONTRACT_LAUNCHERS)
def test_artifact_completion_launchers_use_contract_decision(name):
    path = LAUNCHER_ROOT / name
    text = path.read_text(encoding="utf-8")
    assert '"${DECISION_TOOL}" contract' in text
    assert '"${DECISION_TOOL}" p0' not in text
    assert "--check" in text
    assert "--evidence" in text
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_no_rlinf_launcher_calls_strict_p0_decision():
    offenders = [
        path.name
        for path in LAUNCHER_ROOT.glob("run_*.sh")
        if '"${DECISION_TOOL}" p0' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
