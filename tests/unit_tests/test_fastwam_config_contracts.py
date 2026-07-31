import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "rlinf/config_contracts.py"
SPEC = importlib.util.spec_from_file_location("fastwam_config_contracts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recompute_kv_requires_every_update_weight_sync() -> None:
    MODULE.validate_fastwam_kv_weight_sync("recompute", 1)
    MODULE.validate_fastwam_kv_weight_sync("stored", 4)

    with pytest.raises(ValueError, match="weight_sync_interval: 1"):
        MODULE.validate_fastwam_kv_weight_sync("recompute", 2)


def test_pi05_critic_artifact_config_requires_path_and_hex_digest() -> None:
    MODULE.validate_pi05_critic_artifact_config("/tmp/pi05", "a" * 64)

    with pytest.raises(ValueError, match="non-empty checkpoint path"):
        MODULE.validate_pi05_critic_artifact_config("", "a" * 64)
    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        MODULE.validate_pi05_critic_artifact_config("/tmp/pi05", "z" * 64)


def test_terminal_reward_config_rejects_repeated_success_reward_mode() -> None:
    MODULE.validate_libero_terminal_reward_config(
        ignore_terminations=False,
        use_rel_reward=False,
    )
    MODULE.validate_libero_terminal_reward_config(
        ignore_terminations=True,
        use_rel_reward=True,
    )

    with pytest.raises(ValueError, match="terminal success rewards would repeat"):
        MODULE.validate_libero_terminal_reward_config(
            ignore_terminations=True,
            use_rel_reward=False,
        )


def test_fastwam_resume_uses_consistent_payload_step() -> None:
    assert MODULE.validate_fastwam_resume_steps(
        [7, 7],
        "/tmp/global_step_7",
    ) == 7

    with pytest.raises(ValueError, match="directory/payload step mismatch"):
        MODULE.validate_fastwam_resume_steps([7, 7], "/tmp/global_step_8")

    with pytest.raises(ValueError, match="ranks disagree"):
        MODULE.validate_fastwam_resume_steps([7, 8], "/tmp/global_step_7")

    with pytest.raises(ValueError, match="did not return payload steps"):
        MODULE.validate_fastwam_resume_steps([7, None], "/tmp/global_step_7")
