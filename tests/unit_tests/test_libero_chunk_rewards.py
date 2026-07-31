import importlib.util
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "libero_reward_utils_under_test",
    REPO_ROOT / "rlinf/envs/libero/reward_utils.py",
)
reward_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward_utils)


def test_rewards_after_first_terminal_primitive_are_zeroed() -> None:
    rewards = torch.tensor([[0.0, 1.0, -1.0, 4.0], [0.0, 0.0, 0.0, 2.0]])
    dones = torch.tensor(
        [[False, True, False, False], [False, False, False, True]]
    )

    masked = reward_utils.mask_rewards_after_first_done(rewards, dones)

    assert torch.equal(masked, torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]))


def test_chunk_reward_mask_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        reward_utils.mask_rewards_after_first_done(
            torch.zeros(1, 2),
            torch.zeros(1, 3, dtype=torch.bool),
        )
