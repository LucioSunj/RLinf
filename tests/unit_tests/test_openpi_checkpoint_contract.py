import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf/models/embodiment/openpi/__init__.py"
)
SPEC = importlib.util.spec_from_file_location("openpi_checkpoint_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_load_openpi_state_dict = MODULE._load_openpi_state_dict


class _TinyOpenPi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)


def test_strict_pi05_load_rejects_missing_vlm_key() -> None:
    model = _TinyOpenPi()
    state = dict(model.state_dict())
    state.pop("paligemma_with_expert.paligemma.bias")

    with pytest.raises(ValueError, match="missing_vlm"):
        _load_openpi_state_dict(model, state, strict_vlm_checkpoint=True)


def test_strict_pi05_load_rejects_unexpected_non_value_key() -> None:
    model = _TinyOpenPi()
    state = dict(model.state_dict())
    state["unexpected.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match="unexpected"):
        _load_openpi_state_dict(model, state, strict_vlm_checkpoint=True)


def test_strict_pi05_load_allows_legacy_value_head_weights() -> None:
    model = _TinyOpenPi()
    state = dict(model.state_dict())
    state["value_head.layers.0.weight"] = torch.zeros(1)

    _load_openpi_state_dict(model, state, strict_vlm_checkpoint=True)
