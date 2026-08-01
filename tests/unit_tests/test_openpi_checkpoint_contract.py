# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

MODULE_PATH = Path(__file__).resolve().parents[2] / (
    "rlinf/models/embodiment/openpi/__init__.py"
)
SPEC = importlib.util.spec_from_file_location(
    "openpi_checkpoint_under_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_load_openpi_state_dict = MODULE._load_openpi_state_dict
_merge_vlm_checkpoint_aliases = MODULE._merge_vlm_checkpoint_aliases


_TIED_ALIAS = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
_TIED_CANONICAL = "paligemma_with_expert.paligemma.lm_head.weight"


class _TinyOpenPi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)


class _TinyHierarchicalOpenPi(nn.Module):
    def __init__(self, *, tied: bool) -> None:
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        paligemma = nn.Module()
        paligemma.model = nn.Module()
        paligemma.model.language_model = nn.Module()
        paligemma.model.language_model.embed_tokens = nn.Embedding(4, 2)
        paligemma.lm_head = nn.Linear(2, 4, bias=False)
        paligemma.norm = nn.LayerNorm(2)
        if tied:
            paligemma.lm_head.weight = (
                paligemma.model.language_model.embed_tokens.weight
            )
        self.paligemma_with_expert.paligemma = paligemma
        self.paligemma_with_expert.gemma_expert = nn.Embedding(4, 2)


def _cloned_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_strict_pi05_load_allows_verified_safetensors_tied_alias() -> None:
    model = _TinyHierarchicalOpenPi(tied=True)
    state = _cloned_state_dict(model)
    state.pop(_TIED_ALIAS)
    state[_TIED_CANONICAL] = torch.full((4, 2), 3.0, dtype=torch.bfloat16)

    _load_openpi_state_dict(
        model,
        state,
        strict_vlm_checkpoint=True,
        checkpoint_aliases={_TIED_ALIAS: _TIED_CANONICAL},
    )

    alias = model.get_parameter(_TIED_ALIAS)
    canonical = model.get_parameter(_TIED_CANONICAL)
    assert alias is canonical
    assert torch.equal(alias, torch.full_like(alias, 3.0))


def test_strict_tied_alias_without_safetensors_metadata_still_fails() -> None:
    model = _TinyHierarchicalOpenPi(tied=True)
    state = _cloned_state_dict(model)
    state.pop(_TIED_ALIAS)

    with pytest.raises(ValueError, match="missing_vlm"):
        _load_openpi_state_dict(model, state, strict_vlm_checkpoint=True)


def test_strict_tied_alias_rejects_missing_canonical_tensor() -> None:
    model = _TinyHierarchicalOpenPi(tied=True)
    state = _cloned_state_dict(model)
    state.pop(_TIED_ALIAS)
    state.pop(_TIED_CANONICAL)

    with pytest.raises(ValueError, match="missing_vlm"):
        _load_openpi_state_dict(
            model,
            state,
            strict_vlm_checkpoint=True,
            checkpoint_aliases={_TIED_ALIAS: _TIED_CANONICAL},
        )


def test_strict_tied_alias_rejects_untied_live_parameters() -> None:
    model = _TinyHierarchicalOpenPi(tied=False)
    state = _cloned_state_dict(model)
    state.pop(_TIED_ALIAS)

    with pytest.raises(ValueError, match="missing_vlm"):
        _load_openpi_state_dict(
            model,
            state,
            strict_vlm_checkpoint=True,
            checkpoint_aliases={_TIED_ALIAS: _TIED_CANONICAL},
        )


def test_strict_tied_alias_rejects_wrong_canonical_shape() -> None:
    model = _TinyHierarchicalOpenPi(tied=True)
    state = _cloned_state_dict(model)
    state.pop(_TIED_ALIAS)
    state[_TIED_CANONICAL] = torch.zeros(5, 2)

    with pytest.raises(RuntimeError, match="size mismatch"):
        _load_openpi_state_dict(
            model,
            state,
            strict_vlm_checkpoint=True,
            checkpoint_aliases={_TIED_ALIAS: _TIED_CANONICAL},
        )


def test_strict_tied_alias_rejects_incompatible_canonical_dtype() -> None:
    model = _TinyHierarchicalOpenPi(tied=True)
    state = _cloned_state_dict(model)
    state.pop(_TIED_ALIAS)
    state[_TIED_CANONICAL] = torch.ones(4, 2, dtype=torch.int64)

    with pytest.raises(ValueError, match="missing_vlm"):
        _load_openpi_state_dict(
            model,
            state,
            strict_vlm_checkpoint=True,
            checkpoint_aliases={_TIED_ALIAS: _TIED_CANONICAL},
        )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    (
        (_TIED_ALIAS, "paligemma_with_expert.gemma_expert.weight"),
        ("paligemma_with_expert.gemma_expert.weight", _TIED_CANONICAL),
    ),
)
def test_safetensors_alias_metadata_rejects_vlm_boundary_crossing(
    alias: str,
    canonical: str,
) -> None:
    with pytest.raises(ValueError, match="crosses the VLM boundary"):
        _merge_vlm_checkpoint_aliases(
            {}, {alias: canonical}, source="model.safetensors"
        )


def test_safetensors_alias_metadata_rejects_conflicting_shards() -> None:
    aliases: dict[str, str] = {}
    _merge_vlm_checkpoint_aliases(
        aliases,
        {_TIED_ALIAS: _TIED_CANONICAL},
        source="model-00001-of-00002.safetensors",
    )

    with pytest.raises(ValueError, match="Conflicting"):
        _merge_vlm_checkpoint_aliases(
            aliases,
            {_TIED_ALIAS: ("paligemma_with_expert.paligemma.norm.weight")},
            source="model-00002-of-00002.safetensors",
        )


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
