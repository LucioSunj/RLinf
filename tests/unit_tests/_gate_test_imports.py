"""Load gate modules without importing RLinf's optional Ray runtime."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def load_gate_modules():
    root = Path(__file__).resolve().parents[2]
    packages = {
        "rlinf": root / "rlinf",
        "rlinf.models": root / "rlinf/models",
        "rlinf.models.embodiment": root / "rlinf/models/embodiment",
        "rlinf.models.embodiment.modules": root / "rlinf/models/embodiment/modules",
        "rlinf.models.embodiment.gate_policy": root
        / "rlinf/models/embodiment/gate_policy",
    }
    module_names = (
        "rlinf.models.embodiment.base_policy",
        "rlinf.models.embodiment.modules.utils",
        "rlinf.models.embodiment.modules.value_head",
        "rlinf.models.embodiment.gate_policy.obs_preprocessor",
        "rlinf.models.embodiment.gate_policy.gate_policy",
        "rlinf.models.embodiment.gate_policy.reward",
        "rlinf.models.embodiment.gate_policy.bc",
    )
    touched = list(packages) + list(module_names)
    previous = {name: sys.modules.get(name) for name in touched}
    try:
        for name, path in packages.items():
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package
        loaded = {name: importlib.import_module(name) for name in module_names}
        return types.SimpleNamespace(
            gate=loaded["rlinf.models.embodiment.gate_policy.gate_policy"],
            obs=loaded["rlinf.models.embodiment.gate_policy.obs_preprocessor"],
            reward=loaded["rlinf.models.embodiment.gate_policy.reward"],
            bc=loaded["rlinf.models.embodiment.gate_policy.bc"],
        )
    finally:
        for name in reversed(touched):
            old = previous[name]
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
