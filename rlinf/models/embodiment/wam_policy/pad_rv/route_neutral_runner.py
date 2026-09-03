# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PAD runner that retains the generic reversal-damped branch controller."""

from __future__ import annotations

from rlinf.runners.embodied_runner import EmbodiedRunner

from . import route_neutral_budget as _route_neutral_budget_registration  # noqa: F401
from .runner import PadFrozenRunner


class PadRouteNeutralRunner(PadFrozenRunner):
    """Reuse PAD initialization/resume while keeping config-selected cost control."""

    def __init__(self, *args, **kwargs) -> None:
        # PadFrozenRunner replaces generic cost control with its historical
        # projected dual. This profile intentionally retains the generic
        # config-selected reversal-damped controller instead.
        EmbodiedRunner.__init__(self, *args, **kwargs)


__all__ = ["PadRouteNeutralRunner"]
