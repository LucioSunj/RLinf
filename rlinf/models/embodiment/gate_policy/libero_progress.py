# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Progress callbacks for the concrete paired LIBERO driver."""

from __future__ import annotations


def success_only_progress(*, env, observation, infos) -> float:
    """Binary fallback useful for snapshot smoke and terminal uplift labels.

    This does *not* support claims about one-/three-chunk progress.  Final E3
    collection should replace it with a preregistered task-specific predicate
    callback using the same keyword-only signature.
    """
    del observation, infos
    return float(bool(env.success_once[0]))
