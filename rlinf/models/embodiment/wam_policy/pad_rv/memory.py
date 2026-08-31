# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Explicit host-allocation release points owned by PAD-RV workers."""

from __future__ import annotations

import ctypes
import gc
from typing import Any

import psutil


def release_pad_host_memory(*, schema: str, rank: int, phase: str) -> dict[str, Any]:
    """Collect dead Python objects and return free glibc pages to Linux."""

    collected = gc.collect()
    trimmed = int(ctypes.CDLL("libc.so.6").malloc_trim(0))
    return {
        "schema": str(schema),
        "rank": int(rank),
        "phase": str(phase),
        "gc_collected": int(collected),
        "malloc_trim_result": trimmed,
        "rss_bytes": int(psutil.Process().memory_info().rss),
        "status": "PASS",
    }


__all__ = ["release_pad_host_memory"]
