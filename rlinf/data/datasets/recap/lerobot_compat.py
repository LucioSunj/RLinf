# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility helpers for LeRobot 0.1.x with newer Hugging Face datasets."""

from __future__ import annotations


def ensure_lerobot_column_indexing_compat() -> None:
    """Restore legacy string-column indexing expected by LeRobot 0.1.x.

    LeRobot 0.1.x calls ``torch.stack(dataset["timestamp"])`` during
    construction. With Hugging Face ``datasets>=4``, string indexing returns a
    ``Column`` object instead of the older list-like value, and ``torch.stack``
    rejects that object even though it is iterable. This scoped monkey patch
    converts string-indexed Columns to lists before LeRobot sees them.
    """

    try:
        import datasets
    except ImportError:
        return

    dataset_cls = datasets.arrow_dataset.Dataset
    if getattr(dataset_cls, "_rlinf_lerobot_column_compat", False):
        return

    original_getitem = dataset_cls.__getitem__

    def patched_getitem(self, key):  # type: ignore[no-untyped-def]
        value = original_getitem(self, key)
        if isinstance(key, str) and value.__class__.__name__ == "Column":
            return list(value)
        return value

    dataset_cls.__getitem__ = patched_getitem
    dataset_cls._rlinf_lerobot_column_compat = True

