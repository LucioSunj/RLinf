# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strictly merge per-suite paired-v1 artifacts for one logical Plus manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _suite_bindings(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        suite, separator, path = value.partition("=")
        if not separator or not suite or not path:
            raise ValueError(
                "--suite-paired must use TASK_SUITE=/path/to/paired-v1"
            )
        if suite in result:
            raise ValueError(f"duplicate paired-v1 binding for suite {suite!r}")
        result[suite] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-manifest",
        required=True,
        help="complete logical parent LIBERO-Plus manifest",
    )
    parser.add_argument(
        "--suite-paired",
        action="append",
        required=True,
        help="TASK_SUITE=/path/to/physical/paired-v1 (repeat once per suite)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    from rlinf.envs.libero.episode_manifest import load_frozen_episode_manifest
    from rlinf.models.embodiment.gate_policy.paired_data import (
        merge_paired_suite_datasets,
    )

    manifest = load_frozen_episode_manifest(args.episode_manifest)
    if manifest.parent_manifest_path is not None:
        raise ValueError("--episode-manifest must be the complete logical parent")
    summary = merge_paired_suite_datasets(
        manifest,
        _suite_bindings(args.suite_paired),
        args.out,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_out:
        path = Path(args.summary_out).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
