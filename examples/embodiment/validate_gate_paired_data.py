# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed validator for a paired-v1 counterfactual dataset."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired",
        required=True,
        nargs="+",
        help=(
            "paired-v1 dataset directory (including a logical multi-suite "
            "merge), or tensor shard(s) under one <dataset>/tensors directory"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="optional path for the same machine-readable JSON summary",
    )
    args = parser.parse_args()

    from rlinf.models.embodiment.gate_policy.paired_data import (
        validate_paired_dataset,
    )

    paired = args.paired[0] if len(args.paired) == 1 else args.paired
    summary = validate_paired_dataset(paired)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
