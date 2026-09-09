# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Thin offline-first command entrypoints. Argument help never initializes hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, preflight


def main(command: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Real robot PAD {command} (mock by default)"
    )
    parser.add_argument("--config", default="configs/real_robot/pad_mock.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--tasks", default="configs/real_robot/tasks.yaml", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["always_idm", "always_uncond", "random", "learned"],
    )
    args = parser.parse_args()
    if command == "preflight":
        try:
            report = preflight(load_config(args.config))
        except (ValueError, KeyError) as error:
            parser.exit(
                2, json.dumps({"status": "BLOCKED", "reason": str(error)}) + "\n"
            )
        print(json.dumps(report, indent=2))
        return
    if command == "summarize":
        from .evaluation import summarize_run

        print(
            json.dumps(
                summarize_run(args.run_dir or Path("/tmp/pad_mock_run")), indent=2
            )
        )
        return
    cfg = load_config(args.config)
    if command in {"collect_demos", "prepare_dataset"}:
        from .data import collect_demos, prepare_dataset

        if command == "collect_demos":
            result = collect_demos(
                cfg, args.output or Path("/tmp/pad_mock_demos"), import_dir=args.input
            )
        else:
            if args.input is None:
                parser.error("prepare_dataset requires --input with recorded sessions")
            result = prepare_dataset(
                args.input, args.output or Path("/tmp/pad_mock_dataset")
            )
        print(json.dumps(result, indent=2))
        return
    if cfg["backend"] != "mock":
        parser.exit(
            2,
            "Live CLI execution is blocked: bind the confirmed controller/cameras explicitly through the Python API after site authorization.\n",
        )
    from .runner import run_mock_training

    run_dir = args.run_dir or args.output or Path(cfg["run_dir"])
    if command == "train":
        result = run_mock_training(
            cfg,
            run_dir,
            tasks_path=args.tasks,
            resume=args.resume,
            updates=args.updates,
        )
    elif command == "eval":
        from .evaluation import run_mock_evaluation

        result = run_mock_evaluation(
            cfg,
            run_dir,
            methods=args.methods,
            tasks_path=args.tasks,
            checkpoint=args.checkpoint,
        )
    else:
        raise ValueError(f"Unknown command: {command}")
    print(json.dumps(result, indent=2))
