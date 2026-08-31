# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cheap PAD runtime-asset checks performed before Ray model allocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _required_env_roles(cfg: Any) -> tuple[str, ...]:
    if bool(cfg.runner.get("only_eval", False)):
        return ("eval",)
    roles = ["train"]
    if int(cfg.runner.get("val_check_interval", -1)) > 0:
        roles.append("eval")
    return tuple(roles)


def validate_pad_text_cache_coverage(cfg: Any) -> dict[str, Any]:
    """Require every configured LIBERO prompt cache before allocating GPUs."""

    from libero.libero import benchmark

    model_cfg = (
        cfg.rollout.model
        if bool(cfg.runner.get("only_eval", False))
        else cfg.actor.model
    )
    runtime = model_cfg.runtime
    template = str(runtime.prompt_template)
    if "{task}" not in template:
        raise ValueError("PAD text-cache preflight requires `{task}` in the template.")
    cache_dir = Path(str(runtime.text_embedding_cache_dir))
    context_len = int(runtime.text_embedding_context_len)
    benchmark_dict = benchmark.get_benchmark_dict()
    required = []
    missing = []
    for role in _required_env_roles(cfg):
        env_cfg = OmegaConf.select(cfg, f"env.{role}")
        suite_name = str(env_cfg.task_suite_name)
        if suite_name not in benchmark_dict:
            raise ValueError(
                f"PAD preflight cannot resolve LIBERO suite {suite_name!r}."
            )
        suite = benchmark_dict[suite_name]()
        task_filter = env_cfg.get("task_id_filter", None)
        task_ids = (
            list(range(int(suite.n_tasks)))
            if task_filter is None
            else [int(task_id) for task_id in task_filter]
        )
        for task_id in task_ids:
            task = suite.get_task(task_id)
            prompt = template.format(task=str(task.language))
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
            record = {
                "role": role,
                "suite": suite_name,
                "task_id": task_id,
                "prompt_sha256": digest,
                "path": str(path),
            }
            required.append(record)
            if not path.is_file():
                missing.append(record)
    if missing:
        raise FileNotFoundError(
            "PAD text-cache preflight found missing configured prompts: "
            + json.dumps(missing, sort_keys=True)
        )
    report = {
        "schema": "pad-text-cache-preflight-v1",
        "required_prompt_count": len(required),
        "prompt_sha256": [record["prompt_sha256"] for record in required],
        "status": "PASS",
    }
    print(
        "PAD_TEXT_CACHE_PREFLIGHT=" + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return report
