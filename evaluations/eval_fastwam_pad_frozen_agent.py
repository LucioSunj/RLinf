# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Config-selected standalone evaluation for PAD-Frozen checkpoints."""

import json

import hydra
import torch.multiprocessing as mp
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.models.embodiment.wam_policy.pad_rv import (
    validate_pad_frozen_training_config,
)
from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker
from rlinf.models.embodiment.wam_policy.pad_rv.eval_runner import (
    PadFrozenEvalRunner,
)
from rlinf.models.embodiment.wam_policy.pad_rv.rollout import (
    PadFrozenRolloutWorker,
)
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="libero",
    config_name="libero_10_fastwam_pad_frozen_eval",
)
def main(cfg) -> None:
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    validate_pad_frozen_training_config(cfg, only_eval=True)
    text_cache_preflight = get_method(
        str(cfg.pad_rv_implementation.text_cache_preflight_target)
    )
    text_cache_preflight(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout_cls = get_class(str(cfg.pad_rv_implementation.rollout_worker_target))
    if rollout_cls is not PadFrozenRolloutWorker:
        raise TypeError("PAD evaluation rollout target must be PadFrozenRolloutWorker.")
    env_cls = get_class(str(cfg.pad_rv_implementation.env_worker_target))
    if env_cls is not PadFrozenEnvWorker:
        raise TypeError("PAD evaluation env target must be PadFrozenEnvWorker.")
    runner_cls = get_class(str(cfg.pad_rv_implementation.evaluation_runner_target))
    if runner_cls is not PadFrozenEvalRunner:
        raise TypeError("PAD evaluation runner target must be PadFrozenEvalRunner.")

    rollout = rollout_cls.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = env_cls.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = runner_cls(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
