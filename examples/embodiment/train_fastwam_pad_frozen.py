# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dedicated entrypoint for config-selected PAD-Frozen training."""

import json

import hydra
import torch.multiprocessing as mp
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.models.embodiment.wam_policy.pad_rv import (
    validate_pad_frozen_training_config,
)
from rlinf.models.embodiment.wam_policy.pad_rv.actor import PadFrozenFSDPActor
from rlinf.models.embodiment.wam_policy.pad_rv.env import PadFrozenEnvWorker
from rlinf.models.embodiment.wam_policy.pad_rv.rollout import (
    PadFrozenRolloutWorker,
)
from rlinf.models.embodiment.wam_policy.pad_rv.runner import PadFrozenRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_10_ppo_fastwam_pad_frozen_formal",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    validate_pad_frozen_training_config(cfg)
    if bool(cfg.runner.get("use_training_pipeline", False)):
        raise ValueError("PAD-Frozen does not use the pipeline actor variant.")
    text_cache_preflight = get_method(
        str(cfg.pad_rv_implementation.text_cache_preflight_target)
    )
    text_cache_preflight(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    placement = HybridComponentPlacement(cfg, cluster)
    actor_cls = get_class(str(cfg.pad_rv_implementation.actor_target))
    if actor_cls is not PadFrozenFSDPActor:
        raise TypeError("PAD actor target must resolve to PadFrozenFSDPActor.")
    rollout_cls = get_class(str(cfg.pad_rv_implementation.rollout_worker_target))
    if rollout_cls is not PadFrozenRolloutWorker:
        raise TypeError("PAD rollout target must resolve to PadFrozenRolloutWorker.")
    runner_cls = get_class(str(cfg.pad_rv_implementation.runner_target))
    if runner_cls is not PadFrozenRunner:
        raise TypeError("PAD runner target must resolve to PadFrozenRunner.")
    env_cls = get_class(str(cfg.pad_rv_implementation.env_worker_target))
    if env_cls is not PadFrozenEnvWorker:
        raise TypeError("PAD env target must resolve to PadFrozenEnvWorker.")
    actor = actor_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
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
    runner = runner_cls(
        cfg=cfg,
        actor=actor,
        rollout=rollout,
        env=env,
        reward=None,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
