# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Entrypoint for route-neutral Gate with BC-initialized trainable UNCOND."""

import json

import hydra
import torch.multiprocessing as mp
from hydra.utils import get_class
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.models.embodiment.wam_policy.route_neutral_online.actor import (
    RouteNeutralOnlineIDMBCFSDPActor,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.config import (
    validate_route_neutral_online_idm_bc_training_config,
)
from rlinf.models.embodiment.wam_policy.route_neutral_online.lifecycle import (
    RouteNeutralOnlineEnvWorker,
    RouteNeutralOnlineRolloutWorker,
    RouteNeutralOnlineRunner,
)
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_10_ppo_fastwam_route_neutral_online_formal",
)
def main(cfg) -> None:
    """Launch the standard runner with only the additive actor class selected."""

    cfg = validate_cfg(cfg)
    validate_route_neutral_online_idm_bc_training_config(cfg)
    if bool(cfg.runner.get("use_training_pipeline", False)):
        raise ValueError("Route-neutral trainable profile uses the non-pipeline actor.")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    placement = HybridComponentPlacement(cfg, cluster)
    actor_cls = get_class(str(cfg.route_neutral_online_implementation.actor_target))
    if actor_cls is not RouteNeutralOnlineIDMBCFSDPActor:
        raise TypeError("Route-neutral actor target changed.")
    rollout_cls = get_class(
        str(cfg.route_neutral_online_implementation.rollout_worker_target)
    )
    if rollout_cls is not RouteNeutralOnlineRolloutWorker:
        raise TypeError("Route-neutral rollout worker target changed.")
    env_cls = get_class(str(cfg.route_neutral_online_implementation.env_worker_target))
    if env_cls is not RouteNeutralOnlineEnvWorker:
        raise TypeError("Route-neutral env worker target changed.")
    runner_cls = get_class(str(cfg.route_neutral_online_implementation.runner_target))
    if runner_cls is not RouteNeutralOnlineRunner:
        raise TypeError("Route-neutral runner target changed.")
    actor_group = actor_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    rollout_group = rollout_cls.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env_group = env_cls.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = runner_cls(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=None,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
