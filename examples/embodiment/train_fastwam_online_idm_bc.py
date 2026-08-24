# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dedicated entrypoint for opt-in online IDM-to-UNCOND behavior cloning."""

import json

import hydra
import torch.multiprocessing as mp
from hydra.utils import get_class
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.models.embodiment.wam_policy.online_idm_bc import (
    validate_online_idm_bc_training_config,
)
from rlinf.models.embodiment.wam_policy.online_idm_bc.actor import OnlineIDMBCFSDPActor
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_10_ppo_fastwam_adaptive_formal",
)
def main(cfg) -> None:
    """Launch the normal embodied runner with only the actor subclass swapped."""

    cfg = validate_cfg(cfg)
    validate_online_idm_bc_training_config(cfg)
    if bool(cfg.runner.get("use_training_pipeline", False)):
        raise ValueError("Online IDM BC does not use the pipeline actor variant.")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    placement = HybridComponentPlacement(cfg, cluster)
    actor_worker_cls = get_class(str(cfg.online_idm_bc_implementation.actor_target))
    if actor_worker_cls is not OnlineIDMBCFSDPActor:
        raise TypeError(
            "Online IDM BC actor target must resolve to OnlineIDMBCFSDPActor."
        )
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy("actor"),
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = EmbodiedRunner(
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
