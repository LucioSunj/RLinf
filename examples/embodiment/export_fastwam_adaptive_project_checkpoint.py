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

"""Export a native all-layer FastWAM adaptive project checkpoint at step zero."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.fastwam_checkpoint_export import (
    export_initial_actor_checkpoint,
    validate_initial_checkpoint_export_config,
)
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_10_ppo_fastwam_adaptive",
)
def main(cfg) -> None:
    """Construct one production actor and save its untouched adaptive state."""

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_world_size = component_placement.get_world_size("actor")
    validate_initial_checkpoint_export_config(
        cfg,
        actor_world_size=actor_world_size,
    )
    actor_placement = component_placement.get_strategy("actor")
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=actor_placement,
    )
    actor_dir = export_initial_actor_checkpoint(
        cfg,
        actor_group=actor_group,
        actor_world_size=actor_world_size,
    )
    print(
        json.dumps(
            {
                "schema": "fastwam-adaptive-step0-export-result-v1",
                "step": 0,
                "optimizer_steps": 0,
                "actor_checkpoint_dir": str(actor_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
