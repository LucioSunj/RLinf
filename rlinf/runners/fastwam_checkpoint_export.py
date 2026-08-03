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

"""Native step-zero FastWAM adaptive project-checkpoint export."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def validate_initial_checkpoint_export_config(
    cfg: Any,
    *,
    actor_world_size: int,
) -> Path:
    """Validate the fixed E4 bootstrap checkpoint profile."""

    output_value = OmegaConf.select(
        cfg,
        "runner.bootstrap_project_checkpoint_dir",
        default=None,
    )
    if output_value is None or not str(output_value).strip():
        raise ValueError("runner.bootstrap_project_checkpoint_dir is required.")
    if OmegaConf.select(cfg, "runner.resume_dir", default=None) is not None:
        raise ValueError("Step-zero export forbids runner.resume_dir.")
    if str(OmegaConf.select(cfg, "actor.model.model_type", default="")) != (
        "fastwam_adaptive"
    ):
        raise ValueError("Step-zero export requires fastwam_adaptive.")
    if int(actor_world_size) != 1:
        raise ValueError("Step-zero export requires exactly one actor rank.")

    num_layers = int(
        OmegaConf.select(
            cfg,
            "actor.model.fastwam.action_dit_config.num_layers",
            default=-1,
        )
    )
    if num_layers != 30:
        raise ValueError("E4 step-zero export requires exactly 30 MoT layers.")
    if bool(
        OmegaConf.select(cfg, "actor.model.gate.share_blocks", default=True)
    ):
        raise ValueError("E4 step-zero export requires independent Gate blocks.")
    if int(
        OmegaConf.select(cfg, "actor.model.gate.denoise_last_n", default=-1)
    ) != 1:
        raise ValueError("E4 step-zero export requires gate.denoise_last_n=1.")
    if str(
        OmegaConf.select(cfg, "actor.model.gate.layer_taps.mode", default="")
    ) != "all":
        raise ValueError("E4 step-zero export requires the native all-layer Gate.")
    if OmegaConf.select(
        cfg,
        "actor.model.gate.layer_taps.last_n",
        default=None,
    ) is not None or OmegaConf.select(
        cfg,
        "actor.model.gate.layer_taps.indices",
        default=None,
    ) is not None:
        raise ValueError("E4 all-layer export forbids layer subset arguments.")
    return Path(str(output_value)).expanduser().resolve()


def export_initial_actor_checkpoint(
    cfg: Any,
    *,
    actor_group: Any,
    actor_world_size: int,
) -> Path:
    """Initialize only the actor and invoke its production step-zero save."""

    output_dir = validate_initial_checkpoint_export_config(
        cfg,
        actor_world_size=actor_world_size,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Bootstrap checkpoint output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_dir = output_dir / "actor"
    actor_group.init_worker().wait()
    actor_group.save_checkpoint(str(actor_dir), 0).wait()
    return actor_dir
