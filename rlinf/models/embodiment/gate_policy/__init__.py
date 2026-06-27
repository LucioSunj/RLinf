# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from omegaconf import DictConfig


def build_wam_adapter(wam_cfg: DictConfig):
    """Build a FROZEN fast-wam dual-regime model and wrap it in WAMModeAdapter.

    Required `wam_cfg` keys:
      configs_dir        : path to FastWAM/configs (for hydra compose)
      task               : fastwam task name (e.g. libero_metric_adaptive_joint_2cam224_1e-4)
      backbone_kind      : "joint" | "idm"
      ckpt               : path to the dual-regime checkpoint
    Optional:
      num_video_frames (9), action_horizon (32), k_lo (4), k_hi (20),
      cost_table_path (None), device ("cuda"), dtype ("bfloat16")
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from fastwam.adaptive_gate import WAMModeAdapter

    device = str(wam_cfg.get("device", "cuda"))
    dtype = getattr(torch, str(wam_cfg.get("dtype", "bfloat16")))
    configs_dir = str(wam_cfg.configs_dir)
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        fcfg = compose(config_name="train", overrides=[f"task={wam_cfg.task}"])
    model = instantiate(fcfg.model, model_dtype=dtype, device=device)
    ckpt = wam_cfg.get("ckpt", None)
    if ckpt:
        model.load_checkpoint(str(ckpt))
    model.eval()
    model.requires_grad_(False)  # WAM stays FROZEN; only the gate trains.

    return WAMModeAdapter(
        model,
        backbone_kind=str(wam_cfg.backbone_kind),
        num_video_frames=int(wam_cfg.get("num_video_frames", 9)),
        action_horizon=int(wam_cfg.get("action_horizon", 32)),
        k_lo=int(wam_cfg.get("k_lo", 4)),
        k_hi=int(wam_cfg.get("k_hi", 20)),
        cost_table_path=wam_cfg.get("cost_table_path", None),
        cost_source=wam_cfg.get("cost_source", None),
        default_seed=wam_cfg.get("default_seed", None),
    )


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.gate_policy.gate_policy import GatePolicy

    adapter = build_wam_adapter(cfg.wam)
    gate_cfg = cfg.get("gate", {})
    hidden = tuple(gate_cfg.get("hidden_sizes", (256, 256)))

    policy = GatePolicy(
        world_feat_dim=adapter.world_feat_dim,
        proprio_dim=int(cfg.proprio_dim),
        num_modes=int(gate_cfg.get("num_modes", 3)),
        hidden_sizes=hidden,
        add_value_head=bool(cfg.get("add_value_head", True)),
        use_last_action=bool(gate_cfg.get("use_last_action", False)),
        activation=str(gate_cfg.get("activation", "tanh")),
        wam_adapter=adapter,
        obs_preprocessor=None,  # wired by the env rollout (suite-specific)
    )
    return policy
