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

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from rlinf.utils.fastwam_training_plotter import FastWAMTrainingPlotter

FASTWAM_TENSORBOARD_LAYOUT = {
    "FastWAM outcomes": {
        "Environment reward and return": [
            "Multiline",
            ["env/reward", "env/return", "rollout/rewards_mean"],
        ],
        "Sparse success": [
            "Multiline",
            [
                "env/success_once",
                "rollout/fastwam/reward/successful_trajectory_count",
            ],
        ],
        "Raw and shaped reward": [
            "Multiline",
            [
                "rollout/fastwam/reward/raw_chunk/mean",
                "rollout/fastwam/reward/shaped_chunk/mean",
                "rollout/fastwam/cost/actual_chunk/mean",
            ],
        ],
    },
    "FastWAM value and GAE": {
        "Returns": [
            "Margin",
            ["rollout/returns_mean", "rollout/returns_min", "rollout/returns_max"],
        ],
        "Values": [
            "Margin",
            ["rollout/values_mean", "rollout/values_min", "rollout/values_max"],
        ],
        "Advantages": [
            "Margin",
            [
                "rollout/advantages_mean",
                "rollout/advantages_min",
                "rollout/advantages_max",
            ],
        ],
        "Gate and Flow advantages": [
            "Multiline",
            [
                "rollout/gate_advantages_mean",
                "rollout/flow_advantages_mean",
            ],
        ],
    },
    "FastWAM Gate routing": {
        "IDM probability and realized routes": [
            "Multiline",
            [
                "rollout/fastwam/gate/base_idm_probability_mean",
                "rollout/fastwam/gate/behavior_idm_probability_mean",
                "rollout/fastwam/route/eligible_idm_fraction",
                "rollout/fastwam/route/executed_idm_fraction",
            ],
        ],
        "Base IDM probability range": [
            "Margin",
            [
                "rollout/fastwam/gate/base_idm_probability_mean",
                "rollout/fastwam/gate/base_idm_probability_min",
                "rollout/fastwam/gate/base_idm_probability_max",
            ],
        ],
        "Gate PPO stability": [
            "Multiline",
            [
                "train/gate/entropy",
                "train/gate/approx_kl",
                "train/gate/clip_fraction",
                "train/gate/ratio_abs",
            ],
        ],
    },
    "FastWAM optimization": {
        "Policy losses": [
            "Multiline",
            [
                "train/gate/policy_loss",
                "train/uncond_flow/policy_loss",
                "train/fastwam/regularized_policy_loss",
            ],
        ],
        "Critic": [
            "Multiline",
            [
                "train/critic/value_loss",
                "train/critic/explained_variance",
                "train/critic/value_clip_ratio",
            ],
        ],
        "Gradient and Gate update": [
            "Multiline",
            [
                "train/actor/grad_norm",
                "train/gate/relative_update_l2_norm",
                "train/gate/update_max_abs",
            ],
        ],
        "Learning rates": [
            "Multiline",
            ["train/gate/lr", "train/uncond_flow/lora_lr", "train/critic/lr"],
        ],
    },
    "FastWAM systems and audits": {
        "Step timing": [
            "Multiline",
            [
                "time/step",
                "time/generate_rollouts",
                "time/actor_training",
            ],
        ],
        "K/V replay bytes": [
            "Multiline",
            [
                "rollout/fastwam/kv/eligible_total_bytes",
                "rollout/fastwam/kv/eligible_max_bytes_per_sample",
            ],
        ],
        "Numerical audit errors": [
            "Multiline",
            [
                "rollout/fastwam/cost/identity_max_abs_error",
                "rollout/fastwam/counterfactual/alignment_max_abs_error",
                "train/gate/update_nonfinite_count",
            ],
        ],
    },
}


class _TensorboardLogger:
    def __init__(self, log_path, *, custom_scalars=None, flush_secs: int = 10):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_path, flush_secs=flush_secs)
        if custom_scalars:
            self.writer.add_custom_scalars(custom_scalars)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def flush(self) -> None:
        self.writer.flush()

    def finish(self):
        self.writer.close()


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger_cfg = cfg.runner.logger

        self.log_path = logger_cfg.get("log_path", "logs")
        self.project_name = logger_cfg.get("project_name", "rlinf")
        self.experiment_name = logger_cfg.get("experiment_name", "default")
        self.per_worker_log = bool(cfg.runner.get("per_worker_log", False))
        self.per_worker_log_root = cfg.runner.get(
            "per_worker_log_path", os.path.join(self.log_path, "worker_logs")
        )

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        self.wandb_proxy = logger_cfg.get("wandb_proxy", None)
        self.wandb_entity = logger_cfg.get("wandb_entity", None)
        self.swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        model_type = OmegaConf.select(self.cfg, "actor.model.model_type")
        observability_cfg = logger_cfg.get("fastwam_observability", {}) or {}
        self.fastwam_observability_enabled = str(
            model_type
        ) == "fastwam_adaptive" and bool(observability_cfg.get("enabled", True))
        tensorboard_cfg = observability_cfg.get("tensorboard", {}) or {}
        self.fastwam_custom_scalars_enabled = bool(
            tensorboard_cfg.get("custom_scalars", True)
        )
        self.tensorboard_flush_every_step = bool(
            tensorboard_cfg.get("flush_every_step", True)
        )
        self.tensorboard_flush_secs = int(tensorboard_cfg.get("flush_secs", 10))
        if self.tensorboard_flush_secs < 1:
            raise ValueError("TensorBoard flush_secs must be at least 1.")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"

        self.config = OmegaConf.to_container(cfg, resolve=True)
        self._all_loggers = []
        self._worker_loggers: dict[tuple[str, int], dict] = {}
        self._finished = False
        self.logger = self._create_logger_bundle(
            log_path=self.log_path,
            experiment_name=self.experiment_name,
            log_path_suffix="all" if self.per_worker_log else "",
        )
        self._fastwam_plotter = self._create_fastwam_plotter(observability_cfg)

    def _create_fastwam_plotter(self, observability_cfg):
        if not self.fastwam_observability_enabled:
            return None
        static_cfg = observability_cfg.get("static_plots", {}) or {}
        if not bool(static_cfg.get("enabled", True)):
            return None
        run_dir = Path(self.log_path) / str(self.experiment_name)
        configured_output_dir = static_cfg.get("output_dir", None)
        if configured_output_dir is None:
            output_dir = run_dir / "training_curves"
        else:
            output_dir = Path(str(configured_output_dir)).expanduser()
            if not output_dir.is_absolute():
                output_dir = run_dir / output_dir
        title = static_cfg.get("title", f"FastWAM adaptive RL — {self.experiment_name}")
        return FastWAMTrainingPlotter(
            output_dir,
            title=str(title),
            interval_steps=int(static_cfg.get("interval_steps", 5)),
            smoothing=float(static_cfg.get("smoothing", 0.6)),
            dpi=int(static_cfg.get("dpi", 160)),
            export_all_scalars_on_finish=bool(
                static_cfg.get("export_all_scalars_on_finish", True)
            ),
        )

    def _create_logger_bundle(
        self, log_path: str, experiment_name: str, log_path_suffix: str = ""
    ) -> dict:
        logger = {}
        if "wandb" in self.logger_backends:
            import wandb

            wandb_log_path = os.path.join(log_path, "wandb", log_path_suffix)
            os.makedirs(wandb_log_path, exist_ok=True)

            settings = None
            if self.wandb_proxy:
                settings = wandb.Settings(https_proxy=self.wandb_proxy)
            wandb.init(
                entity=self.wandb_entity,
                project=self.project_name,
                name=experiment_name,
                config=self.config,
                settings=settings,
                dir=wandb_log_path,
                reinit=True,
            )
            logger["wandb"] = wandb

        if "swanlab" in self.logger_backends:
            import swanlab

            swanlab_log_path = os.path.join(log_path, "swanlab", log_path_suffix)
            os.makedirs(swanlab_log_path, exist_ok=True)

            swanlab.init(
                project=self.project_name,
                experiment_name=experiment_name,
                config=self.config,
                logdir=swanlab_log_path,
                mode=self.swanlab_mode,
            )
            logger["swanlab"] = swanlab

        if "tensorboard" in self.logger_backends:
            tensorboard_log_path = os.path.join(
                log_path,
                experiment_name,
                "tensorboard",
                log_path_suffix,
            )
            os.makedirs(tensorboard_log_path, exist_ok=True)

            config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
            OmegaConf.save(self.cfg, config_yaml_path, resolve=True)

            custom_scalars = (
                FASTWAM_TENSORBOARD_LAYOUT
                if self.fastwam_observability_enabled
                and self.fastwam_custom_scalars_enabled
                else None
            )
            logger["tensorboard"] = _TensorboardLogger(
                tensorboard_log_path,
                custom_scalars=custom_scalars,
                flush_secs=self.tensorboard_flush_secs,
            )
        self._all_loggers.append(logger)
        return logger

    def _get_scoped_logger(self, worker_group_name: str, rank: int) -> dict:
        key = (worker_group_name, int(rank))
        if key in self._worker_loggers:
            return self._worker_loggers[key]

        scoped_log_path = os.path.join(
            self.per_worker_log_root,
            worker_group_name,
            f"rank_{int(rank)}",
        )
        scoped_experiment_name = (
            f"{self.experiment_name}-{worker_group_name}-rank_{int(rank)}"
        )
        scoped_logger = self._create_logger_bundle(
            log_path=scoped_log_path,
            experiment_name=scoped_experiment_name,
        )
        self._worker_loggers[key] = scoped_logger
        return scoped_logger

    def log(
        self,
        data,
        step,
        backend=None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ):
        target_logger = self.logger
        if self.per_worker_log and worker_group_name is not None and rank is not None:
            target_logger = self._get_scoped_logger(
                worker_group_name=worker_group_name,
                rank=rank,
            )
        elif self._fastwam_plotter is not None:
            self._fastwam_plotter.record(data, step)
        for default_backend, logger_instance in target_logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def commit_step(self, step: int) -> None:
        """Publish one complete metric step to live and static viewers."""

        if self.tensorboard_flush_every_step:
            for logger in self._all_loggers:
                tensorboard_logger = logger.get("tensorboard")
                if tensorboard_logger is not None:
                    tensorboard_logger.flush()
        if self._fastwam_plotter is not None:
            self._fastwam_plotter.maybe_render(step)

    def log_table(self, df_data, name, step):
        if "wandb" in self.logger_backends:
            table = self.logger["wandb"].Table(dataframe=df_data)
            self.logger["wandb"].log({name: table}, step=step)
        else:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")

    def __del__(self):
        self.finish()

    def finish(self):
        if self._finished:
            return
        if self._fastwam_plotter is not None:
            self._fastwam_plotter.finish()
        for logger in self._all_loggers:
            for logger_instance in logger.values():
                logger_instance.finish()
        self._finished = True
