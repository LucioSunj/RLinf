# Copyright 2026 The RLinf Authors.
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

"""Entry points for the LingBot-VA single-arm experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def configure_devices(selected, *, count=1):
    """Select only user-listed physical GPUs0--6, before CUDA initialization."""
    devices = [int(value) for value in selected.split(",")]
    if (
        len(devices) != count
        or len(set(devices)) != len(devices)
        or any(value not in range(7) for value in devices)
    ):
        raise ValueError(
            f"This stage requires {count} distinct physical GPUs0--6; GPU7 is excluded even from queries."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    if int(os.environ.get("RANK", 0)) != 0:
        return
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if output:
        raise RuntimeError(
            "A selected GPU is owned by another process. Choose idle authorized devices or wait."
        )


def required_assets(cfg, command):
    paths = [
        Path(cfg.model.base_model_path) / name / "config.json"
        for name in ("vae", "text_encoder")
    ]
    paths.append(Path(cfg.model.base_model_path) / "tokenizer/tokenizer_config.json")
    if command == "precompute":
        paths.append(Path(cfg.model.normalizer_path))
    if command not in {"precompute", "train-parent"}:
        paths += [
            Path(cfg.model.parent_path) / "transformer/config.json",
            Path(cfg.model.normalizer_path),
        ]
    if command == "train-parent":
        paths.append(Path(cfg.model.base_model_path) / "transformer/config.json")
    if command in {"train-parent", "train-bc", "precompute"}:
        paths.append(Path(cfg.offline.prepared) / "manifest.json")
    if command not in {"train-parent", "precompute"} and cfg.model.bc_path:
        paths.append(Path(cfg.model.bc_path))
    missing = [str(p) for p in paths if not p.is_file()]
    if cfg.model.get("load_critic", False):
        critic_dir = Path(cfg.model.critic.backbone.model_path)
        critic_weights = [
            *critic_dir.glob("*.safetensors"),
            critic_dir / "model_state_dict/full_weights.pt",
            critic_dir / "actor/model_state_dict/full_weights.pt",
        ]
        if not cfg.model.critic.backbone.model_path or not any(
            p.is_file() for p in critic_weights
        ):
            missing.append(f"Pi0.5 checkpoint weights in {critic_dir}")
        norm = (
            Path(cfg.model.critic.backbone.openpi_data.norm_stats_path)
            / "norm_stats.json"
        )
        if not norm.is_file():
            missing.append(str(norm))
    if missing:
        raise FileNotFoundError(
            "Required experiment assets are absent:\n" + "\n".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "export",
            "precompute",
            "train-parent",
            "train-bc",
            "serve",
            "client",
            "collect",
            "schedule",
            "aggregate",
            "validate",
            "benchmark",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--gpus", help="Explicit idle physical devices0--6, e.g. 0 or 0,1,2,3"
    )
    parser.add_argument(
        "--task",
        choices=("grasp_place", "occluded_drawer", "peg_insertion"),
        default="grasp_place",
    )
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--scenes", type=Path)
    parser.add_argument("--development-rates", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--rl-checkpoint", type=Path)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Serve the four-episode online training loop",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--endpoint", choices=("idm", "uncond"), default="idm")
    args = parser.parse_args()
    for field in {
        "prepare": ("raw",),
        "export": ("prepared",),
        "schedule": ("scenes", "development_rates"),
        "aggregate": ("schedule", "results"),
        "benchmark": ("recording",),
    }.get(args.command, ()):
        if getattr(args, field) is None:
            parser.error(f"{args.command} requires --{field.replace('_', '-')}.")
    if args.train and args.command != "serve":
        parser.error("--train is used with serve.")
    if args.resume and not (args.command == "serve" and args.train):
        parser.error("--resume is used with serve --train.")
    if args.rl_checkpoint and args.command not in {"serve", "validate", "benchmark"}:
        parser.error("--rl-checkpoint is used with serve, validate or benchmark.")
    for path in (args.resume, args.rl_checkpoint, args.recording):
        if path is not None and not path.is_file():
            parser.error(f"Required input file does not exist: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf

    if args.command == "prepare":
        from rlinf.models.embodiment.lingbot_va_route_neutral.data import prepare_split

        prepare_split(args.raw, args.output)
        return
    if args.command == "export":
        from rlinf.models.embodiment.lingbot_va_route_neutral.data import export_lerobot

        export_lerobot(args.prepared, args.output)
        return
    if args.command == "schedule":
        from rlinf.models.embodiment.lingbot_va_route_neutral.evaluation import (
            make_schedule,
        )

        result = make_schedule(
            json.loads(args.scenes.read_text()),
            json.loads(args.development_rates.read_text()),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.command == "aggregate":
        from rlinf.models.embodiment.lingbot_va_route_neutral.evaluation import (
            write_aggregate,
        )

        files = sorted(args.results.glob("**/episodes/*.json"))
        result = write_aggregate(args.schedule, files, args.output)
        print(json.dumps(result, indent=2))
        return
    if args.config is None:
        parser.error("This command requires --config.")
    cfg = OmegaConf.load(args.config)
    if args.command == "train-bc" and cfg.model.bc_path:
        raise ValueError(
            "Stage B starts new adapters; clear model.bc_path for BC training."
        )
    if args.train and args.rl_checkpoint:
        raise ValueError(
            "Training restores --resume; --rl-checkpoint selects a final evaluation endpoint."
        )
    if args.command in {"client", "collect"}:
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        # Robot control belongs in the existing controller environment, with no
        # inference GPU visible. Its Ray configuration is explicit in robot.yaml.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from rlinf.models.embodiment.lingbot_va_route_neutral.client import (
            collect_spacemouse,
            run_robot_client,
        )
        from rlinf.models.embodiment.lingbot_va_route_neutral.robot import (
            build_franka_driver,
        )

        driver = build_franka_driver(cfg)
        if args.command == "collect":
            collect_spacemouse(driver, args.output, args.task)
        else:
            from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
                WebsocketClientPolicy,
            )

            remote = WebsocketClientPolicy(host=args.host, port=args.port)
            schedule = json.loads(args.schedule.read_text()) if args.schedule else None
            run_robot_client(
                remote,
                driver,
                task=args.task,
                output=args.output,
                schedule=schedule,
                endpoint=args.endpoint,
            )
        return
    cfg.model.load_critic = bool(args.command == "serve" and args.train)
    if args.train and not cfg.model.bc_path:
        raise ValueError("Online RL requires the selected new UNCOND BC checkpoint.")
    if args.train and args.schedule:
        raise ValueError("Training and main evaluation use separate server runs.")
    if args.schedule and (args.rl_checkpoint is None or not cfg.model.bc_path):
        raise ValueError(
            "Six-method evaluation requires both the original BC and final RL checkpoint."
        )
    required_assets(cfg, args.command)
    if args.gpus is None:
        parser.error(
            "GPU commands require --gpus with currently idle physical GPUs0--6."
        )
    configure_devices(args.gpus, count=4 if args.command == "train-parent" else 1)
    import torch

    torch.manual_seed(42)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "train-parent":
        from rlinf.models.embodiment.lingbot_va_route_neutral.offline import (
            train_parent,
        )

        stage = OmegaConf.create(
            {
                **OmegaConf.to_container(cfg.offline, resolve=True),
                "base_model_path": cfg.model.base_model_path,
                "output": str(args.output),
            }
        )
        train_parent(stage)
        return
    if args.command == "precompute":
        from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae

        from rlinf.models.embodiment.lingbot_va_route_neutral.contracts import (
            ActionNormalizer,
            RoutingConfig,
        )
        from rlinf.models.embodiment.lingbot_va_route_neutral.data import precompute
        from rlinf.models.embodiment.lingbot_va_route_neutral.encoder import (
            ObservationEncoder,
        )

        base = Path(cfg.model.base_model_path)
        encoder = ObservationEncoder(
            load_vae(str(base / "vae"), torch.bfloat16, "cuda:0"),
            load_text_encoder(str(base / "text_encoder"), torch.bfloat16, "cpu"),
            load_tokenizer(str(base / "tokenizer")),
            RoutingConfig(**dict(cfg.model.routing)),
        )
        encoder.base_model_path = str(base.resolve())
        precompute(
            cfg.offline.prepared,
            encoder,
            ActionNormalizer.load(cfg.model.normalizer_path),
        )
        args.output.write_text(
            json.dumps({"status": "PASS", "prepared": cfg.offline.prepared}) + "\n"
        )
        return
    from rlinf.models import get_model

    policy = get_model(cfg.model)
    if args.command == "train-bc":
        from rlinf.models.embodiment.lingbot_va_route_neutral.offline import train_bc

        stage = OmegaConf.create(
            {
                **OmegaConf.to_container(cfg.offline, resolve=True),
                "output": str(args.output),
            }
        )
        train_bc(policy, stage)
        return
    if args.command in {"validate", "benchmark"}:
        if args.rl_checkpoint:
            from rlinf.models.embodiment.lingbot_va_route_neutral.service import (
                load_rl_weights,
            )

            load_rl_weights(policy, args.rl_checkpoint, args.task)
        from rlinf.models.embodiment.lingbot_va_route_neutral.data import TASKS
        from rlinf.models.embodiment.lingbot_va_route_neutral.validation import (
            native_parity,
            replay_recording,
        )

        result = native_parity(policy)
        if args.recording:
            result["recording"] = replay_recording(
                policy,
                args.recording,
                TASKS[args.task]["instruction"],
                benchmark=args.command == "benchmark",
            )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import (
        WebsocketPolicyServer,
    )

    from rlinf.models.embodiment.lingbot_va_route_neutral.service import (
        PolicyService,
        load_rl_weights,
    )
    from rlinf.models.embodiment.lingbot_va_route_neutral.training import (
        SingleRobotTrainer,
        TrainingConfig,
    )

    trainer = (
        SingleRobotTrainer(policy, TrainingConfig(**dict(cfg.training)), task=args.task)
        if args.train
        else None
    )
    if args.resume:
        if trainer is None:
            raise ValueError(
                "--resume is for online training; evaluation uses --rl-checkpoint."
            )
        trainer.load(args.resume)
    schedule = json.loads(args.schedule.read_text()) if args.schedule else None
    if (args.output / "chunks.jsonl").exists() and not args.resume:
        raise FileExistsError(
            "Use a fresh server output directory; training continuation requires --resume."
        )
    from rlinf.utils.metric_logger import MetricLogger

    cfg.runner.logger.log_path = str(args.output)
    metric_logger = MetricLogger(cfg) if trainer is not None else None
    service = PolicyService(
        policy,
        args.output,
        task=args.task,
        trainer=trainer,
        schedule=schedule,
        metric_logger=metric_logger,
    )
    if args.rl_checkpoint:
        service.final_lora = load_rl_weights(policy, args.rl_checkpoint, args.task)
    (args.output / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    WebsocketPolicyServer(
        service,
        host=args.host,
        port=args.port,
        metadata={"model_type": "lingbot_va_route_neutral", "task": args.task},
    ).serve_forever()


if __name__ == "__main__":
    main()
