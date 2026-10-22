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

"""CPU contracts for the FR3 adapter and the optional EasyWAM data dependency."""

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.wam_policy.fr3_idm import (
    FastWAMFr3Policy,
    Fr3Processor,
    _validate_cached_context,
    validate_training_config,
)


def training_config():
    """Use the exported FR3 recipe without machine-specific artifact paths."""
    meta = {
        "images": [
            {"key": role, "raw_shape": [3, 480, 640], "shape": [3, 224, 224]}
            for role in ("left", "wrist")
        ],
        "action": [{"key": "default", "raw_shape": 8, "shape": 8}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }
    transforms = [
        {"_target_": "data.lerobot.transforms.image.ToTensor"},
        {"_target_": "torchvision.transforms.Resize", "size": [168, 224]},
        {"_target_": "data.lerobot.transforms.image.Pad", "padding": [0, 28, 0, 28]},
    ]
    processor = {
        "_target_": "data.lerobot.processors.wam_processor.WAMProcessor",
        "shape_meta": meta,
        "num_obs_steps": 17,
        "num_output_cameras": 2,
        "action_output_dim": 8,
        "proprio_output_dim": 8,
        "action_state_transforms": [
            {
                "_target_": "data.lerobot.transforms.relative_action.RelativeJointTransform",
                "keys": ["default"],
                "joint_dim": 7,
            }
        ],
        "use_stepwise_action_norm": False,
        "norm_default_mode": "z-score",
        "norm_exception_mode": None,
        "action_state_merger": {
            "_target_": "data.lerobot.transforms.action_state_merger.ConcatLeftAlign"
        },
        "train_transforms": transforms,
        "val_transforms": transforms,
    }
    return OmegaConf.create(
        {
            "model": {
                "_target_": "runtime.create_fastwam_idm",
                "backbone": {"name": "wan22"},
                "state_position": "context",
                "state_dim": 8,
                "action_dit_config": {"action_dim": 8},
            },
            "data": {
                "train": {
                    "num_frames": 17,
                    "action_video_freq_ratio": 4,
                    "global_sample_stride": 1,
                    "concat_multi_camera": "horizontal",
                    "video_size": [224, 448],
                    "processor": processor,
                }
            },
        }
    )


@pytest.fixture
def processor(tmp_path):
    root = Path(__file__).resolve().parents[3] / "EasyWAM"
    if not (root / "src/data/lerobot").is_dir():
        pytest.skip(
            "Native preprocessing integration requires the sibling EasyWAM checkout."
        )
    statistics = tmp_path / "stats.json"
    stats = {
        "state": {"default": {"global_mean": [0.0] * 8, "global_std": [1.0] * 8}},
        "action": {
            "default": {
                "global_mean": [0.1] * 7 + [0.5],
                "global_std": [2.0] * 7 + [0.25],
            }
        },
    }
    statistics.write_text(json.dumps(stats))
    return Fr3Processor(training_config(), statistics, root)


def observation():
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[..., 0] = 255
    wrist = np.zeros_like(left)
    wrist[..., 1] = 255
    return {
        "observation.images.left": left,
        "observation.images.wrist": wrist,
        "observation.state": np.array([0.4] * 7 + [1.0], dtype=np.float32),
        "task": "Insert the tubes.",
        "episode_id": "episode-1",
        "chunk_id": 4,
    }


def test_training_configuration_rejects_wrong_action_transform():
    config = training_config()
    validate_training_config(config)
    config.data.train.processor.action_state_transforms[0].joint_dim = 8
    with pytest.raises(ValueError, match="joint_dim=7"):
        validate_training_config(config)


def test_images_keep_rgb_order_camera_layout_and_black_padding(processor):
    prepared = processor.prepare(observation())
    image = prepared["input_image"]
    assert image.shape == (1, 3, 224, 448)
    torch.testing.assert_close(image[0, :, 40, 40], torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(image[0, :, 40, 300], torch.tensor([-1.0, 1.0, -1.0]))
    assert torch.all(image[:, :, :28] == -1)
    assert torch.all(image[:, :, -28:] == -1)
    assert prepared["proprio"].shape == (1, 8)


def test_action_recovery_preserves_horizon_raw_anchor_and_absolute_gripper(processor):
    obs = observation()
    obs["observation.state"][0] = 8.0
    prepared = processor.prepare(obs)
    assert prepared["proprio"][0, 0] == 5.0
    normalized = torch.zeros(1, 16, 8)
    normalized[0, 1, 0] = 0.1
    result = processor.decode(normalized, prepared["raw_state"])
    assert result.shape == (16, 8)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, 7], 0.5)
    np.testing.assert_allclose(result[:2, 0], [8.1, 8.3], atol=1e-6)
    np.testing.assert_allclose(result[:, 1:7], 0.5, atol=1e-6)


def test_inverse_matches_training_forward_on_physical_labels(processor):
    raw = torch.tensor([[[0.4] * 7 + [1.0]]])
    expected = torch.tensor(
        [[[0.4 + i * 0.01] * 7 + [float(i % 2)] for i in range(16)]]
    )
    batch = {
        "action": {"default": expected[0].clone()},
        "state": {"default": raw[0].clone()},
    }
    batch = processor.processor.action_state_transform(batch)
    batch = processor.processor.normalizer.forward(batch)
    batch = processor.processor.action_state_merger.forward(batch)
    actual = processor.decode(batch["action"].unsqueeze(0), raw)
    np.testing.assert_allclose(actual, expected[0].numpy(), atol=1e-6)


class NativeModel:
    """Stand in for the external expensive model at the adapter boundary."""

    def __init__(self):
        self.encode_calls = 0
        self.calls = []

    def encode_prompt(self, prompt):
        self.encode_calls += 1
        return torch.ones(1, 2, 4096), torch.ones(1, 2, dtype=torch.bool)

    def infer_action_batch(self, **kwargs):
        self.calls.append(kwargs)
        return {"action": torch.zeros(1, 16, 8)}


def test_service_adapter_calls_idm_and_echoes_ids_with_fresh_state(processor):
    native = NativeModel()
    policy = FastWAMFr3Policy(native, processor, checkpoint_step=20000)
    obs = observation()
    first = policy.infer(obs)
    obs["observation.state"][:7] += 0.1
    obs["chunk_id"] += 1
    second = policy.infer(obs)
    assert native.encode_calls == 1
    assert native.calls[0]["action_horizon"] == 16
    assert native.calls[0]["num_video_frames"] == 5
    assert native.calls[0]["prompt"] is None
    assert first["episode_id"] == "episode-1" and first["chunk_id"] == 4
    assert second["chunk_id"] == 5
    assert first["action_space"] == "absolute_joint_position"
    np.testing.assert_allclose(
        second["actions"][:, :7] - first["actions"][:, :7], 0.1, atol=1e-6
    )
    assert policy.metadata["checkpoint_step"] == 20000
    assert policy.metadata["protocol"] == "fastwam_fr3_v1"
    assert policy.metadata["preview_video"] is False
    assert "predicted_video" not in first


def test_video_preview_encodes_the_same_native_joint_prediction(processor):
    from PIL import Image

    class NativeVideoModel(NativeModel):
        def infer_action_batch(self, **kwargs):
            raise AssertionError("Preview must use one native joint call only.")

        def infer_joint_batch(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "action": torch.zeros(1, 16, 8),
                "video": [
                    [Image.new("RGB", (448, 224), (i * 40, 20, 200)) for i in range(5)]
                ],
            }

    native = NativeVideoModel()
    policy = FastWAMFr3Policy(
        native, processor, checkpoint_step=20000, preview_video=True
    )
    result = policy.infer(observation())
    assert len(native.calls) == 1
    assert native.calls[0]["decode_video"] is True
    assert policy.metadata["preview_video"] is True
    video = result["predicted_video"]
    assert video["relative_times_s"] == [0.0, 0.4, 0.8, 1.2, 1.6]
    assert video["camera_roles"] == ["left", "wrist"]
    assert video["first_frame"] == "observation_reconstruction"
    assert result["timing"]["video_pack_s"] >= 0
    with Image.open(BytesIO(video["data"])) as clip:
        assert clip.format == "WEBP" and clip.n_frames == 5
        assert clip.size == (448, 224)
        clip.seek(4)
        np.testing.assert_allclose(np.array(clip)[100, 100, :3], [160, 20, 200], atol=6)


def test_cached_context_is_task_bound_and_needs_no_text_encoder(processor):
    obs = observation()
    payload = {
        "prompt": processor.prompt_template.format(task=obs["task"]),
        "context": torch.zeros(2, 4096),
        "context_mask": torch.tensor([True, False]),
    }
    _validate_cached_context(payload, 128)
    native = NativeModel()
    policy = FastWAMFr3Policy(
        native, processor, checkpoint_step=20000, cached_context=payload
    )
    policy.infer(obs)
    assert native.encode_calls == 0
    assert native.calls[-1]["context_mask"].tolist() == [[True, False]]
    obs["task"] = "A different instruction."
    with pytest.raises(ValueError, match="does not match"):
        policy.infer(obs)


def test_registered_builder_is_lazy_and_explicit_device_avoids_platform_query(
    monkeypatch,
):
    import rlinf.models as registry

    assert "fastwam_idm_fr3" in registry._MODEL_REGISTRY
    instance = object()
    monkeypatch.setattr(FastWAMFr3Policy, "from_config", lambda cfg: instance)

    class ForbiddenPlatform:
        def is_available(self):
            raise AssertionError("Explicit device placement must not query a platform.")

    monkeypatch.setattr(registry.Worker, "torch_platform", ForbiddenPlatform())
    config = OmegaConf.create(
        {
            "model_type": "fastwam_idm_fr3",
            "precision": "bf16",
            "is_lora": False,
            "load_to_device": False,
        }
    )
    assert registry.get_model(config) is instance


def test_native_factory_loads_full_checkpoint_with_local_auxiliary_assets(
    processor, tmp_path, monkeypatch
):
    import runtime
    from data.lerobot.utils.normalizer import save_dataset_stats_to_json

    config = training_config()
    config.model.backbone.model_id = "/training-machine/robot-backbone"
    config.model.backbone.tokenizer_model_id = "/training-machine/tokenizer"
    config.model.backbone.tokenizer_max_len = 128
    config.model.backbone.load_text_encoder = False
    config.model.action_dit_pretrained_path = "/training-machine/action.pt"
    config.model.skip_dit_load_from_pretrain = False
    training_path = tmp_path / "training.yaml"
    OmegaConf.save(config, training_path)
    stats_path = tmp_path / "stats.json"
    save_dataset_stats_to_json(processor.processor.normalizer.stats, stats_path)
    checkpoint = tmp_path / "step.pt"
    checkpoint.touch()
    assets = tmp_path / "wan22"
    assets.mkdir()
    (assets / "Wan2.2_VAE.pth").touch()
    (assets / "models_t5_umt5-xxl-enc-bf16.pth").touch()
    tokenizer = assets / "tokenizer"
    tokenizer.mkdir()
    calls = {}

    class NativeLoadModel(NativeModel):
        def __init__(self):
            super().__init__()
            self.vae = SimpleNamespace()

        def load_checkpoint(self, path):
            calls["checkpoint"] = path
            return {
                "step": 20000,
                "model_variant": "fastwam_idm",
                "backbone_name": "wan22",
                "state_position": "context",
                "text_padding": "masked",
            }

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            calls["grad_enabled"] = enabled
            return self

    def factory(**kwargs):
        calls.update(kwargs)
        return NativeLoadModel()

    monkeypatch.setattr(runtime, "create_fastwam_idm", factory)
    deployment = OmegaConf.create(
        {
            "training_config": str(training_path),
            "checkpoint": str(checkpoint),
            "statistics": str(stats_path),
            "easywam_root": str(Path(__file__).resolve().parents[3] / "EasyWAM"),
            "backbone_dir": str(assets),
            "tokenizer_dir": str(tokenizer),
            "device": "cpu",
            "preview_video": True,
        }
    )
    policy = FastWAMFr3Policy.from_config(deployment)
    assert policy.metadata["checkpoint_step"] == 20000
    assert policy.metadata["preview_video"] is True
    assert calls["checkpoint"] == str(checkpoint)
    assert calls["skip_dit_load_from_pretrain"] is True
    assert calls["action_dit_pretrained_path"] is None
    assert calls["backbone"]["model_id"] == str(assets)
    assert calls["backbone"]["load_text_encoder"] is True
    assert calls["model_dtype"] == torch.bfloat16
    assert calls["grad_enabled"] is False
    assert OmegaConf.load(training_path).model.backbone.model_id.startswith(
        "/training-machine"
    )


@pytest.mark.parametrize("preview_video", [False, True])
def test_tiny_native_idm_matches_direct_inference_and_physical_targets(
    processor, monkeypatch, preview_video
):
    import runtime
    from model.backbone import loader
    from model.backbone.wan22.wan_video_dit import WanVideoDiT
    from model.fastwam import FastWAMIDM
    from model.helpers.inference import configure_model_execution

    class PoolingVAE(torch.nn.Module):
        temporal_downsample_factor = 4
        upsampling_factor = 16

        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(z_dim=3)
            self.decode_calls = 0

        def encode(self, videos, **kwargs):
            if isinstance(videos, list):
                videos = torch.stack(videos)
            return torch.nn.functional.avg_pool3d(videos[:, :, ::4], (1, 16, 16))

        def decode(self, latents, **kwargs):
            assert preview_video, "Action-only inference must not decode video."
            self.decode_calls += 1
            return torch.nn.functional.interpolate(
                latents, size=(5, 224, 448), mode="trilinear", align_corners=False
            )

    video_config = {
        "hidden_dim": 16,
        "in_dim": 3,
        "out_dim": 3,
        "ffn_dim": 32,
        "text_dim": 4096,
        "freq_dim": 8,
        "eps": 1e-6,
        "patch_size": (1, 2, 2),
        "num_heads": 2,
        "attn_head_dim": 8,
        "num_layers": 1,
        "has_image_input": False,
        "seperated_timestep": True,
        "video_attention_mask_mode": "first_frame_causal",
        "attention_backend": "sdpa",
    }
    action_config = {
        "hidden_dim": 8,
        "action_dim": 8,
        "ffn_dim": 16,
        "text_dim": 4096,
        "freq_dim": 8,
        "eps": 1e-6,
        "num_heads": 2,
        "attn_head_dim": 8,
        "num_layers": 1,
        "attention_backend": "sdpa",
    }
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            components = SimpleNamespace(
                dit=WanVideoDiT(**video_config),
                vae=PoolingVAE(),
                text_encoder=None,
                tokenizer=None,
                text_dim=4096,
                block_protocol="main",
                dit_path="synthetic-cpu-transformer",
                vae_path="synthetic-cpu-pooling",
                text_encoder_path=None,
                tokenizer_path=None,
            )
            monkeypatch.setattr(
                loader, "load_easywam_backbone", lambda *args, **kwargs: components
            )
            native = runtime.create_fastwam_idm(
                backbone={"name": "wan22"},
                action_dit_config=action_config,
                state_dim=8,
                state_position="context",
                skip_dit_load_from_pretrain=True,
                device="cpu",
                model_dtype=torch.float32,
            ).eval()
        assert type(native) is FastWAMIDM
        configure_model_execution(native, inference_cross_kv_reuse=True)
        obs = observation()
        obs["observation.state"][0] = 8.0
        prepared = processor.prepare(obs)
        context = torch.linspace(0, 1, 8192).reshape(2, 4096)
        mask = torch.tensor([True, False])
        policy = FastWAMFr3Policy(
            native,
            processor,
            checkpoint_step=20000,
            num_inference_steps=2,
            seed=73,
            preview_video=preview_video,
            cached_context={
                "prompt": prepared["prompt"],
                "context": context,
                "context_mask": mask,
            },
        )
        wrapped = policy.infer(obs)
        direct = native.infer_action_batch(
            prompt=None,
            input_image=prepared["input_image"],
            proprio=prepared["proprio"],
            context=context.unsqueeze(0),
            context_mask=mask.unsqueeze(0),
            action_horizon=16,
            num_video_frames=5,
            num_inference_steps=2,
            seed=73,
            rand_device="cpu",
        )["action"]
        assert direct.shape == (1, 16, 8)
        np.testing.assert_array_equal(wrapped["normalized_actions"], direct[0].numpy())
        stats = processor.processor.normalizer.stats["action"]["default"]
        expected = direct[0] * (stats["global_std"] + 1e-8) + stats["global_mean"]
        expected[:, :7] += prepared["raw_state"][0, 0, :7]
        np.testing.assert_allclose(wrapped["actions"], expected.numpy(), atol=2e-6)
        np.testing.assert_array_equal(
            wrapped["actions"], processor.decode(direct, prepared["raw_state"])
        )
        assert native.vae.decode_calls == int(preview_video)
        if preview_video:
            from PIL import Image

            with Image.open(BytesIO(wrapped["predicted_video"]["data"])) as clip:
                assert clip.n_frames == 5 and clip.size == (448, 224)
    finally:
        torch.set_num_threads(threads)
