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

"""Focused Action-contract tests for FastWAM's LIBERO boundary."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.action_contract import (
    DENORMALIZED_ACTION_STAGE,
    GRIPPER_CONVERTED_ACTION_STAGE,
    NORMALIZED_ACTION_STAGE,
    PREPARED_LIBERO_ACTION_STAGE,
    SUBMITTED_LIBERO_ACTION_STAGE,
    ActionExecutionTrace,
    ActionStageStatistics,
)
from rlinf.envs.action_utils import prepare_actions_for_libero
from rlinf.envs.libero.action_contract import (
    LIBERO_ACTION_CONTRACT_SCHEMA_V1,
    LiberoActionContract,
    inspect_libero_action_contract,
    merge_libero_action_contracts,
)
from rlinf.envs.libero.image_preprocessing import (
    OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
    prepare_libero_camera_batch,
)
from rlinf.models.embodiment.wam_policy.libero_runtime import (
    _convert_fastwam_gripper_to_libero,
    _seeded_randn,
)
from rlinf.workers.env.env_worker import EnvWorker


class _Controller:
    name = "OSC_POSE"
    control_dim = 6
    input_min = np.full(6, -1.0)
    input_max = np.full(6, 1.0)
    output_min = np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5])
    output_max = -output_min


class _Gripper:
    dof = 1
    speed = 0.01


class _RobotModel:
    pass


class _Robot:
    controller = _Controller()
    gripper = _Gripper()
    robot_model = _RobotModel()
    action_limits = (np.full(7, -1.0), np.full(7, 1.0))


class _UnderlyingEnv:
    action_dim = 7
    action_spec = (np.full(7, -1.0), np.full(7, 1.0))
    robots = [_Robot()]
    control_freq = 20
    horizon = 1000


class _OuterEnv:
    env = _UnderlyingEnv()


def _contract():
    return inspect_libero_action_contract(
        _OuterEnv(),
        dependency_versions={
            "libero_revision": "a" * 40,
            "robosuite_version": "1.4.0",
        },
    )


def test_live_libero_contract_records_exact_spec_and_provenance() -> None:
    contract = _contract()

    assert contract.action_dim == 7
    assert contract.low == (-1.0,) * 7
    assert contract.high == (1.0,) * 7
    assert contract.gripper_dimension_index == 6
    assert contract.dimension_names == (
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_axis_angle_x",
        "delta_axis_angle_y",
        "delta_axis_angle_z",
        "gripper",
    )
    artifact = contract.to_artifact()
    assert artifact["source"] == "underlying_env.action_spec"
    assert artifact["controller"]["name"] == "OSC_POSE"
    assert artifact["robot"]["models"] == ["_RobotModel"]
    assert artifact["dependency_versions"]["robosuite_version"] == "1.4.0"
    assert artifact["canonical_sha256"] == contract.canonical_sha256
    assert len(contract.canonical_sha256) == 64


def test_live_contract_v1_round_trip_remains_supported() -> None:
    legacy = replace(
        _contract(),
        schema=LIBERO_ACTION_CONTRACT_SCHEMA_V1,
        robot_models=(),
    )
    artifact = legacy.to_artifact()

    restored = LiberoActionContract.from_artifact(artifact)

    assert restored == legacy
    assert artifact["robot"]["model"] == "_RobotModel"
    assert "models" not in artifact["robot"]


def test_vector_contract_merge_aggregates_robot_identity_not_action_semantics() -> None:
    first = replace(
        _contract(),
        robot_model="OnTheGroundPanda",
        robot_models=("OnTheGroundPanda",),
        underlying_environment_classes=("unit.LivingRoomTask",),
    )
    second = replace(
        first,
        robot_model="MountedPanda",
        robot_models=("MountedPanda",),
        underlying_environment_classes=("unit.KitchenTask",),
    )

    merged = merge_libero_action_contracts((first.to_artifact(), second))

    assert merged.all_robot_models == ("MountedPanda", "OnTheGroundPanda")
    assert merged.underlying_environment_classes == (
        "unit.KitchenTask",
        "unit.LivingRoomTask",
    )
    assert merged.low == (-1.0,) * 7
    assert LiberoActionContract.from_artifact(merged.to_artifact()) == merged

    incompatible = replace(
        second,
        controller_output_high=(
            0.10,
            *second.controller_output_high[1:],
        ),
    )
    with pytest.raises(ValueError, match="different Action contracts"):
        merge_libero_action_contracts((first, incompatible))


def test_live_libero_contract_fails_closed_when_robot_limits_disagree() -> None:
    env = _OuterEnv()
    env.env = _UnderlyingEnv()
    env.env.robots = [_Robot()]
    env.env.robots[0].action_limits = (np.full(7, -0.5), np.full(7, 0.5))

    with pytest.raises(ValueError, match="robot action limits"):
        inspect_libero_action_contract(env)


def test_action_stage_statistics_are_per_dimension_and_batch_round_trip() -> None:
    contract = _contract()
    values = torch.tensor(
        [
            [[-1.25, 0.0, 0.5], [0.25, 1.5, float("inf")]],
            [[-0.5, -2.0, 0.0], [0.75, 0.5, 1.0]],
        ],
        dtype=torch.float32,
    )
    stats = ActionStageStatistics.from_values(
        stage="normalized_action",
        values=values,
        low=(-1.0, -1.0, -1.0),
        high=(1.0, 1.0, 1.0),
        gripper_dimension_index=2,
        action_contract_sha256=contract.canonical_sha256,
    )

    assert stats.shape == torch.Size([2, 3])
    assert stats.per_sample_shape == (2, 3)
    assert stats.dtype == "float32"
    assert stats.minimum.tolist() == [[-1.25, 0.0, 0.5], [-0.5, -2.0, 0.0]]
    assert stats.maximum.tolist() == [[0.25, 1.5, 0.5], [0.75, 0.5, 1.0]]
    assert stats.finite_count.tolist() == [[2, 2, 1], [2, 2, 2]]
    assert stats.below_low_count.tolist() == [[1, 0, 0], [0, 1, 0]]
    assert stats.above_high_count.tolist() == [[0, 1, 1], [0, 0, 0]]
    assert stats.total_value_count.tolist() == [[2, 2, 2], [2, 2, 2]]

    trace = ActionExecutionTrace(stages=(stats,))
    merged = ActionExecutionTrace.cat(trace.split((1, 1), dim=0), dim=0)
    assert merged == trace.cpu()
    time_merged = ActionExecutionTrace.merge_time((trace, trace))
    assert time_merged.stages[0].per_sample_shape == (4, 3)
    assert torch.equal(time_merged.stages[0].minimum, stats.minimum)
    assert torch.equal(time_merged.stages[0].maximum, stats.maximum)
    assert torch.equal(time_merged.stages[0].finite_count, stats.finite_count * 2)
    assert torch.equal(
        time_merged.stages[0].total_value_count, stats.total_value_count * 2
    )
    dimension = trace.record_for_batch_index(0)["stages"]["normalized_action"][
        "dimensions"
    ][2]
    assert dimension == {
        "index": 2,
        "minimum": 0.5,
        "maximum": 0.5,
        "finite_count": 1,
        "below_low_count": 0,
        "above_high_count": 1,
        "total_value_count": 2,
    }


def test_minmax_normalizer_round_trip_does_not_clamp_backward() -> None:
    normalizer = SingleFieldLinearNormalizer(
        stats={
            "min": torch.tensor([-0.9375, 0.0]),
            "max": torch.tensor([0.9375, 1.0]),
        },
        mode="min/max",
    )
    source = torch.tensor([[-0.9375, 0.0], [0.0, 0.5], [0.9375, 1.0]])
    assert torch.allclose(normalizer.backward(normalizer.forward(source)), source)

    restored = normalizer.backward(torch.tensor([[1.2, -1.2]]))
    assert restored[0, 0] > 0.9375
    assert restored[0, 1] < 0.0


def test_gripper_binarization_matches_official_evaluator_without_arm_clamp() -> None:
    actions = torch.tensor(
        [
            [-1.125, 1.0625, 0, 0, 0, 0, -0.03125],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.5],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1.0625],
        ],
        dtype=torch.float32,
    )

    affine = _convert_fastwam_gripper_to_libero(actions, binarize=False)
    binary = _convert_fastwam_gripper_to_libero(actions, binarize=True)

    assert torch.equal(affine[:, -1], torch.tensor([1.0625, 1.0, 0.0, -1.0, -1.125]))
    assert torch.equal(binary[:, -1], torch.tensor([1.0, 1.0, 0.0, -1.0, -1.0]))
    assert torch.equal(binary[:, :-1], actions[:, :-1])
    assert binary[0, 0].item() == -1.125
    assert binary[0, 1].item() == 1.0625


def test_prepare_actions_for_libero_is_identity_and_does_not_clamp() -> None:
    actions = np.array([[[-1.125, 1.0625, 0, 0, 0, 0, 1]]], dtype=np.float32)
    original = actions.copy()

    prepared = prepare_actions_for_libero(actions, "fastwam_adaptive")

    np.testing.assert_array_equal(prepared, original)
    np.testing.assert_array_equal(actions, original)
    assert prepared[0, 0, 0] == np.float32(-1.125)
    assert prepared[0, 0, 1] == np.float32(1.0625)


def _official_center_crop_resize(
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    pil_image = Image.fromarray(image)
    source_width, source_height = pil_image.size
    scale = max(width / source_width, height / source_height)
    resized = pil_image.resize(
        (
            round(source_width * scale),
            round(source_height * scale),
        ),
        resample=Image.BILINEAR,
    )
    resized_width, resized_height = resized.size
    left = max((resized_width - width) // 2, 0)
    top = max((resized_height - height) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + width, top + height)),
        dtype=np.uint8,
    )


def test_camera_preprocessing_is_pixel_exact_with_official_evaluator() -> None:
    values = np.arange(2 * 240 * 320 * 3, dtype=np.uint32)
    images = (values.reshape(2, 240, 320, 3) % 256).astype(np.uint8)
    expected = np.stack(
        [_official_center_crop_resize(image, height=224, width=224) for image in images]
    )

    prepared_bhwc = prepare_libero_camera_batch(
        images,
        height=224,
        width=224,
        resize_mode=OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
    )
    prepared_bchw = prepare_libero_camera_batch(
        torch.from_numpy(images).permute(0, 3, 1, 2),
        height=224,
        width=224,
        resize_mode=OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
    )

    expected_bchw = torch.from_numpy(expected.copy()).permute(0, 3, 1, 2)
    assert prepared_bhwc.dtype is torch.uint8
    assert torch.equal(prepared_bhwc, expected_bchw)
    assert torch.equal(prepared_bchw, expected_bchw)


@pytest.mark.parametrize(
    "bad_images",
    [torch.zeros(3, 224, 224), torch.zeros(1, 3, 224, 224)],
)
def test_camera_preprocessing_fails_closed_on_malformed_input(bad_images) -> None:
    with pytest.raises((TypeError, ValueError)):
        prepare_libero_camera_batch(
            bad_images,
            height=224,
            width=224,
            resize_mode=OFFICIAL_LIBERO_CAMERA_RESIZE_MODE,
        )


def test_seeded_noise_matches_official_cpu_generator() -> None:
    generator = torch.Generator(device="cpu").manual_seed(42)
    expected = torch.randn(
        (2, 3),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    actual = _seeded_randn(
        42,
        (2, 3),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        rand_device="cpu",
    )

    assert torch.equal(actual, expected)


class _TrainingActionTraceEnv:
    def __init__(self, contract):
        self.action_contract = contract
        self.submitted = None

    def chunk_step_with_action_trace(self, actions, contract):
        assert contract is self.action_contract
        self.submitted = torch.as_tensor(actions).clone()
        submitted = ActionStageStatistics.from_values(
            stage=SUBMITTED_LIBERO_ACTION_STAGE,
            values=actions,
            low=contract.low,
            high=contract.high,
            gripper_dimension_index=contract.gripper_dimension_index,
            action_contract_sha256=contract.canonical_sha256,
        )
        batch, horizon = actions.shape[:2]
        observations = [{"state": torch.zeros(batch, 1)}]
        rewards = torch.zeros(batch, horizon)
        terminations = torch.zeros(batch, horizon, dtype=torch.bool)
        truncations = torch.zeros_like(terminations)
        return (observations, rewards, terminations, truncations, [{}]), submitted


def _three_stage_model_trace(actions, contract) -> ActionExecutionTrace:
    return ActionExecutionTrace(
        tuple(
            ActionStageStatistics.from_values(
                stage=stage,
                values=actions,
                low=contract.low,
                high=contract.high,
                gripper_dimension_index=contract.gripper_dimension_index,
                action_contract_sha256=contract.canonical_sha256,
            )
            for stage in (
                NORMALIZED_ACTION_STAGE,
                DENORMALIZED_ACTION_STAGE,
                GRIPPER_CONVERTED_ACTION_STAGE,
            )
        )
    )


def test_guarded_env_step_combines_exact_five_stage_trace_without_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    environment = _TrainingActionTraceEnv(contract)
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "runner": {"fastwam_training_guard": {"enabled": True}},
            "env": {
                "train": {
                    "env_type": "libero",
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
        }
    )
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "fastwam_adaptive",
            "num_action_chunks": 10,
            "action_dim": 7,
        }
    )
    worker.env_list = [environment]
    worker.use_external_reward_model = False
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda *, raw_chunk_actions, **_: raw_chunk_actions,
    )
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.get_env_attr",
        lambda owner, name: getattr(owner, name),
    )
    actions = torch.linspace(-0.75, 0.75, 70).reshape(1, 10, 7)
    model_trace = _three_stage_model_trace(actions, contract)

    _, _, payload = inspect.unwrap(EnvWorker.env_interact_step)(
        worker, actions, 0, action_execution_trace=model_trace
    )

    trace = payload["action_execution_trace"]
    assert trace.stage_names == (
        NORMALIZED_ACTION_STAGE,
        DENORMALIZED_ACTION_STAGE,
        GRIPPER_CONVERTED_ACTION_STAGE,
        PREPARED_LIBERO_ACTION_STAGE,
        SUBMITTED_LIBERO_ACTION_STAGE,
    )
    assert torch.equal(environment.submitted, actions)
    assert trace.stages[-2].record_for_batch_index(0) == (
        trace.stages[-1].record_for_batch_index(0)
    )


def test_guarded_env_step_requires_typed_model_action_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "runner": {"fastwam_training_guard": {"enabled": True}},
            "env": {"train": {"env_type": "libero"}},
        }
    )
    worker.model_cfg = OmegaConf.create(
        {"model_type": "fastwam_adaptive", "num_action_chunks": 10, "action_dim": 7}
    )
    worker.env_list = [_TrainingActionTraceEnv(contract)]
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda *, raw_chunk_actions, **_: raw_chunk_actions,
    )
    actions = torch.zeros(1, 10, 7)
    with pytest.raises(ValueError, match="typed model Action trace"):
        inspect.unwrap(EnvWorker.env_interact_step)(worker, actions, 0)


def test_guarded_training_rollout_input_injects_live_action_contract() -> None:
    contract = _contract()
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {"runner": {"fastwam_training_guard": {"enabled": True}}}
    )
    worker.model_cfg = OmegaConf.create({"model_type": "fastwam_adaptive"})
    worker.env_list = [_TrainingActionTraceEnv(contract)]
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 2
    worker.eval_num_envs_per_stage = 2
    worker._rank = 0
    worker.enable_rlt = False
    observations = {
        "states": torch.zeros(2, 8),
        "task_descriptions": ["first", "second"],
    }

    rollout_input = worker._build_rollout_input_data(
        {
            "obs": observations,
            "final_obs": None,
            "dones": torch.zeros(2, 10, dtype=torch.bool),
        },
        stage_id=0,
    )

    rollout_obs = rollout_input["obs"]
    assert "_fastwam_action_contract_low" not in observations
    assert torch.equal(
        rollout_obs["_fastwam_action_contract_low"],
        torch.tensor(contract.low).expand(2, -1),
    )
    assert torch.equal(
        rollout_obs["_fastwam_action_contract_high"],
        torch.tensor(contract.high).expand(2, -1),
    )
    assert rollout_obs["_fastwam_action_gripper_indices"].tolist() == [6, 6]
    assert rollout_obs["_fastwam_action_contract_sha256"] == [
        contract.canonical_sha256,
        contract.canonical_sha256,
    ]


def test_guarded_training_rollout_input_requires_typed_live_contract() -> None:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {"runner": {"fastwam_training_guard": {"enabled": True}}}
    )
    worker.model_cfg = OmegaConf.create({"model_type": "fastwam_adaptive"})
    worker.env_list = [object()]
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.eval_num_envs_per_stage = 1
    worker._rank = 0
    worker.enable_rlt = False

    with pytest.raises(TypeError, match="typed live LIBERO Action contract"):
        worker._build_rollout_input_data(
            {
                "obs": {"states": torch.zeros(1, 8)},
                "final_obs": None,
                "dones": torch.zeros(1, 10, dtype=torch.bool),
            },
            stage_id=0,
        )
