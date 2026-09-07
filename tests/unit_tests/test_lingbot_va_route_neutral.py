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

"""CPU tests use the actual native Wan architecture, not a fake backbone."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip(
    "wan_va", reason="Native LingBot-VA source is an optional model dependency."
)
pytest.importorskip(
    "fastwam",
    reason="Shared Gate/Flow math requires the experiment's FastWAM checkout.",
)

from rlinf.models.embodiment.lingbot_va_route_neutral.cache import append_cache
from rlinf.models.embodiment.lingbot_va_route_neutral.contracts import (
    ActionNormalizer,
    RoutingConfig,
    action_tensor,
    canonical_quaternions,
    flatten_actions,
    valid_action_mask,
)
from rlinf.models.embodiment.lingbot_va_route_neutral.functional import (
    FunctionalVA,
    mesh,
)


@pytest.fixture
def native():
    from wan_va.modules.model import WanTransformer3DModel

    torch.manual_seed(11)
    return WanTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        action_dim=30,
        text_dim=16,
        freq_dim=16,
        ffn_dim=48,
        num_layers=2,
        attn_mode="torch",
    ).eval()


@pytest.fixture
def functional(native):
    return FunctionalVA(native, rank=2, alpha=2)


@pytest.fixture
def normalizer():
    return ActionNormalizer(
        -torch.ones(30), torch.ones(30), -torch.ones(15), torch.ones(15)
    )


@pytest.mark.parametrize("action", [False, True])
def test_native_frozen_parity(functional, native, action):
    from wan_va.utils.utils import data_seq_to_patch

    x = torch.randn(1, 30 if action else 4, 4, 4, 1 if action else 8)
    text = torch.randn(1, 5, 16)
    times = torch.full((1, 4), 700.0)
    grid, _ = mesh(
        4,
        4 if action else 2,
        1 if action else 4,
        start=3,
        action=action,
        device=x.device,
    )
    expected = native(
        {
            "noisy_latents": x,
            "timesteps": times,
            "text_emb": text,
            "grid_id": grid[None],
        },
        action_mode=action,
    )
    if action:
        expected = expected.reshape(1, 4, 4, 1, 30).permute(0, 4, 1, 2, 3)
    else:
        expected = data_seq_to_patch(native.patch_size, expected, 4, 4, 8)
    actual, _ = functional(x, times, text, action=action, start=3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapters_only_update_action_student(functional):
    x = torch.randn(1, 30, 4, 4, 1)
    text = torch.randn(1, 5, 16)
    reference, _ = functional(x, 500, text, action=True)
    student, _ = functional(x, 500, text, action=True, uncond=True)
    torch.testing.assert_close(student, reference, atol=0, rtol=0)
    student.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in functional.adapters.parameters()
    )
    assert all(p.grad is None for p in functional.parent.parameters())
    with torch.no_grad():
        for p in functional.adapters.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.1)
    again, _ = functional(x, 500, text, action=True)
    changed, _ = functional(x, 500, text, action=True, uncond=True)
    torch.testing.assert_close(again, reference, atol=0, rtol=0)
    assert not torch.equal(changed, reference)
    with pytest.raises(ValueError, match="video"):
        functional(torch.randn(1, 4, 4, 4, 8), 500, text, uncond=True)


def test_cache_read_does_not_evict_or_mutate(functional):
    text = torch.randn(1, 5, 16)
    _, kv = functional(torch.randn(1, 4, 4, 4, 8), 0, text)
    history = append_cache((), kv, max_frames=4)
    saved = [x.key.clone() for x in history]
    for _ in range(3):
        functional(
            torch.randn(1, 30, 4, 4, 1),
            500,
            text,
            action=True,
            uncond=True,
            history=history,
            start=4,
        )
    assert all(torch.equal(x.key, old) for x, old in zip(history, saved))
    _, next_kv = functional(
        torch.randn(1, 4, 2, 4, 8), 0, text, start=4, history=history
    )
    trimmed = append_cache(history, next_kv, max_frames=4)
    assert set(trimmed[0].frames.tolist()) == {2, 3, 4, 5}
    assert set(history[0].frames.tolist()) == {0, 1, 2, 3}


def test_action_codec_and_first_chunk_mask(normalizer):
    physical = torch.randn(16, 8)
    encoded = normalizer.encode(physical)
    torch.testing.assert_close(normalizer.decode(encoded), physical)
    native_actions = action_tensor(encoded)
    torch.testing.assert_close(flatten_actions(native_actions), encoded)
    mask = valid_action_mask(native_actions, first_chunk=True)
    assert not mask[:, :, :1].any()
    assert mask.sum() == 12 * 8
    assert not encoded[:, 7:28].any()


def test_quaternion_sign_and_invalid_pose():
    actions = np.zeros((3, 8))
    actions[:, 6] = [2, -2, 3]
    fixed = canonical_quaternions(actions)
    np.testing.assert_array_equal(fixed[:, 6], 1)
    with pytest.raises(ValueError, match="quaternions"):
        canonical_quaternions(np.zeros((1, 8)))


def test_reference_profile_rejects_temporal_misalignment():
    with pytest.raises(ValueError, match="eight"):
        RoutingConfig(execution_horizon=10)


def test_gpu_selection_rejects_gpu7_before_any_query(monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "examples/embodiment/lingbot_va/run.py"
    spec = importlib.util.spec_from_file_location("lingbot_va_cli", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    calls = []
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        entry.subprocess,
        "check_output",
        lambda args, **kwargs: calls.append(args) or "",
    )
    with pytest.raises(ValueError, match="GPU7"):
        entry.configure_devices("7")
    with pytest.raises(ValueError):
        entry.configure_devices("0,1")
    assert not calls
    entry.configure_devices("1,3,4,6", count=4)
    assert calls[0][calls[0].index("-i") + 1] == "1,3,4,6"


def test_franka_observation_is_measured_state_and_one_rgb_conversion():
    from types import SimpleNamespace

    from rlinf.models.embodiment.lingbot_va_route_neutral.robot import FrankaDriver

    measured = SimpleNamespace(
        arm_joint_position=np.arange(7),
        tcp_pose=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
        gripper_position=0.04,
    )
    env = SimpleNamespace(
        config=SimpleNamespace(step_frequency=20), _franka_state=measured
    )
    driver = FrankaDriver(env, ("cam1", "cam2"))
    frames = {
        "frames": {k: np.array([[[1, 2, 3]]], dtype=np.uint8) for k in ("cam1", "cam2")}
    }
    first = driver._observation(frames)
    np.testing.assert_array_equal(first["images"]["external"], [[[3, 2, 1]]])
    np.testing.assert_array_equal(first["state"][:7], np.arange(7))
    assert first["state"].shape == (15,)
    measured.tcp_pose[-1] = -1
    second = driver._observation(frames)
    np.testing.assert_array_equal(second["state"], first["state"])


class SmallEncoder:
    def reset(self):
        self.first = True

    def text(self, instruction):
        return torch.full((1, 5, 16), float(bool(instruction))), torch.ones(
            1, 5, dtype=torch.bool
        )

    def images(self, observations, current_only=False):
        frames = 1 if current_only or self.first else len(observations) // 4
        if not current_only:
            self.first = False
        return torch.full((1, 4, frames, 4, 8), observations[-1]["pixel"])


@pytest.fixture
def runtime(functional, normalizer):
    from rlinf.models.embodiment.lingbot_va_route_neutral.runtime import (
        LingBotVARuntime,
    )

    config = RoutingConfig(
        gate_layers=(0, 1),
        gate_hidden=24,
        video_steps=2,
        action_steps=3,
        history_frames=4,
    )
    runtime = LingBotVARuntime(functional, SmallEncoder(), normalizer, config)
    runtime.reset_episode("pick", {"pixel": 0.2, "state": np.zeros(15)})
    return runtime


def test_runtime_switches_replay_and_teacher_are_read_only(runtime):
    from rlinf.models.embodiment.lingbot_va_route_neutral.contracts import Route
    from rlinf.models.embodiment.lingbot_va_route_neutral.runtime import ChunkNoise

    observation = {"pixel": 0.4, "state": np.zeros(15)}
    for index, route in enumerate(
        (Route.UNCOND, Route.IDM, Route.UNCOND, Route.UNCOND)
    ):
        features = runtime.prepare_gate_features(observation)
        old_history = tuple(x.key.clone() for x in runtime.history)
        sample = runtime.sample_chunk(
            route, ChunkNoise(1 + index, 10 + index, 20 + index), training=True
        )
        assert sample.actions.shape == (8, 8)
        assert sample.timings["video_forwards"] == (0 if route is Route.UNCOND else 3)
        assert all(torch.equal(a, b.key) for a, b in zip(old_history, runtime.history))
        after = runtime.prepare_gate_features(observation)
        for before_layer, after_layer in zip(
            features.visual.layers, after.visual.layers
        ):
            torch.testing.assert_close(
                before_layer.current_frame_video.key,
                after_layer.current_frame_video.key,
                atol=0,
                rtol=0,
            )
        if sample.flow is not None:
            replayed = runtime.replay_uncond_transition(sample.flow)
            torch.testing.assert_close(
                replayed.detach(), sample.flow.old_log_prob, rtol=0, atol=0
            )
            replayed.backward()
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in runtime.model.adapters.parameters()
            )
            teacher = runtime.sample_from_context(
                Route.IDM, sample.noise, sample.context
            )
            assert runtime.pending is sample
            assert teacher.flow is None
        runtime.commit_execution([observation] * 8, sample.actions)
        rebuilt = runtime.rebuild_history(runtime.snapshot())
        for actual, restored in zip(runtime.history, rebuilt):
            torch.testing.assert_close(actual.key, restored.key, rtol=0, atol=0)
            assert len(set(actual.frames.tolist())) <= 4
    runtime.reset_episode("pick", observation)
    assert not runtime.blocks and not runtime.history and runtime.start == 0


def test_bc_only_updates_lora_and_terminal_prefix_is_not_committed(runtime):
    from rlinf.models.embodiment.lingbot_va_route_neutral.runtime import ChunkNoise

    sample = runtime.sample_chunk("uncond", ChunkNoise(1, 2, 3))
    loss = runtime.bc_loss(sample.context, sample.normalized_plan, 42)
    loss.backward()
    assert all(p.grad is None for p in runtime.model.parent.parameters())
    assert any(p.grad is not None for p in runtime.model.adapters.parameters())
    runtime.commit_execution([{}] * 3, sample.actions[:3], terminated=True)
    assert not runtime.active and not runtime.history and runtime.pending is None


class SmallCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = torch.nn.Linear(15, 1)

    def encode_features(self, observation):
        return observation["states"].detach()

    def value_from_features(self, features):
        return self.value_head(features.detach())


@pytest.fixture
def policy(runtime):
    from rlinf.models.embodiment.lingbot_va_route_neutral.policy import (
        LingBotVARouteNeutralPolicy,
        make_gate,
    )

    return LingBotVARouteNeutralPolicy(
        runtime, make_gate(runtime.model.parent, runtime.config), SmallCritic()
    )


def collect_round(trainer):
    observation = {
        "pixel": 0.3,
        "state": np.ones(15),
        "images": {
            "external": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    }
    episodes = []
    for index in range(4):
        trainer.policy.reset_episode("pick", observation)
        episode = []
        for chunk in range(2):
            decision = trainer.policy.decide(
                observation, training=True, warmup=trainer.warmup
            )
            trainer.policy.commit_execution(
                [observation] * 8,
                decision.sample.actions,
                terminated=chunk == 1,
                success=chunk == 1 and index % 2 == 0,
            )
            episode.append(decision)
        episodes.append(episode)
    return episodes


def test_gate_ratio_warmup_joint_update_and_resume(policy, tmp_path):
    from rlinf.models.embodiment.lingbot_va_route_neutral.training import (
        SingleRobotTrainer,
    )

    trainer = SingleRobotTrainer(policy)
    initial_gate = {k: v.clone() for k, v in policy.gate.state_dict().items()}
    initial_lora = {k: v.clone() for k, v in policy.core.adapters.state_dict().items()}
    initial_critic = {
        k: v.clone() for k, v in policy.critic.value_head.state_dict().items()
    }
    for _ in range(5):
        episodes = collect_round(trainer)
        assert all(d.probability == 0.5 for e in episodes for d in e)
        trainer.update(episodes)
    assert all(
        torch.equal(v, policy.gate.state_dict()[k]) for k, v in initial_gate.items()
    )
    assert all(
        torch.equal(v, policy.core.adapters.state_dict()[k])
        for k, v in initial_lora.items()
    )
    assert any(
        not torch.equal(v, policy.critic.value_head.state_dict()[k])
        for k, v in initial_critic.items()
    )
    assert trainer.controller.price == 0
    saved = trainer.save(tmp_path / "step5")
    episodes = collect_round(trainer)
    for episode in episodes:
        decision = episode[0]
        replay = policy(decision=decision)
        torch.testing.assert_close(
            replay["gate"].logprob.detach().cpu(), decision.log_prob, rtol=0, atol=0
        )
    metrics = trainer.update(episodes)
    assert metrics["teacher_samples"] > 0 and trainer.step == 6
    assert any(
        not torch.equal(v, policy.gate.state_dict()[k]) for k, v in initial_gate.items()
    )
    assert any(
        not torch.equal(v, policy.core.adapters.state_dict()[k])
        for k, v in initial_lora.items()
    )
    trainer.load(saved)
    assert trainer.step == policy.actor_version == 5
    assert (
        not policy.runtime.active
        and not policy.runtime.blocks
        and not policy.runtime.history
    )
    assert all(
        torch.equal(v, policy.core.adapters.state_dict()[k])
        for k, v in initial_lora.items()
    )
    assert trainer.optimizers["critic"].state
    assert not trainer.optimizers["lora"].state


def test_b50_clears_large_history_before_bounding_feedback():
    from rlinf.models.embodiment.lingbot_va_route_neutral.training import (
        B50Controller,
        padded_minibatches,
    )

    controller = B50Controller(price=0.09, last_side=1)
    assert controller.update(0.1) == pytest.approx(-0.001)
    assert controller.reversals == 1
    assert controller.update(0.50) == pytest.approx(-0.001)
    controller.update(0.9)
    assert controller.price == pytest.approx(0.001)
    batches = list(padded_minibatches(11, 8, torch.Generator().manual_seed(1)))
    assert batches[-1][1].sum() == 3
    assert batches[-1][0][3:] == [-1] * 5


def test_absolute_rotation_grip_and_termination_stop_commands():
    from scipy.spatial.transform import Rotation

    from rlinf.models.embodiment.lingbot_va_route_neutral.robot import (
        ChunkExecutor,
        absolute_to_delta,
    )

    current = np.r_[
        0.4, 0.0, 0.2, Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_quat()
    ]
    target = np.r_[
        0.41, 0.02, 0.19, Rotation.from_euler("xyz", [-0.1, 0.2, 0.3]).as_quat(), 0.07
    ]
    scales = [0.05, 0.5, 1.0]
    command, submitted = absolute_to_delta(target, current, scales)
    np.testing.assert_allclose(
        current[:3] + command[:3] * scales[0], target[:3], atol=1e-7
    )
    actual = Rotation.from_euler("xyz", command[3:6] * scales[1]) * Rotation.from_quat(
        current[3:]
    )
    np.testing.assert_allclose(
        actual.as_matrix(), Rotation.from_quat(target[3:7]).as_matrix(), atol=1e-7
    )
    assert command[6] == 1 and submitted[7] == pytest.approx(0.08)

    class Driver:
        calls = 0

        def submit(self, target):
            self.calls += 1
            return (
                {"timestamp": float(self.calls)},
                target,
                np.zeros(7),
                {"success": self.calls == 3},
            )

    driver = Driver()
    feedback = ChunkExecutor(driver, hz=1e9).execute(
        np.repeat(target[None], 8, axis=0), remaining_steps=320
    )
    assert driver.calls == len(feedback.executed_actions) == 3
    assert feedback.terminated and feedback.success
    driver.calls = 0
    feedback = ChunkExecutor(driver, hz=1e9).execute(
        np.repeat(target[None], 8, axis=0),
        remaining_steps=320,
        stop_status=lambda: {"takeover": driver.calls >= 2},
    )
    assert driver.calls == 2 and not feedback.autonomous and feedback.terminated


def test_action_exception_never_sends_remaining_plan():
    from rlinf.models.embodiment.lingbot_va_route_neutral.robot import ChunkExecutor

    class Driver:
        calls = 0

        def submit(self, target):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("controller stopped")
            return {"timestamp": 0}, target, np.zeros(7), {}

    driver = Driver()
    result = ChunkExecutor(driver, hz=1e9).execute(
        np.zeros((8, 8)), remaining_steps=320
    )
    assert driver.calls == 2 and len(result.executed_actions) == 1
    assert not result.autonomous and result.terminated


def test_episode_split_statistics_and_shared_window_indices(tmp_path):
    from rlinf.models.embodiment.lingbot_va_route_neutral.data import (
        TASKS,
        DemonstrationDataset,
        prepare_split,
        write_episode,
    )

    for task in TASKS:
        for i in range(3):
            states = np.zeros((17, 15), dtype=np.float32)
            states[:, 13] = 1
            observations = [
                {
                    "timestamp": j / 20,
                    "state": states[j],
                    "images": {
                        key: np.full((8, 8, 3), j, dtype=np.uint8)
                        for key in ("external", "wrist")
                    },
                }
                for j in range(17)
            ]
            actions = np.zeros((16, 8), dtype=np.float32)
            actions[:, 0], actions[:, 6] = 100 * i, 1
            write_episode(
                tmp_path / "raw",
                task,
                i,
                observations,
                actions,
                np.arange(16) / 20,
                success=True,
                reset_metadata={"scene": i},
            )
    manifest = prepare_split(
        tmp_path / "raw", tmp_path / "prepared", per_task=3, validation_count=1
    )
    assert sum(r["split"] == "train" for r in manifest["episodes"]) == 6
    expected = torch.tensor(
        [100.0 * r["episode_id"] for r in manifest["episodes"] if r["split"] == "train"]
    ).repeat_interleave(16)
    normalizer = ActionNormalizer.load(tmp_path / "prepared/normalizer.json")
    assert normalizer.q01[0] == torch.quantile(expected, 0.01)
    assert normalizer.q99[0] == torch.quantile(expected, 0.99)
    frames = 100
    episode = {
        "latents": torch.arange(frames).reshape(1, 1, -1, 1, 1),
        "actions": torch.arange(frames).reshape(1, 1, -1, 1, 1).expand(1, 30, -1, 4, 1),
        "action_mask": torch.ones(1, 30, frames, 4, 1, dtype=torch.bool),
        "frame_ids": torch.arange(frames) * 4,
        "text": torch.ones(1, 5, 16),
        "negative_text": torch.zeros(1, 5, 16),
    }
    window = DemonstrationDataset.sft_window(episode, 42)
    assert window["latents"].shape[2] == 64
    torch.testing.assert_close(window["frame_ids"], window["latents"].flatten() * 4)
    torch.testing.assert_close(
        window["actions"][0, 0, :, 0, 0], window["latents"].flatten()
    )
    context, target = DemonstrationDataset.bc_window(episode, 42)
    assert target.shape[2] == 4
    if context.start:
        assert context.blocks[0].video.shape[2] == 3
        assert all(b.video.shape[2] == 2 for b in context.blocks[1:])
        assert (
            context.blocks[-1].start + context.blocks[-1].video.shape[2]
            == context.start
        )
    assert all((b.video.flatten() < context.start).all() for b in context.blocks)


def test_sft_masks_placeholder_after_normalization():
    from rlinf.models.embodiment.lingbot_va_route_neutral.offline import (
        _schedulers,
        native_joint_inputs,
    )

    actions = torch.randn(1, 30, 4, 4, 1)
    mask = valid_action_mask(actions, True)
    window = {
        "latents": torch.randn(1, 4, 4, 4, 8),
        "actions": actions * mask,
        "action_mask": mask,
        "text": torch.ones(1, 5, 16),
        "negative_text": torch.zeros(1, 5, 16),
        "start": 0,
    }
    inputs = native_joint_inputs(
        window, *_schedulers(), reference=torch.zeros(()), seed=42
    )
    for key in ("noisy_latents", "latent", "targets"):
        assert not inputs["action_dict"][key][~mask].any()


def test_six_method_schedule_and_incomplete_results_are_not_success():
    from rlinf.models.embodiment.lingbot_va_route_neutral.data import TASKS
    from rlinf.models.embodiment.lingbot_va_route_neutral.evaluation import (
        aggregate_results,
        evaluation_route,
        make_schedule,
    )

    scenes = [
        {
            "task": task,
            "scene_id": str(i),
            "reset_photo": f"{task}/{i}.png",
            "reset_description": "marked fixture",
        }
        for task in TASKS
        for i in range(20)
    ]
    schedule = make_schedule(scenes, dict.fromkeys(TASKS, 0.4))
    assert (
        len(schedule["trials"])
        == len({r["trial_id"] for r in schedule["trials"]})
        == 360
    )
    assert aggregate_results(schedule, [])["status"] == "INCOMPLETE"
    assert (
        sum(
            evaluation_route("periodic", i, 0.4)["route"].value == "idm"
            for i in range(100)
        )
        == 40
    )


def test_native_websocket_wire_protocol_collects_four_episodes(policy, tmp_path):
    from wan_va.utils.Simple_Remote_Infer.deploy.msgpack_numpy import Packer, unpackb

    from rlinf.models.embodiment.lingbot_va_route_neutral.service import PolicyService
    from rlinf.models.embodiment.lingbot_va_route_neutral.training import (
        SingleRobotTrainer,
    )

    service = PolicyService(
        policy,
        tmp_path / "server",
        task="grasp_place",
        trainer=SingleRobotTrainer(policy),
    )
    packer = Packer()

    def call(request):
        return unpackb(packer.pack(service.infer(unpackb(packer.pack(request)))))

    observation = {
        "pixel": 0.3,
        "state": np.ones(15),
        "timestamp": 100.0,
        "images": {
            key: np.zeros((8, 8, 3), dtype=np.uint8) for key in ("external", "wrist")
        },
    }
    for i in range(4):
        call(
            {
                "operation": "reset",
                "observation": observation,
                "episode_started": 100.0,
                "reset_metadata": {"scene": i},
            }
        )
        decision = call({"operation": "decide", "observation": observation})
        result = call(
            {
                "operation": "commit",
                "episode_finished": 102.0,
                "feedback": {
                    "observed_frames": [observation] * 8,
                    "executed_actions": decision["actions"],
                    "submitted_commands": np.zeros((8, 7)),
                    "terminated": True,
                    "success": i % 2 == 0,
                    "autonomous": True,
                    "reason": "success" if i % 2 == 0 else "step_limit",
                    "stage_progress": 0.5,
                    "action_timestamps": [101.0 + j * 0.05 for j in range(8)],
                    "observation_timestamps": [101.05 + j * 0.05 for j in range(8)],
                },
            }
        )
        assert result["step"] == (1 if i == 3 else 0)
    assert (tmp_path / "server/rollouts/update_1.pt").exists()
    assert len((tmp_path / "server/chunks.jsonl").read_text().splitlines()) == 4


def test_actual_streaming_vae_current_encoder_isolated_and_native_normalization():
    from diffusers import AutoencoderKLWan

    from rlinf.models.embodiment.lingbot_va_route_neutral.encoder import (
        ObservationEncoder,
    )

    torch.manual_seed(42)
    vae = AutoencoderKLWan(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        in_channels=12,
        out_channels=12,
        patch_size=2,
        latents_mean=[0.13, 0.17, -0.11, 0.23],
        latents_std=[1.17, 1.23, 0.77, 0.83],
    ).eval()
    config = RoutingConfig(height=32, width=32)
    encoder = ObservationEncoder(vae, torch.nn.Linear(1, 1), None, config)
    observations = [
        {
            "images": {
                key: np.full((32, 32, 3), i * 20, dtype=np.uint8)
                for key in ("external", "wrist")
            }
        }
        for i in range(9)
    ]
    anchor = encoder.images(observations[:1])
    current = encoder.images([observations[-1]], current_only=True)
    next_latents = encoder.images(observations[1:])
    again = encoder.images([observations[-1]], current_only=True)
    torch.testing.assert_close(current, again, atol=0, rtol=0)
    images = (
        torch.tensor(
            np.stack(
                [
                    [o["images"][key] for o in observations]
                    for key in ("external", "wrist")
                ]
            )
        )
        .permute(0, 4, 1, 2, 3)
        .float()
        / 255
        * 2
        - 1
    )
    with torch.no_grad():
        mu = vae.encode(images).latent_dist.mean
    mean = torch.tensor(vae.config.latents_mean)[None, :, None, None, None]
    std = torch.tensor(vae.config.latents_std)[None, :, None, None, None]
    native = torch.cat((((mu.float() - mean) * (1 / std)).to(mu)).split(1), dim=-1)
    torch.testing.assert_close(
        torch.cat((anchor, next_latents), dim=2), native, atol=0, rtol=0
    )


def test_bf16_backbone_keeps_fp32_lora_and_exact_flow_replay(runtime):
    from rlinf.models.embodiment.lingbot_va_route_neutral.runtime import ChunkNoise

    runtime.model.parent.to(torch.bfloat16)
    sample = runtime.sample_chunk("uncond", ChunkNoise(1, 2, 3), training=True)
    replay = runtime.replay_uncond_transition(sample.flow)
    torch.testing.assert_close(
        replay.detach(), sample.flow.old_log_prob, atol=0, rtol=0
    )
    replay.backward()
    assert all(p.dtype == torch.float32 for p in runtime.model.adapters.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in runtime.model.adapters.parameters()
    )


def test_lerobot_export_round_trip(tmp_path):
    import json

    lerobot = pytest.importorskip(
        "lerobot.datasets.lerobot_dataset",
        reason="Export requires native lerobot==0.3.3.",
    )
    from rlinf.models.embodiment.lingbot_va_route_neutral.data import (
        export_lerobot,
        write_episode,
    )

    state = np.zeros(15, dtype=np.float32)
    state[13] = 1
    observations = [
        {
            "timestamp": i / 20,
            "state": state,
            "images": {
                key: np.full((32, 32, 3), i * 10, dtype=np.uint8)
                for key in ("external", "wrist")
            },
        }
        for i in range(17)
    ]
    actions = np.zeros((16, 8), dtype=np.float32)
    actions[:, 6] = 1
    raw = write_episode(
        tmp_path / "raw",
        "grasp_place",
        0,
        observations,
        actions,
        np.arange(16) / 20,
        success=True,
        reset_metadata={"scene": "test"},
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task": "grasp_place",
                        "episode_id": 0,
                        "raw_path": str(raw),
                        "split": "train",
                        "instruction": "pick",
                    }
                ]
            }
        )
    )
    exported = export_lerobot(prepared, tmp_path / "lerobot")
    dataset = lerobot.LeRobotDataset(
        repo_id="local/lingbot_va_franka", root=exported, video_backend="pyav"
    )
    assert len(dataset) == 17
    assert dataset[0]["observation.images.external"].shape == (3, 256, 256)
    torch.testing.assert_close(dataset[0]["action"], torch.tensor(actions[0]))
    episode = json.loads((exported / "meta/episodes.jsonl").read_text())
    assert episode["action_config"] == [{"start_frame": 0, "end_frame": 16}]
    assert episode["split"] == "train"
