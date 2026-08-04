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

from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "evaluations/libero"


def test_standard_libero_fastwam_eval_preset_is_one_env_no_critic(monkeypatch) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("FASTWAM_PROJECT_CHECKPOINT", "/tmp/project/rank_0.pt")
    monkeypatch.setenv("FASTWAM_EVAL_OUTPUT_DIR", "/tmp/fastwam-eval")
    monkeypatch.setenv("FASTWAM_EVAL_LEDGER", "/tmp/ledger.json")
    monkeypatch.setenv("FASTWAM_EVAL_RUN_ID", "preset-unit")
    monkeypatch.setenv("FASTWAM_TEXT_EMBEDDING_CACHE", "/tmp/text-contexts")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="libero_10_fastwam_adaptive_eval")

    assert cfg.runner.task_type == "embodied_eval"
    assert cfg.runner.only_eval is True
    assert cfg.runner.ckpt_path == "/tmp/project/rank_0.pt"
    assert cfg.cluster.component_placement["env,rollout"] == 0
    assert cfg.env.eval.total_num_envs == 1
    assert cfg.env.eval.is_eval is True
    assert cfg.env.eval.use_fixed_reset_state_ids is True
    assert cfg.env.eval.use_ordered_reset_state_ids is True
    assert cfg.env.eval.ordered_reset_state_ids == [0]
    assert cfg.env.eval.ignore_terminations is False
    assert cfg.env.eval.video_cfg.save_video is False
    assert cfg.rollout.model.model_type == "fastwam_adaptive"
    assert cfg.rollout.model.eval_routing_mode == "learned_threshold"
    assert cfg.rollout.model.gate_epsilon == 0.0
    assert cfg.rollout.model.kv_replay.backend == "stored"
    assert cfg.rollout.model.runtime.binarize_gripper is True
    assert cfg.rollout.model.runtime.num_inference_steps == 10
    assert cfg.rollout.model.runtime.seeded_noise_device == "cpu"
    assert cfg.rollout.model.runtime.camera_resize_mode == "official_pil_center_crop"
    assert cfg.runner.evaluation_collector.noise_seed_mode == "stateless_per_chunk"
    assert cfg.rollout.model.fastwam.load_text_encoder is False
    assert cfg.rollout.model.runtime.text_embedding_cache_dir == "/tmp/text-contexts"
    assert cfg.rollout.model.critic.load_for_eval is False
    assert cfg.rollout.collect_prev_infos is False
    assert cfg.critic.use_critic_model is False
    assert cfg.runner.evaluation_collector._target_.endswith(
        "FastWAMLiberoEvalCollector"
    )


def test_eval_preset_accepts_explicit_forced_and_matched_random_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    monkeypatch.setenv("FASTWAM_CHECKPOINT", "/tmp/fastwam.pt")
    monkeypatch.setenv("FASTWAM_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("FASTWAM_DATASET_STATS", "/tmp/dataset_stats.json")
    monkeypatch.setenv("FASTWAM_PROJECT_CHECKPOINT", "/tmp/project/rank_0.pt")
    monkeypatch.setenv("FASTWAM_EVAL_OUTPUT_DIR", "/tmp/fastwam-eval")
    monkeypatch.setenv("FASTWAM_EVAL_LEDGER", "/tmp/ledger.json")
    monkeypatch.setenv("FASTWAM_EVAL_RUN_ID", "preset-unit")
    monkeypatch.setenv("FASTWAM_TEXT_EMBEDDING_CACHE", "/tmp/text-contexts")

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        forced = compose(
            config_name="libero_10_fastwam_adaptive_eval",
            overrides=["rollout.model.eval_routing_mode=forced_uncond"],
        )
        matched = compose(
            config_name="libero_10_fastwam_adaptive_eval",
            overrides=[
                "rollout.model.eval_routing_mode=matched_random",
                "rollout.model.eval_random_idm_probability=0.375",
                "rollout.model.eval_routing_seed=19",
                "env.eval.ordered_reset_state_ids=[0,51]",
                "env.eval.total_num_envs=2",
            ],
        )

    assert forced.rollout.model.eval_routing_mode == "forced_uncond"
    assert forced.rollout.model.eval_idm_threshold == 0.5
    assert matched.rollout.model.eval_random_idm_probability == 0.375
    assert matched.rollout.model.eval_routing_seed == 19
    assert matched.env.eval.ordered_reset_state_ids == [0, 51]
    assert matched.env.eval.total_num_envs == 2
