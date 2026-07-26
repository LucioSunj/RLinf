"""WS3.3: gate snapshots as rollout INITIAL STATES (`reset_from_snapshot`).

Fake-worker tests, no simulator: the outer ``LiberoEnv`` runs against in-process
fake LIBERO workers wired through the real ``gate_snapshot`` capture/restore
machinery, mirroring ``test_libero_gate_snapshot.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


snapshot_mod = _load(
    "libero_gate_snapshot_start_test", "rlinf/envs/libero/gate_snapshot.py"
)
manifest_mod = _load(
    "libero_episode_manifest_start_test", "rlinf/envs/libero/episode_manifest.py"
)
phase_mod = _load("libero_gate_phase_start_test", "rlinf/envs/libero/gate_phase.py")


def _nested_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _list_of_dict_to_dict_of_list(items):
    return {key: [item[key] for item in items] for key in items[0]}


def _to_tensor(value):
    if isinstance(value, dict):
        return {key: _to_tensor(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_tensor(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value.copy()))
    return torch.as_tensor(value)


def _load_libero_env_with_stubs(monkeypatch):
    gym = types.ModuleType("gym")
    gym.Env = object
    monkeypatch.setitem(sys.modules, "gym", gym)

    utils = types.ModuleType("rlinf.envs.libero.utils")
    utils.build_interleaved_eval_reset_state_ids = lambda *args, **kwargs: None
    utils.distribute_reset_state_ids_round_robin = lambda *args, **kwargs: None
    utils.get_benchmark_overridden = lambda *_: None
    utils.get_libero_image = lambda obs: obs["agentview_image"]
    utils.get_libero_type = lambda: "standard"
    utils.get_libero_wrist_image = lambda obs: obs["robot0_eye_in_hand_image"]
    utils.quat2axisangle = lambda value: np.asarray(value, dtype=np.float64)[:3]
    utils.record_completed_episode_task_stats = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.libero.episode_manifest", manifest_mod
    )
    monkeypatch.setitem(sys.modules, "rlinf.envs.libero.gate_phase", phase_mod)
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.libero.gate_snapshot", snapshot_mod
    )

    venv = types.ModuleType("rlinf.envs.libero.venv")
    venv.ReconfigureSubprocEnv = object
    monkeypatch.setitem(sys.modules, venv.__name__, venv)
    env_utils = types.ModuleType("rlinf.envs.utils")
    env_utils.list_of_dict_to_dict_of_list = _list_of_dict_to_dict_of_list
    env_utils.to_tensor = _to_tensor
    monkeypatch.setitem(sys.modules, env_utils.__name__, env_utils)

    libero = types.ModuleType("libero")
    libero.__path__ = []
    libero_core = types.ModuleType("libero.libero")
    libero_core.__path__ = []
    benchmark = types.ModuleType("libero.libero.benchmark")
    benchmark.Benchmark = object
    monkeypatch.setitem(sys.modules, "libero", libero)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_core)
    monkeypatch.setitem(sys.modules, benchmark.__name__, benchmark)
    return _load("libero_env_snapshot_start_test", "rlinf/envs/libero/libero_env.py")


class _Controller:
    def __init__(self):
        self.goal_pos = np.array([0.1, 0.2])


class _Robot:
    def __init__(self):
        self.controller = _Controller()


class _Sim:
    def __init__(self, env):
        self.env = env

    def set_state_from_flattened(self, state):
        self.env.state = np.asarray(state).copy()

    def forward(self):
        pass


class _FakeWorker:
    """Deterministic fake OffScreenRenderEnv satisfying the snapshot contract."""

    def __init__(self, bddl, value):
        self.bddl_file_name = str(bddl)
        self.state = np.array([float(value), 0.0])
        self.timestep = 0
        self.robots = [_Robot()]
        self.sim = _Sim(self)
        self.np_random = np.random.default_rng(17 + int(value))

    def get_sim_state(self):
        return self.state.copy()

    def advance(self):
        self.state[0] += 1.0
        self.timestep += 1

    def _get_observations(self):
        pixel = int(round(float(self.state[0]))) % 256
        return {
            "agentview_image": np.full((4, 4, 3), pixel, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full((4, 4, 3), pixel, dtype=np.uint8),
            "robot0_eef_pos": np.array([float(self.state[0]), 0.0, 0.0]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
        }


class _FakeVectorEnv:
    """In-process stand-in for ReconfigureSubprocEnv's snapshot/step surface."""

    def __init__(self, workers):
        self.workers = workers

    def capture_gate_snapshots(self, id=None):
        return [snapshot_mod.capture_worker_snapshot(self.workers[i]) for i in id]

    def restore_gate_snapshots(self, snapshots, id=None):
        return [
            snapshot_mod.restore_worker_snapshot(self.workers[i], snapshots[j])
            for j, i in enumerate(id)
        ]

    def step(self, actions, env_idx=None):
        indices = range(len(self.workers)) if env_idx is None else env_idx
        raw_obs = []
        for i in indices:
            self.workers[i].advance()
            raw_obs.append(self.workers[i]._get_observations())
        count = len(raw_obs)
        return (
            raw_obs,
            np.zeros(count),
            np.zeros(count, dtype=bool),
            [{} for _ in range(count)],
        )


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def _make_env(libero_env_mod, tmp_path, *, num_envs=2, max_episode_steps=10):
    bddl = tmp_path / "task.bddl"
    if not bddl.exists():
        bddl.write_text("task-v1")
    workers = [_FakeWorker(bddl, 10 * (i + 1)) for i in range(num_envs)]
    env = object.__new__(libero_env_mod.LiberoEnv)
    env.cfg = _Cfg(
        max_episode_steps=max_episode_steps,
        reward_coef=1.0,
        task_suite_name="libero_10",
    )
    env.num_envs = num_envs
    env.seed = 0
    env.is_eval = False
    env.auto_reset = False
    env.ignore_terminations = False
    env.use_rel_reward = False
    env.use_step_penalty = False
    env.episode_manifest = None
    env.test_episode_manifest = None
    env._gate_phase_spec = None
    env.task_ids = np.zeros(num_envs, dtype=np.int64)
    env.trial_ids = np.zeros(num_envs, dtype=np.int64)
    env.reset_state_ids = np.zeros(num_envs, dtype=np.int64)
    env.task_descriptions = ["pick up the mug"] * num_envs
    env._manifest_entries = [None] * num_envs
    env._generator = np.random.default_rng(1)
    env._generator_ordered = np.random.default_rng(0)
    env._manifest_cursor = 0
    env._manifest_order = np.asarray([], dtype=np.int64)
    env._last_manifest_batch = []
    env.start_idx = 0
    env._is_start = False
    env.prev_step_reward = np.zeros(num_envs)
    env._init_metrics()
    env._elapsed_steps = np.zeros(num_envs, dtype=np.int32)
    env._episode_generation = np.zeros(num_envs, dtype=np.int64)
    env.env = _FakeVectorEnv(workers)
    env.current_raw_obs = [worker._get_observations() for worker in workers]
    return env, workers


def _step(env, num_envs=None):
    count = env.num_envs if num_envs is None else num_envs
    return env.step(np.zeros((count, 7)))


def test_reset_from_snapshot_roundtrip_matches_restore_then_steps(
    tmp_path, monkeypatch
):
    libero_env_mod = _load_libero_env_with_stubs(monkeypatch)
    env, workers = _make_env(libero_env_mod, tmp_path)
    for _ in range(3):
        _step(env)
    env.returns[:] = 2.5
    env.success_once[:] = True
    env.prev_step_reward[:] = 0.5
    snapshot = env.capture_gate_snapshot()
    snapshot_uid = env._episode_uid(0)
    restore_obs = env.restore_gate_snapshot(copy.deepcopy(snapshot))

    for _ in range(2):
        _step(env)
    env.start_idx = 7
    env._manifest_cursor = 4
    env._task_success_stats = {3: {"success": 1, "count": 2}}
    random.seed(23)
    np.random.seed(24)
    torch.manual_seed(25)
    process_rng_before = snapshot_mod.capture_process_rng_state()
    generator_before = copy.deepcopy(env._generator.bit_generator.state)

    obs, infos = env.reset_from_snapshot(snapshot, horizon_mode="remaining")

    # Same observation content as the counterfactual restore path.
    assert infos == {}
    assert torch.equal(obs["main_images"], restore_obs["main_images"])
    assert torch.equal(obs["wrist_images"], restore_obs["wrist_images"])
    for state, expected in zip(obs["states"], restore_obs["states"]):
        assert torch.equal(state, expected)
    assert torch.equal(
        obs["gate_context"]["elapsed_steps"],
        restore_obs["gate_context"]["elapsed_steps"],
    )

    # Rollout-start bookkeeping: metrics zeroed, clock kept, new episode uid.
    assert np.array_equal(env._elapsed_steps, np.array([3, 3]))
    assert not env.success_once.any()
    assert not env.fail_once.any()
    assert np.array_equal(env.returns, np.zeros(2))
    assert np.array_equal(env.prev_step_reward, np.zeros(2))
    assert np.array_equal(env.success_episode_len, np.zeros(2))
    assert env._episode_uid(0) != snapshot_uid
    first_generation = env._episode_generation.copy()

    # Global scheduling/RNG state is preserved, not rewound to snapshot time.
    assert env.start_idx == 7
    assert env._manifest_cursor == 4
    assert env._task_success_stats == {3: {"success": 1, "count": 2}}
    assert _nested_equal(
        snapshot_mod.capture_process_rng_state(), process_rng_before
    )
    assert _nested_equal(env._generator.bit_generator.state, generator_before)

    # The env steps normally afterwards from the restored simulator state.
    obs, _reward, _term, trunc, step_infos = _step(env)
    assert np.array_equal(env._elapsed_steps, np.array([4, 4]))
    assert not trunc.any()
    assert workers[0].state[0] == pytest.approx(14.0)  # 10 + 3 steps + 1
    assert int(obs["main_images"][0][0, 0, 0]) == 14
    assert torch.equal(
        step_infos["episode"]["episode_len"], torch.tensor([4, 4], dtype=torch.int32)
    )

    # A second start from the same snapshot gets yet another episode identity.
    env.reset_from_snapshot(snapshot, horizon_mode="remaining")
    assert (env._episode_generation > first_generation).all()


def test_remaining_mode_ends_at_original_horizon(tmp_path, monkeypatch):
    libero_env_mod = _load_libero_env_with_stubs(monkeypatch)
    env, _workers = _make_env(libero_env_mod, tmp_path, max_episode_steps=5)
    for _ in range(3):
        _step(env)
    snapshot = env.capture_gate_snapshot()
    for _ in range(2):
        _step(env)

    env.reset_from_snapshot(snapshot, horizon_mode="remaining")
    assert np.array_equal(env._elapsed_steps, np.array([3, 3]))

    _obs, _reward, _term, trunc, _infos = _step(env)
    assert not trunc.any()
    _obs, _reward, _term, trunc, _infos = _step(env)
    assert trunc.all()  # restored at t = T-2 truncates after exactly 2 steps


def test_fresh_mode_zeroes_clock_and_grants_full_horizon(tmp_path, monkeypatch):
    libero_env_mod = _load_libero_env_with_stubs(monkeypatch)
    env, _workers = _make_env(libero_env_mod, tmp_path, max_episode_steps=6)
    for _ in range(3):
        _step(env)
    snapshot = env.capture_gate_snapshot()

    obs, _infos = env.reset_from_snapshot(snapshot, horizon_mode="fresh")
    assert np.array_equal(env._elapsed_steps, np.array([0, 0]))
    assert torch.equal(
        obs["gate_context"]["elapsed_steps"], torch.zeros(2, dtype=torch.int64)
    )
    assert torch.equal(
        obs["gate_context"]["decision_index"], torch.zeros(2, dtype=torch.int64)
    )

    for step_index in range(6):
        _obs, _reward, _term, trunc, _infos = _step(env)
        assert bool(trunc.all()) == (step_index == 5)


def test_reset_from_snapshot_fails_closed_without_mutation(tmp_path, monkeypatch):
    libero_env_mod = _load_libero_env_with_stubs(monkeypatch)
    env, workers = _make_env(libero_env_mod, tmp_path, max_episode_steps=5)
    for _ in range(3):
        _step(env)
    snapshot = env.capture_gate_snapshot()

    def _assert_untouched():
        assert np.array_equal(env._elapsed_steps, np.array([3, 3]))
        assert workers[0].state[0] == pytest.approx(13.0)
        assert workers[1].state[0] == pytest.approx(23.0)

    with pytest.raises(ValueError, match="horizon_mode"):
        env.reset_from_snapshot(snapshot, horizon_mode="forever")
    _assert_untouched()

    bad_schema = copy.deepcopy(snapshot)
    bad_schema["schema"] = "libero-gate-snapshot-v0"
    with pytest.raises(ValueError, match="schema"):
        env.reset_from_snapshot(bad_schema, horizon_mode="remaining")
    _assert_untouched()

    mixed = copy.deepcopy(snapshot)
    mixed["outer"]["per_env"]["_elapsed_steps"] = np.array([2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="one shared elapsed_steps"):
        env.reset_from_snapshot(mixed, horizon_mode="remaining")
    _assert_untouched()

    exhausted = copy.deepcopy(snapshot)
    exhausted["outer"]["per_env"]["_elapsed_steps"] = np.array(
        [5, 5], dtype=np.int32
    )
    with pytest.raises(ValueError, match="no remaining episode budget"):
        env.reset_from_snapshot(exhausted, horizon_mode="remaining")
    _assert_untouched()

    missing = copy.deepcopy(snapshot)
    del missing["outer"]["per_env"]["_elapsed_steps"]
    with pytest.raises(ValueError, match="_elapsed_steps"):
        env.reset_from_snapshot(missing, horizon_mode="remaining")
    _assert_untouched()


def test_subset_restart_requires_synchronized_vector_clocks(tmp_path, monkeypatch):
    libero_env_mod = _load_libero_env_with_stubs(monkeypatch)
    env, workers = _make_env(libero_env_mod, tmp_path, max_episode_steps=10)
    for _ in range(3):
        _step(env)
    subset_snapshot = env.capture_gate_snapshot(env_idx=np.array([0]))
    for _ in range(2):
        _step(env)

    # Untouched slot 1 sits at t=5 while slot 0 would restart at t=3: the gate
    # horizon contract cannot honor asynchronous per-slot clocks.
    with pytest.raises(ValueError, match="asynchronous episode clocks"):
        env.reset_from_snapshot(subset_snapshot, horizon_mode="remaining")
    assert np.array_equal(env._elapsed_steps, np.array([5, 5]))
    assert workers[0].state[0] == pytest.approx(15.0)

    # Fresh mode needs the untouched slots at clock zero; t=5 also fails.
    with pytest.raises(ValueError, match="asynchronous episode clocks"):
        env.reset_from_snapshot(subset_snapshot, horizon_mode="fresh")

    # Aligning the untouched slot with the target clock makes the same subset
    # restart legal, and only slot 0 is rewound.
    env._elapsed_steps[1] = 3
    env.reset_from_snapshot(subset_snapshot, horizon_mode="remaining")
    assert np.array_equal(env._elapsed_steps, np.array([3, 3]))
    assert workers[0].state[0] == pytest.approx(13.0)
    assert workers[1].state[0] == pytest.approx(25.0)  # slot 1 untouched
