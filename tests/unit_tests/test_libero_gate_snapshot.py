from __future__ import annotations

import importlib.util
import json
import random
import sys
import types
from dataclasses import replace
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


snapshot_mod = _load("libero_gate_snapshot_test", "rlinf/envs/libero/gate_snapshot.py")
manifest_mod = _load("libero_episode_manifest_test", "rlinf/envs/libero/episode_manifest.py")
phase_mod = _load("libero_gate_phase_test", "rlinf/envs/libero/gate_phase.py")


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


def _load_libero_env_with_stubs(monkeypatch):
    gym = types.ModuleType("gym")
    gym.Env = object
    monkeypatch.setitem(sys.modules, "gym", gym)

    utils = types.ModuleType("rlinf.envs.libero.utils")
    utils.build_interleaved_eval_reset_state_ids = lambda *args, **kwargs: None
    utils.distribute_reset_state_ids_round_robin = lambda *args, **kwargs: None
    utils.get_benchmark_overridden = lambda *_: None
    utils.get_libero_image = lambda obs: obs
    utils.get_libero_type = lambda: "standard"
    utils.get_libero_wrist_image = lambda obs: obs
    utils.quat2axisangle = lambda value: value
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
    env_utils.list_of_dict_to_dict_of_list = lambda value: value
    env_utils.to_tensor = lambda value: value
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
    return _load("libero_env_reset_test", "rlinf/envs/libero/libero_env.py")


class _Controller:
    def __init__(self):
        self.goal_pos = np.array([0.1, 0.2])
        self.goal_ori = np.eye(2)
        self.unrelated_object = object()


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


class _FakeLibero:
    def __init__(self, bddl):
        self.bddl_file_name = str(bddl)
        self.state = np.array([1.0, 2.0])
        self.timestep = 3
        self.robots = [_Robot()]
        self.sim = _Sim(self)
        self.np_random = np.random.default_rng(17)
        self.secondary_rng = np.random.RandomState(18)
        self.python_rng = random.Random(19)
        self.torch_rng = torch.Generator().manual_seed(20)

    def get_sim_state(self):
        return self.state.copy()

    def _get_observations(self):
        pixel = int(round(float(self.state[0])))
        return {
            "agentview_image": np.full((3, 4, 3), pixel, dtype=np.uint8),
            "robot0_eef_pos": self.state.astype(np.float64).copy(),
        }


def test_frozen_plus_reset_installs_manifest_reset_state(monkeypatch):
    libero_env_mod = _load_libero_env_with_stubs(monkeypatch)
    monkeypatch.setenv("LIBERO_TYPE", "plus")

    class TaskSuite:
        def get_task_init_states(self, task_id):
            assert task_id == 0
            return [np.array([10.0]), np.array([20.0])]

    class VectorEnv:
        def __init__(self):
            self.installed = None

        def reconfigure_env_fns(self, params, env_idx):
            assert len(params) == 1
            assert np.array_equal(env_idx, np.array([0]))

        def reset(self, *, id):
            assert np.array_equal(id, np.array([0]))

        def set_init_state(self, *, init_state, id):
            self.installed = (init_state, id.copy())

    env = object.__new__(libero_env_mod.LiberoEnv)
    env.cfg = {"libero_variant": "plus"}
    env.episode_manifest = object()
    env.task_ids = np.array([0])
    env.trial_ids = np.array([0])
    env.trial_id_bins = [2]
    env.cumsum_trial_id_bins = np.array([2])
    env.is_eval = True
    env.seed = 0
    env.task_suite = TaskSuite()
    env.env = VectorEnv()
    env.get_env_fn_params = lambda env_idx: [{"slot": int(env_idx[0])}]

    env._reconfigure(np.array([1]), np.array([0]))

    init_states, slots = env.env.installed
    assert np.array_equal(init_states[0], np.array([20.0]))
    assert np.array_equal(slots, np.array([0]))


def test_worker_snapshot_restores_pixels_controller_and_rng(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("task-v1")
    env = _FakeLibero(bddl)
    random.seed(8)
    np.random.seed(9)
    torch.manual_seed(10)
    snapshot = snapshot_mod.capture_worker_snapshot(env)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_env = float(env.np_random.random())
    expected_secondary = float(env.secondary_rng.random_sample())
    expected_python_env = env.python_rng.random()
    expected_torch_env = float(torch.rand((), generator=env.torch_rng))
    expected_torch = float(torch.rand(()))
    env.state[:] = 7
    env.timestep = 19
    env.robots[0].controller.goal_pos[:] = 9
    random.random()
    np.random.random()
    env.np_random.random()
    env.secondary_rng.random_sample()
    env.python_rng.random()
    torch.rand((), generator=env.torch_rng)
    torch.rand(())

    observation = snapshot_mod.restore_worker_snapshot(env, snapshot)
    assert np.array_equal(env.state, np.array([1.0, 2.0]))
    assert env.timestep == 3
    assert np.array_equal(env.robots[0].controller.goal_pos, np.array([0.1, 0.2]))
    assert np.array_equal(observation["agentview_image"], np.ones((3, 4, 3), dtype=np.uint8))
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert float(env.np_random.random()) == expected_env
    assert float(env.secondary_rng.random_sample()) == expected_secondary
    assert env.python_rng.random() == expected_python_env
    assert float(torch.rand((), generator=env.torch_rng)) == expected_torch_env
    assert float(torch.rand(())) == expected_torch


def test_worker_snapshot_capture_is_observational(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("task-v1")

    class MutatingObservationEnv(_FakeLibero):
        def _get_observations(self):
            self.timestep += 1
            random.random()
            np.random.random()
            torch.rand(())
            self.np_random.random()
            self.secondary_rng.random_sample()
            self.python_rng.random()
            torch.rand((), generator=self.torch_rng)
            return super()._get_observations()

    env = MutatingObservationEnv(bddl)
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    expected_process = snapshot_mod.capture_process_rng_state()
    expected_env_rng = snapshot_mod._capture_rng_state(env)
    snapshot_mod.capture_worker_snapshot(env)
    assert env.timestep == 3

    observed_process = snapshot_mod.capture_process_rng_state()
    assert _nested_equal(observed_process, expected_process)
    observed_env_rng = snapshot_mod._capture_rng_state(env)
    assert _nested_equal(observed_env_rng, expected_env_rng)


def test_worker_snapshot_restore_undoes_rerender_controller_mutation(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("task-v1")

    class MutatingObservationEnv(_FakeLibero):
        def _get_observations(self):
            self.robots[0].controller.goal_pos[:] += 1
            return super()._get_observations()

    env = MutatingObservationEnv(bddl)
    expected_goal = env.robots[0].controller.goal_pos.copy()
    snapshot = snapshot_mod.capture_worker_snapshot(env)
    env.robots[0].controller.goal_pos[:] = 9

    snapshot_mod.restore_worker_snapshot(env, snapshot)

    assert np.array_equal(env.robots[0].controller.goal_pos, expected_goal)


def test_worker_snapshot_rejects_changed_bddl(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("task-v1")
    env = _FakeLibero(bddl)
    snapshot = snapshot_mod.capture_worker_snapshot(env)
    bddl.write_text("task-v2")
    with pytest.raises(RuntimeError, match="BDDL identity"):
        snapshot_mod.restore_worker_snapshot(env, snapshot)


def test_pre_treatment_phase_callback_normalizes_and_fails_closed():
    observed = {}

    def callback(**context):
        observed.update(context)
        return {"phase": "contact", "phase_reliable": True}

    assert phase_mod.evaluate_gate_phase(callback, raw_observation={"x": 1}) == (
        "contact_alignment",
        True,
    )
    assert observed["raw_observation"] == {"x": 1}
    assert phase_mod.evaluate_gate_phase(lambda **_: ("bad", True)) == (
        "unknown",
        False,
    )
    assert phase_mod.evaluate_gate_phase(
        lambda **_: (_ for _ in ()).throw(RuntimeError("predicate failed"))
    ) == ("unknown", False)


def test_worker_phase_callback_is_fully_observational(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.libero.gate_snapshot", snapshot_mod
    )
    module = types.ModuleType("fake_gate_phase_callback")

    def predicate(*, env, raw_observation, elapsed_steps, **_):
        assert env.marker == "worker-env"
        assert raw_observation["contact"] == 1
        assert elapsed_steps == 20
        env.state[:] = 8
        env.timestep = 99
        env.robots[0].controller.goal_pos[:] = 7
        env.np_random.random()
        return "transport", True

    module.predicate = predicate
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class WorkerEnv(_FakeLibero):
        marker = "worker-env"

        def __init__(self, bddl):
            super().__init__(bddl)
            self.np_random = np.random.default_rng(41)

        def _get_observations(self):
            self.timestep += 1
            self.robots[0].controller.goal_pos[:] += 1
            self.np_random.random()
            return {**super()._get_observations(), "contact": 1}

    bddl = tmp_path / "task.bddl"
    bddl.write_text("task-v1")
    env = WorkerEnv(bddl)
    state_before = env.state.copy()
    timestep_before = env.timestep
    controller_before = env.robots[0].controller.goal_pos.copy()
    rng_before = snapshot_mod._capture_rng_state(env)
    assert phase_mod.evaluate_worker_gate_phase(
        env, f"{module.__name__}:predicate", {"elapsed_steps": 20}
    ) == ("transport_completion", True)
    assert np.array_equal(env.state, state_before)
    assert env.timestep == timestep_before
    assert np.array_equal(
        env.robots[0].controller.goal_pos, controller_before
    )
    assert _nested_equal(snapshot_mod._capture_rng_state(env), rng_before)


def _manifest_payload(bddl, sha):
    return {
        "schema": manifest_mod.EPISODE_MANIFEST_SCHEMA,
        "libero_plus_commit": "a" * 40,
        "split": "test",
        "episodes": [
            {
                "episode_id": "episode-0",
                "base_task": "pick_up_the_mug",
                "task_suite_name": "libero_10",
                "task_id": 0,
                "factor": "object-layout",
                "level": "L1",
                "bddl_path": bddl.name,
                "bddl_sha256": sha,
                "reset_state_id": 0,
                "trial_id": 0,
                "env_seed": 11,
                "perturbation_id": "layout-001",
                "asset_ids": ["mug-red", "table-a"],
                "allowed_extra_field": {"kept_by_canonical_freezer": True},
            }
        ],
    }


def test_frozen_episode_manifest_preserves_canonical_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    monkeypatch.delenv("LIBERO_PERTURBATION", raising=False)
    bddl = tmp_path / "episode.bddl"
    bddl.write_text("frozen")
    payload = _manifest_payload(bddl, manifest_mod.sha256_file(bddl))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    manifest = manifest_mod.load_frozen_episode_manifest(
        path,
        libero_plus_root=tmp_path,
        libero_plus_commit="a" * 40,
        verify_git=False,
        verify_import=False,
    )
    episode = manifest.episodes[0]
    assert episode.base_task == "pick_up_the_mug"
    assert episode.asset_ids == ("mug-red", "table-a")
    assert episode.bddl_path == str(bddl.resolve())


def test_suite_partition_keeps_logical_parent_sha_and_rejects_incomplete_subset(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    bddl = tmp_path / "episode.bddl"
    bddl.write_text("frozen")
    payload = _manifest_payload(bddl, manifest_mod.sha256_file(bddl))
    second = json.loads(json.dumps(payload["episodes"][0]))
    second.update(episode_id="episode-1", reset_state_id=1, trial_id=1, env_seed=12)
    payload["episodes"].append(second)
    payload["num_entries"] = 2
    parent_path = tmp_path / "plus_full.json"
    parent_path.write_text(json.dumps(payload))
    parent_sha = manifest_mod.sha256_file(parent_path)

    child_payload = {
        **payload,
        "num_entries": 2,
        "suite_partition": {
            "task_suite_name": "libero_10",
            "parent_manifest_path": parent_path.name,
            "parent_manifest_sha256": parent_sha,
            "parent_num_entries": 2,
        },
    }
    child_path = tmp_path / "libero_10.json"
    child_path.write_text(json.dumps(child_payload))
    kwargs = dict(
        libero_plus_root=tmp_path,
        libero_plus_commit="a" * 40,
        verify_git=False,
        verify_import=False,
    )
    manifest = manifest_mod.load_frozen_episode_manifest(child_path, **kwargs)
    assert manifest.sha256 == parent_sha
    assert manifest.file_sha256 == manifest_mod.sha256_file(child_path)
    assert manifest.file_sha256 != manifest.sha256
    assert manifest.parent_manifest_path == str(parent_path.resolve())
    assert manifest.task_suite_partition == "libero_10"

    child_payload["episodes"] = child_payload["episodes"][:1]
    child_payload["num_entries"] = 1
    child_path.write_text(json.dumps(child_payload))
    with pytest.raises(ValueError, match="complete ordered"):
        manifest_mod.load_frozen_episode_manifest(child_path, **kwargs)

    child_payload["episodes"] = payload["episodes"]
    child_payload["num_entries"] = 2
    child_path.write_text(json.dumps(child_payload))
    parent_path.write_text(json.dumps({**payload, "unexpected_mutation": True}))
    with pytest.raises(ValueError, match="parent SHA256 mismatch"):
        manifest_mod.load_frozen_episode_manifest(child_path, **kwargs)


def test_manifest_verifies_official_libero_import_origin(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    bddl = tmp_path / "episode.bddl"
    bddl.write_text("frozen")
    payload = _manifest_payload(bddl, manifest_mod.sha256_file(bddl))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    package_file = tmp_path / "libero" / "__init__.py"
    package_file.parent.mkdir()
    package_file.write_text("# frozen official package")
    fake = types.ModuleType("libero")
    fake.__file__ = str(package_file)
    monkeypatch.setitem(sys.modules, "libero", fake)
    kwargs = dict(
        libero_plus_root=tmp_path,
        libero_plus_commit="a" * 40,
        verify_git=False,
        verify_import=True,
    )
    manifest_mod.load_frozen_episode_manifest(path, **kwargs)

    outside = types.ModuleType("libero")
    outside.__file__ = str((tmp_path.parent / "wrong" / "__init__.py").resolve())
    monkeypatch.setitem(sys.modules, "libero", outside)
    with pytest.raises(ValueError, match="outside LIBERO_PLUS_ROOT"):
        manifest_mod.load_frozen_episode_manifest(path, **kwargs)


def test_frozen_episode_manifest_rejects_dynamic_all_and_bad_sha(tmp_path, monkeypatch):
    bddl = tmp_path / "episode.bddl"
    bddl.write_text("frozen")
    payload = _manifest_payload(bddl, "0" * 64)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        manifest_mod.load_frozen_episode_manifest(
            path,
            libero_plus_root=tmp_path,
            libero_plus_commit="a" * 40,
            verify_git=False,
            verify_import=False,
        )
    payload["episodes"][0]["bddl_sha256"] = manifest_mod.sha256_file(bddl)
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("LIBERO_SUFFIX", "all")
    with pytest.raises(ValueError, match="incompatible"):
        manifest_mod.load_frozen_episode_manifest(
            path,
            libero_plus_root=tmp_path,
            libero_plus_commit="a" * 40,
            verify_git=False,
            verify_import=False,
        )


def test_frozen_episode_manifest_requires_asset_identity(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    bddl = tmp_path / "episode.bddl"
    bddl.write_text("frozen")
    payload = _manifest_payload(bddl, manifest_mod.sha256_file(bddl))
    payload["episodes"][0]["asset_ids"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-empty"):
        manifest_mod.load_frozen_episode_manifest(
            path,
            libero_plus_root=tmp_path,
            libero_plus_commit="a" * 40,
            verify_git=False,
            verify_import=False,
        )


def test_train_test_manifest_disjoint_audit(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    train_bddl = tmp_path / "train.bddl"
    test_bddl = tmp_path / "test.bddl"
    train_bddl.write_text("train")
    test_bddl.write_text("test")
    train_payload = _manifest_payload(
        train_bddl, manifest_mod.sha256_file(train_bddl)
    )
    train_payload["split"] = "train"
    test_payload = _manifest_payload(test_bddl, manifest_mod.sha256_file(test_bddl))
    test_payload["split"] = "test"
    test_entry = test_payload["episodes"][0]
    test_entry.update(
        episode_id="episode-test",
        reset_state_id=1,
        trial_id=1,
        env_seed=22,
        perturbation_id="layout-999",
        asset_ids=["mug-blue"],
    )
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(json.dumps(train_payload))
    test_path.write_text(json.dumps(test_payload))
    kwargs = dict(
        libero_plus_root=tmp_path,
        libero_plus_commit="a" * 40,
        verify_git=False,
        verify_import=False,
    )
    train = manifest_mod.load_frozen_episode_manifest(train_path, **kwargs)
    heldout = manifest_mod.load_frozen_episode_manifest(test_path, **kwargs)
    assert manifest_mod.validate_manifest_disjoint(train, heldout)["env_seed"] == ()
    validation = replace(train, split="validation")
    assert manifest_mod.validate_manifest_disjoint(validation, heldout)["env_seed"] == ()
    shared_bddl_heldout = replace(
        heldout,
        episodes=(
            replace(
                heldout.episodes[0],
                bddl_path=train.episodes[0].bddl_path,
                bddl_sha256=train.episodes[0].bddl_sha256,
            ),
        ),
    )
    # Reusing an immutable base-task BDDL is legal; only the four frozen split
    # identities are required to be disjoint.
    manifest_mod.validate_manifest_disjoint(train, shared_bddl_heldout)
    overlapping = json.loads(json.dumps(test_payload))
    overlapping["episodes"][0]["env_seed"] = 11
    test_path.write_text(json.dumps(overlapping))
    heldout = manifest_mod.load_frozen_episode_manifest(test_path, **kwargs)
    with pytest.raises(ValueError, match="env_seed"):
        manifest_mod.validate_manifest_disjoint(train, heldout)


def test_disjoint_reset_identity_is_scoped_by_suite_and_task(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_SUFFIX", raising=False)
    train_bddl = tmp_path / "train.bddl"
    test_bddl = tmp_path / "test.bddl"
    train_bddl.write_text("train")
    test_bddl.write_text("test")
    train_payload = _manifest_payload(
        train_bddl, manifest_mod.sha256_file(train_bddl)
    )
    train_payload["split"] = "train"
    test_payload = _manifest_payload(test_bddl, manifest_mod.sha256_file(test_bddl))
    test_payload["split"] = "test"
    test_payload["episodes"][0].update(
        episode_id="episode-test",
        task_id=1,
        # reset 0 is a different identity in task 1 than task 0.
        reset_state_id=0,
        env_seed=22,
        perturbation_id="layout-999",
        asset_ids=["mug-blue"],
    )
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(json.dumps(train_payload))
    test_path.write_text(json.dumps(test_payload))
    kwargs = dict(
        libero_plus_root=tmp_path,
        libero_plus_commit="a" * 40,
        verify_git=False,
        verify_import=False,
    )
    train = manifest_mod.load_frozen_episode_manifest(train_path, **kwargs)
    heldout = manifest_mod.load_frozen_episode_manifest(test_path, **kwargs)
    assert manifest_mod.validate_manifest_disjoint(train, heldout)["reset_state"] == ()

    test_payload["episodes"][0]["task_id"] = 0
    test_path.write_text(json.dumps(test_payload))
    heldout = manifest_mod.load_frozen_episode_manifest(test_path, **kwargs)
    with pytest.raises(ValueError, match="reset_state"):
        manifest_mod.validate_manifest_disjoint(train, heldout)
