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

"""W14 wiring: offline ``examples/embodiment`` consumers of frozen manifests.

The W7 guard (``rlinf/utils/test_set_guard.py``) protected only the simulator
entry point; the offline tools under ``examples/embodiment`` stayed
split-blind.  These tests lock the W14 wiring:

- TRAINING-SIDE consumers (``collect_gate_paired_states.py``,
  ``merge_gate_paired_data.py``, ``smoke_libero_gate_snapshot.py``) must call
  ``assert_training_manifest`` immediately after loading the manifest - with no
  ``allow_test_split`` escape hatch - and must refuse to run without a
  committed ``dev-test-disjoint-audit-v1`` record (``--disjoint-audit``).
- The FINAL-EVAL consumer (``build_gate_mode_manifest.py``) routes through the
  two-key lock: an explicit ``--final`` flag supplies ``allow_test_split=True``
  and the operator must separately export ``STAGE2_FINAL_EVAL=1``.

Structural (AST) tests kill nesting/reordering mutations; behavioral tests run
each entry point's ``main()`` against fake manifests with the heavy simulator
and model modules stubbed out.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

RLINF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.libero.episode_manifest import FrozenEpisodeManifest  # noqa: E402
from rlinf.utils.test_set_guard import (  # noqa: E402
    DISJOINT_AUDIT_SCHEMA,
    DISJOINT_AUDIT_VALIDATOR,
    FINAL_EVAL_ENV_VALUE,
    FINAL_EVAL_ENV_VAR,
    HeldOutSplitError,
)

EXAMPLES_DIR = RLINF_ROOT / "examples" / "embodiment"
LAUNCHER_DIR = EXAMPLES_DIR / "adaptive_gate"

COMMIT = "c" * 40
TRAIN_SHA = "a" * 64
VAL_SHA = "b" * 64
TEST_SHA = "d" * 64

#: Wiring contract per training-side consumer: which variable holds the loaded
#: manifest, which variable is handed to the audit (the LOGICAL manifest - the
#: parent for a per-suite partition, because the committed audit is keyed on
#: ``file_sha256``), and the first expensive call the guards must precede.
TRAINING_SIDE_WIRING = {
    "collect_gate_paired_states.py": dict(
        manifest_var="manifest",
        audit_var="assignment_manifest",
        sink="_load_factory",
    ),
    "merge_gate_paired_data.py": dict(
        manifest_var="manifest",
        audit_var="manifest",
        sink="merge_paired_suite_datasets",
    ),
    "smoke_libero_gate_snapshot.py": dict(
        manifest_var="manifest",
        audit_var="audit_manifest",
        sink="build_libero_fastwam_driver",
    ),
}

FINAL_EVAL_SCRIPT = "build_gate_mode_manifest.py"


@pytest.fixture(autouse=True)
def _clear_final_eval_env(monkeypatch):
    """Every test starts with the final-evaluation lock closed."""
    monkeypatch.delenv(FINAL_EVAL_ENV_VAR, raising=False)


# --------------------------------------------------------------------------- #
# AST helpers (mirroring the wiring-test style in test_test_set_guard.py)
# --------------------------------------------------------------------------- #
def _script_source(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text(encoding="utf-8")


def _main_body(tree: ast.Module) -> list:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node.body
    raise AssertionError("script has no top-level main()")


def _calls_named(node, name: str) -> list:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
    ]


def _stmt_index(body, predicate, description: str) -> int:
    for index, stmt in enumerate(body):
        if predicate(stmt):
            return index
    raise AssertionError(f"missing statement: {description}")


def _is_assign_of(stmt, var: str, call_name: str) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == var
            for target in stmt.targets
        )
        and bool(_calls_named(stmt, call_name))
    )


def _guard_expr_call(stmt, func: str, first_arg: str):
    """Return the Call if ``stmt`` is a direct ``func(first_arg, ...)`` guard."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    if not (isinstance(call.func, ast.Name) and call.func.id == func):
        return None
    if not (
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == first_arg
    ):
        return None
    return call


def _imports_guard(tree: ast.Module, names: set[str]) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "rlinf.utils.test_set_guard"
        and names <= {alias.name for alias in node.names}
        for node in ast.walk(tree)
    )


# --------------------------------------------------------------------------- #
# structural wiring: training-side consumers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(TRAINING_SIDE_WIRING))
def test_training_side_guard_is_wired_on_the_manifest_load_path(name):
    """The guard + audit must be direct statements between load and use.

    Direct membership in ``main``'s statement list (not merely presence in the
    subtree) kills the mutation that wraps a guard in a skippable condition;
    the index ordering kills the mutation that moves it after the expensive
    driver/merge work has already started.
    """
    spec = TRAINING_SIDE_WIRING[name]
    source = _script_source(name)
    tree = ast.parse(source)
    assert _imports_guard(
        tree, {"assert_training_manifest", "assert_disjoint_audit"}
    ), f"{name} must import assert_training_manifest and assert_disjoint_audit"

    body = _main_body(tree)
    load_idx = _stmt_index(
        body,
        lambda stmt: _is_assign_of(
            stmt, spec["manifest_var"], "load_frozen_episode_manifest"
        ),
        f"{name}: {spec['manifest_var']} = load_frozen_episode_manifest(...)",
    )
    guard_idx = _stmt_index(
        body,
        lambda stmt: _guard_expr_call(
            stmt, "assert_training_manifest", spec["manifest_var"]
        )
        is not None,
        f"{name}: assert_training_manifest({spec['manifest_var']}, ...)",
    )
    audit_idx = _stmt_index(
        body,
        lambda stmt: _guard_expr_call(
            stmt, "assert_disjoint_audit", spec["audit_var"]
        )
        is not None,
        f"{name}: assert_disjoint_audit({spec['audit_var']}, ...)",
    )
    sink_idx = _stmt_index(
        body,
        lambda stmt: bool(_calls_named(stmt, spec["sink"])),
        f"{name}: call to {spec['sink']}",
    )
    assert load_idx < guard_idx < sink_idx, (
        f"{name}: the split guard must sit between the manifest load and "
        f"{spec['sink']} (got load={load_idx}, guard={guard_idx}, "
        f"sink={sink_idx})"
    )
    assert audit_idx < sink_idx, (
        f"{name}: the audit check must run before {spec['sink']}"
    )

    audit_call = _guard_expr_call(
        body[audit_idx], "assert_disjoint_audit", spec["audit_var"]
    )
    assert (
        len(audit_call.args) >= 2
        and isinstance(audit_call.args[1], ast.Attribute)
        and audit_call.args[1].attr == "disjoint_audit"
        and isinstance(audit_call.args[1].value, ast.Name)
        and audit_call.args[1].value.id == "args"
    ), f"{name}: the audit path must come from args.disjoint_audit"
    assert "--disjoint-audit" in source, (
        f"{name}: must expose a --disjoint-audit CLI argument"
    )


@pytest.mark.parametrize("name", sorted(TRAINING_SIDE_WIRING))
def test_training_side_consumers_have_no_test_split_escape_hatch(name):
    """No training-side guard call may pass allow_test_split at all.

    The two-key lock exists only for the final evaluation; a training-side
    consumer that forwards any value for ``allow_test_split`` has reintroduced
    the bypass this wiring exists to close.
    """
    tree = ast.parse(_script_source(name))
    calls = _calls_named(tree, "assert_training_manifest")
    assert calls, f"{name} must call assert_training_manifest"
    for call in calls:
        assert not any(
            keyword.arg == "allow_test_split" for keyword in call.keywords
        ), f"{name}: training-side guard calls must not pass allow_test_split"


# --------------------------------------------------------------------------- #
# structural wiring: final-eval consumer
# --------------------------------------------------------------------------- #
def test_final_eval_entry_wires_the_two_key_lock():
    source = _script_source(FINAL_EVAL_SCRIPT)
    tree = ast.parse(source)
    assert _imports_guard(tree, {"assert_training_manifest"}), (
        f"{FINAL_EVAL_SCRIPT} must import assert_training_manifest"
    )

    calls = _calls_named(tree, "assert_training_manifest")
    assert len(calls) == 1, "expected exactly one guard call"
    call = calls[0]
    assert (
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "episode_manifest"
    ), "the guard must be handed the loaded episode_manifest"
    flag = next(
        (kw.value for kw in call.keywords if kw.arg == "allow_test_split"), None
    )
    assert flag is not None, "the guard call must pass allow_test_split"
    assert (
        isinstance(flag, ast.Attribute)
        and flag.attr == "final"
        and isinstance(flag.value, ast.Name)
        and flag.value.id == "args"
    ), "allow_test_split must be exactly the --final CLI flag (args.final)"

    body = _main_body(tree)
    load_idx = _stmt_index(
        body,
        lambda stmt: _is_assign_of(
            stmt, "episode_manifest", "load_frozen_episode_manifest"
        ),
        "episode_manifest = load_frozen_episode_manifest(...)",
    )
    guard_idx = _stmt_index(
        body,
        lambda stmt: _guard_expr_call(
            stmt, "assert_training_manifest", "episode_manifest"
        )
        is not None,
        "assert_training_manifest(episode_manifest, ...)",
    )
    sink_idx = _stmt_index(
        body,
        lambda stmt: bool(_calls_named(stmt, "make_mode_schedule_manifest")),
        "schedule construction",
    )
    assert load_idx < guard_idx < sink_idx, (
        "the two-key lock must run after the load and before any schedule work"
    )

    # The --final flag is a plain store_true switch whose help documents the
    # environment half; the tool itself must never touch the environment.
    final_args = [
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "add_argument"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "--final"
            for arg in candidate.args
        )
    ]
    assert len(final_args) == 1, "expected exactly one --final argument"
    action = next(
        (kw.value for kw in final_args[0].keywords if kw.arg == "action"), None
    )
    assert isinstance(action, ast.Constant) and action.value == "store_true", (
        "--final must be a store_true flag"
    )
    assert FINAL_EVAL_ENV_VAR in source, (
        "--help must document the STAGE2_FINAL_EVAL operator half of the lock"
    )
    assert "os.environ" not in source, (
        "the final-eval tool must never set or read the env key itself"
    )


# --------------------------------------------------------------------------- #
# launcher threading
# --------------------------------------------------------------------------- #
def test_launchers_thread_the_new_guard_arguments():
    collect = (LAUNCHER_DIR / "run_e3_collect_paired_states.sh").read_text(
        encoding="utf-8"
    )
    assert collect.count('--disjoint-audit "${DEV_TEST_DISJOINT_AUDIT}"') == 2, (
        "collect launcher must thread the audit into BOTH the per-suite "
        "collection and the logical merge commands"
    )
    assert "require_env DEV_TEST_DISJOINT_AUDIT" in collect

    smoke = (LAUNCHER_DIR / "run_e3_snapshot_smoke.sh").read_text(encoding="utf-8")
    assert smoke.count('--disjoint-audit "${DEV_TEST_DISJOINT_AUDIT}"') == 1
    assert "require_env DEV_TEST_DISJOINT_AUDIT" in smoke

    e6 = (LAUNCHER_DIR / "run_e6_forced_and_random.sh").read_text(encoding="utf-8")
    assert re.search(r"build_gate_mode_manifest\.py\s+--final\b", e6), (
        "the E6 launcher must pass --final to build_gate_mode_manifest.py"
    )
    # A comment may DOCUMENT the operator's export; an actual export/assignment
    # statement would collapse the two keys into one and is forbidden.
    assert not re.search(rf"(?m)^\s*(export\s+)?{FINAL_EVAL_ENV_VAR}=", e6), (
        "the environment half of the lock must come from the operator, "
        "never from the launcher"
    )

    for name in (
        "run_e3_collect_paired_states.sh",
        "run_e3_snapshot_smoke.sh",
        "run_e6_forced_and_random.sh",
    ):
        subprocess.run(["bash", "-n", str(LAUNCHER_DIR / name)], check=True)


# --------------------------------------------------------------------------- #
# behavioral helpers
# --------------------------------------------------------------------------- #
def _load_script(name: str):
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(f"w14_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_fake_loader(monkeypatch):
    """Replace the frozen-manifest loader with a lightweight JSON reader.

    The scripts import ``load_frozen_episode_manifest`` inside ``main()``, so
    patching the source module attribute is picked up at call time.  The fake
    returns a REAL ``FrozenEpisodeManifest`` so the guard sees the production
    attribute surface (``split``, ``file_sha256``, ``libero_plus_commit``).
    """
    import rlinf.envs.libero.episode_manifest as manifest_mod

    def _fake_load(path, **kwargs):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return FrozenEpisodeManifest(
            path=str(path),
            sha256=payload.get("sha256", payload["file_sha256"]),
            file_sha256=payload["file_sha256"],
            libero_plus_root="/plus",
            libero_plus_commit=payload["libero_plus_commit"],
            split=payload["split"],
            episodes=(),
            parent_manifest_path=payload.get("parent_manifest_path"),
        )

    monkeypatch.setattr(manifest_mod, "load_frozen_episode_manifest", _fake_load)


def _write_manifest(tmp_path, name, *, split, file_sha256, parent=None):
    payload = {
        "split": split,
        "file_sha256": file_sha256,
        "libero_plus_commit": COMMIT,
    }
    if parent is not None:
        payload["parent_manifest_path"] = str(parent)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit_entry(sha: str, split: str) -> dict:
    return {
        "path": f"/frozen/{split}.json",
        "file_sha256": sha,
        "split": split,
        "episodes": 1,
    }


def _audit_pair(primary: str, heldout: str) -> dict:
    return {
        "primary": primary,
        "heldout": heldout,
        "validator": DISJOINT_AUDIT_VALIDATOR,
        "status": "disjoint",
        "detail": None,
    }


def _write_audit(tmp_path, *, train_sha=TRAIN_SHA):
    record = {
        "schema": DISJOINT_AUDIT_SCHEMA,
        "generated_by": "scripts/stage2/split_plus_assets.py audit",
        "libero_plus_commit": COMMIT,
        "manifests": {
            "dev_train": _audit_entry(train_sha, "train"),
            "dev_validation": _audit_entry(VAL_SHA, "validation"),
            "test": _audit_entry(TEST_SHA, "test"),
        },
        "pairs": [
            _audit_pair("dev_train", "test"),
            _audit_pair("dev_validation", "test"),
            _audit_pair("dev_train", "dev_validation"),
        ],
        "ok": True,
    }
    path = tmp_path / "disjoint_audit.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", list(argv))
    module.main()


# --------------------------------------------------------------------------- #
# behavioral: merge_gate_paired_data (training-side)
# --------------------------------------------------------------------------- #
def _stub_paired_data(monkeypatch):
    calls = []
    _stub_module(
        monkeypatch,
        "rlinf.models.embodiment.gate_policy.paired_data",
        merge_paired_suite_datasets=lambda *args, **kwargs: calls.append(args)
        or {"merged": True},
    )
    return calls


def test_merge_refuses_a_test_split_manifest(tmp_path, monkeypatch):
    _install_fake_loader(monkeypatch)
    calls = _stub_paired_data(monkeypatch)
    module = _load_script("merge_gate_paired_data.py")
    manifest = _write_manifest(
        tmp_path, "test.json", split="test", file_sha256=TEST_SHA
    )
    audit = _write_audit(tmp_path)
    argv = [
        "merge",
        "--episode-manifest", str(manifest),
        "--disjoint-audit", str(audit),
        "--out", str(tmp_path / "out"),
        "--suite-paired", "libero_10=/paired/libero_10",
    ]
    with pytest.raises(HeldOutSplitError, match="refusing a split="):
        _run_main(module, argv, monkeypatch)
    assert calls == [], "the merge sink must never see a split=test manifest"


def test_merge_refuses_a_training_manifest_without_an_audit(tmp_path, monkeypatch):
    """The W7 review's third clause: split=train with no committed audit."""
    _install_fake_loader(monkeypatch)
    calls = _stub_paired_data(monkeypatch)
    module = _load_script("merge_gate_paired_data.py")
    manifest = _write_manifest(
        tmp_path, "train.json", split="train", file_sha256=TRAIN_SHA
    )
    argv = [
        "merge",
        "--episode-manifest", str(manifest),
        "--out", str(tmp_path / "out"),
        "--suite-paired", "libero_10=/paired/libero_10",
    ]
    with pytest.raises(HeldOutSplitError, match="disjointness audit"):
        _run_main(module, argv, monkeypatch)
    assert calls == []


def test_merge_passes_with_train_split_and_committed_audit(tmp_path, monkeypatch):
    _install_fake_loader(monkeypatch)
    calls = _stub_paired_data(monkeypatch)
    module = _load_script("merge_gate_paired_data.py")
    manifest = _write_manifest(
        tmp_path, "train.json", split="train", file_sha256=TRAIN_SHA
    )
    audit = _write_audit(tmp_path)
    argv = [
        "merge",
        "--episode-manifest", str(manifest),
        "--disjoint-audit", str(audit),
        "--out", str(tmp_path / "out"),
        "--suite-paired", "libero_10=/paired/libero_10",
    ]
    _run_main(module, argv, monkeypatch)
    assert len(calls) == 1, "a fully audited train manifest must merge normally"
    assert calls[0][0].split == "train"


# --------------------------------------------------------------------------- #
# behavioral: collect_gate_paired_states (training-side)
# --------------------------------------------------------------------------- #
def _load_collect(monkeypatch):
    _stub_module(
        monkeypatch,
        "rlinf.models.embodiment.gate_policy.paired_collector",
        PairedStateCollector=object,
    )
    return _load_script("collect_gate_paired_states.py")


def test_collect_refuses_test_split_before_any_driver_work(tmp_path, monkeypatch):
    _install_fake_loader(monkeypatch)
    module = _load_collect(monkeypatch)
    factory_calls = []
    monkeypatch.setattr(
        module, "_load_factory", lambda spec: factory_calls.append(spec)
    )
    manifest = _write_manifest(
        tmp_path, "test.json", split="test", file_sha256=TEST_SHA
    )
    argv = [
        "collect",
        "--episode-manifest", str(manifest),
        "--disjoint-audit", str(_write_audit(tmp_path)),
        "--out", str(tmp_path / "paired"),
    ]
    with pytest.raises(HeldOutSplitError, match="collect_gate_paired_states"):
        _run_main(module, argv, monkeypatch)
    assert factory_calls == [], "no driver may be built for a split=test manifest"


def test_collect_refuses_a_training_manifest_without_an_audit(tmp_path, monkeypatch):
    _install_fake_loader(monkeypatch)
    module = _load_collect(monkeypatch)
    factory_calls = []
    monkeypatch.setattr(
        module, "_load_factory", lambda spec: factory_calls.append(spec)
    )
    manifest = _write_manifest(
        tmp_path, "train.json", split="train", file_sha256=TRAIN_SHA
    )
    argv = [
        "collect",
        "--episode-manifest", str(manifest),
        "--out", str(tmp_path / "paired"),
    ]
    with pytest.raises(HeldOutSplitError, match="disjointness audit"):
        _run_main(module, argv, monkeypatch)
    assert factory_calls == []


def test_collect_train_split_with_audit_passes_the_guards(tmp_path, monkeypatch):
    """An audited train manifest reaches the driver-contract check.

    The stub factory returns an object with none of the required driver
    methods, so ``main`` raising the driver-incomplete ``TypeError`` proves the
    guards passed and control moved on to real work.
    """
    _install_fake_loader(monkeypatch)
    module = _load_collect(monkeypatch)
    monkeypatch.setattr(
        module, "_load_factory", lambda spec: (lambda **kwargs: object())
    )
    manifest = _write_manifest(
        tmp_path, "train.json", split="train", file_sha256=TRAIN_SHA
    )
    argv = [
        "collect",
        "--episode-manifest", str(manifest),
        "--disjoint-audit", str(_write_audit(tmp_path)),
        "--out", str(tmp_path / "paired"),
    ]
    with pytest.raises(TypeError, match="incomplete"):
        _run_main(module, argv, monkeypatch)


# --------------------------------------------------------------------------- #
# behavioral: smoke_libero_gate_snapshot (training-side)
# --------------------------------------------------------------------------- #
class _UntouchableDriver:
    """Any real use of the driver marks progression past the guards."""

    def close(self):  # reached via ``finally`` - must not mask the real error
        pass

    def __getattr__(self, name):
        raise RuntimeError(f"driver.{name} must not be reached")


def _load_smoke(monkeypatch):
    driver_calls = []
    _stub_module(
        monkeypatch,
        "rlinf.models.embodiment.gate_policy.libero_paired_driver",
        build_libero_fastwam_driver=lambda **kwargs: driver_calls.append(kwargs)
        or _UntouchableDriver(),
    )
    return _load_script("smoke_libero_gate_snapshot.py"), driver_calls


def test_smoke_refuses_test_split_before_driver_construction(tmp_path, monkeypatch):
    _install_fake_loader(monkeypatch)
    module, driver_calls = _load_smoke(monkeypatch)
    manifest = _write_manifest(
        tmp_path, "test.json", split="test", file_sha256=TEST_SHA
    )
    argv = [
        "smoke",
        "--episode-manifest", str(manifest),
        "--disjoint-audit", str(_write_audit(tmp_path)),
        "--progress-fn", "pkg.mod:fn",
    ]
    with pytest.raises(HeldOutSplitError, match="smoke_libero_gate_snapshot"):
        _run_main(module, argv, monkeypatch)
    assert driver_calls == []


def test_smoke_partition_audit_binds_the_logical_parent(tmp_path, monkeypatch):
    """A per-suite partition is vouched for by its parent's committed audit.

    The partition file's own ``file_sha256`` never appears in the committed
    audit (partitions are materialized per run), so the wiring must audit the
    logical parent.  Reaching the driver (which then trips on the empty
    episode list) proves both guards accepted the partition + parent pair.
    """
    _install_fake_loader(monkeypatch)
    module, driver_calls = _load_smoke(monkeypatch)
    parent = _write_manifest(
        tmp_path, "parent.json", split="train", file_sha256=TRAIN_SHA
    )
    partition = _write_manifest(
        tmp_path,
        "partition.json",
        split="train",
        file_sha256="9" * 64,
        parent=parent,
    )
    argv = [
        "smoke",
        "--episode-manifest", str(partition),
        "--disjoint-audit", str(_write_audit(tmp_path)),
        "--progress-fn", "pkg.mod:fn",
    ]
    with pytest.raises(IndexError):
        _run_main(module, argv, monkeypatch)
    assert len(driver_calls) == 1, "the guards must accept a parent-audited partition"


# --------------------------------------------------------------------------- #
# behavioral: build_gate_mode_manifest (final-eval, two-key lock)
# --------------------------------------------------------------------------- #
def _stub_mode_selectors(monkeypatch):
    calls = []

    class _Selector:
        def schedule_for(self, *args, **kwargs):
            return []

    _stub_module(
        monkeypatch,
        "rlinf.models.embodiment.gate_policy.mode_selectors",
        REFERENCE_MATCH_METHODS=frozenset(),
        build_eval_mode_selector=lambda cfg: _Selector(),
        load_canonical_reference_trace=lambda path: [],
        make_mode_schedule_manifest=lambda **kwargs: calls.append(("schedule", kwargs))
        or {"schema": "stub"},
        make_reference_matched_mode_manifest=lambda **kwargs: calls.append(
            ("reference", kwargs)
        )
        or {"schema": "stub"},
        sha256_file=lambda path: "0" * 64,
        write_json_atomic=lambda out, payload: calls.append(("write", str(out))),
    )
    return calls


def _build_argv(tmp_path, manifest, *, final: bool):
    checkpoint = tmp_path / "gate.pt"
    checkpoint.write_bytes(b"stub-checkpoint")
    argv = [
        "build",
        "--episode-manifest", str(manifest),
        "--checkpoint", str(checkpoint),
        "--out", str(tmp_path / "mode_manifest.json"),
        "--kind", "forced",
        "--mode", "0",
    ]
    if final:
        argv.append("--final")
    return argv


def test_build_with_only_the_final_flag_is_refused(tmp_path, monkeypatch):
    _install_fake_loader(monkeypatch)
    calls = _stub_mode_selectors(monkeypatch)
    module = _load_script(FINAL_EVAL_SCRIPT)
    manifest = _write_manifest(
        tmp_path, "test.json", split="test", file_sha256=TEST_SHA
    )
    with pytest.raises(HeldOutSplitError, match=FINAL_EVAL_ENV_VAR):
        _run_main(module, _build_argv(tmp_path, manifest, final=True), monkeypatch)
    assert calls == []


def test_build_with_only_the_environment_key_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(FINAL_EVAL_ENV_VAR, FINAL_EVAL_ENV_VALUE)
    _install_fake_loader(monkeypatch)
    calls = _stub_mode_selectors(monkeypatch)
    module = _load_script(FINAL_EVAL_SCRIPT)
    manifest = _write_manifest(
        tmp_path, "test.json", split="test", file_sha256=TEST_SHA
    )
    with pytest.raises(HeldOutSplitError, match="allow_test_split"):
        _run_main(module, _build_argv(tmp_path, manifest, final=False), monkeypatch)
    assert calls == []


def test_build_with_both_keys_unlocks_the_test_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv(FINAL_EVAL_ENV_VAR, FINAL_EVAL_ENV_VALUE)
    _install_fake_loader(monkeypatch)
    calls = _stub_mode_selectors(monkeypatch)
    module = _load_script(FINAL_EVAL_SCRIPT)
    manifest = _write_manifest(
        tmp_path, "test.json", split="test", file_sha256=TEST_SHA
    )
    _run_main(module, _build_argv(tmp_path, manifest, final=True), monkeypatch)
    kinds = [kind for kind, _ in calls]
    assert "schedule" in kinds and "write" in kinds, (
        "both keys together must let the final-eval schedule build complete"
    )


def test_build_validation_manifest_needs_no_keys(tmp_path, monkeypatch):
    """DEV schedule building keeps working with neither key present."""
    _install_fake_loader(monkeypatch)
    calls = _stub_mode_selectors(monkeypatch)
    module = _load_script(FINAL_EVAL_SCRIPT)
    manifest = _write_manifest(
        tmp_path, "validation.json", split="validation", file_sha256=VAL_SHA
    )
    _run_main(module, _build_argv(tmp_path, manifest, final=False), monkeypatch)
    assert [kind for kind, _ in calls].count("schedule") == 1
