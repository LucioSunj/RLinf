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

import ast
import json
import sys
from pathlib import Path

import pytest

RLINF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.libero.episode_manifest import (  # noqa: E402
    FrozenEpisode,
    FrozenEpisodeManifest,
    validate_manifest_disjoint,
)
from rlinf.utils.test_set_guard import (  # noqa: E402
    DISJOINT_AUDIT_SCHEMA,
    DISJOINT_AUDIT_VALIDATOR,
    FINAL_EVAL_ENV_VALUE,
    FINAL_EVAL_ENV_VAR,
    REQUIRED_AUDIT_DIMENSIONS,
    HeldOutSplitError,
    assert_disjoint_audit,
    assert_manifest_membership,
    assert_training_manifest,
    final_eval_unlocked,
)

COMMIT = "c" * 40


def _episode(index: int, *, suite: str = "libero_10") -> FrozenEpisode:
    return FrozenEpisode(
        episode_id=f"ep-{index}",
        base_task="pick_up_the_book",
        task_suite_name=suite,
        task_id=index,
        factor="texture",
        level="3",
        bddl_path=f"/plus/bddl/{index}.bddl",
        bddl_sha256="b" * 64,
        reset_state_id=index,
        trial_id=0,
        env_seed=1000 + index,
        perturbation_id=f"pert-{index}",
        asset_ids=(f"asset-{index}",),
    )


def _manifest(
    *,
    split: str,
    indices,
    file_sha256: str,
    sha256: str | None = None,
    commit: str = COMMIT,
) -> FrozenEpisodeManifest:
    return FrozenEpisodeManifest(
        path=f"/frozen/{split}.json",
        sha256=sha256 if sha256 is not None else file_sha256,
        file_sha256=file_sha256,
        libero_plus_root="/plus",
        libero_plus_commit=commit,
        split=split,
        episodes=tuple(_episode(i) for i in indices),
    )


def _train_manifest(**kwargs) -> FrozenEpisodeManifest:
    kwargs.setdefault("split", "train")
    kwargs.setdefault("indices", (1, 2))
    kwargs.setdefault("file_sha256", "a" * 64)
    return _manifest(**kwargs)


def _validation_manifest(**kwargs) -> FrozenEpisodeManifest:
    kwargs.setdefault("split", "validation")
    kwargs.setdefault("indices", (4, 5))
    kwargs.setdefault("file_sha256", "b" * 64)
    return _manifest(**kwargs)


def _test_manifest(**kwargs) -> FrozenEpisodeManifest:
    kwargs.setdefault("split", "test")
    kwargs.setdefault("indices", (7, 8))
    kwargs.setdefault("file_sha256", "d" * 64)
    return _manifest(**kwargs)


def _audit_manifest_entry(manifest) -> dict:
    return {
        "path": manifest.path,
        "file_sha256": manifest.file_sha256,
        "split": manifest.split,
        "episodes": len(manifest.episodes),
    }


def _audit_pair(primary: str, heldout: str, *, relabeled: bool = False) -> dict:
    return {
        "primary": primary,
        "heldout": heldout,
        "validator": DISJOINT_AUDIT_VALIDATOR,
        "heldout_relabeled_to_test_for_schema_guard": relabeled,
        "status": "disjoint",
        "detail": None,
    }


def _audit_record(train, heldout, *, validation=None, commit=None) -> dict:
    """Mirror ``scripts/stage2/split_plus_assets.py audit`` output exactly.

    The producer always audits three manifests (``dev_train``,
    ``dev_validation``, ``test``) and three pairs, with the DEV-internal
    train-vs-validation pair relabelled for the validator's schema guard.
    """
    if validation is None:
        validation = _validation_manifest(commit=train.libero_plus_commit or COMMIT)
    return {
        "schema": DISJOINT_AUDIT_SCHEMA,
        "generated_by": "scripts/stage2/split_plus_assets.py audit",
        "libero_plus_commit": commit
        if commit is not None
        else train.libero_plus_commit,
        "workspace_rlinf_module": (
            "/workspace/RLinf/rlinf/envs/libero/episode_manifest.py"
        ),
        "manifests": {
            "dev_train": _audit_manifest_entry(train),
            "dev_validation": _audit_manifest_entry(validation),
            "test": _audit_manifest_entry(heldout),
        },
        "pairs": [
            _audit_pair("dev_train", "test"),
            _audit_pair("dev_validation", "test"),
            _audit_pair("dev_train", "dev_validation", relabeled=True),
        ],
        "ok": True,
    }


def _write_audit(tmp_path: Path, record) -> Path:
    path = tmp_path / "disjoint_audit.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clear_final_eval_env(monkeypatch):
    """Every test starts with the final-evaluation lock closed."""
    monkeypatch.delenv(FINAL_EVAL_ENV_VAR, raising=False)


# --------------------------------------------------------------------------- #
# assert_training_manifest
# --------------------------------------------------------------------------- #
def test_training_splits_pass_without_any_unlock():
    for split in ("train", "validation"):
        manifest = _train_manifest(split=split)
        assert assert_training_manifest(manifest, context="unit") == split


def test_test_split_is_refused_on_a_training_entry_point():
    manifest = _test_manifest()
    with pytest.raises(HeldOutSplitError) as excinfo:
        assert_training_manifest(manifest, context="E5 GRPO periodic validation")

    message = str(excinfo.value)
    # The operator must learn which entry point refused, what is missing, and
    # what to do instead - not merely that something was denied.
    assert "E5 GRPO periodic validation" in message
    assert "allow_test_split" in message
    assert FINAL_EVAL_ENV_VAR in message
    assert "split=train/validation" in message


def test_only_the_call_site_flag_is_not_enough(monkeypatch):
    monkeypatch.delenv(FINAL_EVAL_ENV_VAR, raising=False)
    with pytest.raises(HeldOutSplitError, match=FINAL_EVAL_ENV_VAR):
        assert_training_manifest(
            _test_manifest(), context="unit", allow_test_split=True
        )


def test_only_the_environment_key_is_not_enough(monkeypatch):
    monkeypatch.setenv(FINAL_EVAL_ENV_VAR, FINAL_EVAL_ENV_VALUE)
    with pytest.raises(HeldOutSplitError, match="allow_test_split"):
        assert_training_manifest(
            _test_manifest(), context="unit", allow_test_split=False
        )


def test_both_keys_together_unlock_the_test_split(monkeypatch):
    monkeypatch.setenv(FINAL_EVAL_ENV_VAR, FINAL_EVAL_ENV_VALUE)
    assert (
        assert_training_manifest(
            _test_manifest(), context="E6 headline", allow_test_split=True
        )
        == "test"
    )
    assert final_eval_unlocked(allow_test_split=True) is True


@pytest.mark.parametrize("value", ["0", "true", "TRUE", "yes", "", " 1", "1 "])
def test_environment_key_must_be_exactly_the_documented_value(monkeypatch, value):
    monkeypatch.setenv(FINAL_EVAL_ENV_VAR, value)
    assert final_eval_unlocked(allow_test_split=True) is False
    with pytest.raises(HeldOutSplitError):
        assert_training_manifest(
            _test_manifest(), context="unit", allow_test_split=True
        )


@pytest.mark.parametrize("flag", [1, "1", "true", "yes", [1], {"a": 1}])
def test_truthy_non_true_flag_does_not_unlock(monkeypatch, flag):
    """A stray truthy value must not stand in for an explicit opt-in.

    ``bool(flag)`` is True for every value here, so a truthiness check would
    open the held-out split on a stringly-typed config value.
    """
    monkeypatch.setenv(FINAL_EVAL_ENV_VAR, FINAL_EVAL_ENV_VALUE)
    assert final_eval_unlocked(allow_test_split=flag) is False
    with pytest.raises(HeldOutSplitError):
        assert_training_manifest(
            _test_manifest(), context="unit", allow_test_split=flag
        )


def test_libero_plus_import_module_no_longer_trips_the_guard(monkeypatch):
    """Reverse lock for the 2026-07-26 semantics change.

    An earlier design asserted that LIBERO_PLUS_IMPORT_MODULE was unset during
    training. Since DEV perturbations are now sourced from LIBERO-Plus, training
    legitimately sets it, and reinstating that assertion would break every DEV
    rollout. This test fails loudly if someone restores the old rule from stale
    documentation.
    """
    monkeypatch.setenv("LIBERO_PLUS_IMPORT_MODULE", "liberoplus")
    monkeypatch.setenv("LIBERO_PLUS_ROOT", "/plus")
    monkeypatch.setenv("LIBERO_PLUS_COMMIT", COMMIT)
    for split in ("train", "validation"):
        assert (
            assert_training_manifest(_train_manifest(split=split), context="unit")
            == split
        )


def test_unknown_split_is_refused():
    manifest = _train_manifest(split="holdout")
    with pytest.raises(HeldOutSplitError, match="unknown manifest split"):
        assert_training_manifest(manifest, context="unit")


def test_accepts_a_mapping_or_a_manifest_file(tmp_path):
    assert assert_training_manifest({"split": "train"}, context="unit") == "train"

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"split": "validation"}), encoding="utf-8")
    assert assert_training_manifest(path, context="unit") == "validation"
    assert assert_training_manifest(str(path), context="unit") == "validation"


def test_missing_or_unreadable_split_is_refused(tmp_path):
    with pytest.raises(HeldOutSplitError, match="cannot determine"):
        assert_training_manifest(tmp_path / "absent.json", context="unit")

    empty = tmp_path / "no_split.json"
    empty.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    with pytest.raises(HeldOutSplitError, match="no explicit string 'split'"):
        assert_training_manifest(empty, context="unit")

    not_object = tmp_path / "list.json"
    not_object.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(HeldOutSplitError, match="must be an object"):
        assert_training_manifest(not_object, context="unit")


# --------------------------------------------------------------------------- #
# assert_disjoint_audit
# --------------------------------------------------------------------------- #
def test_valid_audit_is_accepted(tmp_path):
    train, heldout = _train_manifest(), _test_manifest()
    path = _write_audit(tmp_path, _audit_record(train, heldout))
    record = assert_disjoint_audit(train, path, context="unit")
    assert record["schema"] == DISJOINT_AUDIT_SCHEMA


def test_valid_audit_vouches_for_the_validation_manifest_too(tmp_path):
    """The DEV-validation half is a training split and holds the same audit."""
    in_hand = _validation_manifest()
    record = _audit_record(_train_manifest(), _test_manifest(), validation=in_hand)
    path = _write_audit(tmp_path, record)
    assert assert_disjoint_audit(in_hand, path, context="unit")["ok"] is True


def test_missing_audit_is_refused(tmp_path):
    with pytest.raises(HeldOutSplitError, match="cannot read the disjointness audit"):
        assert_disjoint_audit(
            _train_manifest(), tmp_path / "absent.json", context="unit"
        )


def test_audit_for_another_manifest_is_refused(tmp_path):
    audited, in_hand = _train_manifest(), _train_manifest(file_sha256="e" * 64)
    path = _write_audit(tmp_path, _audit_record(audited, _test_manifest()))
    with pytest.raises(HeldOutSplitError, match="written for a different"):
        assert_disjoint_audit(in_hand, path, context="unit")


def test_identity_is_keyed_on_file_sha256_not_the_logical_sha(tmp_path):
    """A suite partition shares its parent's logical ``sha256`` by design.

    ``FrozenEpisodeManifest.sha256`` is the parent manifest's hash for a
    per-suite partition, so keying identity on it would let an audit written for
    one partition vouch for a different one. ``file_sha256`` is the file's own
    bytes and is the only safe key.
    """
    shared_logical = "f" * 64
    audited = _train_manifest(file_sha256="1" * 64, sha256=shared_logical)
    sibling = _train_manifest(file_sha256="2" * 64, sha256=shared_logical)
    path = _write_audit(tmp_path, _audit_record(audited, _test_manifest()))

    assert audited.sha256 == sibling.sha256
    with pytest.raises(HeldOutSplitError, match="written for a different"):
        assert_disjoint_audit(sibling, path, context="unit")


def test_the_previously_invented_schema_is_refused(tmp_path):
    """The consumer once invented ``stage2-disjoint-audit-v1``; never again.

    The preregistered producer contract is ``dev-test-disjoint-audit-v1``
    (outer ``scripts/stage2/split_plus_assets.py``, ``docs/stage2/DEV_SET.md``).
    An audit carrying the invented name is by definition not a W6 product.
    """
    train, heldout = _train_manifest(), _test_manifest()
    for wrong in ("stage2-disjoint-audit-v1", "dev-test-disjoint-audit-v2", None):
        record = _audit_record(train, heldout)
        if wrong is None:
            del record["schema"]
        else:
            record["schema"] = wrong
        with pytest.raises(HeldOutSplitError, match="audit schema must be"):
            assert_disjoint_audit(
                train, _write_audit(tmp_path, record), context="unit"
            )


@pytest.mark.parametrize("ok", [False, None, 1, "true", "True"])
def test_audit_ok_must_be_literally_true(tmp_path, ok):
    """``ok`` is the producer's overall verdict; anything but True is a No."""
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["ok"] = ok
    with pytest.raises(HeldOutSplitError, match="ok"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_overlap_pair_is_refused(tmp_path):
    """A pair the producer marked OVERLAP can never vouch, even if ok is forged."""
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["pairs"][0]["status"] = "OVERLAP"
    record["pairs"][0]["detail"] = "asset overlap between train and test: asset-7"
    # The producer would have written ok:false; a forged ok:true must not help.
    record["ok"] = True
    with pytest.raises(HeldOutSplitError, match="status='OVERLAP'"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_overlap_on_the_dev_internal_pair_is_also_refused(tmp_path):
    """ok:true with ANY overlapping row is internally inconsistent - forged."""
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["pairs"][2]["status"] = "OVERLAP"
    record["pairs"][2]["detail"] = "env_seed overlap inside DEV"
    with pytest.raises(HeldOutSplitError, match="status='OVERLAP'"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_disjoint_row_with_non_null_detail_is_refused(tmp_path):
    """The producer writes detail=null exactly when a pair is disjoint."""
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["pairs"][0]["detail"] = "checked only asset_id"
    with pytest.raises(HeldOutSplitError, match="non-null"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_missing_in_hand_vs_test_pair_is_refused(tmp_path):
    """An audit that never compared this manifest against TEST proves nothing."""
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["pairs"] = [
        row for row in record["pairs"] if not (
            row["primary"] == "dev_train" and row["heldout"] == "test"
        )
    ]
    with pytest.raises(HeldOutSplitError, match="no pair covering"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_missing_or_empty_pairs_list_is_refused(tmp_path):
    train, heldout = _train_manifest(), _test_manifest()
    for mutate in (
        lambda record: record.pop("pairs"),
        lambda record: record.update(pairs=[]),
        lambda record: record.update(pairs={"not": "a list"}),
    ):
        record = _audit_record(train, heldout)
        mutate(record)
        with pytest.raises(HeldOutSplitError, match="'pairs'"):
            assert_disjoint_audit(
                train, _write_audit(tmp_path, record), context="unit"
            )


def test_pair_from_an_unknown_validator_is_refused(tmp_path):
    """Only ``validate_manifest_disjoint`` output is preregistered evidence."""
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["pairs"][0]["validator"] = "somebody.elses.checker"
    with pytest.raises(HeldOutSplitError, match="was not produced by"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_audit_without_a_test_split_entry_is_refused(tmp_path):
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["manifests"]["test"]["split"] = "validation"
    with pytest.raises(HeldOutSplitError, match="exactly one"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")

    record = _audit_record(train, heldout)
    del record["manifests"]["test"]
    with pytest.raises(HeldOutSplitError, match="exactly one"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_the_test_manifest_itself_cannot_borrow_the_audit(tmp_path):
    """Handing the held-out manifest to a training path must still fail."""
    heldout = _test_manifest()
    record = _audit_record(_train_manifest(), heldout)
    path = _write_audit(tmp_path, record)
    with pytest.raises(HeldOutSplitError, match="expected one of"):
        assert_disjoint_audit(heldout, path, context="unit")


def test_audit_entry_split_must_match_the_manifest_in_hand(tmp_path):
    """A relabelled audit entry cannot vouch for a differently split manifest."""
    train, heldout = _train_manifest(), _test_manifest()
    in_hand = _manifest(
        split="validation", indices=(1, 2), file_sha256=train.file_sha256
    )
    path = _write_audit(tmp_path, _audit_record(train, heldout))
    with pytest.raises(HeldOutSplitError, match="relabelled"):
        assert_disjoint_audit(in_hand, path, context="unit")


def test_ambiguous_duplicate_file_sha256_is_refused(tmp_path):
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout)
    record["manifests"]["dev_validation"]["file_sha256"] = train.file_sha256
    with pytest.raises(HeldOutSplitError, match="ambiguous"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


def test_audit_commit_must_match_the_manifest(tmp_path):
    train, heldout = _train_manifest(), _test_manifest()
    record = _audit_record(train, heldout, commit="9" * 40)
    with pytest.raises(HeldOutSplitError, match="libero_plus_commit"):
        assert_disjoint_audit(train, _write_audit(tmp_path, record), context="unit")


# --------------------------------------------------------------------------- #
# drift locks against the preregistered producer contract
# --------------------------------------------------------------------------- #
def test_audit_schema_is_the_preregistered_producer_schema():
    """Lock the schema constant to the W6 producer contract.

    The producer is the outer repo's ``scripts/stage2/split_plus_assets.py``
    (``AUDIT_SCHEMA``), preregistered in ``docs/stage2/DEV_SET.md`` with the
    explicit lane-D instruction that the consumer must not invent its own
    schema. This consumer once shipped ``stage2-disjoint-audit-v1``, which
    rejected every genuine W6 audit; this test pins the fix.
    """
    assert DISJOINT_AUDIT_SCHEMA == "dev-test-disjoint-audit-v1"


def test_audit_validator_constant_matches_the_real_function():
    """The ``validator`` string in every pair row must name the real function."""
    assert DISJOINT_AUDIT_VALIDATOR == (
        f"{validate_manifest_disjoint.__module__}"
        f".{validate_manifest_disjoint.__qualname__}"
    )


def test_required_audit_dimensions_match_validate_manifest_disjoint():
    """Lock the validator's dimension coverage to the preregistered set.

    The ``dev-test-disjoint-audit-v1`` payload does not enumerate dimensions;
    its coverage IS whatever ``validate_manifest_disjoint`` checks when the
    producer runs it. If a dimension is ever added to (or dropped from) the
    validator, this test fails, forcing an explicit review instead of
    previously written audits silently covering something other than what was
    preregistered.
    """
    train = _train_manifest(indices=(1, 2))
    heldout = _test_manifest(indices=(7, 8))
    reported = validate_manifest_disjoint(train, heldout)
    assert frozenset(reported) == REQUIRED_AUDIT_DIMENSIONS

    # Positive-content half: the validator genuinely raises on an overlap, so
    # a producer-run 'disjoint' verdict mirrors a real check, not a constant.
    colliding = _test_manifest(indices=(1, 8))  # shares episode identity 1
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest_disjoint(train, colliding)


# --------------------------------------------------------------------------- #
# LiberoEnv wiring
# --------------------------------------------------------------------------- #
def _body_containing(tree, predicate):
    """Return the statement list that DIRECTLY contains a matching statement."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list) and any(predicate(stmt) for stmt in body):
                return body
    return None


def _is_guard_statement(stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "assert_training_manifest"
    )


def _is_split_branch(stmt) -> bool:
    return isinstance(stmt, ast.If) and any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "split"
        for node in ast.walk(stmt.test)
    )


def test_libero_env_calls_the_guard_before_branching_on_split():
    """Structural lock on the LiberoEnv wiring.

    ``rlinf/envs/libero/libero_env.py`` imports ``gym`` and the LIBERO simulator,
    so ``LiberoEnv.__init__`` cannot be exercised in a CPU-only unit test. This
    parses the source instead.

    An earlier version of this test asserted only presence, keyword shape, and
    line ordering, and an adversarial review showed three mutations that fully
    revert the guard while keeping it green: flipping the ``cfg.get`` default to
    ``True`` (no config anywhere sets ``allow_test_split``, so the default IS the
    production value), wrapping the call in ``if not self.is_eval:`` (disabling
    it for exactly the eval envs it exists to protect), and passing a literal
    ``{"split": "train"}`` instead of the real manifest. Each assertion below
    exists to kill one of those.
    """
    source = (RLINF_ROOT / "rlinf" / "envs" / "libero" / "libero_env.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    imported = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "rlinf.utils.test_set_guard"
        and any(alias.name == "assert_training_manifest" for alias in node.names)
    ]
    assert imported, "libero_env.py must import assert_training_manifest"

    init = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "LiberoEnv"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_training_manifest"
    ]
    assert len(calls) == 1, "expected exactly one guard call in LiberoEnv.__init__"
    call = calls[0]

    # Kills mutation C: the guard must inspect the real manifest, not a literal.
    assert call.args, "the guard call must pass the manifest positionally"
    target = call.args[0]
    assert (
        isinstance(target, ast.Attribute)
        and target.attr == "episode_manifest"
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ), "the guard must be handed self.episode_manifest"

    # Kills mutation A: no config sets allow_test_split, so the cfg.get default
    # is the production value and must be closed.
    flag = next(
        (kw.value for kw in call.keywords if kw.arg == "allow_test_split"), None
    )
    assert flag is not None, "the guard call must pass allow_test_split"
    assert (
        isinstance(flag, ast.Call)
        and isinstance(flag.func, ast.Attribute)
        and flag.func.attr == "get"
        and len(flag.args) == 2
    ), "allow_test_split must be read from config via a two-argument cfg.get"
    default = flag.args[1]
    assert isinstance(default, ast.Constant) and default.value in (False, None), (
        f"the allow_test_split default must be closed, got {ast.dump(default)}"
    )

    # Kills mutation B: the guard must be a sibling of the split branching, not
    # nested inside a condition that can skip it.
    guard_body = _body_containing(init, _is_guard_statement)
    split_body = _body_containing(init, _is_split_branch)
    assert guard_body is not None, "guard call must be a standalone statement"
    assert split_body is not None, "expected LiberoEnv.__init__ to branch on split"
    assert guard_body is split_body, (
        "the guard must sit in the same statement list as the split branching, "
        "not nested inside a condition that could skip it"
    )
    guard_index = next(
        i for i, stmt in enumerate(guard_body) if _is_guard_statement(stmt)
    )
    split_index = next(i for i, stmt in enumerate(split_body) if _is_split_branch(stmt))
    assert guard_index < split_index, "the guard must run before the split branching"


# --------------------------------------------------------------------------- #
# audit hardening cases from the adversarial reviews
# --------------------------------------------------------------------------- #
def test_audit_missing_top_level_commit_is_refused(tmp_path):
    """An audit that does not pin the LIBERO-Plus checkout cannot vouch.

    The producer always writes a top-level ``libero_plus_commit`` (it is a
    required CLI argument), so its absence means the record is not a genuine
    W6 audit. The earlier consumer silently skipped the comparison; this pins
    the fail-closed behaviour.
    """
    train, heldout = _train_manifest(), _test_manifest()
    for missing in (None, ""):
        record = _audit_record(train, heldout)
        if missing is None:
            del record["libero_plus_commit"]
        else:
            record["libero_plus_commit"] = missing
        with pytest.raises(HeldOutSplitError, match="libero_plus_commit"):
            assert_disjoint_audit(
                train, _write_audit(tmp_path, record), context="unit"
            )


def test_manifest_without_a_commit_is_refused(tmp_path):
    """A manifest that cannot say which checkout froze it gets no audit pass.

    The earlier consumer skipped the commit comparison whenever the manifest
    exposed no ``libero_plus_commit``, which let an unpinnable manifest borrow
    any audit. Fail-closed instead: every ``FrozenEpisodeManifest`` carries the
    commit, so a missing one is not a legitimate caller.
    """
    heldout = _test_manifest()
    bare = _train_manifest(commit="")
    path = _write_audit(tmp_path, _audit_record(bare, heldout, commit=COMMIT))
    with pytest.raises(HeldOutSplitError, match="exposes no libero_plus_commit"):
        assert_disjoint_audit(bare, path, context="unit")


def test_partition_audit_with_parent_logical_sha_is_accepted(tmp_path):
    """Positive half of the file_sha256 identity choice.

    For a per-suite partition ``sha256`` is deliberately the PARENT manifest's
    hash while ``file_sha256`` is the partition file's own bytes. An audit
    written for the partition must be accepted even though the two hashes
    differ - an implementation that mixed the keys would refuse every
    legitimate partition audit, and only this test would catch it.
    """
    parent_logical = "f" * 64
    partition = _train_manifest(file_sha256="1" * 64, sha256=parent_logical)
    assert partition.sha256 != partition.file_sha256
    path = _write_audit(tmp_path, _audit_record(partition, _test_manifest()))
    record = assert_disjoint_audit(partition, path, context="unit")
    assert record["manifests"]["dev_train"]["file_sha256"] == "1" * 64


# --------------------------------------------------------------------------- #
# manifest membership (executed episodes vs declared manifest)
# --------------------------------------------------------------------------- #
def test_membership_accepts_entries_from_the_declared_manifest():
    manifest = _train_manifest(indices=(1, 2, 3))
    assert_manifest_membership(list(manifest.episodes), manifest, context="unit")


def test_membership_is_a_noop_without_a_manifest():
    assert_manifest_membership([_episode(9)], None, context="unit")


def test_membership_refuses_a_foreign_episode():
    manifest = _train_manifest(indices=(1, 2))
    foreign = _episode(7)  # id not present in the manifest
    with pytest.raises(HeldOutSplitError, match="not part of the declared"):
        assert_manifest_membership([foreign], manifest, context="unit")


def test_membership_refuses_a_forged_record_borrowing_a_legitimate_id():
    """Same episode_id, different frozen fields - a borrowed identity."""
    import dataclasses

    manifest = _train_manifest(indices=(1, 2))
    forged = dataclasses.replace(manifest.episodes[0], env_seed=999999)
    with pytest.raises(HeldOutSplitError, match="differs from the frozen record"):
        assert_manifest_membership([forged], manifest, context="unit")


def test_assign_manifest_entries_is_wired_to_membership_check():
    """Structural lock: the membership check guards the executed-episode path."""
    source = (RLINF_ROOT / "rlinf" / "envs" / "libero" / "libero_env.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assign = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "LiberoEnv"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_assign_manifest_entries"
    )
    calls = [
        node
        for node in ast.walk(assign)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_manifest_membership"
    ]
    assert len(calls) == 1, (
        "_assign_manifest_entries must call assert_manifest_membership"
    )
    # The check must be the FIRST statement so no entry is assigned before it.
    first = assign.body[0]
    while isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        first = assign.body[assign.body.index(first) + 1]  # skip docstring
    assert (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "assert_manifest_membership"
    ), "membership check must be the first statement of _assign_manifest_entries"
