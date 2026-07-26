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

"""Fail-closed guard keeping frozen ``split=test`` episodes out of training.

The boundary this module enforces is the manifest ``split`` field, NOT the
presence of LIBERO-Plus.  Stage-2 sources its DEV perturbations from LIBERO-Plus
and splits them by asset instance, so LIBERO-Plus appearing in a training path is
expected and correct; ``LIBERO_PLUS_IMPORT_MODULE`` is legitimately set during
training and is therefore useless as a tripwire.

``LiberoEnv`` already refuses a ``split=train`` manifest on eval envs, refuses a
``split=validation`` manifest without a held-out audit, and runs
``validate_manifest_disjoint`` inline for both.  The one case it cannot decide on
its own is ``split=test``: the headline E6 evaluation and an E5 run's periodic
in-training validation are both ``is_eval=true`` envs with ``group_size=1``.
Telling them apart needs an explicit intent signal, which is what
:func:`assert_training_manifest` requires.

Known limitations
-----------------
This module raises the cost of reaching the held-out split; it does not make it
impossible.  Two gaps are known and deliberately left to follow-up work rather
than papered over here:

1. The two unlock keys are **not** independent in this stack.  A cluster config
   can inject environment variables into the worker process
   (``NodeGroup.get_node_env_vars``, ``rlinf/scheduler/cluster/node.py:208``),
   so one Hydra override list can supply both halves.  A genuinely independent
   second key would have to be bound to a preregistered evidence artifact whose
   hash is recorded in the run manifest.
2. The guard sees only one env sub-config and never learns whether it is
   protecting ``env.train`` or ``env.eval``.  Deciding that needs a whole-run
   view, i.e. validation in ``rlinf/config.py`` where ``runner.only_eval`` and
   the presence of ``env.train.episode_manifest_path`` are both visible.

A third gap - ``LiberoEnv.reset`` accepting caller-supplied ``manifest_entries``
that never belonged to the declared manifest - is closed by
:func:`assert_manifest_membership`, wired into ``_assign_manifest_entries``.

:func:`assert_disjoint_audit` currently has **no production caller**.  It defines
the consumer side of the committed-audit contract that the stage-2 asset-split
tool must satisfy; the offline manifest consumers under ``examples/`` are still
split-blind and wiring them is separate work.
"""

import json
import os
from pathlib import Path
from typing import Any, Mapping

# The module name matches pytest's default `python_files = test_*.py` glob, so
# tell pytest this is production code and not a test module.
__test__ = False

#: Schema of the committed disjointness audit consumed by
#: :func:`assert_disjoint_audit`.  This is the PREREGISTERED producer contract
#: emitted by the outer repo's ``scripts/stage2/split_plus_assets.py audit``
#: subcommand and documented in ``docs/stage2/DEV_SET.md``; the lane-D work
#: order explicitly forbids inventing a different schema here.  Locked by a
#: drift test in ``tests/unit_tests/test_test_set_guard.py``.
DISJOINT_AUDIT_SCHEMA = "dev-test-disjoint-audit-v1"

#: Fully qualified name of the validator the audit producer runs for every
#: manifest pair, recorded verbatim in each ``pairs`` row.  An audit whose rows
#: were produced by anything else is not evidence of disjointness on the
#: preregistered dimensions.  Locked to the real function by a drift test.
DISJOINT_AUDIT_VALIDATOR = (
    "rlinf.envs.libero.episode_manifest.validate_manifest_disjoint"
)

#: Environment half of the two-key final-evaluation lock.
FINAL_EVAL_ENV_VAR = "STAGE2_FINAL_EVAL"

#: The only value of :data:`FINAL_EVAL_ENV_VAR` that unlocks the test split.
FINAL_EVAL_ENV_VALUE = "1"

#: Splits that may drive training, reward, or model-selection paths.
TRAINING_SPLITS = frozenset({"train", "validation"})

#: The held-out split that may only be executed by a final evaluation.
TEST_SPLIT = "test"

#: Every split ``load_frozen_episode_manifest`` is allowed to return.
KNOWN_SPLITS = frozenset(TRAINING_SPLITS | {TEST_SPLIT})

#: Preregistered dimension coverage of ``validate_manifest_disjoint``.  The
#: ``dev-test-disjoint-audit-v1`` payload does not enumerate dimensions - its
#: coverage comes from the producer invoking the validator itself - so this
#: constant locks the VALIDATOR's coverage via a drift test in
#: ``tests/unit_tests/test_test_set_guard.py``.  If the validator's coverage
#: ever changes, that test fails, forcing an explicit schema/coverage review
#: instead of previously written audits silently becoming weaker (or stronger)
#: than what was preregistered.
REQUIRED_AUDIT_DIMENSIONS = frozenset(
    {"env_seed", "reset_state", "perturbation_id", "asset_id"}
)


class HeldOutSplitError(RuntimeError):
    """Raised when frozen ``split=test`` data reaches a non-final-eval path."""


def _split_of(manifest_or_path: Any, *, context: str) -> str:
    """Return the ``split`` of a manifest object, mapping, or manifest file.

    A path is read as plain JSON rather than through
    ``load_frozen_episode_manifest`` on purpose: the guard must stay usable in
    offline tooling that has no LIBERO-Plus checkout, and reading one string
    cannot be weaker than the full loader because the loader is still the thing
    that decides whether the manifest is usable at all.

    Path-likes are matched before any attribute duck-typing because ``str``
    itself exposes a ``.split`` method, which would otherwise be mistaken for a
    manifest split field.
    """
    if isinstance(manifest_or_path, (str, os.PathLike)):
        try:
            payload = json.loads(Path(manifest_or_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HeldOutSplitError(
                f"{context}: cannot determine the manifest split. Pass a "
                "FrozenEpisodeManifest, a mapping with a 'split' key, or a "
                f"readable manifest JSON path; got {manifest_or_path!r}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise HeldOutSplitError(
                f"{context}: manifest JSON must be an object, got "
                f"{type(payload).__name__}"
            )
        split = payload.get("split")
    elif isinstance(manifest_or_path, Mapping):
        split = manifest_or_path.get("split")
    else:
        split = getattr(manifest_or_path, "split", None)
    if not isinstance(split, str) or not split:
        raise HeldOutSplitError(
            f"{context}: manifest has no explicit string 'split'; refusing to "
            "guess whether this is held-out test data"
        )
    return split


def final_eval_unlocked(*, allow_test_split: bool) -> bool:
    """Report whether both halves of the final-evaluation lock are present.

    Requiring two keys means a single copy-pasted config flag does not open the
    held-out split on its own.  It is deliberately NOT a claim of independence:
    a cluster config can inject environment variables into the worker process
    (``NodeGroup.get_node_env_vars``), so both halves are reachable from one
    Hydra override list.  See the module docstring's known limitations.

    ``allow_test_split`` is compared against literal ``True`` rather than
    evaluated for truthiness, so a stringly-typed config value such as
    ``"false"`` fails closed instead of being coerced to ``True``.

    Args:
        allow_test_split: The call-site half, set explicitly by a final
            evaluation entry point.

    Returns:
        ``True`` only when ``allow_test_split`` is literally ``True`` and the
        environment variable half is exactly :data:`FINAL_EVAL_ENV_VALUE`.
    """
    if allow_test_split is not True:
        return False
    return os.environ.get(FINAL_EVAL_ENV_VAR) == FINAL_EVAL_ENV_VALUE


def assert_training_manifest(
    manifest_or_path: Any,
    *,
    context: str,
    allow_test_split: bool = False,
) -> str:
    """Refuse held-out ``split=test`` data outside a final evaluation.

    Args:
        manifest_or_path: A ``FrozenEpisodeManifest``, a mapping carrying a
            ``split`` key, or a path to a frozen manifest JSON file.
        context: Human-readable description of the call site, echoed in the
            error so an operator can tell which entry point refused.
        allow_test_split: The call-site half of the final-evaluation lock.
            Leave it ``False`` on every training, reward, early-stopping, and
            model-selection path.

    Returns:
        The validated split string.

    Raises:
        HeldOutSplitError: If the split is unknown, or if it is ``test`` and
            the two-key final-evaluation lock is not fully satisfied.
    """
    split = _split_of(manifest_or_path, context=context)
    if split not in KNOWN_SPLITS:
        raise HeldOutSplitError(
            f"{context}: unknown manifest split {split!r}; expected one of "
            f"{sorted(KNOWN_SPLITS)}"
        )
    if split in TRAINING_SPLITS:
        return split

    if final_eval_unlocked(allow_test_split=allow_test_split):
        return split

    missing = []
    if allow_test_split is not True:
        missing.append("the call-site flag allow_test_split=true")
    if os.environ.get(FINAL_EVAL_ENV_VAR) != FINAL_EVAL_ENV_VALUE:
        actual = os.environ.get(FINAL_EVAL_ENV_VAR)
        missing.append(
            f"{FINAL_EVAL_ENV_VAR}={FINAL_EVAL_ENV_VALUE} (currently {actual!r})"
        )
    raise HeldOutSplitError(
        f"{context}: refusing a split={TEST_SPLIT!r} manifest. The held-out "
        "half may only be executed by a final evaluation, which must supply "
        f"BOTH keys; missing {' and '.join(missing)}. If this is a training, "
        "reward, early-stopping, or model-selection run, point it at the "
        "split=train/validation manifest instead - reading the held-out half "
        "here invalidates the headline result."
    )


def _audit_entry_text(
    entry: Mapping[str, Any], key: str, *, context: str, name: str
) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise HeldOutSplitError(
            f"{context}: disjointness audit manifests[{name!r}].{key} must be "
            "a non-empty string"
        )
    return value


def assert_disjoint_audit(
    train_manifest: Any,
    audit_path: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate a committed ``dev-test-disjoint-audit-v1`` record.

    ``LiberoEnv`` recomputes disjointness inline whenever it drives the
    simulator, so this function exists for the paths that do not: offline label,
    uplift, and reporting tools that consume a frozen manifest without ever
    constructing an environment.  It verifies that the audit really pertains to
    *this* manifest and that the producer found it disjoint from the frozen
    held-out manifest, so a stale or mismatched record cannot be passed off as
    evidence.

    The payload shape is the producer's, verbatim (outer repo,
    ``scripts/stage2/split_plus_assets.py audit``): top-level ``schema``,
    ``libero_plus_commit``, ``manifests`` (name -> ``{path, file_sha256, split,
    episodes}``), ``pairs`` (rows with ``primary``/``heldout`` manifest names,
    ``validator``, ``status`` of ``"disjoint"`` or ``"OVERLAP"``, and ``detail``
    that is ``null`` exactly when the pair is disjoint), and ``ok`` which the
    producer sets to true only when NO pair overlapped.

    Identity is keyed on ``file_sha256`` rather than ``sha256``: for a per-suite
    partition ``sha256`` is deliberately the *parent* manifest's hash, so it
    cannot distinguish a Plus-Full suite partition from Plus-Full itself.

    Args:
        train_manifest: The ``split=train``/``validation`` manifest being used.
        audit_path: Path to the committed audit JSON.
        context: Human-readable description of the call site.

    Returns:
        The parsed audit record.

    Raises:
        HeldOutSplitError: On any schema, identity, commit, or overlap failure.
    """
    try:
        record = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise HeldOutSplitError(
            f"{context}: cannot read the disjointness audit at {audit_path!r}. "
            "A training manifest without a committed audit is refused."
        ) from exc
    if not isinstance(record, Mapping):
        raise HeldOutSplitError(
            f"{context}: disjointness audit must be a JSON object, got "
            f"{type(record).__name__}"
        )
    if record.get("schema") != DISJOINT_AUDIT_SCHEMA:
        raise HeldOutSplitError(
            f"{context}: disjointness audit schema must be "
            f"{DISJOINT_AUDIT_SCHEMA!r}, got {record.get('schema')!r}"
        )
    # Literal True, not truthiness: the producer writes ``ok`` only after every
    # validator pair came back clean, and a stringly "true" (or 1) is a forged
    # or hand-edited record, not evidence.
    if record.get("ok") is not True:
        raise HeldOutSplitError(
            f"{context}: disjointness audit records ok={record.get('ok')!r}; "
            "only an audit whose producer found every pair disjoint (ok: true) "
            "can vouch for a training manifest"
        )

    manifests = record.get("manifests")
    if not isinstance(manifests, Mapping) or not manifests:
        raise HeldOutSplitError(
            f"{context}: disjointness audit is missing the 'manifests' object"
        )

    actual_sha = getattr(train_manifest, "file_sha256", None)
    if not isinstance(actual_sha, str) or not actual_sha:
        raise HeldOutSplitError(
            f"{context}: cannot verify the disjointness audit because the "
            "manifest in hand exposes no file_sha256"
        )

    matches: list[str] = []
    test_names: list[str] = []
    for name, entry in manifests.items():
        if not isinstance(entry, Mapping):
            raise HeldOutSplitError(
                f"{context}: disjointness audit manifests[{name!r}] must be "
                "an object"
            )
        entry_split = _audit_entry_text(entry, "split", context=context, name=name)
        entry_sha = _audit_entry_text(entry, "file_sha256", context=context, name=name)
        if entry_sha == actual_sha:
            matches.append(name)
        if entry_split == TEST_SPLIT:
            test_names.append(name)

    if not matches:
        raise HeldOutSplitError(
            f"{context}: disjointness audit was written for a different "
            f"manifest (no manifests entry has file_sha256={actual_sha}). "
            "Regenerate the audit for the manifest actually being used."
        )
    if len(matches) > 1:
        raise HeldOutSplitError(
            f"{context}: disjointness audit is ambiguous - manifests entries "
            f"{sorted(matches)} all carry file_sha256={actual_sha}"
        )
    train_name = matches[0]
    train_split = manifests[train_name]["split"]
    if train_split not in TRAINING_SPLITS:
        raise HeldOutSplitError(
            f"{context}: disjointness audit entry {train_name!r} matching the "
            f"manifest in hand has split {train_split!r}; expected one of "
            f"{sorted(TRAINING_SPLITS)}"
        )
    declared_split = getattr(train_manifest, "split", None)
    if (
        isinstance(declared_split, str)
        and declared_split
        and declared_split != train_split
    ):
        raise HeldOutSplitError(
            f"{context}: disjointness audit entry {train_name!r} records split "
            f"{train_split!r} but the manifest in hand declares "
            f"{declared_split!r}; refusing a relabelled record"
        )

    if len(test_names) != 1:
        raise HeldOutSplitError(
            f"{context}: disjointness audit must contain exactly one "
            f"split={TEST_SPLIT!r} manifests entry, found {sorted(test_names)}"
        )
    test_name = test_names[0]

    # The audit must pin the LIBERO-Plus checkout and it must be the one the
    # manifest in hand was frozen against.  Both sides are required: an audit
    # without a commit, or a manifest that cannot say which checkout it came
    # from, cannot be tied to the frozen held-out data and is refused.
    actual_commit = getattr(train_manifest, "libero_plus_commit", None)
    if not isinstance(actual_commit, str) or not actual_commit:
        raise HeldOutSplitError(
            f"{context}: the manifest in hand exposes no libero_plus_commit, "
            "so the audit's pinned checkout cannot be verified; refusing"
        )
    audited_commit = record.get("libero_plus_commit")
    if not isinstance(audited_commit, str) or not audited_commit:
        raise HeldOutSplitError(
            f"{context}: disjointness audit has no top-level "
            "libero_plus_commit; an audit that does not pin the LIBERO-Plus "
            "checkout cannot vouch for a frozen manifest"
        )
    if audited_commit != actual_commit:
        raise HeldOutSplitError(
            f"{context}: disjointness audit libero_plus_commit "
            f"{audited_commit!r} does not match the manifest's "
            f"{actual_commit!r}"
        )

    pairs = record.get("pairs")
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise HeldOutSplitError(
            f"{context}: disjointness audit is missing the 'pairs' list"
        )
    wanted = {train_name, test_name}
    covered = False
    for index, row in enumerate(pairs):
        if not isinstance(row, Mapping):
            raise HeldOutSplitError(
                f"{context}: disjointness audit pairs[{index}] must be an object"
            )
        primary = row.get("primary")
        heldout = row.get("heldout")
        status = row.get("status")
        detail = row.get("detail")
        # The producer writes status="disjoint" with detail=null for a clean
        # pair and status="OVERLAP" with the validator's message otherwise.
        # ``ok`` is true only when no pair overlapped, so ANY non-disjoint row
        # in an ok-true audit is a forged or hand-edited record.
        if status != "disjoint":
            raise HeldOutSplitError(
                f"{context}: disjointness audit pair ({primary!r} vs "
                f"{heldout!r}) reports status={status!r}"
                + (f" with detail {detail!r}" if detail is not None else "")
                + "; the manifests share held-out identities and must be "
                "regenerated"
            )
        if detail is not None:
            raise HeldOutSplitError(
                f"{context}: disjointness audit pair ({primary!r} vs "
                f"{heldout!r}) is marked disjoint but carries a non-null "
                f"detail {detail!r}; refusing an ambiguous row"
            )
        if row.get("validator") != DISJOINT_AUDIT_VALIDATOR:
            raise HeldOutSplitError(
                f"{context}: disjointness audit pair ({primary!r} vs "
                f"{heldout!r}) was not produced by "
                f"{DISJOINT_AUDIT_VALIDATOR!r} (got {row.get('validator')!r})"
            )
        if {primary, heldout} == wanted:
            covered = True
    if not covered:
        raise HeldOutSplitError(
            f"{context}: disjointness audit contains no pair covering "
            f"({train_name!r} vs {test_name!r}); it cannot vouch for "
            "disjointness between the manifest in hand and the held-out test "
            "manifest"
        )
    return dict(record)


def assert_manifest_membership(entries, manifest, *, context: str) -> None:
    """Refuse executed episodes that are not members of the declared manifest.

    The construction-time guard validates the manifest an env *declares*;
    ``LiberoEnv.reset`` then accepts caller-supplied ``manifest_entries``.
    Without a membership check, a custom paired-collection driver can declare a
    ``split=train`` manifest (so every disjointness audit passes and provenance
    records ``manifest_split="train"``) while actually executing episodes taken
    from the held-out half.  Membership is compared on the full frozen record,
    not only the episode id, so a foreign episode cannot borrow a legitimate id.

    Args:
        entries: Episodes about to be executed.
        manifest: The declared ``FrozenEpisodeManifest`` (``None`` skips the
            check - a manifest-free env has no declared identity to protect).
        context: Human-readable description of the call site.

    Raises:
        HeldOutSplitError: If any entry is absent from the manifest or differs
            from the frozen record stored under the same episode id.
    """
    if manifest is None:
        return
    known = {episode.episode_id: episode for episode in manifest.episodes}
    for entry in entries:
        expected = known.get(entry.episode_id)
        if expected is None:
            raise HeldOutSplitError(
                f"{context}: episode {entry.episode_id!r} is not part of the "
                f"declared split={manifest.split!r} manifest; executing foreign "
                "episodes would falsify the recorded provenance"
            )
        if expected != entry:
            raise HeldOutSplitError(
                f"{context}: episode {entry.episode_id!r} differs from the "
                "frozen record in the declared manifest; refusing a forged entry"
            )
