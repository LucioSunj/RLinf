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
#: :func:`assert_disjoint_audit`.  Produced by the stage-2 asset-split tool.
DISJOINT_AUDIT_SCHEMA = "stage2-disjoint-audit-v1"

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

#: Dimension keys a disjointness audit must report, mirroring the keys returned
#: by ``rlinf.envs.libero.episode_manifest.validate_manifest_disjoint``.  The two
#: are locked together by a drift test in
#: ``tests/unit_tests/test_test_set_guard.py``; if a dimension is ever added to
#: the validator, that test fails and previously written audits stop being
#: accepted here rather than silently passing with an unchecked dimension.
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


def _audit_side(
    record: Mapping[str, Any], key: str, *, context: str
) -> Mapping[str, Any]:
    side = record.get(key)
    if not isinstance(side, Mapping):
        raise HeldOutSplitError(
            f"{context}: disjointness audit is missing the {key!r} object"
        )
    return side


def _audit_text(side: Mapping[str, Any], key: str, *, context: str, where: str) -> str:
    value = side.get(key)
    if not isinstance(value, str) or not value:
        raise HeldOutSplitError(
            f"{context}: disjointness audit {where}.{key} must be a non-empty string"
        )
    return value


def assert_disjoint_audit(
    train_manifest: Any,
    audit_path: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate a committed disjointness audit against the manifest in hand.

    ``LiberoEnv`` recomputes disjointness inline whenever it drives the
    simulator, so this function exists for the paths that do not: offline label,
    uplift, and reporting tools that consume a frozen manifest without ever
    constructing an environment.  It verifies that the audit really pertains to
    *this* manifest and that it covers every dimension the validator checks
    today, so a stale or mismatched record cannot be passed off as evidence.

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
        HeldOutSplitError: On any schema, identity, coverage, or
            non-empty-intersection failure.
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

    train_side = _audit_side(record, "train_manifest", context=context)
    test_side = _audit_side(record, "test_manifest", context=context)

    train_split = _audit_text(
        train_side, "split", context=context, where="train_manifest"
    )
    if train_split not in TRAINING_SPLITS:
        raise HeldOutSplitError(
            f"{context}: disjointness audit train side has split {train_split!r}; "
            f"expected one of {sorted(TRAINING_SPLITS)}"
        )
    test_split = _audit_text(test_side, "split", context=context, where="test_manifest")
    if test_split != TEST_SPLIT:
        raise HeldOutSplitError(
            f"{context}: disjointness audit held-out side has split "
            f"{test_split!r}; expected {TEST_SPLIT!r}"
        )

    audited = _audit_text(
        train_side, "file_sha256", context=context, where="train_manifest"
    )
    actual = getattr(train_manifest, "file_sha256", None)
    if not isinstance(actual, str) or not actual:
        raise HeldOutSplitError(
            f"{context}: cannot verify the disjointness audit because the "
            "manifest in hand exposes no file_sha256"
        )
    if audited != actual:
        raise HeldOutSplitError(
            f"{context}: disjointness audit was written for a different "
            f"manifest (audit file_sha256={audited}, in-hand={actual}). "
            "Regenerate the audit for the manifest actually being used."
        )

    audited_commit = train_side.get("libero_plus_commit") or record.get(
        "libero_plus_commit"
    )
    actual_commit = getattr(train_manifest, "libero_plus_commit", None)
    if isinstance(actual_commit, str) and actual_commit:
        if audited_commit != actual_commit:
            raise HeldOutSplitError(
                f"{context}: disjointness audit libero_plus_commit "
                f"{audited_commit!r} does not match the manifest's "
                f"{actual_commit!r}"
            )

    dimensions = record.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise HeldOutSplitError(
            f"{context}: disjointness audit is missing the 'dimensions' object"
        )
    reported = frozenset(dimensions)
    if reported != REQUIRED_AUDIT_DIMENSIONS:
        missing = sorted(REQUIRED_AUDIT_DIMENSIONS - reported)
        extra = sorted(reported - REQUIRED_AUDIT_DIMENSIONS)
        raise HeldOutSplitError(
            f"{context}: disjointness audit dimensions must be exactly "
            f"{sorted(REQUIRED_AUDIT_DIMENSIONS)}; missing={missing}, "
            f"unexpected={extra}"
        )
    for name in sorted(REQUIRED_AUDIT_DIMENSIONS):
        overlap = dimensions[name]
        if not isinstance(overlap, (list, tuple)):
            raise HeldOutSplitError(
                f"{context}: disjointness audit dimension {name!r} must be a "
                f"list of overlapping values, got {type(overlap).__name__}"
            )
        if overlap:
            raise HeldOutSplitError(
                f"{context}: disjointness audit reports a non-empty overlap on "
                f"{name!r}: {list(overlap)[:10]}. The train/validation manifest "
                "shares held-out identities and must be regenerated."
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
