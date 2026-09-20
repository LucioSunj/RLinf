# RLinf review criteria

Read the section relevant to the resolved PR or local diff. The baseline and
proposed state come from [SKILL.md](SKILL.md), not an assumed `origin/main`.
For FastWAM, apply the Outer project's scoped scientific contracts as well.

## Runtime and integration

Trace the changed behavior through its callers and the closest existing
implementation. In distributed/numerical work, pay attention to the invariants
the change touches: collective participation, tensor shape/dtype and masks,
actor/rollout placement, gradient ownership, replay/state restoration, and
worker/offload lifetime. An uncommon supported path still matters; a merely
constructible input or unrelated hardening idea is not a reason to broaden a fix.

For new components, verify the applicable registration and interface:
`register_advantage`, `register_policy_loss`, or `register_reward`;
`SupportedModel` in `rlinf/config.py`; `SupportedEnvType` and `get_env_cls()`
in `rlinf/envs/__init__.py`; embodied `BasePolicy` forwards; or scheduler
`Worker` initialization and group launch. Check the actual current signatures.
Use `self.log_*` in workers and the project logger elsewhere.

Public YAML follows existing hierarchy, contains static values, and remains
read-only in code. Compare refactors against the resolved baseline and distinguish
changed semantics from existing behavior. Suggest an existing helper when it
actually replaces duplicated logic; avoid introducing abstractions for hypothetical
future uses.

## Documentation and contribution

[CONTRIBUTING.md](../../../CONTRIBUTING.md) is authoritative for style, tests,
documentation, sign-off, and PR acceptance. User-facing behavior needs tests and
documentation validated by a reviewer for reproducibility. Include relevant
training performance/stability evidence when the change affects it.

Changed user documentation must match the proposed code/configuration and its
EN/ZH counterpart: commands, paths, keys, supported names, claims, metrics,
dataset/trial counts, and structure. Use [docs-check](../docs-check/SKILL.md)
for those pages. Agent Markdown and research artifacts outside the Sphinx trees
do not acquire Sphinx layout/build rules from this review.

New model/env integrations need install, Docker, and e2e/CI coverage unless the
project explicitly exempts that target. Review changed installation paths with
[install-check](../install-check/SKILL.md); do not start runtime tests solely
because the review mentions CI.

Style/metadata findings should identify a concrete requirement violation.
Use Google docstrings/type hints and project logging; preserve third-party
licenses and existing headers on moved files. Newly authored RLinf source files
use the current-year RLinf license header in a format appropriate to the file.
PR-only title, template, and commit checks apply when a PR/commit exists;
Conventional Commits and `Signed-off-by:` remain required.

A review does not post comments, request reviewers, or merge by itself.
If large dependencies require maintainer involvement, identify that need in the
review/PR preparation without sending an unsolicited message.

## Model-weight examples

For changed public example recipes under `examples/embodiment/config/`, follow
the matching documentation's download command for the exact environment, task
suite, and model family. Where the recipe downloads into a repo-named directory,
the `/path/to/<repo-name>` placeholder and Hugging Face comment must agree.
Distinguish base weights from LoRA adapters in `model_path`, `lora_path`,
`backbone_model_path`, and `wan_wm_hf_ckpt_path`.

Verify each distinct changed model reference against its official Hub API and
check a returned canonical id for renames. An authentication/access failure
limits verification; it does not by itself prove the repo is absent. Intentional
user-selected checkpoint placeholders or task-specific deployment paths follow
their own recipe/run contract. Do not replace them with a generic public model.

Report a mismatch with its exact config/doc location and correction. Keep failed
access distinct from a confirmed wrong model or broken public download.
