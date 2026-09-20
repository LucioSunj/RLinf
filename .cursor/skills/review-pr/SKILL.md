---
name: review-pr
description: Review a specified RLinf PR or local diff for correctness and contribution requirements.
---

# Review an RLinf change

Deliver an evidence-backed review of the requested change. Resolve its scope
first: a PR uses its actual base/head; a local review uses the user's staged,
working-tree, commit, or branch comparison. A local review needs no PR URL.
Inspect repository context before asking about a materially ambiguous baseline.

Read the diff and the relevant baseline/proposed files. Use `git show` or
authorized repository tools without switching the checkout or overwriting WIP.
Fetch only missing or stale refs. If one source is unavailable, use another
authorized source for that same target and state any remaining evidence gap.

Prioritize changed behavior, supported inputs, numerical/distributed correctness,
and the nearest existing implementation. Report every substantiated defect in
scope, including uncommon cases reachable through supported use. Keep fixes
proportionate to the project's scope; do not manufacture findings or require a
quota of bugs in any category.

## Load detail when it applies

- For algorithms, policies, workers, configuration, or runtime changes, read the
  relevant [RLinf review criteria](reference.md#runtime-and-integration).
- For user-facing behavior or documentation, read
  [documentation and contribution requirements](reference.md#documentation-and-contribution).
  Use [docs-check](../docs-check/SKILL.md) for affected user documentation.
- For installer/Docker changes, use [install-check](../install-check/SKILL.md).
  Read the install/CI wiring skill only for new integration work.
- For changed public model-weight examples, read
  [model reference checks](reference.md#model-weight-examples).

Required tests and reviewer reproducibility remain acceptance conditions.
Choose checks that resolve a concrete concern in this diff; existing evidence
may suffice. A review does not itself authorize installations, GPU/e2e runs,
training, or external posting. State `NOT-RUN` where required evidence is absent.

Open with the exact scope and a brief explanation of the resulting behavior.
Then give actionable findings in severity order, with file/line evidence, the
trigger and impact, and a concrete fix. Distinguish newly introduced defects from
baseline issues. If the checked scope is correct, say so; disclose material
validation limits without turning them into unsupported findings.
