---
name: install-check
description: Review or fix RLinf install.sh and Dockerfile conventions. Use test-install for runtime validation.
---

# Review installation conventions

Review the requested install/Docker change; fix it when the user requests edits.
Check affected helpers, dependency pins, system packages, and target coverage.
Read the relevant [conventions](reference.md#conventions) before changing those
parts. For new install/Docker/CI wiring, use
[add-install-docker-ci-e2e](../add-install-docker-ci-e2e/SKILL.md).

The existing [check.sh](check.sh) statically flags convention candidates and Docker
coverage gaps. Run it from the RLinf root when those are the concrete failures
being checked:

```bash
bash .cursor/skills/install-check/check.sh requirements/install.sh docker/Dockerfile
```

Its heuristics over-report; a nonzero result is not automatically a defect in the
requested diff. Read each relevant hit in context, distinguish baseline issues,
and fix or explain substantiated findings. Do not require unrelated cleanup to
make the entire historical tree pass. After installer edits, use `bash -n` on
changed shell scripts and rerun affected static checks when needed.

Keep real requirements: shared install utilities, system deps in `sys_deps.sh`,
pinned or RLinf-forked git dependencies, sanctioned torch overrides, and a Docker
stage for each new model/env unless an explicit target exception applies.
The [reference](reference.md) explains accepted exceptions and harness limits.

Resolve choices from existing helpers, recipes, and recorded decisions. Ask only
if a remaining core-dependency conflict, supported-env set, or Docker exception
requires a material decision; reuse a choice already settled for this change.

Report evidence and concrete fixes, or state that the checked change conforms.
Static review does not install dependencies, build images, or run GPU/e2e tests.
