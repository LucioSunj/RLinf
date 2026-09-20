---
name: test-install
description: Execute a requested RLinf model/environment installation or embodied CI e2e validation.
---

# Validate a requested install or embodied e2e run

Use the existing [driver.py](driver.py), from the RLinf root. It reads
[embodied-e2e-tests.yml](../../../.github/workflows/embodied-e2e-tests.yml).
Select the requested model/env, config, platform, and venv from that workflow and
the task's authorization. Do not infer an execution budget from a skill example.

## Inspect and prepare

With an existing interpreter that has PyYAML, `list`, `resolve`, and
`check-paths` inspect CI/configuration without launching a runtime:

```bash
python3 .cursor/skills/test-install/driver.py resolve <model> <env>
python3 .cursor/skills/test-install/driver.py check-paths <config>
```

Use `list` when the target is unknown, and `check-all` only for a requested
whole-CI inventory. The driver bootstraps PyYAML with `uv` if absent; choose an
already provisioned interpreter for read-only work instead of causing an install.

Before execution, preview the selected `install`, `test`, or `run` command with
`--dry-run` and inspect the actual CI shell and runner it will invoke.
Read [driver behavior](reference.md) for scope, paths, mirrors, and completion.

## Execute within the task's scope

A real test starts Ray workers and trains; installation can download/build
dependencies, write shared caches, and alter git mirror configuration. Reuse
existing authorization that covers those operations and resources. If required
authorization is missing, finish the preview and preparation first, then ask
about the concrete run. Continue independent preparation while waiting.

Honor the package's physical GPU exclusions even for device queries. Confirm
that the selected shell/config respects the authorized device set and process
ownership before executing. A path-only or source-review request does not imply
a GPU probe, dummy training run, cache cleanup, or global configuration repair.

Use a fresh per-test venv when a clean install is requested; do not erase an
existing environment to obtain one. After a failure, diagnose it and retry only
within the authorized scope and retry budget. A missing CI target or required
model artifact is a specific blocker; derive an available authorized alternative
from project evidence before asking for a material choice.

Finish through actual process termination and the requested e2e assertions.
A log directory, progress bar, or successful launcher alone is not proof of a
passed test. Retain commands, exit status, and result evidence; distinguish
installation success, missing assets, and e2e success. Mark unexecuted stages
`NOT-RUN`.

Clean only the temporary venv and processes created and owned by this test,
unless the user asked to retain them. Preserve pre-existing venvs, shared caches,
and result artifacts.
