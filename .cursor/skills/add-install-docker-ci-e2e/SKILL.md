---
name: add-install-docker-ci-e2e
description: Add RLinf install, Docker, and CI wiring for a new embodied model, environment, or supported combination.
---

# Add install and CI support

Deliver coherent install, Docker, and e2e support for the requested target.
Resolve the model/environment names and supported combinations from the current
registries, install helpers, and neighboring recipes.

- Register the target in `requirements/install.sh` and wire its installer through
  the appropriate model/env branches. Reuse shared helpers and keep system
  packages in `requirements/embodied/sys_deps.sh`.
- Add the matching Docker build stage and Docker CI target. A new target needs
  Docker coverage unless the project has an explicit target-specific exception.
- Add the relevant e2e config and workflow job for the target/platform. Preserve
  the configured runner and required asset paths.

Read only the matching sections of [reference.md](reference.md) for install
locations, Docker layers, or CI wiring. In particular, multiple installations in
an image must share one `RUN` when uv uses hardlinks across their cache: separate
layers break that layout.

Use [install-check](../install-check/SKILL.md) for install conventions and static
validation of the changed wiring. If runtime validation is requested and
authorized, use [test-install](../test-install/SKILL.md) with the selected target
and resources. Editing CI configuration does not authorize launching its jobs or
executing its cleanup on the current machine.

Completion includes the requested files and required validation evidence.
Distinguish configuration/static checks from Docker, install, and GPU/e2e results;
mark those `NOT-RUN` until executed. Preserve reviewer reproducibility requirements.
