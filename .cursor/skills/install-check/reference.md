# Install conventions and harness limits

## Conventions

- Reuse `clone_or_reuse_repo`, `install_flash_attn`, `install_apex`,
  `create_and_sync_venv`, and `install_common_embodied_deps` where applicable.
  Use the existing CUDA/ROCm detection and mirror helpers.
- A model installer dispatches on `ENV_NAME` and calls the supporting env
  installers. Env-only support belongs in `install_env_only`.
- Put system packages in `requirements/embodied/sys_deps.sh`, covering its
  supported apt/dnf/yum/pacman functions. Do not add an inline host-package
  installation to `install.sh`.
- Pin git dependencies to an explicit supported revision/tag/branch or use the
  project's RLinf fork convention. A checkout after cloning can supply the pin;
  inspect the whole function before reporting a floating dependency.
- Modify RLinf's torch dependency through `apply_torch_override`. Do not rewrite
  RLinf's `pyproject.toml` ad hoc. An edit to a dependency's cloned checkout after
  `cd` is different; determine which project is being edited.
- Avoid per-env overrides of core dependencies such as `ray` and loose `torch`
  pins. Check the shared dependency definitions and actual target requirements.
  Existing sanctioned overrides outside RLinf's checkout must be judged in context.
- A newly supported model/env requires matching Docker coverage, except for an
  explicit target-specific project exception. Follow
  [Docker/CI wiring](../add-install-docker-ci-e2e/reference.md) for that change.

Use target recipes and recorded decisions to resolve exceptions. Historical
examples or a heuristic hit do not establish that the same exception is valid
for a new target.

## Harness limits

[check.sh](check.sh) uses static grep/awk patterns and reports candidates with
line numbers. Exit 1 means candidates were found, not that every candidate is
a bug; exit 2 can mean the requested input file is absent.

The harness cannot fully resolve shell working directories, later git checkouts,
or legitimate multi-repo workspaces. Docker coverage is inferred from
`--model <name>` and `--env <name>` invocations, so unusual wiring needs direct
inspection. Review hits against the requested diff and current source; do not
treat a dated list of previously missing targets as live findings.

A source review should finish with concrete defects/fixes or a conforming result.
Running `install.sh`, building Docker, changing dependency pins beyond the
request, or repairing global configuration requires its own task scope.
