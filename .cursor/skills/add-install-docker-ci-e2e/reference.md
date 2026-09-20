# Install, Docker, and CI wiring

## Installer

Read the current `SUPPORTED_MODELS` and `SUPPORTED_ENVS` arrays in
`requirements/install.sh`; do not copy an old enumerated list.

A model has an `install_<model>_model()` function dispatched by `main()` and
branches on `ENV_NAME`. Add environment setup to the supporting model installers
and to `install_env_only` when the env supports that mode. Reuse the helpers
described in [install-check](../install-check/reference.md#conventions).
Keep the help text consistent with the arrays.

## Docker layers

Follow the nearest compatible target in `docker/Dockerfile`. Add its
`base-image-embodied-<target>` alias when needed and the matching
`embodied-<target>-image` stage. Choose the base image from actual platform/target
requirements, rather than a fixed CUDA example from an old guide.

When multiple installs share uv's hardlink cache, place them in one layer:

```dockerfile
RUN bash requirements/install.sh embodied --venv <venv1> --model <model1> --env <env> && \
    bash requirements/install.sh embodied --venv <venv2> --model <model2> --env <env>
```

Separate install layers break that hardlink layout. Keep required assets and the
default venv activation with the target. The final
`FROM ${BUILD_TARGET}-image AS final-image` already selects a named target;
the CI `BUILD_TARGET` must match the new stage.

## Docker CI

In `.github/workflows/docker-build.yml`, follow a matching platform job's
checkout/Buildx/build setup. Keep job id, build target, and image tag consistent.
Preserve that job's mirror and cache-only output behavior where applicable.
Adding wiring does not request publishing an image or dispatching a build.

## Embodied e2e CI

Place the config in `tests/e2e_tests/embodied/<name>.yaml` and reference it in
`.github/workflows/embodied-e2e-tests.yml`. Follow the correct existing platform
job for install env vars, assets, runner labels, timeout, and test launcher.
The synchronous `run.sh <name>` loads from that config directory; async/offline
tests use their own runner. Do not change training/evaluation scientific settings
to make an unrelated install test pass.

CI cleanup steps are scoped to their disposable CI job. They are not a recipe
for pruning shared caches or stopping other jobs on a collaborator's host.
Use [test-install](../test-install/SKILL.md) to prepare a separately authorized
local reproduction and record unexecuted runtime validation as `NOT-RUN`.
