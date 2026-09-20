# Driver behavior and runtime evidence

## Select the smallest requested command

| Driver command | Effect |
|---|---|
| `list` / `resolve <model> <env>` | Read CI jobs and show target/configuration. |
| `check-paths <config>` | Inspect required inputs versus output paths for that config. |
| `check-all` | Inspect every e2e config; use for a requested inventory. |
| `install <model> <env> --venv <path>` | Execute the selected CI installer shell. |
| `test <config> --venv <path>` | Replay the CI step containing that config. |
| `run <model> <env> --venv <path>` | Install and run matching tests on the selected platform. |

Use `--dry-run` before a selected execution command. A `run` preview can still
report missing model paths; this is an artifact-readiness issue, not an executed
installation failure.

The driver selects the first matching install job; `run` gathers tests from
matching jobs on its platform. `test <config>` replays the containing CI shell
step, which may include other configs or multiple invocations. Inspect the full
preview against the authorized test count and resources. If it exceeds the
request, use the project's narrower authorized runner recipe, or resolve the
missing scope before launching; do not silently run the larger step.

A config outside CI may fall back to `run.sh`. Establish the correct
`run`/`run_async`/`run_offline` launcher from existing configuration and pass
`--runner` when necessary. Do not treat a default fallback as evidence of fit.

## Environment and artifacts

Use an absolute venv path. The installer can reuse an existing venv; a clean
test needs a fresh task-owned path, not deletion of a pre-existing environment.
CI shell variables may write to shared `/workspace/dataset/` uv caches and
replay target-specific asset copying or robot setup.

The driver adds `--use-mirror` by default; the installer may change global git
mirror configuration. `--no-mirror` suppresses the added flag, but inspect the
original CI step for one already present. Do not unconditionally unset proxies
or clear global git config as a troubleshooting step. Diagnose a concrete
connection/configuration failure and keep any repair within the user's scope.

With PyYAML already available, inspection/preview does not execute CI shell.
Without it, the driver's import fallback runs `uv` to provision PyYAML; choose
an existing suitable interpreter for read-only preparation.

Missing required model/input paths are different from output directories that
the run will create. Report the precise missing input. Do not skip the path
failure or download large artifacts unless that action fits the request.

## Completion and cleanup

A real test starts Ray and runs training, even for a dummy configuration.
Wait for the actual owned process and inspect its exit status and requested test
assertions/results. Fresh TensorBoard files or a progress bar only establish
progress. Foreground launcher status matters; if it detaches, follow the owned
training process to its terminal state.

Preserve failure logs and result artifacts. Separate install, asset readiness,
and e2e outcomes, and state which stages were not run. Cleanup covers only
task-created venvs and owned processes; shared caches and pre-existing venvs stay
under their existing ownership.
