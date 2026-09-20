# RLinf agent guide

RLinf uses Ray for distributed workers and Hydra for configuration. Apply the
Outer project's scientific and run-authorization contracts when working on
FastWAM. Paths below are relative to this RLinf repository.

## Working agreements

- Complete the authorized implementation, documentation, and relevant checks.
  Resolve ordinary reversible choices from the repository and prior decisions;
  ask only about a material unresolved requirement or authorization. A TODO
  records a real blocker or an out-of-scope follow-up.
- [CONTRIBUTING.md](CONTRIBUTING.md) requires tests and documentation for
  user-facing changes, with reproducibility validated by a reviewer. Prepare a
  complete reviewable change; reviewer acceptance is a review/merge condition.
- Local edits and suitable CPU checks are part of implementation. Installations,
  Docker builds, Ray launches, GPU/e2e runs, and shared-resource cleanup must fit
  the user's authorized target and resource scope. A command in a guide or skill
  is an example, not permission to run it. Preserve owned-process restrictions.
- Use Google Python style, Ruff, and public API docstrings/type hints. Use
  `rlinf.utils.logging.get_logger()` or a Worker's `self.log_*` methods.
- Keep user-facing config fields read-only in code and YAML values static;
  follow a current sibling configuration. Public behavior needs corresponding
  tests and docs. Use Conventional Commits and `Signed-off-by:` when committing;
  fill the PR template and include relevant performance/stability evidence.

## Read by task

| Task | Starting points |
|---|---|
| Policy/model integration | `rlinf/models/embodiment/`, `BasePolicy`, `SupportedModel` in `rlinf/config.py`; [FSDP model guide](docs/source-en/rst_source/extending/new_model_fsdp.rst) or [Megatron guide](docs/source-en/rst_source/extending/new_model_megatron.rst). |
| Environment/action integration | `SupportedEnvType` and `get_env_cls()` in `rlinf/envs/__init__.py`, `rlinf/envs/action_utils.py`; [environment guide](docs/source-en/rst_source/extending/new_env.rst). |
| Advantage, loss, or reward | `rlinf/algorithms/registry.py`, `advantages.py`, `losses.py`, and `rewards/`; use the existing registries and actor call signatures. |
| Runner, worker, or placement | `rlinf/runners/`, `rlinf/workers/`, `rlinf/scheduler/`, `rlinf/utils/placement.py`; [cluster guide](docs/source-en/rst_source/guides/hetero.rst). |
| Resume or metric handling | [resume guide](docs/source-en/rst_source/guides/resume.rst), [logger guide](docs/source-en/rst_source/guides/logger.rst), and the affected runner. FastWAM also requires the Outer checkpoint/scientific contracts. |
| Install/Docker/CI wiring or review | Select `add-install-docker-ci-e2e` or `install-check` from the [skill index](.cursor/skills/README.md). `test-install` is for requested runtime validation. |
| Sphinx pages, examples, publications, or doc review | Select the matching [documentation skill](.cursor/skills/README.md). [STYLE_GUIDE.md](docs/STYLE_GUIDE.md) governs `docs/source-en/` and `docs/source-zh/`, including their paired builds. |
| PR or local diff review | [review-pr](.cursor/skills/review-pr/SKILL.md), using the actual PR base/head or the requested local baseline. |

For distributed work, preserve the non-obvious lifecycle contracts: set
`RLINF_NODE_RANK` before starting Ray on each node, because Ray captures the
environment at startup; allocate worker groups through the scheduler and launch
the entry script only on the head node. Single-node configs use
`cluster.num_nodes: 1`. Read the relevant launch/config files before an
authorized run; do not tune fixed scientific batch or rollout settings as an
incidental OOM fix.

Keep this guide limited to standing agreements and useful entry points. Skill
details have one maintained source under `.cursor/skills/`; the other skill
locations point there. Load only skills relevant to the requested deliverable.
