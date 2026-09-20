# Sources for RLinf documentation checks

Use only the locations relevant to the changed claims.

## Code and configuration

- Model identifiers: `SupportedModel` in `rlinf/config.py`.
- Environment identifiers: `SupportedEnvType` and `get_env_cls()` in
  `rlinf/envs/__init__.py`.
- Training recipes: referenced YAML under `examples/embodiment/config/` or the
  corresponding reasoning/agent examples.
- Commands: the actual scripts under `examples/`, `evaluations/`,
  `toolkits/`, `ray_utils/`, or `requirements/` named by the page.
- Published weight downloads: the exact model variant in the recipe and official
  Hub metadata; see [model-weight criteria](../review-pr/reference.md#model-weight-examples)
  when those references change.

Resolve PR facts against its actual proposed revision. A config name existing
somewhere in the tree does not prove the documented command loads it; trace the
referenced entrypoint's config path.

## Language and navigation

Sphinx sources are paired by relative path under `docs/source-en/` and
`docs/source-zh/`. README counterparts are `README.md` and `README.zh-CN.md`.
Compare the changed claims and their counterparts, including technical tokens,
numbers, capability statements, and section structure. A missing required
counterpart is a concrete finding, not permission to rewrite unrelated pages.

Use [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md) for site ownership,
page anatomy, and the paired-build gate. New pages belong in their owning index;
keep its visible cards/table consistent with the hidden toctree. The actual
category indexes are listed in the
[example placement reference](../add-example-doc-model-env/reference.md#placement-and-navigation).

Within Sphinx use `:doc:`/`:ref:` or valid relative links. In READMEs and other
out-of-tree Markdown, use public versioned-site URLs where appropriate.
A regex match for `readthedocs.io` alone is not a broken-link finding.

## Evidence and output

Judge severity by the effect on the documented workflow. Support findings with
the changed page, corresponding code/page, and a concrete correction.
Do not present a static path lookup as proof that an installation, GPU command,
or benchmark ran. Preserve source values when translating; missing runtime
evidence remains unverified.
