# Example placement and source guidance

## Placement and navigation

Both language trees use
`docs/source-<en|zh>/rst_source/examples/<category>/<name>.rst`.
Choose the owning index by the reader's starting point:

| Example | Category | Owning index under `examples/` |
|---|---|---|
| Simulator or benchmark | `embodied` | `simulators_index.rst` |
| Physical robot | `embodied` | `real_world_index.rst` |
| Model or policy family | `embodied` | `vla_wam_index.rst` |
| Training method | `embodied` | `methods_index.rst` |
| SFT recipe | `embodied` | `sft_index.rst` |
| Agent/reasoning example | `agentic` | `agentic/index.rst` |
| System/placement example | `system` | `system/index.rst` |

The embodied gallery indexes live directly under `examples/`; an entry is
`embodied/<name>`. Entries in `agentic/index.rst` or `system/index.rst` are
relative to that category. The top-level `examples/index.rst` routes to galleries
and usually does not change for a single new example.

Use the current category index's hidden toctree plus visible cards/table. Register
the page once in its owning gallery, with the same placement and order in EN/ZH.
For Franka variants, use the existing nested Franka hierarchy rather than adding
another top-level robot category.

Page anatomy and gallery card schemas are maintained in
[docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md#example--recipe-page-requirements).
Use that guide and a current sibling such as
[LIBERO](../../../docs/source-en/rst_source/examples/embodied/libero.rst);
do not reuse the obsolete Environment/Algorithm/Quick Start template.

## Sources and links

For embodied training, inspect `examples/embodiment/run_embodiment.sh` and the
selected YAML before documenting its invocation. Reasoning, async, and SFT
recipes use their own entrypoints. Verify the actual config name and source path.

Standalone benchmark evaluation belongs in `rst_source/evaluations/`; link to
the appropriate guide. Use relative `:doc:` references from the page's actual
directory. Technical tokens and metric values stay identical in the EN/ZH pair.

For new supported capabilities, update both README announcement/feature entries.
README links are public URLs, for example
`https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/<name>.html`
and its `/zh-cn/` counterpart. Include the category segment. Use the supported
capability's actual announcement date; a new doc page alone is not a new release.
