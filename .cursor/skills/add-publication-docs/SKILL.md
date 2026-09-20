---
name: add-publication-docs
description: Add or update RLinf Sphinx publication pages and their English/Chinese index entries.
---

# Add or update an RLinf publication

Use the supplied paper/report and existing publication pages as sources. Create
or update `docs/source-en/rst_source/resources/publications/<slug>.rst` and its
Chinese counterpart, with a lowercase underscore-separated slug.

Read the relevant parts of [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md).
Preserve the publication structure: title and paper/documentation link, Overview,
Results, Quickstart, and Citation when applicable. Keep claims and reported
numbers grounded in the supplied source, and preserve technical EN/ZH parity.

Quickstart contains exactly one link to the corresponding example page; link to
the maintained recipe instead of copying installation steps. If that example is
missing, create it with [add-example-doc-model-env](../add-example-doc-model-env/SKILL.md)
when covered by the request. If it requires an unresolved technical choice or
additional scope, prepare the publication and identify that dependency before
asking; do not invent a runnable example.

Update both publication indexes. Use the same order for the hidden toctree and
the visible cards/table, as required by the style guide. Do not reintroduce a
bullet-list index from an older template.

Use `list-table` for results where appropriate; any `:widths:` list must match
the column count. Credit figures and preserve citation metadata. Existing
results need no new experiment to document them.

Complete the style guide's paired Sphinx builds with zero new warnings and
[docs-check](../docs-check/SKILL.md) for source and language consistency.
This skill concerns publication pages in RLinf's site; research-paper drafting
or review outside those trees uses the corresponding research workflow.
