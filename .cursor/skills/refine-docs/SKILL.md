---
name: refine-docs
description: Write or restyle RLinf Sphinx pages using docs/STYLE_GUIDE.md. Use docs-check for factual review.
---

# Write or refine an RLinf Sphinx page

Apply [docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md) to the requested page or
section in `docs/source-en/` and `docs/source-zh/`. Read its shared voice/parity
rules and the section for the page type: landing/index, recipe/example, or prose.
The guide owns the exact structure and style requirements; do not duplicate them
in this skill.

Preserve the requested scope and technical meaning. Update paired language pages
together and keep code identifiers, commands, claims, and results aligned.
A wording fix does not request a new page, release announcement, or site redesign.

Use [add-example-doc-model-env](../add-example-doc-model-env/SKILL.md) for a new
example's placement and [add-publication-docs](../add-publication-docs/SKILL.md)
for publication-specific content. Existing source-backed facts need no unrelated
research or model/environment execution.

Complete the style guide's review gate after the edits: check factual/EN-ZH
consistency with [docs-check](../docs-check/SKILL.md), build both language trees
with zero new warnings, and inspect the affected presentation for rendering
issues. Rebuild after a fix that changes the output; do not repeat successful
builds without a new change or unresolved concern. Report missing build tooling
as a validation limitation rather than silently installing dependencies.

This skill's site layout and build requirements do not apply to `AGENTS.md`,
skills, experiment records, or research papers outside the two Sphinx trees.
