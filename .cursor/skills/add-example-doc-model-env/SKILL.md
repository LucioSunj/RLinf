---
name: add-example-doc-model-env
description: Add RLinf model or environment example documentation and its English/Chinese gallery entries.
---

# Add an RLinf example

Create the requested example in both Sphinx language trees, using the category
and owning gallery defined in [reference.md](reference.md#placement-and-navigation).
Read the relevant recipe and parity sections of
[docs/STYLE_GUIDE.md](../../../docs/STYLE_GUIDE.md); it owns the page anatomy.
Use a current sibling in the same gallery for working syntax and supported facts.

Deliver the paired RST pages, their matching category-index entries, and any
evaluation link needed for a newly supported benchmark. Keep standalone evaluation
instructions in the Evaluation section. Avoid duplicating a page across galleries.

Verify commands and configuration against the actual entrypoints and YAML.
Use the [source and link guidance](reference.md#sources-and-links), rather than an
old generic template or a hardcoded model list. Never invent benchmark numbers.

For a new supported capability, update the corresponding EN/ZH README feature
entries and dated announcement as part of its documentation. A documentation-only
addition for existing functionality does not imply a new release announcement.

Complete the Sphinx style guide's EN/ZH review and build gate, using
[docs-check](../docs-check/SKILL.md) for factual consistency. Documenting launch
commands does not authorize running training or evaluation.
