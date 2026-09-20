---
name: docs-check
description: Check RLinf user documentation against code and its English/Chinese counterpart.
---

# Check RLinf user documentation

Use the requested pages or changed documentation as the scope. Check their
commands, configuration, paths, capability claims, and technical EN/ZH parity.
This skill covers the Sphinx trees and related user-facing README content;
agent instructions, research manuscripts, and run records have their own contracts.

Resolve technical facts from the referenced implementation/configuration and the
corresponding language page. Use the actual reviewed revision for PR work.
Read [reference.md](reference.md) only for the relevant source locations,
navigation rules, or model-weight checks.

For Sphinx pages, keep technical tokens, results, and structure aligned across EN
and ZH, and use stable `:doc:`/`:ref:` or relative links within the site.
Public documentation URLs are appropriate in a README or other document outside
the Sphinx tree. Judge the impact of a wrong link rather than assigning a fixed
severity to every absolute URL.

A review reports mismatches; an editing request fixes source-supported mismatches
in scope. Do not change implementation to fit incorrect prose. Resolve ordinary
details from sources, and ask only about an unresolved material contract.

For Sphinx edits, retain the [style guide's review gate](../../../docs/STYLE_GUIDE.md#review-gate),
including both builds with zero new warnings. Reading or reviewing a page alone
does not require executing its installation, training, or evaluation commands.

Report actionable issues with exact file/line evidence, impact, and corrected
values or wording. If none are found, say the checked scope is consistent.
Distinguish unverified facts from confirmed errors; include material limits.
