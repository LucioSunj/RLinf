# RLinf project skills

Maintain skill content here, under `RLinf/.cursor/skills/`. The existing
`.claude/skills/` and `.codex/skills/` locations point to this same directory;
`.agents/skills/` exposes the same skills for Codex discovery. The Outer workspace
also exposes these skills through its own `.agents/skills/` entry points.

| Requested work | Skill |
|---|---|
| Review a PR or specified local diff | [review-pr](review-pr/SKILL.md) |
| Check user documentation against code and EN/ZH counterparts | [docs-check](docs-check/SKILL.md) |
| Write or restyle a Sphinx page | [refine-docs](refine-docs/SKILL.md) |
| Add model/environment example documentation | [add-example-doc-model-env](add-example-doc-model-env/SKILL.md) |
| Add or update a publication page/listing | [add-publication-docs](add-publication-docs/SKILL.md) |
| Add install, Docker, and CI support | [add-install-docker-ci-e2e](add-install-docker-ci-e2e/SKILL.md) |
| Review or fix installer/Docker conventions | [install-check](install-check/SKILL.md) |
| Execute a requested installation or embodied e2e validation | [test-install](test-install/SKILL.md) |

Read only the matching skill and the references it needs. Paths described as
repository-relative mean the RLinf root, containing `requirements/install.sh`;
run the skill's shell commands from there. When using an Outer discovery link,
resolve its target before interpreting links to RLinf repository documents.

`AGENTS.md`, `SKILL.md`, research papers, and experiment records are not Sphinx
pages. Their editing does not trigger a documentation-site restyle/build or an
installation/e2e run.

Keep descriptions short and distinguishable. Put substantial conditional details
in linked references, and keep the existing executable helpers with their skills.
Do not copy the source files back into tool-specific directories. Validate changed
skill frontmatter and links; test a helper only when its behavior changes.

Codex supports [repository skill discovery and symlinked skill folders](https://learn.chatgpt.com/docs/build-skills).
If a skill is absent from the selector, it can be read by its source path;
restart Codex if its discovery list has not refreshed.
