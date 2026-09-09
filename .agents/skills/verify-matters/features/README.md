# Matters verification map

Read this index before driving Matters. Each feature file lists user entry points and observable proof. Report each exercised entry point separately. A CLI pass does not establish a browser pass, and a search-result click does not establish canvas picking.

## Baseline preconditions

- Follow [Launch and Doctor](../SKILL.md), including its CLI-only option when appropriate. Use one disposable graph per run and one owned server per browser run.
- The seed contains resolved `define_project_goal`, actionable `ship_first_version`, and the edge `define_project_goal -> ship_first_version`.
- Keep `MATTERS_CONFIG` pointed at the empty temporary config. Always pass the scratch `--state` to CLI commands.
- `tab` is the owned browser tab. `launchUrl` is the complete token-bearing URL retained in memory. `VERIFY_SCRATCH` and `VERIFY_EVIDENCE` come from the run's environment file.
- Start each independent recipe with a fresh run unless it explicitly builds on a previous recipe. Never restore the seed while a browser is submitting edits.

## Driving conventions

Use the CUA Playwright API documented by `mcp__cua_repl.js`. Read the live DOM before applying selectors. Capture the action, its visible result, and persisted side effects. Keep screenshots, DOM snapshots, command output, and state copies in the named evidence directory.

Each feature file uses four H2 sections. Its driving section states preconditions, exact actions, and expected outcomes. If an entry point is unavailable, record the attempted action and the unmet precondition. Do not count it as covered by another entry point.

## Features

- [Create a matter](create-matter.md) covers browser creation, explicit IDs, initial conditions, and CLI expressions or stdin.
- [Edit conditions](conditions.md) covers truth toggles, renamed and added conditions, and the corresponding CLI commands.
- [Manage dependencies](dependencies.md) covers adding and removing edges, CLI links, and rejected cycles.
- [Navigate and inspect the graph](navigation.md) covers search, graph modes, scopes, status filters, and working reports.
- [Switch graphs and use the terminal](workspace.md) covers the graph path form, isolated files, and the browser's real shell.

This initial map excludes extraction, ToTs, sharing, static exports, and destructive CLI deletion commands. Add a feature recipe when work touches those paths. Keep the map current with `/maintain-verification-skill`.
