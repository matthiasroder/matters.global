---
name: matters
description: Manage the things in life or in the world that matter by turning concerns, goals, problems, decisions, and responsibilities into matters with conditions, dependencies, resolved/unresolved state, universe, frontier, and horizon.
---

# Matters

Use this skill when the user wants to manage what matters: personal concerns, world problems, goals, decisions, responsibilities, unresolved situations, or systems of related matters.

## Purpose

Help turn vague importance into a working map:

- what the matter is
- what would make it resolved
- what must be resolved first
- what is actionable now
- what is visible further downstream

## Core Model

- A matter is a thing that matters enough to track: a concern, goal, problem, decision, responsibility, risk, or question.
- A condition is a named truth criterion: what must be true for a matter to count as resolved, plus its current true/false state.
- A dependency `(a, b)` means matter `a` must be resolved before matter `b` can be resolved.
- A matter is resolved when all conditions are true and all prerequisite matters are resolved.
- `universe` is all unresolved matters that are actionable anywhere now.
- `frontier(r)` is the actionable unresolved level-1 dependents of a resolved matter `r`.
- `horizon(r)` is the farthest unresolved descendants visible downstream from `r`.

## Workflow

1. Name the matters clearly enough that each one can be revisited.
2. Define named conditions for resolution; keep them observable and concrete.
3. Discover dependencies by comparing the matter against existing matters.
   - Ask what must already be resolved before this matter can count as resolved.
   - Ask what existing matters this matter would unlock or block.
   - Treat "no dependencies" as an explicit conclusion, not the default absence of data.
4. Add dependencies only where one matter must actually be resolved before another.
5. Compute or explain the universe, frontier, and horizon to decide what takes attention next.
6. Promote a condition into a matter when it needs its own decomposition, ownership, or sequence.

## Unlock Workflow

Use this workflow when the user asks to unlock, advance, resolve, or work toward goals in a matters tree.

1. Load the relevant state file.
2. Scan all unresolved matters, their conditions, and their dependencies.
3. Compute the universe: unresolved matters whose prerequisites are resolved.
4. Prefer actionable matters with larger downstream impact.
5. For each prioritized matter, propose concrete actions aimed at making false conditions true.
6. Separate work the agent can start autonomously from work that requires human confirmation, external access, payment, sending, publishing, or a decision.
7. Return a short proposal or progress report.
8. Do not change persisted state unless the user explicitly asked you to update it, or unless you can verify that a condition has become true through completed work in the current task.

## Extraction Workflow

Use this workflow when the user asks to extract matters from a PDF, AI conversation, blog post, notes, pasted text, or another source.

1. Convert the source to readable text first. For PDFs or documents, use an available parser or export path before extraction.
2. Extract candidate matters with stable ids, clear names, short descriptions, and observable false resolution conditions.
3. Compare candidates against existing matters in the selected state file.
4. Propose possible dependencies where names, topics, or conditions overlap, but do not silently add them.
5. Show the proposed candidates, conditions, and dependency candidates to the user for confirmation.
6. Persist only after explicit confirmation, unless the user has already asked for an update and every change is directly verifiable.

## Public Sharing Workflow

Use this workflow when the user asks to publish, share, or separate public matters from private matters.

1. Keep private state as the source of truth.
2. Use a visibility map where each matter is `private`, `shared`, or `public`.
3. Treat conditions as inheriting their matter's visibility.
4. Export dependency edges only when both endpoint matters are public.
5. Review the generated public state for accidental private matter ids before committing or publishing it.
6. Use `matters export-public --state <private-state> --visibility <visibility.json>` when the CLI is installed.
7. Use `matters merge-public --state <private-state> --public-state <public-state> --visibility <visibility.json>` to merge reviewed public edits back into private state.

## Persistence Behavior

When the user asks about or mentions a matter in a matter-management context:

1. Find the JSON state file.
   - Use the path the user gave, if any.
   - Otherwise use the `MATTERS_STATE` environment variable, if set.
   - Otherwise use project-local `.matters/matters.json`, if present.
   - Otherwise use `~/.local/share/matters/matters.json`.
   - If the user wants a different state file, use that path explicitly.

2. Load the JSON and check whether the matter already exists.
   - Match by the matter's stable name or id.
   - If there are near matches, show them and ask whether the user means an existing matter or a new one.

3. If the matter does not exist, prepare a proposed new matter.
   - Required: a clear matter name/id.
   - Required: named resolution conditions, each with a label and current true/false state, unless the user explicitly wants a placeholder.
   - Required: a dependency review against existing matters.
     - Ask whether any existing matter must be resolved before this one can be resolved.
     - Ask whether this matter must be resolved before any existing matter can be resolved.
     - If likely dependencies can be inferred from names or conditions, show the candidates and ask the user to confirm or reject them.
     - If no dependency is found, include `No dependencies` in the proposal before saving.
   - If required information is missing, ask concise follow-up questions before creating anything.

4. Before saving, show the exact matter, named conditions, and dependencies that will be added.
   - Ask the user to confirm or correct the proposed addition.
   - Do not persist unconfirmed additions.

5. Once the user confirms, update the JSON state.
   - Add the matter if missing.
   - Add its named condition truth values and dependencies.
   - Save the JSON.
   - Reload or validate the state, then report the updated universe/frontier/horizon when relevant.

Do not persist exploratory conversation by default. Persist when the user is managing matters over time, references the JSON state, asks to save/update/track/record a matter, or confirms a proposed addition.

## Implementation Guidance

- Use the installed `matters` Python package or `matters` CLI when code or storage is needed.
- The skill does not bundle the engine. The reusable implementation lives in the `matters` package.
- Keep persisted state to the primitives only: matters, condition labels and truth values, and dependencies.
- Compute all derived concepts from the loaded graph.
- Use JSON persistence through the resolved state path unless the user gives a different path.
- Persist each condition as an object with `label` and `truth`; do not save new conditions as bare booleans.
- Treat legacy boolean conditions as unlabeled data that must be normalized before saving; callable condition predicates are runtime-only.
- For unlock-style reports, prefer `matters unlock --state <path>` when the CLI is installed, or the `unlock_report` API from the `matters` package when working from source.
- For extraction, prefer `matters extract <source-text-file> --source-type <kind> --state <path>` when the CLI is installed, or the `extraction_proposal` API from the `matters` package when working from source.
- For public sharing, prefer `matters export-public --state <private-state> --visibility <visibility.json>`, or the `public_state` API from the `matters` package when working from source.
- For public edit intake, prefer `matters merge-public --state <private-state> --public-state <public-state> --visibility <visibility.json>`, or the `merge_public_state` API from the `matters` package when working from source.
- If `matters` is not installed, ask the user to install the `matters.global` package before performing persisted operations:
  `python -m pip install -e /Users/matthias/code/matters.global`.
