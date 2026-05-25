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
- If `matters` is not installed, ask the user to install the `matters.global` package before performing persisted operations:
  `python -m pip install -e /Users/matthias/code/matters.global`.
