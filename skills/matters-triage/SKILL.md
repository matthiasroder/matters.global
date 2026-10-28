---
name: matters-triage
description: Execute authorised work across a Matters graph, verify completion, and continue past individual blockers. Use when asked to run graph triage, work through matters, finish what can be automated, or resume a triage run.
---

# Matters triage

Complete as much of the selected graph as the available evidence, tools and authority permit. Defer individual blockers and continue with independent work. Stop when a full pass finds no remaining executable work or useful preparation, the user's limit is reached, or the user stops the run.

Read [matters](../matters/SKILL.md) once at the start. It owns graph semantics, persistence and authorised execution. This skill selects and coordinates work; a suitable execution skill or tool performs each task.

## Scope and authority

Distinguish an execution request from a review, plan, dry run or request to edit this skill. Reviews only inspect and report. An explicit request to run triage or complete work on an identified graph or branch activates the base skill's authorised execution mode, subject to the user's limits. State the scope and applicable limits briefly; do not request permission already supplied.

Only work on the selected matters. Read prerequisites outside the scope as context, but do not work on or mark them without authority. Resolve a genuinely ambiguous graph or scope before dependent actions; do not silently expand it.

Judge the concrete action against current user instructions and available access. Existing decisions and approvals can be verified from evidence. New decisions reserved for the user, or sending, paying, publishing, deploying, changing DNS or deleting data, require authority for that action; a graph condition alone supplies none. Defer an unauthorised action, prepare a useful reviewable result where possible, and continue elsewhere. Reuse existing authorisation when it covers the action.

## Commands

An explicit user-selected graph takes precedence. Use the installed CLI with `--state <path>`, or a project wrapper only after verifying that it selects that same graph. Otherwise prefer the intended project's `./scripts/matters` wrapper when it selects that project's graph; outside a project, use `MATTERS_STATE` or the CLI default. Confirm the selection with `state-path` and retain the same command route throughout the run. Do not add a conflicting `--state` to a wrapper.

Read through `list --json`, `show <matter> --json` and `unlock --json`; inspect every unresolved matter in scope and its prerequisites. `unlock.items` contains universe matters only. Read blocked matters with `show` rather than assuming their conditions are unavailable. Never edit the state JSON.

CLI `show` text and `mark` use 1-based condition numbers; JSON indexes are 0-based. Each `unlock.items` entry's `false_conditions` contains labels, not condition numbers. Use a freshly verified exact label for marking, or the current 1-based number when a label is ambiguous.

## Select work

For each false condition, inspect existing evidence and identify the next concrete action and the acceptance check. Distinguish:

- **Complete now:** the condition can be verified already, or satisfied and verified with current inputs and authority.
- **Prepare now:** useful bounded work can be done, but the full criterion still needs an input, approval or external event. Leave it false unless the criterion explicitly asks for that preparatory deliverable.
- **Wait:** no useful action remains until a specific input, decision, permission or prerequisite changes.
- **Failed:** an attempt failed; retain the evidence and a concrete recovery or retry trigger.

Treat `agent_can_start` and `needs_human_input` as hints only. The CLI uses substring heuristics, not knowledge of tools, evidence or authority. Words such as "confirmed" or "payment" neither prove that a new human action is needed nor authorise one.

Consider all false conditions; their stored order does not create dependencies. Prefer work that can complete or unlock matters, using downstream impact, remaining blockers, effort and confidence to choose. `impact` counts downstream matters; it is not a measure of immediate unlocks or task value. Use stable matter/condition order to break otherwise equal ties.

Prefer executable universe work, then inspect blocked matters for independently satisfiable conditions or useful preparation. A dependency prevents the dependent matter from resolving; it does not forbid every action on it. Do not assume an unresolved prerequisite's outcome or use this distinction to bypass required approval, access or facts. A fully verified own condition may become true while its matter remains blocked.

## Execute and verify

1. Freshly read the selected condition and required context. Reuse existing work and evidence before creating anything. Define an observable check for the entire criterion, including actors, thresholds and approvals.
2. Do the bounded task using appropriate installed skills and tools. Poteto mode is optional when available; its absence must not block other execution methods. Read a selected skill once per run and load only the relevant playbook. A task-specific pause defers that task unless it affects the whole run.
3. Verify the actual outcome. A document is evidence of its content, not proof that an operational process works. Use relevant tests or representative executions for operational claims, authoritative records for external facts, and actual approval records for approvals. A draft qualifies when a draft is the requested result; it cannot stand in for approval or a working system. Do not invent missing evidence or weaken the condition.
4. Record the outcome and evidence. If the entire condition is true, follow Marking. Otherwise retain useful artifacts, leave it false and record the remaining blocker.
5. Refresh affected matters and `unlock --json`, then choose again across the eligible work. Continue while an independent action remains; do not repeatedly choose an unchanged blocker.

Independent tasks may run in parallel when their files, resources and dependencies do not conflict. Assign each worker explicit ownership and acceptance checks. Workers return artifacts and evidence; one coordinator owns all graph writes and revalidates worker results against the current condition.

## Marking

Under the base skill's authorised execution mode, mark verified existing conditions true without another approval round. This does not authorise changing criteria, dependencies or scope. Outside that mode, collect proposed marks under the normal persistence rules and continue any independently authorised work.

Immediately before marking, re-read the exact condition, its truth and relevant prerequisites. If its wording or required context changed, reassess the evidence. If it is already true, record that without another write. Do not use a condition number retained from an earlier snapshot. The CLI lock protects a write, but current CLI `mark` has no expected-revision check; avoid concurrent graph edits and never blindly retry a stale write.

Use the selected command route:

```sh
matters mark <matter> <fresh-condition-ref> true --state <selected-path>
matters show <matter> --state <selected-path>
```

With a verified project wrapper, substitute `./scripts/matters` and omit `--state`. Relay the result and verify the intended condition is true. On a rejection or unexpected read-back, preserve the error and inspect the current state. Correct a clear input error only after revalidation; defer an unresolved condition-specific failure and continue independent work. If graph identity or integrity cannot be trusted, stop graph writes and report the problem. Never repair it by editing JSON.

## Resume and retry

For execution runs only, keep a compact run record outside graph state and installed skill folders, preferably alongside the selected graph in `triage-runs/<run-id>.jsonl`. Reviews and dry runs do not create or update this record. Use the project's existing task tracker for task tracking; the run record is an execution/evidence log, not a second graph or backlog. Record:

- The selected graph, scope, user instruction authorising the run and any resource/action limits.
- Each attempted condition's exact label and observed position, acceptance check, action, attempt count and outcome.
- Artifact/evidence locations, verified graph writes, outstanding inputs and the event that would justify another attempt.

On resumption, establish authority from the actual user instructions, then reread the graph and validate that prior evidence still applies. A saved record does not grant permission or prove current truth. Reuse artifacts, reconcile changed criteria and do not repeat successful work or unchanged requests for approval.

Bound retries for transient failures: default to one retry of the same failed action unless the user or relevant execution skill sets another limit. A changed method or new input may justify a new attempt; name that change and still respect the run's limits. Missing facts, decisions and permissions wait for a relevant change. Do not regenerate the same draft or repeatedly poll an unchanged blocker as "progress".

## Finish the run

Before declaring a pass exhausted, collect and reconcile in-flight worker results, then refresh eligibility. After a complete pass yields no executable work or useful preparation, report that the run has exhausted its current scope and authority. Distinguish that from all matters being resolved. An empty universe on a valid graph may mean completion; verify scoped unresolved counts rather than treating every stop as success. If a user limit or interruption ends the run sooner, stop outstanding delegated work, record its actual status and artifacts, and say what eligible or unfinished work remains for resumption.

Report completed conditions and newly resolved matters with evidence/artifact links, useful preparation that remains false, and deferred or failed work with reasons. Consolidate remaining decisions, access needs and approvals into a short batch, prioritised by what they would unlock. Include the run-record location for resumption. Omit empty categories and do not claim hypothetical preparation as completion.

When an answer could unlock useful work during the run, ask a concise question while continuing independent tasks. Do not stop the whole run merely because that answer is pending.
