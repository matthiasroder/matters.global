---
name: matters-triage
description: Work through eligible conditions in a Matters graph, verify outcomes, and record completion. Use for graph triage, advancing a target and its prerequisites, or resuming a triage run.
---

# Matters triage

Read [matters](../matters/SKILL.md) for graph semantics, persistence and authorised execution. Identify what can advance in the selected graph, act, verify the outcome, record verified conditions and refresh eligibility.

## Scope and authority

An explicit request to run triage or complete work on an identified graph or branch activates the base skill's authorised execution mode, within the user's limits. Reviews and dry runs only inspect and report. A request for a plan authorises planning, with a durable plan written only when requested.

State the selected scope and limits briefly. When asked to work toward one target, include its transitive prerequisites unless the user narrowed the scope. Read prerequisites beyond an explicit boundary for context. Check existing authorisation before taking external or destructive action. If authorisation or access is missing, defer that action and continue useful independent work.

## Establish the run

Select the graph using the user's path, otherwise the verified project wrapper, `MATTERS_STATE`, or CLI default. Confirm with `state-path`, resolve to an absolute canonical path, and require an existing file. Pin the command route and graph for the run. Preserve any graph selection fixed by a wrapper.

**Direct triage is the default.** Use the Matters graph and existing evidence in the current session. Read-only reviews can use the graph on its own. For execution or durable planning, use [the Beads workflow](references/beads-workflow.md) when the user requests linked task tracking, resumes a Beads-backed run, or the work belongs to a project that requires it.

**For an opt-in Codex session that keeps following one target**, use `matters follow <matter-id> --state <canonical-path>` from the work directory when the user asks for that mode and authorises work on the target and its prerequisites. Pass `--max-continuations` when the user sets that limit. For narrower scope or action limits that must be present before any work starts, work in the current session, which already has those instructions. On first use the user reviews the `Stop` hook in Codex. It continues when an eligible condition remains and yields at resolution, a blocker, no new verified condition, or the continuation limit. In an already active follow session, do the work there and let its hook decide whether to continue. When a blocker ends the run, report the exact missing input or event and put `[MATTERS_WAITING]` on the final line.

## Read and select work

Use `unlock --json` and `show <matter> --json`; `list --json` can help identify scope. The unresolved set is the union of `unlock.universe` and `unlock.blocked`. `unlock.items` has details only for universe matters, so inspect blocked matters with `show` too. Include relevant prerequisites. Derive eligibility from the full graph before filtering work to scope.

The `revision` fields of `unlock` and every `show` used in one planning or verification pass must match. If a read differs, refresh that pass. Repeated edits that prevent a consistent pass defer planning or marking until a stable view is available. If the CLI lacks revisions or `mark --expected-revision`, report the required CLI update and defer revision-dependent planning and marks. Use the CLI for graph reads and writes.

For every false condition, identify a concrete acceptance check and classify the next move:

- **Complete:** already supported by evidence, or satisfiable and verifiable with current inputs and authority.
- **Prepare:** useful work is possible, but the whole condition still needs an input or event.
- **Wait:** no useful action remains until a specific input, decision or prerequisite changes.
- **Failed:** an attempt failed; retain evidence and a concrete recovery trigger.

Prefer executable universe work, then independent work or useful preparation on blocked matters. A dependency prevents the matter from resolving; it does not forbid all work on its own conditions. Do not assume a prerequisite's outcome. A blocked matter whose own conditions are already true needs no invented task.

Use downstream impact, remaining blockers, effort and confidence to choose a bounded useful batch, with stable matter/condition order breaking ties. `impact` counts descendants, not immediate unlocks. `agent_can_start` and `needs_human_input` are label heuristics, not decisions about capability or authority. Stored condition order does not create work dependencies.

## Execute

Reuse relevant evidence and artifacts before repeating work. Carry out the selected actions with the available tools and record what was verified. Useful preparation can be delivered while the corresponding condition remains false. Defer failed or waiting work with the evidence and the specific change that would let it advance. Continue independent eligible work within scope.

## Verify and reconcile

Reconcile incrementally as evidence becomes available; do not wait for the entire plan to finish. Under authorised execution, verified existing false conditions may be marked true without another approval round. This does not authorise changing criteria, dependencies or scope.

1. Collect results and evidence. Verify the entire condition, including named actors, thresholds and approvals. Use demonstrated behaviour for operational claims and authoritative records for external facts. A linked task's closure triggers condition verification.
2. Freshly read the exact target and relevant prerequisites at one revision. Revalidate that the evidence still applies to the current condition and, for tracked work, its task binding. Reassess changed wording, reordered or duplicate labels, deleted targets and changed prerequisite context. Defer ambiguous bindings.
3. If already true, report the observation. For a tracked run, journal the intended update, evidence and checked revision before writing. Use the selected command route:

   ```sh
   matters mark <matter> <fresh-condition-ref> true --state <canonical-path> --expected-revision <checked-revision>
   matters show <matter> --json --state <canonical-path>
   ```

4. Verify the read-back and record the result. On a revision conflict, read again and reassess the evidence before retrying. Defer a condition-specific failure while continuing independent work. If graph identity or integrity is uncertain, stop graph writes.
5. Refresh `unlock --json` and choose again. Matters resolve automatically when their conditions and prerequisites resolve; there is no separate matter-completion write.

JSON condition indexes are 0-based; `show` text and `mark` positions are 1-based. `unlock.items[*].false_conditions` are labels, not positions. Use a freshly verified exact unique label or current 1-based position; digit-only labels are parsed as positions. Each mark commits separately and changes the revision, so a later mark needs a refreshed verification pass.

## Resume and finish

On resumption, recover scope and authority from the user request, reread the graph and inspect existing evidence before repeating earlier work. Verify the outcome of uncertain external actions before retrying them. For a tracked run, reacquire coordinator ownership and use the recovery table. Unchanged blockers wait for their recorded trigger.

Reconcile any work still in flight and refresh eligibility. Stop when the target resolves, a full pass finds no executable work or useful preparation within scope, the user's limit is reached, or the user stops the run. Report verified conditions and resolved matters with evidence, useful preparation, and what would unblock remaining work. Include task and journal references for tracked runs.
