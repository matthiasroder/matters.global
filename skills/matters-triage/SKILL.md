---
name: matters-triage
description: Plan authorised Matters work in Beads, execute or delegate tasks, verify outcomes, and reconcile completion. Use for graph triage, working through matters, or resuming a triage run.
---

# Matters triage

Read Matters → plan in Beads → execute → verify → update Matters → repeat. The calling agent coordinates and performs the work; this skill does not require a `matters triage` command or a model inside the CLI.

Read [matters](../matters/SKILL.md) once. It owns graph semantics, persistence and authorised execution. Matters holds outcome criteria; Beads holds executable tasks and their progress. Closing a Bead triggers verification, not a completion mark.

## Scope and authority

An explicit request to run triage or complete work on an identified graph or branch activates the base skill's authorised execution mode, within the user's limits. Reviews and dry runs only inspect and report. A planning request does not authorise execution or completion marks; create a durable Beads plan only when requested or as part of an authorised execution run.

State the selected scope and limits briefly. Read prerequisites outside that scope as context; do not work on or mark them without authority. A graph condition alone does not authorise sending, paying, publishing, deploying, changing accounts or deleting data. Reuse existing authorisation when it covers the concrete action; otherwise defer that action and continue useful independent work. A saved run record does not grant authority.

## Establish the run

1. Select the graph using the user's path, otherwise the verified project wrapper, `MATTERS_STATE`, or CLI default. Confirm with `state-path`, resolve to an absolute canonical path, and require an existing file. A missing selected graph is an error, not an empty workload. Pin the command route and graph for the run; do not add a conflicting `--state` to a wrapper.
2. Resolve and pin the Beads workspace with `bd where --json`. Use an explicit `-C <workspace>` or verified database route thereafter. Follow the available Beads skill and `bd prime`. If Beads is unavailable, report that prerequisite rather than creating a substitute backlog. Reads and a review remain possible.
3. For execution or durable planning on a POSIX host, choose a run ID and acquire the local coordinator lock in a retained interactive tool session:

   ```sh
   python3 <skill-directory>/scripts/coordinator_lock.py --state <canonical-path> --run-id <run-id>
   ```

   Require its acquired JSON response **and a still-running session**; stdin EOF releases the lock, so a completed piped invocation is not ownership. Keep the interactive session alive. Contention refuses the run: do not delete the lock file or start another coordinator. This POSIX lock coordinates local triage agents only; it does not stop browser edits or coordinate separate machines. Use a single supported host for an active run. If the helper cannot acquire ownership, defer new planning and graph writes rather than inventing a weaker lock. If the lock holder exits unexpectedly, pause new assignments and graph writes, reconcile workers, and reacquire before continuing.
4. Read [the Beads and recovery contract](references/beads-workflow.md) before creating tasks or resuming an execution run. Keep a compact reconciliation journal outside the graph and installed skill, preferably `triage-runs/<run-id>.jsonl` alongside the graph. Beads remains the only task backlog.

## Read and select work

Use `unlock --json` and `show <matter> --json`; `list --json` can help identify scope. The unresolved set is the union of `unlock.universe` and `unlock.blocked`. `unlock.items` has details only for universe matters, so inspect blocked matters with `show` too. Include relevant prerequisites. Derive eligibility from the full graph before filtering work to scope.

The `revision` fields of `unlock` and every `show` used in one planning or verification pass must match. If a read differs, refresh that pass. Repeated edits that prevent a consistent pass defer planning/marking until a stable view is available. If the CLI lacks revisions or `mark --expected-revision`, report the required CLI update and defer revision-dependent planning and automated marks. Independently authorised Beads work may continue when graph identity and integrity remain sound; do not silently fall back to unchecked writes. Never read or edit the state JSON directly.

For every false condition, identify a concrete acceptance check and classify the next move:

- **Complete:** already supported by evidence, or satisfiable and verifiable with current inputs and authority.
- **Prepare:** useful work is possible, but the whole condition still needs an input or event.
- **Wait:** no useful action remains until a specific input, decision or prerequisite changes.
- **Failed:** an attempt failed; retain evidence and a concrete recovery trigger.

Prefer executable universe work, then independent work or useful preparation on blocked matters. A dependency prevents the matter from resolving; it does not forbid all work on its own conditions. Do not assume a prerequisite's outcome. A blocked matter whose own conditions are already true needs no invented task.

Use downstream impact, remaining blockers, effort and confidence to choose a bounded useful batch, with stable matter/condition order breaking ties. `impact` counts descendants, not immediate unlocks. `agent_can_start` and `needs_human_input` are label heuristics, not decisions about capability or authority. Stored condition order does not create work dependencies.

## Plan and execute in Beads

Search existing tasks across all statuses before creating any. Reuse relevant tasks, artifacts and acceptance evidence. Create tasks for concrete actions, not a mechanical copy of every condition. A task may serve several conditions; several tasks may contribute to one. Group a substantial plan with an epic when useful and add dependencies only for actual task ordering.

Use the linked reference's metadata at initial creation, with explicit acceptance criteria. Already-satisfied conditions can be verified without inventing implementation tasks. Claim each task before execution. Always give explicit issue IDs to updates and closure.

One coordinator owns task creation, reuse decisions and all Matters writes. Workers get explicit tasks, file/resource ownership, acceptance checks and authority boundaries. They update assigned Beads and return artifacts and evidence; they do not mark the graph. Run independent tasks in parallel only when their resources and dependencies permit it.

Use appropriate installed tools and execution skills. Close a Bead only when its own acceptance criteria are met, recording evidence first. Record partial work, failure or waiting in Beads and continue elsewhere. A completed draft satisfies a draft task, not an approval or operational condition. Cancelled, superseded or duplicate tasks supply no completion evidence merely because they are closed.

## Verify and reconcile

Reconcile incrementally as evidence becomes available; do not wait for the entire plan to finish. Under authorised execution, verified existing false conditions may be marked true without another approval round. This does not authorise changing criteria, dependencies or scope.

1. Collect task results and evidence. Verify the entire condition, including named actors, thresholds and approvals. Use demonstrated behaviour for operational claims and authoritative records for external facts. Task status and an agent's assertion are not substitutes for the acceptance check.
2. Freshly read the exact target and relevant prerequisites at one revision. Revalidate the stored task binding and whether the evidence still applies. Changed wording, reordered/duplicate labels, deleted targets or changed prerequisite context require reassessment; never silently transfer evidence by a saved position or a near-name match. Defer an ambiguous binding until it is explicitly resolved.
3. If already true, record that observation without another write or claiming this run caused it. Otherwise record the intended update, evidence and checked revision in the journal, then use the selected command route:

   ```sh
   matters mark <matter> <fresh-condition-ref> true --state <canonical-path> --expected-revision <checked-revision>
   matters show <matter> --json --state <canonical-path>
   ```

4. Verify the read-back and record the result. A revision conflict requires a new read and evidence assessment, not just replacing the revision and retrying. Preserve other errors and defer a condition-specific failure while continuing independent work. If graph identity or integrity is uncertain, stop graph writes.
5. Refresh `unlock --json` and choose again. Matters resolve automatically when their conditions and prerequisites resolve; there is no separate matter-completion write.

JSON condition indexes are 0-based; `show` text and `mark` positions are 1-based. `unlock.items[*].false_conditions` are labels, not positions. Use a freshly verified exact unique label or current 1-based position; digit-only labels are parsed as positions. Each mark commits separately and changes the revision, so a later mark needs a refreshed verification pass.

## Resume and finish

On resumption, recover scope and authority from the actual user request, reacquire coordinator ownership, and reconcile Beads, artifacts and the current graph using the reference's recovery table. Do not replay successful work or uncertain external actions blindly. By default retry the same transient failure at most once; a changed method or input may justify a new attempt. Unchanged blockers wait for their recorded trigger.

Before declaring a pass exhausted, collect and reconcile workers, then refresh eligibility. Stop when a full pass finds no executable work or useful preparation within scope, the user's limit is reached, or the user stops the run. An empty universe alone does not prove all scoped matters are resolved.

On interruption, stop further assignments, settle or stop delegated work and record its actual state. Release the retained lock session with Ctrl-C or stdin EOF only after worker reconciliation. Leave the lock file in place; the process releases the kernel lock. Reviews and dry runs create neither tasks nor a journal.

Report verified conditions and newly resolved matters with evidence, useful preparation still false, and deferred/failed work with the input that would unblock it. Include relevant Bead references and journal location for resumption. Consolidate outstanding decisions; ask when an answer would unlock work while continuing independent tasks.
