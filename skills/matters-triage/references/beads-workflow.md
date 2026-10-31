# Beads linkage and recovery

Use this contract for durable planning, execution and resumption. Beads stores tasks, ownership, dependencies, blockers, attempts and work evidence. The triage journal records graph linkage and reconciliation events, not another task status system.

## Pin and reuse

Record the graph's canonical path, scope, pinned Beads workspace/database, command routes, run ID and user-authorised limits at the start of the journal. Resolve Beads redirects with `bd where --json`. A cross-workspace task reference is the workspace/database plus issue ID, never just the issue ID.

Search all statuses, including completed, cancelled/superseded and deferred tasks. The default list limit is not exhaustive:

```sh
bd -C <workspace> list --all --limit 0 --json
```

Filter by metadata/labels where useful, then inspect candidate issues with `show`. Search for both exact recorded task keys and existing work that serves the same acceptance criterion. If several records match, reconcile them rather than creating another. Task closure is not a reason to discard evidence or automatically create a replacement.

Only the coordinator creates/reuses tasks. The local coordinator lock prevents competing triage planners; per-task claims do not. Do not assume `external-ref` is unique or that repeating `create --id` is a safe create-if-absent operation.

## Task metadata

Choose a stable task key once for a new action and record the intended creation in the journal before calling Beads. Include metadata in the initial `bd create`, so recovery can find a task even if the response is lost. Use structured JSON from a file to preserve quoting:

```sh
bd -C <workspace> create 'Concrete action' --type task \
  --acceptance 'Observable result that completes this task' \
  --metadata @<metadata-file.json> --json
```

Use a namespaced metadata object and an array of condition targets. For example:

```json
{
  "matters_triage": {
    "schema_version": 1,
    "task_key": "a-stable-unique-key-for-this-action",
    "created_in_run": "run-id",
    "targets": [
      {
        "state_path": "/absolute/path/to/matters.json",
        "matter_id": "customer_offer",
        "condition_label": "Customer has approved the offer",
        "condition_position": 1,
        "observed_condition_labels": ["Customer has approved the offer"],
        "source_revision": "revision-from-the-coherent-read",
        "role": "prepares"
      }
    ]
  }
}
```

`condition_position` is the observed **1-based** position, not a permanent identifier. `observed_condition_labels` retains the whole ordered label list to expose insertion, deletion, reorder and duplicate ambiguity. `source_revision` identifies the original graph facts; it is not the revision to reuse for a later mark. Valid roles are `satisfies`, `contributes`, and `prepares`; even `satisfies` requires current-condition verification.

Store the task's acceptance criterion in Beads' acceptance field. Evidence notes must identify the observed result, the check performed and artifact/record locations. Use several targets when a task serves multiple conditions. Conditions do not have stable IDs: preserve the original binding, and record any reviewed remapping and its reason rather than overwriting its provenance silently. Relevant prerequisite changes require evidence reassessment even if the labels still match.

For an existing task, preserve unrelated metadata and add the linkage only after inspecting its current data. Do not replace an entire metadata object with stale data. Keep tasks in their existing useful hierarchy; a resumed run does not need a new epic or a reparenting of reused tasks.

Always use explicit issue IDs:

```sh
bd -C <workspace> update <issue-id> --claim
bd -C <workspace> show <issue-id> --json
bd -C <workspace> close <issue-id> --reason 'Acceptance verified; evidence: <location>'
```

Claims coordinate work on one task; use distinct worker identities or coordinator-managed assignment so different workers do not appear to own the same task. A failed claim is not permission to execute anyway. Keep actual task dependencies in Beads; do not blindly mirror Matters edges, because useful preparation can precede resolution of a prerequisite matter.

## Reconciliation journal

Write JSONL events outside graph state and installed skills. Retain only what is needed to recover:

- Run binding: graph, scope, Beads route, instruction authorising work, limits.
- Creation intent and result: stable task key and full workspace-qualified Bead reference.
- Condition binding/remapping and evidence references.
- Mark intent: exact current label/position, checked revision and the acceptance evidence.
- Mark outcome/read-back, conflicts, recovery observations and run stop/release.

Persist evidence in Beads and the intended graph update in the journal **before** marking. Never treat the journal as authority or the graph's current state. There is no transaction spanning Beads, graph state, the journal and external systems. Each graph mark is an independent commit; reconcile actual state on recovery.

## Recovery decisions

| Observed state | Next action |
|---|---|
| Creation response was lost | Search all statuses for the recorded task key before retrying creation; reconcile multiple matches. |
| Artifact exists but task is open | Verify the artifact against task acceptance, then record/close without repeating successful work. |
| Task is closed but condition is false | Inspect closure reason and evidence, then verify the whole current condition. Preparation or cancellation alone does not justify a mark. |
| Mark response or journal outcome was lost | Reread the graph. If true, record observed truth; do not rerun work or invent attribution. If false, reassess evidence before a new checked mark. |
| Condition binding or relevant prerequisites changed | Review the new criterion/context and explicitly rebind only when justified. Ambiguity leaves the condition unmarked. |
| Bead reopened or a prior result was invalidated | Review the evidence and scope; do not automatically set a Matters condition false. That change follows the base skill's normal persistence rules. |
| External action may have succeeded | Inspect the external record before retrying. If the outcome is unknown, defer and retain the uncertainty. |
| Coordinator lock holder exited | Pause new work and writes, settle outstanding workers, then reacquire before resuming. Never remove a lock file to bypass ownership. |

Keep retries bounded and record what new input or event permits another attempt. A task can be fully complete while its condition remains false; a condition can be true while its matter remains blocked. Report these states accurately.
