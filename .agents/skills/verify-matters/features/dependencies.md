# Manage dependencies

Connect a prerequisite to a dependent matter, remove that connection, and reject an edge that would create a cycle.

## Sub-features

- `dependency-add` persists a selected prerequisite and dependent.
- `dependency-remove` removes only that edge.
- `dependency-cycle` rejects an edge that closes a cycle without changing state.
- `dependency-cli` performs the same operations with `link` and `unlink`.

## How to get to it (user POV)

- Expand **Dependencies** in the right panel. Choose the two matters and use **Add edge** or **Remove edge**.
- Run `matters link` or `matters unlink` with the dependent first and prerequisite second.

## Driving it with CUA and the Matters CLI

Preconditions:

- The seed edge is `define_project_goal -> ship_first_version`. Keep a copy of the graph before each action.

- **Choose the seed edge.** Expand **Dependencies** with `tab.playwright.getByText('Dependencies', {exact: true}).click()`. Select `define_project_goal` with `tab.playwright.getByRole('combobox', {name: 'Prerequisite matter', exact: true}).selectOption('define_project_goal')`. Select `ship_first_version` in **Dependent matter**. Capture the selected values.
- **Remove.** Click `tab.playwright.getByRole('button', {name: 'Remove edge', exact: true})`. Require the saved `dependencies` array to be empty. Re-select `ship_first_version` through search and require no prerequisite in the inspector.
- **Add.** Reselect the same two values after inspecting the updated DOM. Click **Add edge**. Require the one original edge in JSON and `define_project_goal` in the dependent's inspector. Capture the visible result.
- **Reject a cycle.** Select `ship_first_version` as prerequisite and `define_project_goal` as dependent, then click **Add edge**. Require an error in `#operation-output` and a byte-for-byte unchanged state file compared with the copy made immediately before the attempt. Both existing matters remain visible.
- **CLI removal.** On a fresh seed, run `.venv/bin/python -m matters.cli unlink ship_first_version define_project_goal --state "$VERIFY_SCRATCH/matters.json"`. Require exit code zero and no saved edges.
- **CLI addition.** Run `.venv/bin/python -m matters.cli link ship_first_version define_project_goal --state "$VERIFY_SCRATCH/matters.json"`. Require the original edge restored. Try the reversed pair and require a nonzero exit, a cycle error, and unchanged graph bytes.

## Gotchas

- CLI arguments are dependent then prerequisite. Browser fields and saved edges use prerequisite then dependent.
- Re-read selected values after mutations because the form options are rebuilt.
- Capture a graph screenshot as well as JSON when the change affects edge drawing. A file comparison cannot establish that the arrow is visible or points the right way.
