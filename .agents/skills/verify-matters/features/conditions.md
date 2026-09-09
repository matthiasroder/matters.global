# Edit conditions

Conditions determine whether a matter is resolved. A user can change truth values, revise labels, and add another condition in the inspector or CLI.

## Sub-features

- `condition-truth` changes a false condition to true and updates resolution.
- `condition-label` saves a new label without changing truth.
- `condition-add` appends a false condition and makes a resolved matter unresolved again.
- `condition-cli` exposes each operation from the terminal.

## How to get to it (user POV)

- Select a graph node, cloud label, **Search matters** result, or prerequisite/dependent link in the inspector.
- Run `matters mark`, `matters edit-condition`, or `matters add-condition`.

## Driving it with CUA and the Matters CLI

Preconditions:

- Use the seed. `ship_first_version` has one false condition. To choose it deterministically, fill **Search matters** with `ship first version`, inspect the results, and click `#search-results button[data-matter-id="ship_first_version"]`.

- **Toggle.** Capture the inspector, then run `await tab.playwright.locator('#inspector').getByRole('button', {name: 'False', exact: true}).click()`. Inspect the result. The button now says **True**, and the status is resolved. `show ship_first_version --json --state "$VERIFY_SCRATCH/matters.json"` confirms the truth value through the CLI.
- **Rename.** Run `await tab.playwright.getByRole('textbox', {name: 'Condition label', exact: true}).fill('First version reviewed by a real user')`, then click the inspector's **Save** button. Capture the result. The saved condition has the new label and remains true.
- **Add.** Fill `tab.playwright.getByPlaceholder('New condition', {exact: true})` with `Reviewer can reopen the result`, then click **Add condition**. The inspector shows both conditions and one **False** button. Read the JSON and require the new condition at the end with `truth: false`.
- **CLI truth.** On a fresh seed, run `.venv/bin/python -m matters.cli mark ship_first_version 1 true --state "$VERIFY_SCRATCH/matters.json"`. Require exit code zero and true in a subsequent `show`.
- **CLI label.** Run `.venv/bin/python -m matters.cli edit-condition ship_first_version 1 'First version reviewed by a real user' --state "$VERIFY_SCRATCH/matters.json"`. Require the renamed label with true unchanged.
- **CLI add.** Run `.venv/bin/python -m matters.cli add-condition ship_first_version 'Reviewer can reopen the result' --state "$VERIFY_SCRATCH/matters.json"`. Require the second false condition. Reopen the browser using the full launch URL and inspect the same facts.
- **Node selection entry point.** When a change affects canvas picking, also select the visibly labeled node in the graph using a fresh screenshot and CUA click. Require the inspector heading to match. Search selection alone does not verify this path.
- **Cloud label and linked matter.** On the seed cloud, click the visible button named `Ship first version, actionable`. Require inspector heading `Ship first version`. Then click its `Define project goal` prerequisite link and require that matter's inspector and true condition. Inspect the current DOM for these labels before clicking.

## Gotchas

- CLI numeric condition references start at one. JSON `show` indices start at zero.
- Truth controls are buttons labeled **True** or **False**, not checkboxes. With multiple conditions, scope the button to the matching `.condition-item` by its **Condition label** textbox value.
- A resolved matter can disappear from an actionable-only scope. Clear filters before treating absence as data loss.
