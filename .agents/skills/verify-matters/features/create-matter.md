# Create a matter

Create a matter with an observable condition, then find it again after reopening the browser. The CLI also accepts compact dependency chains and piped expressions.

## Sub-features

- `create-browser` saves the form with a title, explicit ID, and false initial condition.
- `create-generated-id` derives an ID when Optional id is empty.
- `create-cli` accepts a quoted expression as an argument or through stdin.
- `create-persistence` returns the same saved facts through a fresh page and `show`.

## How to get to it (user POV)

- Expand **Create Matter** in the right panel and submit **Create matter**.
- Run `matters create` with a quoted expression, or pipe an expression to it.

## Driving it with CUA and the Matters CLI

Preconditions:

- Launch and Doctor passed with the seed. There is no `verification_release` matter.

- **Fill the browser form.** Run the following through CUA, then capture `create-before.png` and `create-before.dom.txt`. The form shows the values that will be submitted.

```js
await tab.playwright.getByText('Create Matter', {exact: true}).click();
await tab.playwright.getByPlaceholder('Matter title', {exact: true}).fill('Verification release');
await tab.playwright.getByPlaceholder('Optional id', {exact: true}).fill('verification_release');
await tab.playwright.getByPlaceholder('One condition per line', {exact: true}).fill('Release is visible to one reviewer');
nodeRepl.write(await tab.playwright.domSnapshot());
```

- **Save.** Run `await tab.playwright.getByRole('button', {name: 'Create matter', exact: true}).click()`. Inspect the DOM. The inspector heading is `Verification release`, the condition is `Release is visible to one reviewer`, the truth button says **False**, and the operation output says `Matter created.`
- **Reopen.** Run `await tab.goto(launchUrl)`, inspect the page, and run `await tab.playwright.getByPlaceholder('Search matters', {exact: true}).fill('verification release')`. Inspect the search results, then click `tab.playwright.locator('#search-results button[data-matter-id="verification_release"]')`. The inspector shows the same condition. Save `create-after.png` and `create-after.dom.txt`.
- **Check persistence.** Run `.venv/bin/python -m matters.cli show verification_release --json --state "$VERIFY_SCRATCH/matters.json"`. Require exit code zero and the same false condition. Copy the graph to `state-after.json`. Its `matters` list contains `verification_release`, its `conditions` entry contains the submitted label and `false`, and the seed edge remains unchanged.
- **Generated ID.** In a separate fresh run, repeat the form with title `Verification generated`, leave Optional id empty, and use condition `Generated matter is saved`. Require `Verification generated` in the inspector and `verification_generated` in the saved graph.
- **CLI argument.** Run `.venv/bin/python -m matters.cli create 'CLI release (CLI release is checked) > prepare CLI proof (CLI proof is ready)' --state "$VERIFY_SCRATCH/matters.json"`. Require the two created IDs and the saved edge `prepare_cli_proof -> cli_release`.
- **CLI stdin.** Run `printf '%s\n' 'stdin proof (Piped matter is saved)' | .venv/bin/python -m matters.cli create --state "$VERIFY_SCRATCH/matters.json"`. Require `stdin_proof` in `show stdin_proof --json` with the same explicit state flag. Capture the command, output, and exit code for each CLI entry point.

## Gotchas

- The browser humanizes IDs for its labels. The inspector shows `Verification release` for `verification_release`, while search results show `verification release`. Do not assume the submitted title is stored as separate metadata.
- Quote `>` in CLI expressions so the shell cannot interpret it as file redirection. The expression reads dependent first, while saved edges put the prerequisite first.
- Reopening a bare URL does not restore API auth. Reuse the full launch URL kept in memory.
- The form has no dedicated cancel action. Collapsing its disclosure does not prove the draft was discarded.
