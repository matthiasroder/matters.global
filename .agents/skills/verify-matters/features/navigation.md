# Navigate and inspect the graph

Find matters, change graph views and filters, and inspect actionable work through Universe, Unlock, Frontier, and Horizon.

## Sub-features

- `navigation-search` finds a matter by label, ID, or condition and handles no matches.
- `navigation-cloud` opens the 3D cloud, focuses a selection in 2D, and returns to the prior cloud camera.
- `navigation-filters` changes graph scope and status filters.
- `navigation-reports` compares browser working views with CLI results.

## How to get to it (user POV)

- Use **Search matters**, **Status filter**, and **Graph scope** in the toolbar.
- Use **3D cloud**, **Whole cloud**, **2D focus**, **Open 2D focus**, **Back to overview**, zoom, and **Reset**. The cloud button reads **Overview** when the canvas fallback is active.
- Use **Universe**, **Unlock**, **Frontier**, and **Horizon**. The last two require a selected matter.
- Run `matters universe`, `matters unlock`, `matters frontier`, or `matters horizon`.

## Driving it with CUA and the Matters CLI

Preconditions:

- Use the unchanged seed. The initial universe contains only `ship_first_version`.

- **Search.** Fill **Search matters** with `ship`, inspect the result, and click `#search-results button[data-matter-id="ship_first_version"]`. Require inspector heading `Ship first version`. Repeat with `First version is usable` to exercise condition search. Fill with `no-such-verification-matter` and require `No matching matters`. Clear the search.
- **Status.** Select `resolved` in `tab.playwright.getByRole('combobox', {name: 'Status filter', exact: true})`, then search `define`. Require `define_project_goal` as a result. Search `ship` and require no matches. Restore status `all` and clear search.
- **Scopes.** Click **2D focus** before exercising scopes. Select `universe` in **Graph scope** and capture the visible actionable graph. Select `all` to include the resolved prerequisite. Select `attention` to restore Attention. Check the visible graph and selected option after each action.
- **Cloud and 2D focus.** Click **3D cloud**, search and select `ship_first_version`, then clear search. Capture the cloud and inspector, including **Open 2D focus**. Click **Open 2D focus** and require the dependency view and **Back to overview**. Click **Back to overview** and confirm the prior cloud selection. Capture screenshots to compare the camera. Repeat entry through the toolbar's **2D focus** while a matter is selected. In the cloud, click **Whole cloud** and require both seed matters within the scene. Exercise `#zoom-in`, `#zoom-out`, and **Reset** when zoom behavior is in scope. Use fresh screenshot coordinates for cloud drag, right-drag, and wheel checks.
- **Canvas fallback.** If the toolbar says **Overview** and there is no **Whole cloud** control, record that the canvas fallback ran. Use **Overview** and **Open 2D focus** for its navigation checks. Its pan gesture is Shift-drag. A fallback pass does not prove the WebGL cloud.
- **Universe.** Click `tab.playwright.getByRole('button', {name: 'Universe', exact: true})`. Require `ship_first_version` in `#operation-output`. Compare with `.venv/bin/python -m matters.cli universe --state "$VERIFY_SCRATCH/matters.json"`.
- **Unlock.** Click **Unlock**. Require an actionable report for `ship_first_version` and its false condition. Compare with `.venv/bin/python -m matters.cli unlock --json --state "$VERIFY_SCRATCH/matters.json"`.
- **Frontier and Horizon.** Restore scope `all`, search `define project goal`, and select `#search-results button[data-matter-id="define_project_goal"]`. Click **Frontier** and require `ship_first_version`. Reselect the root if needed, click **Horizon**, and require the same matter for this seed. Compare with `.venv/bin/python -m matters.cli frontier define_project_goal --state "$VERIFY_SCRATCH/matters.json"` and the equivalent `horizon` command.
- **Proof.** Preserve screenshots and operation output for every exercised entry point. Compare state bytes with `state-before.json`. Navigation and reports must not mutate the graph.

## Gotchas

- The cloud and canvas Overview fade filtered-out nodes while preserving their positions. Focus view changes which nodes are shown. Do not assert identical filtering visuals.
- The cloud saves positions and camera in browser local storage. A navigation proof can leave the graph file unchanged while changing browser display state. Preserve relevant storage evidence for layout-persistence work and remove only this run's storage entries during cleanup.
- The two-node seed proves basic navigation only. Layout, camera behavior, or performance work needs a suitable larger fixture and a separate visual check.
- Canvas nodes are not normal DOM buttons. Search-result buttons provide a stable selection path, but do not prove canvas hit-testing.
- The README mentions a chat-style command panel. The current HTML has no such panel. Verify the visible toolbar and CLI paths rather than inventing a command input.
