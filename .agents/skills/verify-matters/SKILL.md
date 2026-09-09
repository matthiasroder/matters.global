---
name: verify-matters
description: Verify the Matters local browser graph editor and CLI with isolated state, real user actions, saved evidence, and teardown. Use after changes to graph editing, navigation, reports, or workspace controls.
---

# Verify Matters

Use the browser editor as the primary path. The CLI is a second user-facing path and a read-only check of persisted graph behavior. Read [the feature map](features/README.md) before choosing a recipe. The map covers graph editing and workspace controls. Extraction, ToTs, sharing, and static HTML exports need their own targeted checks.

## Launch

Run all shell commands from the repository root. The package requires Python 3.10 or later. Use the existing `.venv` when it imports this checkout. Otherwise prepare it with the documented editable install:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,providers]'
```

The installation may use the network. Core browser editing does not call a model provider. The browser loads 3d-force-graph, Cytoscape, Dagre, and xterm from `cdn.jsdelivr.net`, so browser access to that host is required. The default view is **3D cloud**. If the cloud cannot load, the app falls back to canvas **Overview**. Record which renderer actually ran. A responsive HTML page alone does not prove the editor loaded.

Prepare a unique graph, empty model configuration, and separate evidence directory:

```sh
mkdir -p artifacts/verification/matters
VERIFY_EVIDENCE=$(mktemp -d "$PWD/artifacts/verification/matters/run.XXXXXX")
VERIFY_SCRATCH=$(mktemp -d "${TMPDIR:-/tmp/}matters-verify.XXXXXX")
cp examples/matters.example.json "$VERIFY_SCRATCH/matters.json"
cp examples/matters.example.json "$VERIFY_EVIDENCE/state-before.json"
: > "$VERIFY_SCRATCH/config.toml"
printf 'export VERIFY_EVIDENCE=%q\nexport VERIFY_SCRATCH=%q\nexport MATTERS_CONFIG=%q\n' \
  "$VERIFY_EVIDENCE" "$VERIFY_SCRATCH" "$VERIFY_SCRATCH/config.toml" > "$VERIFY_EVIDENCE/run.env"
git rev-parse HEAD > "$VERIFY_EVIDENCE/revision.txt"
.venv/bin/python -c 'import matters; print(matters.__file__)' > "$VERIFY_EVIDENCE/package-path.txt"
printf '%s\n' "$VERIFY_EVIDENCE" "$VERIFY_SCRATCH"
```

Use Bash or Zsh for `%q`. Keep the two printed paths. In subsequent shell calls, source this run's `run.env` by its exact path. Check that `package-path.txt` points to this checkout's `src/matters/__init__.py`.

For CLI-only verification, source `run.env` and run `.venv/bin/python -m matters.cli show ship_first_version --json --state "$VERIFY_SCRATCH/matters.json"` as the read-only doctor. Require the seed's false condition and `define_project_goal` prerequisite. Then run the mapped CLI commands and skip browser startup. Keep command evidence and clean up the scratch directory afterward.

Start the actual CLI in a retained PTY with `exec_command`, `tty: true`, and `yield_time_ms: 1000`. Keep its session ID for cleanup. Source the run's environment first if this is a new shell:

```sh
exec env MATTERS_CONFIG="$VERIFY_SCRATCH/config.toml" \
  .venv/bin/python -u -m matters.cli web \
  --host 127.0.0.1 --port 0 --no-open \
  --state "$VERIFY_SCRATCH/matters.json" \
  --terminal-workspace "$VERIFY_SCRATCH" --terminal-shell /bin/sh
```

Readiness prints `Serving matters web UI at` with the actual port and a token-bearing URL, followed by the state path and terminal workspace. Port zero lets the OS select an unused port. The default port is 8765, but this recipe does not claim it. Keep the launch URL in memory. Do not save it in evidence or commit it.

If the sandbox denies `socket.bind`, record the failed attempt and remove its scratch directory using Cleanup. Start a fresh run with the execution tool's local socket permission. Do not change app code or bind a public address to solve that error.

Each run owns its server, graph file, config, browser tab, and port. Never drive an existing user's instance. Two servers must not share a state path. `--terminal-workspace` selects a directory, not a filesystem sandbox. The terminal still runs as the current user.

The cloud saves layout and camera data in browser local storage, keyed by graph identity. The unique scratch path gives each run its own graph identity. Retain that identity for browser-storage cleanup.

## Doctor

Run this read-only check after launch and whenever auth, graph state, or rendering looks wrong. Source the run's `run.env` in an inspection shell, and set `VERIFY_LAUNCH_URL` to the exact URL retained from this run's PTY output. Keep that value out of saved command transcripts. Export it only for the following Python process:

```sh
export VERIFY_LAUNCH_URL
.venv/bin/python - <<'PY'
import json, os, pathlib, urllib.parse, urllib.request

url = urllib.parse.urlsplit(os.environ['VERIFY_LAUNCH_URL'])
assert url.hostname == '127.0.0.1', 'Expected the owned loopback instance'
token = urllib.parse.parse_qs(url.query)['token'][0]
request = urllib.request.Request(
    f'http://{url.netloc}/api/state',
    headers={'Authorization': f'Bearer {token}'},
)
with urllib.request.urlopen(request, timeout=5) as response:
    assert response.status == 200
    payload = json.load(response)
expected = pathlib.Path(os.environ['VERIFY_SCRATCH'], 'matters.json').resolve()
assert pathlib.Path(payload['state_path']).resolve() == expected, payload['state_path']
assert payload['graph_id'], 'Missing graph identity'
assert {'define_project_goal', 'ship_first_version'} <= {n['id'] for n in payload['nodes']}
evidence = pathlib.Path(os.environ['VERIFY_EVIDENCE'])
(evidence / 'doctor.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps({'origin': f'http://{url.netloc}', 'state_path': str(expected),
                  'graph_id': payload['graph_id'], 'nodes': len(payload['nodes'])}))
PY
unset VERIFY_LAUNCH_URL
```

Require the retained server session to remain alive. The per-run bearer token ties this response to that launch. Confirm the browser displays the same scratch path and renders a graph. A 401 means the token is missing or stale. Open the full launch URL again. A bare reload loses the token because the app removes it from the address bar and keeps it only in memory.

This doctor expects the seeded graph. During the graph-switching recipe, check the intentionally selected `second.json` path instead, then return to `matters.json` before running doctor again. For HTTP diagnostics of a mutation, `X-Matters-Graph` must match `graph_id`. Browser actions set that header themselves.

## Drive

Use the available `mcp__cua_repl.js` browser control. Read its returned API documentation. Create one background tab at the retained launch URL with `cua.createBrowserTab('iab', launchUrl, {visible: false})`, and retain it as `tab`. Here `launchUrl` is the exact string read from the PTY, not the bare origin. Use the browser selected by the user when specified.

Use `tab.playwright.domSnapshot()` to inspect controls and `tab.playwright` locators to act. The feature files contain the real labels and selectors. After each meaningful action, inspect the resulting DOM or screenshot before deciding the next action. Prefer search result buttons with `data-matter-id` to canvas coordinates. Do not invoke app setters, dispatch synthetic DOM events, or substitute API writes for browser actions.

The shortest mutation proof is [create a matter](features/create-matter.md). Open **Create Matter**, fill its fields, capture the filled form, submit, inspect the new matter, and reopen the full launch URL. Find the saved matter with **Search matters** and compare its condition with the JSON file. A screenshot of the success message alone is insufficient.

Run CLI commands in their own retained PTY when CLI behavior is under test. Always pass `--state "$VERIFY_SCRATCH/matters.json"` and inherit this run's `MATTERS_CONFIG`. Record the command, output, and exit code. The built-in browser terminal is a separate entry point.

## Evidence

Keep proof in the run's `artifacts/verification/matters/run.XXXXXX/` directory, outside the scratch graph directory. Save the exact directory path in the task result. This directory is git-ignored and survives Cleanup.

Capture the feature ID, entry point, revision, actions, observed outcome, and skipped paths in `proof.md`. Save a DOM snapshot and screenshot before submission and after the resulting state. In CUA, set `evidencePath` to the exact directory printed during Launch, then capture the filled creation form:

```js
let proofFs = await import('node:fs/promises');
let beforeDom = await tab.playwright.domSnapshot();
let beforePng = await tab.screenshot({fullPage: false});
await proofFs.writeFile(evidencePath + '/create-before.dom.txt', beforeDom);
await proofFs.writeFile(evidencePath + '/create-before.png', beforePng);
await nodeRepl.emitImage(beforePng);
```

After reopening the saved matter, repeat with `create-after` filenames. Inspect the screenshots as well as the DOM. Do not retain a token-bearing address bar or launch transcript.

For graph mutations, copy `$VERIFY_SCRATCH/matters.json` to `$VERIFY_EVIDENCE/state-after.json` and inspect the exact condition or edge. Use a fresh browser load and a read-only CLI report as additional persistence checks. For terminal writes, capture the terminal action and the resulting file. Record CLI stdout, stderr, and exit status together.

Use actual user paths. Only mock an external service at an existing production boundary, and label what the mock leaves unverified. This recipe creates real local files and starts a real shell if Terminal is opened. It is not a dry run. If verifying `extract --no-llm`, compare the graph before and after and capture the proposal. If claiming no network or browser activity for any command, observe that boundary as well. A flag name is not evidence.

## Cleanup

Run cleanup after successful and failed attempts. First capture the final state while the scratch directory exists.

For a cloud run, remove only this run's local-storage entry. Use the `graph_id` in `doctor.json`. Navigate to the same-origin favicon first so the cloud's page-exit handler saves its camera and stops running. Read the CUA CDP capability documentation, then use this cleanup-only operation. It must not substitute for feature actions:

```js
let cleanupDoctor = JSON.parse(await proofFs.readFile(evidencePath + '/doctor.json', 'utf8'));
let cleanupKey = 'matters-cloud:v1:' + cleanupDoctor.graph_id;
await tab.goto(new URL(launchUrl).origin + '/favicon.svg');
let cleanupCdp = await tab.capabilities.get('cdp');
let removed = await cleanupCdp.send('Runtime.evaluate', {
  expression: 'localStorage.removeItem(' + JSON.stringify(cleanupKey) + '); localStorage.getItem(' + JSON.stringify(cleanupKey) + ')',
  returnByValue: true
});
if (removed.result.value !== null) throw new Error('Cloud storage cleanup failed');
```

For graph-switching checks, retain each visited scratch graph's `graph_id` from a read-only `/api/state` response and remove its exact key as well. Do not clear other keys or the browser's whole origin. Save layout evidence before removal when layout persistence is under test. If the current tool cannot remove storage, record that remaining key rather than claiming full cleanup. Close only the tab created for this run with `await tab.close()`.

Send Ctrl-C with `write_stdin` to the retained server session and wait for its exit. Require `Stopping matters web UI` and process completion. This invokes terminal cleanup and closes the listener. Never kill by process name. If interrupted teardown requires a PID, verify that the PID belongs to this run before signaling it. Confirm the captured loopback port no longer accepts a connection.

Source the exact run's `run.env`, confirm `VERIFY_SCRATCH` is the temporary directory created above, then remove it. Do not remove `VERIFY_EVIDENCE`:

```sh
test -n "$VERIFY_SCRATCH" && test -d "$VERIFY_SCRATCH" && rm -r "$VERIFY_SCRATCH"
test ! -e "$VERIFY_SCRATCH"
test -s "$VERIFY_EVIDENCE/state-before.json"
find "$VERIFY_EVIDENCE" -maxdepth 1 -type f
```

Confirm a successful browser run still has its screenshots, DOM snapshots, `state-after.json`, and `proof.md`. A CLI-only run keeps its command evidence and state copies and has no browser or server to stop. For failed startup, keep the failure record and baseline instead. Do not clear unrelated stashes, files, servers, or browser tabs.

## Helpers

This skill ships recipes rather than executable helper files. Use the real `.venv/bin/python -m matters.cli`, the read-only doctor above, and the browser control API. Existing regression checks are `.venv/bin/python -m pytest` and `node --test tests/web_app.test.cjs`. The Node tests use a simulated browser environment and do not replace the live browser proof.

Use `/maintain-verification-skill` when controls, commands, or user entry points change.
