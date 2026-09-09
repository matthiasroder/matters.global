# Switch graphs and use the terminal

Choose which graph file to edit, and open a real shell in the verification workspace without touching the user's graph or shell session.

## Sub-features

- `workspace-switch` changes the active graph path and the displayed matters.
- `workspace-terminal` opens a shell and verifies its working directory and file output.
- `workspace-terminal-hide` hides and reopens the same shell.
- `workspace-terminal-restart` replaces the shell session.

## How to get to it (user POV)

- Fill **Graph state path** in the **Graph** panel and click **Switch graph**.
- Click **Terminal** in the toolbar. Use **Hide**, **Hide Terminal**, or **Restart** in the terminal controls.

## Driving it with CUA and the Matters CLI

Preconditions:

- Use the seeded run. Prepare a second disposable graph through the public CLI with `.venv/bin/python -m matters.cli create 'alternate graph (Alternate graph is selected)' --state "$VERIFY_SCRATCH/second.json"`.
- Set the CUA variable `scratchPath` to the exact scratch directory printed during Launch. Never substitute a private graph path.

- **Switch.** Fill `tab.playwright.getByRole('textbox', {name: 'Graph state path', exact: true})` with `scratchPath + '/second.json'`, then click **Switch graph**. Require the displayed path to end in `second.json`. Search for `alternate graph` and require that matter's result and inspector. The original seed file remains unchanged. Switch back to `scratchPath + '/matters.json'` and rerun Doctor.
- **Open terminal.** Click the toolbar's **Terminal** button. Require the **Workspace terminal** region and a status showing the scratch path. Inspect the DOM for xterm's input textarea. Send `pwd` and Return through its visible `.xterm-helper-textarea` using `pressSequentially` and `press('Enter')`. Capture the terminal screenshot and require the scratch directory in its output.
- **Write a proof file.** Type `printf 'matters terminal proof\n' > terminal-proof.txt` and Return in that input. Read `$VERIFY_SCRATCH/terminal-proof.txt` through the shell and require exactly `matters terminal proof` plus a newline. Copy it to `$VERIFY_EVIDENCE/terminal-proof.txt` before cleanup. This verifies a real shell side effect.
- **Hide and reopen.** Set a shell variable with `VERIFY_TERMINAL_MARK=retained`. Click the drawer's **Hide**, reopen **Terminal**, and type `printf '%s\n' "$VERIFY_TERMINAL_MARK"`. Require `retained`. Repeat using the toolbar's **Hide Terminal** entry when testing both hide controls.
- **Restart.** Click **Restart** within **Workspace terminal**. Type the same variable-print command and require an empty line. Run `pwd` again and require the scratch workspace. The proof file still exists. Capture before and after terminal states.

## Gotchas

- Graph switching changes shared state for every tab on this server. A stale tab's mutation can return HTTP 409. Use separate servers for independent verification runs.
- Switching graph files does not move the explicitly selected terminal workspace. The terminal remains in the directory passed to `--terminal-workspace`.
- **Hide** does not stop the shell. Server cleanup closes terminal sessions. Do not use a hidden drawer as proof of process cleanup.
- The shell can access files outside its initial working directory. Use only the bounded commands above unless the requested feature needs more.
