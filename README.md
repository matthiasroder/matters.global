# matters.global

`matters.global` contains the reusable matters engine and the first-party agent skill that uses it.

## Core Concepts

A matter is any concern, goal, decision, responsibility, risk, or question worth tracking. Each matter has observable conditions that define what must be true for it to count as resolved.

Dependencies connect matters when one must be resolved before another can be resolved. A dependency `(a, b)` means `a` has to resolve before `b` can resolve.

The engine derives three working views from those primitives:

- `universe`: unresolved matters whose prerequisites are already resolved, so they are actionable now.
- `frontier(root)`: immediately actionable downstream matters unlocked by a resolved root.
- `horizon(root)`: the farthest unresolved descendants visible downstream from a root.

## Example

```json
{
  "schema_version": 2,
  "matters": ["define_offer", "send_proposal"],
  "conditions": {
    "define_offer": [
      { "label": "Scope and price are written down", "truth": true }
    ],
    "send_proposal": [
      { "label": "Proposal has been sent to the client", "truth": false }
    ]
  },
  "dependencies": [["define_offer", "send_proposal"]]
}
```

In this example `send_proposal` is in the universe because its prerequisite is resolved and its own condition is still false.

## Layout

```text
src/matters/         # reusable Python package
skills/matters/      # agent skill instructions
examples/            # non-personal example state
scripts/             # repository maintenance scripts
tests/               # engine and storage tests
```

The skill is intentionally thin. Shared behavior belongs in `src/matters`, not in a skill directory.

## Local Development

```sh
python -m pip install -e .
python -m pytest
```

The package installs a `matters` CLI:

```sh
matters create 'go to Mars (human lands and stays on Mars for at least one year) > build spaceship that can fly to Mars > assemble spaceship in earth orbit'
matters universe --state examples/matters.example.json
matters frontier root --state examples/matters.example.json
matters horizon root --state examples/matters.example.json
matters unlock --state examples/matters.example.json
matters extract notes.txt --source-type notes --state examples/matters.example.json
matters export-public --state private.matters.json --visibility visibility.json
matters merge-public --state private.matters.json --public-state public.matters.json --visibility visibility.json
matters web --state examples/matters.example.json
```

## State Files

Runtime state should live outside installed skill directories. Use one of:

- an explicit `--state` path
- the `MATTERS_STATE` environment variable
- project-local `.matters/matters.json`
- `~/.local/share/matters/matters.json`

## Creating Matters

`matters create` writes new matters to the selected state file. The compact
form is:

```sh
matters create 'goal matter (observable resolution condition) > prerequisite matter > earlier prerequisite'
```

The chain reads left-to-right as "depends on", while saved dependency edges use
the engine direction `prerequisite -> dependent`. For example:

```sh
matters create 'go to Mars (human lands and stays on Mars for at least one year) > build spaceship that can fly to Mars > assemble spaceship in earth orbit'
```

creates:

```text
assemble_spaceship_in_earth_orbit -> build_spaceship_that_can_fly_to_mars -> go_to_mars
```

Parentheses at the end of a segment define that matter's first false condition.
Segments without parentheses get a default condition of `Resolved: <matter>`.
Because `>` is shell redirection when unquoted, quote the expression or pipe it:

```sh
printf '%s\n' 'go to Mars (human lands and stays on Mars for at least one year) > build spaceship that can fly to Mars' | matters create
```

## Unlock Reports

`matters unlock` scans the unresolved tree, finds currently actionable matters, and proposes concrete next actions for each false condition. Actions are marked as either `agent_can_start` or `needs_human_input`.

The text format is meant for a quick agent planning pass:

```sh
matters unlock --state ~/.local/share/matters/matters.json
```

Use JSON output when another tool should consume the report:

```sh
matters unlock --json --state ~/.local/share/matters/matters.json
```

## Browser Graph UI

Run a local web UI to inspect and edit the real matters graph in a browser:

```sh
matters web --state examples/matters.example.json
```

The UI shows matters in an organic force-directed 3D layout with dependency
edges as directional arrows. Link distance, charge, collision, and gentle
status drift keep related matters readable without forcing them into a rigid
grid. It supports rotate, pan, zoom, text search, status filtering, node
inspection, condition toggles, matter creation, and dependency creation/removal.
Selecting a node keeps directly connected matters prominent while unrelated
nodes and edges fade into the visual background. Node dragging is disabled as
graph editing state, so grabbing a node rotates/orbits the graph instead of
moving only that matter. Use the reset button to restore the camera.

The top toolbar and the chat-style command panel both expose common graph
operations:

```text
universe
unlock
frontier <matter_id>
horizon <matter_id>
create goal matter (observable condition) > prerequisite matter
extract source text to inspect
```

The command panel is local and engine-backed in this first version. It does not
launch or control Codex, Claude, or other agents directly yet; those integrations
can be added through a future adapter layer.

## Extraction Proposals

`matters extract` turns source text into candidate matters and dependency candidates. It always prints a proposal and does not save anything to the state file.

```sh
matters extract notes.txt --source-type notes --state ~/.local/share/matters/matters.json
```

Use `-` to read from stdin:

```sh
pbpaste | matters extract - --source-type conversation
```

For PDFs, blog posts, documents, and AI conversations, extract or paste the readable text first, then pass the source type as context. The output includes candidate matter ids, descriptions, initial false resolution conditions, dependency candidates against existing matters, and `requires_confirmation: true`.

The extractor recognizes explicit markers like `Goal:`, `Problem:`, `Decision:`, `Risk:`, `Responsibility:`, and `Matter:`. In conversation exports, speaker-prefixed lines such as `Agent: Goal: Map creativity interventions` are also recognized. See `examples/creativity_research/` for a small corpus and expected extraction-quality notes.

## Public Sharing

The first multi-user sharing layer is documented in [docs/multi-user.md](docs/multi-user.md). A private state can be exported into a world-readable public state with a visibility map:

```json
{
  "publish_matters_global_system": "public",
  "resolve_car_insurance_issue": "private"
}
```

```sh
matters export-public --state ~/.local/share/matters/matters.json --visibility visibility.json
matters merge-public --state ~/.local/share/matters/matters.json --public-state public.matters.json --visibility visibility.json
```

The export includes only matters marked `public`, their conditions, and dependency edges where both endpoints are public. The merge path accepts edits to public matters and rejects incoming matter ids that are not marked public.

## Walkthrough

See [docs/walkthrough.md](docs/walkthrough.md) for a small end-to-end example covering `universe`, `unlock`, `extract`, `export-public`, and `merge-public`.

## Publishing the Skill

The canonical skill source lives in this repo at `skills/matters`. To publish it into the local `SKILLS` repository:

```sh
scripts/sync_skill_to_skills_repo.sh
```
