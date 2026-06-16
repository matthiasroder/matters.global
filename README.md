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
python -m pip install -e '.[test]'
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

Runtime state should live outside installed skill directories. By default,
`matters` uses `~/.local/share/matters/matters.json`. Override that with:

- an explicit `--state` path
- the `MATTERS_STATE` environment variable

Project-local files such as `.matters/matters.json` are supported when selected
explicitly with `--state` or `MATTERS_STATE`; they are not auto-selected by
default.

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

The UI shows matters in a Cytoscape.js-powered directed graph, with dependency
edges drawn as arrows from prerequisite matters to dependent matters. Large
graphs open in an Attention scope: high-impact actionable matters and the
matters they unlock, rather than an unreadable all-node hairball. The graph
scope menu can switch between Attention, Universe, and All graph. Selecting a
node focuses the graph on that node's prerequisites and dependents. Focused
views use a Dagre hierarchical layout for dependency readability; explicit
all-graph overviews use Cytoscape's CoSE force layout. The UI supports pan,
zoom, text search, status filtering, node inspection, condition toggles, matter
creation, and dependency creation/removal.

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

Every proposal includes candidate matter ids, names, descriptions, resolution
conditions, dependency candidates against existing matters,
`requires_confirmation: true`, and an `engine` field naming which extractor
produced it. LLM-engine candidates also carry a resolution `status`
(`resolved` or `open`) and a truth value per condition; marker-engine
candidates are always unresolved.

### Two extraction engines

- **LLM engine** (default when an API key is available): reads prose — paper
  abstracts, sections, blog posts — and extracts the source's actual claims,
  contributions, findings, and open questions as matters. Each matter is judged
  **status-aware**: settled results come back `resolved` (their conditions
  marked true), while gaps, unmet goals, and open questions come back `open`
  (at least one condition false) — so a graph captures both what a field has
  established and what it leaves open. Conditions are evidence-grounded and
  dependency candidates are semantic. This is the right engine for scientific
  papers, which rarely contain explicit markers. It calls the Anthropic API, so
  it needs `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the environment.
  The model defaults to `claude-sonnet-4-6` and can be overridden with `--model`
  or the `MATTERS_EXTRACT_MODEL` environment variable.
- **Marker engine** (deterministic fallback): recognizes explicit markers like
  `Goal:`, `Problem:`, `Decision:`, `Risk:`, `Responsibility:`, and `Matter:`,
  plus speaker-prefixed lines such as `Agent: Goal: Map creativity
  interventions`. It runs with no network access and no key.

The selection is automatic: `matters extract` uses the LLM engine when a key is
present and falls back to the marker engine when the key is missing or the API
call fails (the proposal then carries `engine: "marker"` and a
`fallback_reason`). Pass `--no-llm` to force the marker engine.

```sh
ANTHROPIC_API_KEY=... matters extract paper.txt --source-type paper
matters extract notes.txt --no-llm   # deterministic, offline
```

For PDFs and documents, extract the readable text first (v1 is text-only), then
pipe it in. See `examples/creativity_research/` for a small corpus and expected
extraction-quality notes.

## Matter Identity and Reconciliation

When matters from many sources are merged into one graph (for example, extracting
across a whole corpus of papers), matters.global recognizes when a new matter is
the *same* as an existing one by **meaning**, not just by a matching slug — and
lets later evidence resolve earlier open matters. This is a reusable library
layer (`src/matters/identity.py`), used by ingestion pipelines rather than the
`matters extract` CLI.

- **Embedding identity.** Each matter is embedded with a local `model2vec`
  model by default (no API key; a small model downloads on first use; override
  with `MATTERS_EMBED_MODEL`). Candidates are matched by cosine similarity over
  a persisted `.npz` sidecar store kept next to the state file, so reworded
  duplicates collapse into one matter instead of piling up.
- **Relationship-aware reconciliation.** For each new matter and its nearest
  existing neighbours, an LLM classifies the relationship as one of: **same**
  (merge the duplicate), **resolves** (the new matter satisfies an existing
  *open* matter's conditions, so those conditions flip to true — cross-source
  resolution), **link** (complementary, e.g. a problem and its solution — add a
  directed dependency edge), or **distinct**.
- **Role/status guard.** A deterministic check never merges a `resolved` matter
  with an `open` one, a problem with its solution, or a method with a finding
  that uses it — regardless of what the classifier proposes.
- Without an embedding backend or API key, identity degrades safely to slug
  matching, and reconciliation merges only on very high similarity.

Reusable APIs: `get_embedder`, `EmbeddingStore`, `match_candidate`,
`ingest_candidates`, and `reconcile_candidates`.

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
