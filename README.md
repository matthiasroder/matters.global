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
python -m pip install -e '.[test,providers]'
python -m pytest
```

Core installation does not include provider SDKs. Install `.[openai]`,
`.[anthropic]`, or `.[providers]` for the corresponding API adapters. The
`codex-cli` adapter has no additional Python dependency.

The package installs a `matters` CLI:

```sh
matters create 'go to Mars (human lands and stays on Mars for at least one year) > build spaceship that can fly to Mars > assemble spaceship in earth orbit'
matters universe --state examples/matters.example.json
matters frontier root --state examples/matters.example.json
matters horizon root --state examples/matters.example.json
matters unlock --state examples/matters.example.json
matters extract notes.txt --source-type notes --state examples/matters.example.json
matters tots open_question --context evidence.txt --state examples/matters.example.json
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
scope menu can switch between Attention, Universe, and All graph.

The default **Focus** view uses Cytoscape and a readable Dagre dependency
layout. Selecting a matter shows its prerequisite context and direct
dependents. **Overview** is an explicit, stable spatial atlas of every matter
and dependency: drag to rotate, Shift-drag to pan, use the wheel or toolbar to
zoom, and select a matter to emphasize its complete prerequisite ancestry.
**Focus here** moves from that selection into the dependency view; **Back to
overview** restores the previous overview camera and selection. Search and
status filters fade nonmatches in Overview without rearranging the atlas. Its
solar-system layout gives each independent prerequisite family a representative
terminal goal as its system centre, arranges related matters on orbital rings,
and places cross-system joins between their contributing suns. Large graphs
open as a landmark map of system centres; zooming reveals inner orbits and then
the complete systems. The UI also supports node inspection, condition toggles,
matter creation, and dependency creation/removal in either view.

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

### Model provider configuration

Semantic generation is selected explicitly through named profiles in a TOML
file. Matters checks `--config`, then `MATTERS_CONFIG`, then the platform user
configuration directory shown by `matters config path`. It never auto-loads a
repository-local file or activates a provider merely because an API key exists.

```toml
[llm]
default_profile = "personal"

[llm.profiles.personal]
provider = "codex-cli"
model = "gpt-5.6-sol"
auth = "chatgpt"
timeout_seconds = 300

[llm.profiles.openai]
provider = "openai-api"
model = "gpt-5.6"
api_key_env = "OPENAI_API_KEY"

[llm.profiles.claude]
provider = "anthropic-api"
model = "claude-sonnet-4-6"
api_key_env = "ANTHROPIC_API_KEY"

[llm.workflows.extraction]
profile = "personal"
on_unavailable = "marker"

[llm.workflows.reconciliation]
profile = "personal"
on_unavailable = "skip"

[llm.workflows.tots]
profile = "personal"
on_unavailable = "error"
```

Configuration stores only environment-variable names, never credential values.
For subscription-backed Codex use, run `codex login` with ChatGPT. Inspect the
resolved path and provider readiness without making a model request:

```sh
matters config path
matters config check
matters config check --profile personal
```

For a new Codex/ChatGPT installation, create the user configuration without
putting credentials in it:

```sh
codex login
CONFIG_PATH="$(matters config path)"
mkdir -p "$(dirname "$CONFIG_PATH")"
"${EDITOR:-vi}" "$CONFIG_PATH"
```

Use the `personal` Codex profile and workflow sections from the TOML example
above, then run `matters config check --profile personal`. The check verifies
the saved ChatGPT authentication and the isolated Codex command without making
a model request.

The Codex adapter and its fail-closed capability denylist are tested with Codex
CLI 0.145. After upgrading Codex, run `matters config check` and the provider
tests before relying on live generation; new tool-bearing capabilities may need
to be added to the denylist.

Use `--llm-profile` to select another profile and `--model` for a one-run model
override. `MATTERS_EXTRACT_MODEL` and `MATTERS_TOTS_MODEL` remain model-only
overrides; they do not select a provider.

### Two extraction engines

- **LLM engine** (when an extraction profile is configured): reads prose — paper
  abstracts, sections, blog posts — and extracts the source's actual claims,
  contributions, findings, and open questions as matters. Each matter is judged
  **status-aware**: settled results come back `resolved` (their conditions
  marked true), while gaps, unmet goals, and open questions come back `open`
  (at least one condition false) — so a graph captures both what a field has
  established and what it leaves open. Conditions are evidence-grounded and
  dependency candidates are semantic. This is the right engine for scientific
  papers, which rarely contain explicit markers. It can use Codex with ChatGPT
  authentication, the OpenAI Responses API, or the Anthropic Messages API.
- **Marker engine** (deterministic fallback): recognizes explicit markers like
  `Goal:`, `Problem:`, `Decision:`, `Risk:`, `Responsibility:`, and `Matter:`,
  plus speaker-prefixed lines such as `Agent: Goal: Map creativity
  interventions`. It runs with no network access and no key.

`matters extract` uses the selected extraction profile and falls back to the
marker engine when no profile is configured or the configured provider is
unavailable (the proposal then carries `engine: "marker"` and, after a provider
failure, a redacted `fallback_reason`). Pass `--no-llm` to force marker mode.

```sh
matters extract paper.txt --source-type paper --llm-profile personal
matters extract notes.txt --no-llm   # deterministic, offline
```

For PDFs and documents, extract the readable text first (v1 is text-only), then
pipe it in. See `examples/creativity_research/` for a small corpus and expected
extraction-quality notes.

## Matters ToTs

`matters tots` performs a bounded Tree-of-Thoughts exploration for one
unresolved matter. It uses the target's false conditions, prerequisite context,
direct dependents, and optional evidence text to generate structured hypotheses,
reflect on their grounding and testability, compare them, and expand promising
but distinct directions.

```sh
matters tots open_question \
  --state ~/.local/share/matters/matters.json \
  --context evidence.txt
```

The default search generates four initial candidates, expands to at most eight
nodes over two levels, and permits at most 24 ordered comparisons. Override the
bounds with `--breadth`, `--depth`, `--max-candidates`, and
`--max-comparisons`. Use `--llm-profile` to select a configured provider and
`--model` or `MATTERS_TOTS_MODEL` to override that profile's model. ToTs fails
explicitly when no semantic provider is configured or ready; it never
substitutes a non-semantic fallback.

Start with a bounded live smoke test on a real unresolved matter:

```sh
matters config check --profile personal
matters tots MATTER_ID \
  --state /path/to/matters.json \
  --llm-profile personal \
  --breadth 2 \
  --depth 1 \
  --max-candidates 2 \
  --max-comparisons 2 \
  > matters-tots-smoke.json
```

When both candidates remain viable, this smoke configuration currently makes
six structured model calls: one generation, two reflections, one proximity
clustering, and the same pairwise judgment in both presentation orders. The
default search can make substantially more calls because it adds candidates,
expansion rounds, and ordered comparisons. `config check` is non-generating;
`tots` is a live provider operation and may consume subscription or API quota.

Each candidate includes its hypothesis, mechanism, assumptions, discriminating
predictions, evidence references, and a falsification-oriented next test.
Graph and context references are validated before evaluation. Candidates are
judged pairwise in both presentation orders; contradictory order judgments
become ties, and the outcomes are aggregated with a tie-aware
Bradley-Terry-Davidson model. Diverse proximity clusters are preserved among the
finalists. External evaluator failures, when supplied through the library API,
exclude candidates before the model tournament. Evaluator results may also set
an explicit `promote`, `neutral`, or `demote` preference; that preference is
applied before the model tournament rank and remains visible in the output.

The JSON output always carries `requires_confirmation: true`,
`state_modified: false`, and
`ranking_semantics: "search_priority_not_truth"`. A high rank means that a
branch appears useful to investigate next. It is not evidence that the
hypothesis is true. The command never changes the selected state file or adds
its candidates to the graph.

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
- Without an embedding backend or configured reconciliation provider, identity
  degrades safely to slug matching, and reconciliation merges only on very high
  similarity.

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
