# Kenett Creativity Matters Graph

A real-world example of LLM-based matter extraction (`build_extraction_proposal`,
the `engine: "llm"` path) applied to a scientific corpus.

It compiles the published papers of creativity researcher **Yoed N. Kenett**
(Technion) from Semantic Scholar, runs the LLM extractor over every abstract,
and assembles the candidate matters and dependency candidates into a single
dedicated matters graph.

## Contents

| File | What it is |
|---|---|
| `pipeline.py` | The end-to-end script: fetch corpus → extract matters → assemble graph. Paths are user-specific (`~/.local/share/matters/...`); treat it as a worked example, not a turnkey tool. |
| `corpus.json` | 166 fetched papers (title, abstract, year, citation count, venue, DOI); 102 have abstracts. |
| `creativity_graph.json` | The assembled matters state (`schema_version: 2`). |
| `matter_sources.json` | Provenance sidecar: each matter id → its source paper, year, DOI, and the model's name/description. |

## The graph

Built from the **102 abstracts** (extracted with `claude-sonnet-4-6`):

- **750 matters**, **2,681 conditions**, **980 dependencies**
- Acyclic; 201 matters actionable now (no unresolved prerequisites)
- **380 / 980 dependencies link matters from different papers** — cross-corpus
  synthesis, not just within-paper structure
- Hubs reflect Kenett's core program: semantic-memory structure ↔ verbal
  creativity, the association-correlation network method, associative thinking
  as the core mechanism, and extensions of Mednick's (1962) associative theory

## Explore it

```sh
matters web --state examples/kenett_creativity/creativity_graph.json
matters universe --state examples/kenett_creativity/creativity_graph.json
```

## Reproduce / extend

`pipeline.py` caches each paper's extraction under
`~/.local/share/matters/kenett/extractions/`, so a re-run resumes instead of
re-extracting. Requires `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the
environment; `KENETT_EXTRACT_CAP` caps how many of the most-cited abstracts are
processed (default 25; this graph used 102 = all abstracts). Conditions are
extracted as `truth: false` — nothing is marked resolved; a human reviews before
treating any matter as done.
