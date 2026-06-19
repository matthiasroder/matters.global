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
| `enrich_dependencies.py` | Second pass: shows the model the full matter inventory and asks for additional prerequisite edges the per-paper extraction couldn't see, validating ids, duplicates, and cycles before adding them. |
| `corpus.json` | 166 fetched papers (title, abstract, year, citation count, venue, DOI); 102 have abstracts. |
| `creativity_graph.json` | The assembled matters state (`schema_version: 2`), after dependency enrichment. |
| `matter_sources.json` | Provenance sidecar: each matter id → its source paper, year, DOI, and the model's name/description. |
| `added_dependencies.json` | The 854 edges added by the enrichment pass, each with the model's one-line reason. |
| `~/.local/share/matters/kenett/run_manifest.json` | Extraction-run audit manifest: corpus hash, selected papers, extraction-cache hashes, model, script hash, and output hashes. |
| `~/.local/share/matters/kenett/enrich_manifest.json` | Dependency-enrichment audit manifest: input graph hash, provenance hash, batch-cache hashes, rejected proposal counts, and final output hashes. |

## The graph

Built from the **102 abstracts** (extracted with `claude-sonnet-4-6`):

- **1,057 matters**, **2,672 conditions**, **1,917 dependencies** (1,063 from
  extraction + 854 from the enrichment pass)
- 599 resolved and 458 open matters; conditions are 1,916 true and 756 false
- Acyclic; 233 matters actionable now (no unresolved prerequisites)
- **468 / 1,917 dependencies link matters from different papers** — cross-corpus
  synthesis, not just within-paper structure
- Average degree 3.6; 1 isolated matter
- Hubs reflect Kenett's core program: semantic-memory structure ↔ verbal
  creativity, forward flow as a measure of free thought, semantic-network
  rigidity, individual semantic-network construction, and computational network
  science tools

## Explore it

```sh
matters web --state examples/kenett_creativity/creativity_graph.json
matters universe --state examples/kenett_creativity/creativity_graph.json
```

## Reproduce / extend

`pipeline.py` caches each paper's extraction under
`~/.local/share/matters/kenett/extractions/`, so a re-run resumes instead of
re-extracting. Cached proposals are reused only when they include `kind`,
`status`, condition `truth` values, and status/truth-consistent conditions;
stale unresolved-only cache files are ignored and re-extracted. Requires
`ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the environment;
`KENETT_EXTRACT_CAP` caps how many of the most-cited abstracts are processed
(default 25; this graph used 102 = all abstracts).

Current LLM extraction preserves source-grounded condition truth states:
resolved findings or delivered methods have true conditions, while open
questions, gaps, risks, or goals retain false conditions for what remains
unresolved. Fresh and cached proposals are accepted only when matter status and
condition truth states are consistent.

For reproducibility, treat stored LLM responses as the replayable source. A
model re-run may produce different text, but rebuilding from the cached
extraction and enrichment JSON should reproduce the same graph hashes recorded
in the manifests. Enrichment batch caches include a context hash over the model,
matter inventory, batch, and already-existing inbound dependencies, so stale
batch proposals are ignored after graph changes.
