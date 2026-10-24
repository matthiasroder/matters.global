# Creativity Matters Graph

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
| `~/.local/share/matters/creativity_matters_graph/run_manifest.json` | Extraction-run audit manifest: corpus hash, selected papers, extraction-cache hashes, model, script hash, and output hashes. |
| `~/.local/share/matters/creativity_matters_graph/enrich_manifest.json` | Dependency-enrichment audit manifest: input graph hash, provenance hash, batch-cache hashes, rejected proposal counts, and final output hashes. |

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
matters web --state examples/creativity_matters_graph/creativity_graph.json
matters universe --state examples/creativity_matters_graph/creativity_graph.json
```

## Reproduce / extend

`pipeline.py` caches each paper's extraction under
`~/.local/share/matters/creativity_matters_graph/extractions/`, so a re-run resumes instead of
re-extracting. Cached proposals must include `kind`, `status`, boolean condition
truth values, and status/truth-consistent conditions. Reuse also requires the same
source text, provider/model/profile, extraction prompt and schema, and existing
matter inventory. Each cache records those inputs and their fingerprint; changes
trigger fresh extraction. Older caches without fingerprints are re-extracted.
Changes to citation counts alone do not invalidate extraction. Requires
`ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) in the environment;
`CREATIVITY_MATTERS_EXTRACT_CAP` caps how many of the most-cited abstracts are
processed (default 25; this graph used 102 = all abstracts). `KENETT_EXTRACT_CAP`
is still accepted as a legacy alias.

Current LLM extraction preserves source-grounded condition truth states:
resolved findings or delivered methods have true conditions, while open
questions, gaps, risks, or goals retain false conditions for what remains
unresolved. Fresh and cached proposals are accepted only when matter status and
condition truth states are consistent.

For an explicit historical replay, set `CREATIVITY_MATTERS_REPLAY_CACHE=1` when
running `pipeline.py`. Replay uses valid cached responses without selecting or
calling a model, and fails if a selected paper has no valid cache. It preserves
the cache bytes and records `cache_mode: "replay"` in the manifest. Provenance
uses the original cached paper metadata; legacy caches without that metadata
are marked `source_verified: false` rather than attributing old claims to the
current text. A cached corpus is still needed to select the papers without a
Semantic Scholar fetch. Normal runs validate inputs; replay deliberately retains
the old source/model outputs. A model re-run may produce different text.

Enrichment batch caches include a context hash over the model,
matter inventory, batch, and already-existing inbound dependencies, so stale
batch proposals are ignored after graph changes.
