# Creativity Research Extraction Corpus

This corpus is a small fixture set for testing whether `matters extract` can turn
creativity research material into useful candidate matters.

The corpus intentionally mixes source styles:

- `research_notes.txt`: structured notes from a research planning session.
- `conversation.txt`: an AI/human planning conversation with speaker prefixes.
- `paper_excerpt.txt`: prose-like research text with explicit matter markers.
- `expected.json`: expected high-level matters, condition quality expectations,
  and dependency-review expectations.
- `initial_graph_proposal.json`: a reviewed proposal that can be accepted,
  edited, or rejected before writing anything into a live matters state file.

The fixtures are not meant to be a complete creativity research graph. They are a
minimum bar for testing whether extraction output is specific enough to start
building one.

Run manually from the repository root:

```sh
matters extract examples/creativity_research/research_notes.txt --source-type notes --state /Users/matthias/.local/share/matters/matters.json
matters extract examples/creativity_research/conversation.txt --source-type conversation --state /Users/matthias/.local/share/matters/matters.json
matters extract examples/creativity_research/paper_excerpt.txt --source-type paper --state /Users/matthias/.local/share/matters/matters.json
```
