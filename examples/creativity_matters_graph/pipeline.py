"""Compile Yoed N. Kenett's paper corpus and build a creativity matters graph.

Step A (no key): fetch the author's papers from Semantic Scholar, keep those
with abstracts, write corpus.json (cached).
Step B (needs Anthropic key): run the LLM matters extractor over selected
abstracts, accumulate candidates + dependencies, write a dedicated matters
state file plus a provenance sidecar.
"""

import json
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from matters import build_extraction_proposal, save_state
from matters.llm import resolve_generator
from matters.llm_extraction import DEFAULT_MODEL, EXTRACTION_SCHEMA, SYSTEM_PROMPT

AUTHOR_ID = "3000599"  # Yoed N. Kenett (Technion), h-index 43
WORK_DIR = os.environ.get(
    "CREATIVITY_MATTERS_WORK_DIR",
    "/Users/matthias/.local/share/matters/creativity_matters_graph",
)
CORPUS_PATH = os.path.join(WORK_DIR, "corpus.json")
GRAPH_PATH = "/Users/matthias/.local/share/matters/creativity.json"
PROVENANCE_PATH = os.path.join(WORK_DIR, "matter_sources.json")
MANIFEST_PATH = os.path.join(WORK_DIR, "run_manifest.json")
EXTRACT_DIR = os.path.join(WORK_DIR, "extractions")
EXTRACT_CAP = int(
    os.environ.get(
        "CREATIVITY_MATTERS_EXTRACT_CAP",
        os.environ.get("KENETT_EXTRACT_CAP", "25"),
    )
)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def stable_hash(data):
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s2_get(url):
    headers = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=40
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (403, 429) and headers:
                headers = {}  # drop the key and retry anonymously
            if error.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if attempt < 5:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Semantic Scholar request failed: {url}")


def fetch_corpus():
    papers = []
    offset = 0
    fields = "title,abstract,year,citationCount,venue,externalIds"
    while True:
        url = (
            f"https://api.semanticscholar.org/graph/v1/author/{AUTHOR_ID}/papers?"
            + urllib.parse.urlencode({"fields": fields, "limit": 100, "offset": offset})
        )
        batch = s2_get(url)
        rows = batch.get("data", [])
        papers.extend(rows)
        log(f"  fetched {len(papers)} papers...")
        if "next" not in batch or not rows:
            break
        offset = batch["next"]
        time.sleep(1.0)
    cleaned = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        cleaned.append(
            {
                "paperId": paper.get("paperId"),
                "title": title,
                "abstract": abstract,
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount") or 0,
                "venue": paper.get("venue"),
                "doi": (paper.get("externalIds") or {}).get("DOI"),
            }
        )
    return cleaned


def load_or_fetch_corpus():
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH) as handle:
            corpus = json.load(handle)
        log(f"Loaded cached corpus: {len(corpus)} papers")
        return corpus
    log("Fetching corpus from Semantic Scholar...")
    corpus = fetch_corpus()
    with open(CORPUS_PATH, "w") as handle:
        json.dump(corpus, handle, indent=2)
    log(f"Saved corpus.json: {len(corpus)} papers")
    return corpus


def select_for_extraction(corpus):
    with_abstract = [p for p in corpus if p["abstract"]]
    with_abstract.sort(key=lambda p: p["citationCount"], reverse=True)
    return with_abstract[:EXTRACT_CAP]


def summarize_extraction_cache(paper):
    cache_path = os.path.join(EXTRACT_DIR, f"{paper['paperId']}.json")
    summary = {
        "paperId": paper["paperId"],
        "path": cache_path,
        "exists": os.path.exists(cache_path),
    }
    if not summary["exists"]:
        return summary

    with open(cache_path) as handle:
        proposal = json.load(handle)

    status_counts = {}
    truth_counts = {"true": 0, "false": 0}
    for candidate in proposal.get("candidates", []):
        status = candidate.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
        for condition in candidate.get("conditions", []):
            key = "true" if condition.get("truth") else "false"
            truth_counts[key] += 1

    summary.update(
        {
            "sha256": file_hash(cache_path),
            "current": cache_is_current(proposal),
            "engine": proposal.get("engine"),
            "model": proposal.get("model"),
            "candidates": len(proposal.get("candidates", [])),
            "dependency_candidates": len(proposal.get("dependency_candidates", [])),
            "status_counts": status_counts,
            "condition_truth_counts": truth_counts,
            "cache_inputs": proposal.get("cache_inputs"),
            "input_sha256": proposal.get("input_sha256"),
            "source_matches_current_paper": (proposal.get("cache_inputs") or {}).get("source_sha256")
            == stable_hash(f"{paper['title']}\n\n{paper['abstract']}"),
        }
    )
    return summary


def condition_truth_counts(conditions):
    counts = {"true": 0, "false": 0}
    for condition_list in conditions.values():
        for condition in condition_list:
            key = "true" if condition.get("truth") else "false"
            counts[key] += 1
    return counts


def write_manifest(corpus, selected, matters, conditions, edges, provenance, *, replay_cache=False):
    with_abstract = [paper for paper in corpus if paper["abstract"]]
    manifest = {
        "schema_version": 1,
        "kind": "creativity_matters_graph_extraction_run",
        "author_id": AUTHOR_ID,
        "cache_mode": "replay" if replay_cache else "validate_inputs",
        "extract_cap": EXTRACT_CAP,
        "models": {
            "extraction": os.environ.get("MATTERS_EXTRACT_MODEL") or DEFAULT_MODEL,
        },
        "code": {
            "pipeline_sha256": file_hash(__file__),
        },
        "corpus": {
            "path": CORPUS_PATH,
            "sha256": file_hash(CORPUS_PATH),
            "papers_total": len(corpus),
            "papers_with_abstracts": len(with_abstract),
        },
        "selected_papers": [
            {
                "paperId": paper["paperId"],
                "title": paper["title"],
                "year": paper["year"],
                "citationCount": paper["citationCount"],
                "doi": paper["doi"],
                "abstract_sha256": stable_hash(paper["abstract"]),
            }
            for paper in selected
        ],
        "extraction_cache": [summarize_extraction_cache(paper) for paper in selected],
        "outputs": {
            "graph_path": GRAPH_PATH,
            "graph_sha256": file_hash(GRAPH_PATH),
            "provenance_path": PROVENANCE_PATH,
            "provenance_sha256": file_hash(PROVENANCE_PATH),
            "matters": len(matters),
            "conditions": sum(len(items) for items in conditions.values()),
            "condition_truth_counts": condition_truth_counts(conditions),
            "dependencies": len(edges),
            "provenance_entries": len(provenance),
        },
    }
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def has_path(edges, start, goal):
    stack = [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(d for (p, d) in edges if p == node)
    return False


def extraction_cache_inputs(paper, client, existing_matters):
    """Describe all semantic inputs; credentials are never cache metadata."""
    selection = resolve_generator("extraction", injected=client)
    model = None
    if selection is not None:
        with selection:
            model = {"provider": selection.provider, "profile": selection.profile,
                     "model": selection.model}
    return {
        "schema_version": 1,
        "source_type": "paper",
        "source_sha256": stable_hash(f"{paper['title']}\n\n{paper['abstract']}"),
        "paper": dict(paper),
        "model": model,
        "prompt_sha256": stable_hash(SYSTEM_PROMPT),
        "schema_sha256": stable_hash(EXTRACTION_SCHEMA),
        "existing_matters": sorted(existing_matters),
    }


def extract_paper(paper, client, existing_matters, *, replay_cache=False):
    """Return (proposal, was_cached). Caches each paper's LLM proposal so an
    interrupted run resumes instead of re-extracting, and a single failing call
    is retried then skipped rather than killing the whole run."""
    cache_path = os.path.join(EXTRACT_DIR, f"{paper['paperId']}.json")
    inputs = None if replay_cache else extraction_cache_inputs(paper, client, existing_matters)
    fingerprint = (
        stable_hash({key: value for key, value in inputs.items() if key != "paper"})
        if inputs is not None else None
    )
    if os.path.exists(cache_path):
        with open(cache_path) as handle:
            proposal = json.load(handle)
        if cache_is_current(proposal) and (replay_cache or proposal.get("input_sha256") == fingerprint):
            return proposal, True
        log(f"      ignoring stale extraction cache: {paper['paperId']}")
    if replay_cache:
        raise ValueError(f"no valid extraction cache to replay for {paper['paperId']}")

    last_reason = None
    for attempt in range(4):
        proposal = build_extraction_proposal(
            f"{paper['title']}\n\n{paper['abstract']}",
            source_type="paper",
            existing_matters=existing_matters,
            client=client,
        )
        if proposal.get("engine") == "llm" and cache_is_current(proposal):
            proposal = {**proposal, "cache_inputs": inputs, "input_sha256": fingerprint}
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(proposal, handle, indent=2)
            os.replace(tmp, cache_path)
            return proposal, False
        last_reason = proposal.get("fallback_reason") or (
            "llm proposal failed status/truth validation"
        )
        log(f"      retry {attempt + 1}/4 after fallback: {last_reason}")
        time.sleep(3 * (attempt + 1))

    log(f"      SKIPPED (extractor kept falling back: {last_reason})")
    return None, False


def cache_is_current(proposal):
    if proposal.get("engine") != "llm":
        return False
    for candidate in proposal.get("candidates", []):
        status = candidate.get("status")
        if status not in {"resolved", "open"}:
            return False
        if not candidate.get("kind"):
            return False
        condition_truths = []
        for condition in candidate.get("conditions", []):
            if not isinstance(condition, dict) or not isinstance(condition.get("truth"), bool):
                return False
            condition_truths.append(condition["truth"])
        if status == "resolved" and not all(condition_truths):
            return False
        if status == "open" and condition_truths and all(condition_truths):
            return False
    return True


def build_graph(selected, client, *, replay_cache=False):
    matters = []
    conditions = {}
    provenance = {}
    pending_deps = []
    cached = 0

    for index, paper in enumerate(selected, 1):
        if replay_cache:
            proposal, was_cached = extract_paper(paper, client, matters, replay_cache=True)
        else:
            proposal, was_cached = extract_paper(paper, client, matters)
        cached += int(was_cached)
        log(
            f"  [{index}/{len(selected)}] "
            f"{'cached' if was_cached else 'extracted'}: {paper['title'][:64]}"
        )
        if proposal is None:
            continue
        recorded_paper = (proposal.get("cache_inputs") or {}).get("paper")
        source_paper = recorded_paper or ({} if replay_cache else paper)
        for candidate in proposal["candidates"]:
            cid = candidate["id"]
            if cid in conditions:
                continue
            matters.append(cid)
            conditions[cid] = candidate["conditions"]
            provenance[cid] = {
                "name": candidate["name"],
                "description": candidate.get("description", ""),
                "kind": candidate.get("kind", ""),
                "status": candidate.get("status", ""),
                "source_type": candidate.get("source_type", ""),
                "paper": source_paper.get("title"),
                "paperId": paper["paperId"],
                "year": source_paper.get("year"),
                "doi": source_paper.get("doi"),
                "extraction_input_sha256": proposal.get("input_sha256"),
                "cache_replay": replay_cache,
                "source_verified": recorded_paper is not None or not replay_cache,
            }
        pending_deps.extend(proposal.get("dependency_candidates", []))
        if not was_cached:
            time.sleep(0.3)

    log(f"  ({cached} loaded from cache, {len(selected) - cached} freshly extracted)")
    matter_set = set(matters)
    edges = []
    for dep in pending_deps:
        prereq, dependent = dep["prerequisite"], dep["dependent"]
        if prereq not in matter_set or dependent not in matter_set:
            continue
        if prereq == dependent or [prereq, dependent] in edges:
            continue
        if has_path(edges, dependent, prereq):  # would create a cycle
            continue
        edges.append([prereq, dependent])

    return matters, conditions, edges, provenance


def main():
    corpus = load_or_fetch_corpus()
    with_abstract = [p for p in corpus if p["abstract"]]
    log(
        f"Corpus: {len(corpus)} papers total, {len(with_abstract)} with abstracts."
    )

    replay_cache = os.environ.get("CREATIVITY_MATTERS_REPLAY_CACHE") == "1"
    if not replay_cache and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        log("\nNO ANTHROPIC KEY in environment — corpus is ready, extraction skipped.")
        log("Set ANTHROPIC_API_KEY and re-run this script to build the graph.")
        return 0

    selected = select_for_extraction(corpus)
    log(f"\nExtracting matters from top {len(selected)} papers by citation count...")
    if replay_cache:
        matters, conditions, edges, provenance = build_graph(selected, None, replay_cache=True)
    else:
        import anthropic

        with anthropic.Anthropic() as client:
            matters, conditions, edges, provenance = build_graph(selected, client)

    save_state(matters, conditions, edges, path=GRAPH_PATH)
    with open(PROVENANCE_PATH, "w") as handle:
        json.dump(provenance, handle, indent=2)
    manifest = write_manifest(corpus, selected, matters, conditions, edges, provenance,
                              replay_cache=replay_cache)

    log("\n=== DONE ===")
    log(f"Graph: {GRAPH_PATH}")
    log(f"  matters: {len(matters)}")
    log(f"  dependencies: {len(edges)}")
    log(f"Provenance sidecar: {PROVENANCE_PATH}")
    log(f"Run manifest: {MANIFEST_PATH}")
    print(json.dumps({
        "matters": len(matters),
        "dependencies": len(edges),
        "papers_extracted": len(selected),
        "graph_path": GRAPH_PATH,
        "manifest_path": MANIFEST_PATH,
        "manifest_sha256": stable_hash(manifest),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
