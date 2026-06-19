"""Enrich the creativity matters graph with additional sensible dependencies.

Per-paper extraction could only see one paper at a time, so cross-paper
prerequisite relationships are under-captured. This script shows the model the
full inventory of every matter and asks, batch by batch (by dependent), for
additional genuine prerequisite edges. Each proposal is validated: both ids must
exist, no self-loops, no duplicates, and no cycles. Resumable via a per-batch
cache.
"""

import collections
import hashlib
import json
import os
import sys

import anthropic

from matters import load_state, save_state

GRAPH = "/Users/matthias/.local/share/matters/creativity.json"
PROV = "/Users/matthias/.local/share/matters/kenett/matter_sources.json"
CACHE_DIR = "/Users/matthias/.local/share/matters/kenett/enrich_cache"
ADDED_SIDECAR = "/Users/matthias/.local/share/matters/kenett/added_dependencies.json"
MANIFEST = "/Users/matthias/.local/share/matters/kenett/enrich_manifest.json"
MODEL = os.environ.get("MATTERS_EXTRACT_MODEL", "claude-opus-4-8")
BATCH = 60
REQUEST_TIMEOUT = float(os.environ.get("MATTERS_EXTRACT_TIMEOUT", "300"))


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


SCHEMA = {
    "type": "object",
    "properties": {
        "dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prerequisite": {"type": "string"},
                    "dependent": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["prerequisite", "dependent", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dependencies"],
    "additionalProperties": False,
}


def inventory_text(matter_ids, prov):
    lines = []
    for mid in matter_ids:
        pr = prov.get(mid, {})
        name = pr.get("name") or mid
        desc = (pr.get("description") or "").strip()
        lines.append(f"{mid} :: {name}" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def system_prompt(inv):
    return (
        "You are refining a dependency graph of research \"matters\" extracted "
        "from one creativity researcher's papers. A dependency (A, B) means A "
        "must be established or resolved before B can be — A is a conceptual or "
        "methodological PREREQUISITE for B. Good edges: a measure/method -> "
        "findings that rely on it; a foundational claim -> its extensions or "
        "applications; a general framework -> specific instances; evidence -> a "
        "claim that rests on that evidence; a building-block construct -> a "
        "composite one. Do NOT create an edge for mere topical similarity, "
        "co-occurrence, or two-way relatedness — only genuine, directional "
        "prerequisite relationships. Be willing to link matters that came from "
        "different papers. Use exact matter ids from the inventory. Never "
        "propose an edge that would create a cycle.\n\n"
        "FULL MATTER INVENTORY (id :: name — description):\n" + inv
    )


def has_path(start, goal, adj):
    stack = [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adj.get(node, ()))
    return False


def batch_proposals(client, system, batch, existing_into_batch):
    user = (
        "Propose ADDITIONAL dependencies whose DEPENDENT is one of the matters "
        "below. The prerequisite may be ANY matter id in the full inventory "
        "(including from other papers). Aim for genuine prerequisites; quality "
        "over quantity.\n\nDEPENDENT matters for this batch:\n"
        + "\n".join(f"- {b}" for b in batch)
        + "\n\nEdges that ALREADY exist into these (do not repropose):\n"
        + ("\n".join(existing_into_batch) or "(none)")
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=12000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        timeout=REQUEST_TIMEOUT,
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text).get("dependencies", [])


def cache_summary(cache_path):
    summary = {"path": cache_path, "exists": os.path.exists(cache_path)}
    if not summary["exists"]:
        return summary
    cached = json.load(open(cache_path))
    proposals = cached.get("proposals", []) if isinstance(cached, dict) else cached
    summary.update(
        {
            "sha256": file_hash(cache_path),
            "schema_version": (
                cached.get("schema_version") if isinstance(cached, dict) else None
            ),
            "context_sha256": (
                cached.get("context_sha256") if isinstance(cached, dict) else None
            ),
            "proposals": len(proposals),
        }
    )
    return summary


def cache_context(matter_inventory_sha, batch, existing_into):
    return stable_hash(
        {
            "model": MODEL,
            "matter_inventory_sha256": matter_inventory_sha,
            "batch": batch,
            "existing_into": sorted(existing_into),
        }
    )


def load_cached_proposals(cache_path, context_sha):
    if not os.path.exists(cache_path):
        return None
    cached = json.load(open(cache_path))
    if not isinstance(cached, dict):
        return None
    if cached.get("context_sha256") != context_sha:
        return None
    return cached.get("proposals", [])


def write_cached_proposals(cache_path, context_sha, proposals):
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(
            {
                "schema_version": 1,
                "context_sha256": context_sha,
                "proposals": proposals,
            },
            handle,
            indent=2,
        )
    os.replace(tmp, cache_path)


def write_manifest(
    *,
    graph_input_sha,
    prov_sha,
    matter_ids,
    dependencies_before,
    final_deps,
    added,
    rejected,
    cache_paths,
):
    manifest = {
        "schema_version": 1,
        "kind": "kenett_creativity_dependency_enrichment_run",
        "model": MODEL,
        "batch_size": BATCH,
        "request_timeout_seconds": REQUEST_TIMEOUT,
        "code": {
            "enrich_dependencies_sha256": file_hash(__file__),
        },
        "inputs": {
            "graph_path": GRAPH,
            "graph_input_sha256": graph_input_sha,
            "provenance_path": PROV,
            "provenance_sha256": prov_sha,
            "matter_inventory_count": len(matter_ids),
            "matter_inventory_sha256": stable_hash(matter_ids),
            "dependencies_before": len(dependencies_before),
        },
        "cache": [cache_summary(path) for path in cache_paths],
        "outputs": {
            "graph_path": GRAPH,
            "graph_output_sha256": file_hash(GRAPH),
            "added_dependencies_path": ADDED_SIDECAR,
            "added_dependencies_sha256": file_hash(ADDED_SIDECAR),
            "added_dependencies": len(added),
            "dependencies_after": len(final_deps),
            "rejected": dict(rejected),
        },
    }
    with open(MANIFEST, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def main():
    graph_input_sha = file_hash(GRAPH)
    prov_sha = file_hash(PROV)
    matters, conditions, dependencies = load_state(path=GRAPH)
    prov = json.load(open(PROV))
    matter_ids = sorted(matters)
    id_set = set(matter_ids)

    existing = set(tuple(edge) for edge in dependencies)
    adj = collections.defaultdict(list)
    for prereq, dependent in existing:
        adj[prereq].append(dependent)

    os.makedirs(CACHE_DIR, exist_ok=True)
    client = anthropic.Anthropic()
    inventory = inventory_text(matter_ids, prov)
    inventory_sha = stable_hash(inventory)
    system = system_prompt(inventory)

    added = []
    rejected = collections.Counter()
    batches = [matter_ids[i : i + BATCH] for i in range(0, len(matter_ids), BATCH)]
    cache_paths = []

    for bi, batch in enumerate(batches):
        cache_path = os.path.join(CACHE_DIR, f"batch_{bi:03d}.json")
        cache_paths.append(cache_path)
        batch_set = set(batch)
        existing_into = [
            f"{p} -> {d}" for (p, d) in existing if d in batch_set
        ]
        context_sha = cache_context(inventory_sha, batch, existing_into)
        proposals = load_cached_proposals(cache_path, context_sha)
        if proposals is None:
            proposals = batch_proposals(client, system, batch, existing_into)
            write_cached_proposals(cache_path, context_sha, proposals)

        kept = 0
        for dep in proposals:
            p, d = dep.get("prerequisite"), dep.get("dependent")
            if p not in id_set or d not in id_set:
                rejected["unknown_id"] += 1
                continue
            if p == d:
                rejected["self_loop"] += 1
                continue
            if (p, d) in existing:
                rejected["duplicate"] += 1
                continue
            if has_path(d, p, adj):  # d already reaches p -> adding p->d is a cycle
                rejected["cycle"] += 1
                continue
            existing.add((p, d))
            adj[p].append(d)
            added.append({"prerequisite": p, "dependent": d, "reason": dep.get("reason", "")})
            kept += 1
        log(f"  batch {bi + 1}/{len(batches)}: {len(proposals)} proposed, {kept} kept")

    final_deps = [list(edge) for edge in sorted(existing)]
    save_state(matters, conditions, final_deps, path=GRAPH)
    with open(ADDED_SIDECAR, "w") as handle:
        json.dump(added, handle, indent=2)
    manifest = write_manifest(
        graph_input_sha=graph_input_sha,
        prov_sha=prov_sha,
        matter_ids=matter_ids,
        dependencies_before=dependencies,
        final_deps=final_deps,
        added=added,
        rejected=rejected,
        cache_paths=cache_paths,
    )

    log("\n=== ENRICH DONE ===")
    log(f"  added {len(added)} dependencies; total now {len(final_deps)}")
    log(f"  rejected: {dict(rejected)}")
    log(f"  manifest: {MANIFEST}")
    print(json.dumps({
        "added": len(added),
        "total_deps": len(final_deps),
        "manifest_path": MANIFEST,
        "manifest_sha256": stable_hash(manifest),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
