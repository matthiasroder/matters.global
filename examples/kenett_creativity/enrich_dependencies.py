"""Enrich the creativity matters graph with additional sensible dependencies.

Per-paper extraction could only see one paper at a time, so cross-paper
prerequisite relationships are under-captured. This script shows the model the
full inventory of every matter and asks, batch by batch (by dependent), for
additional genuine prerequisite edges. Each proposal is validated: both ids must
exist, no self-loops, no duplicates, and no cycles. Resumable via a per-batch
cache.
"""

import collections
import json
import os
import sys

import anthropic

from matters import load_state, save_state

GRAPH = "/Users/matthias/.local/share/matters/creativity.json"
PROV = "/Users/matthias/.local/share/matters/kenett/matter_sources.json"
CACHE_DIR = "/Users/matthias/.local/share/matters/kenett/enrich_cache"
ADDED_SIDECAR = "/Users/matthias/.local/share/matters/kenett/added_dependencies.json"
MODEL = os.environ.get("MATTERS_EXTRACT_MODEL", "claude-opus-4-8")
BATCH = 60


def log(msg):
    print(msg, file=sys.stderr, flush=True)


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
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text).get("dependencies", [])


def main():
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
    system = system_prompt(inventory_text(matter_ids, prov))

    added = []
    rejected = collections.Counter()
    batches = [matter_ids[i : i + BATCH] for i in range(0, len(matter_ids), BATCH)]

    for bi, batch in enumerate(batches):
        cache_path = os.path.join(CACHE_DIR, f"batch_{bi:03d}.json")
        if os.path.exists(cache_path):
            proposals = json.load(open(cache_path))
        else:
            batch_set = set(batch)
            existing_into = [
                f"{p} -> {d}" for (p, d) in existing if d in batch_set
            ]
            proposals = batch_proposals(client, system, batch, existing_into)
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(proposals, handle, indent=2)
            os.replace(tmp, cache_path)

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

    log("\n=== ENRICH DONE ===")
    log(f"  added {len(added)} dependencies; total now {len(final_deps)}")
    log(f"  rejected: {dict(rejected)}")
    print(json.dumps({"added": len(added), "total_deps": len(final_deps)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
