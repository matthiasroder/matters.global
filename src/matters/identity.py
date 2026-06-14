"""Embedding-based matter identity.

Replaces slug first-write-wins dedup: a newly extracted matter is recognized as
an existing one when their *meanings* are close, not just when their slugs
collide. Embeddings come from a pluggable backend (local model2vec by default, a
deterministic fake for tests); nearest-neighbour search is brute-force cosine
over a persisted sidecar store. The borderline-similarity band is escalated to
the LLM, and with no embedder (or no LLM for the middle band) the logic degrades
safely toward slug-only behaviour.

Slice 1 is identity/dedup only: matches collapse duplicates and never mutate the
canonical matter's conditions. Cross-paper resolution (flipping condition truths
when later evidence resolves an open matter) is slice 2.
"""

import collections
import hashlib
import json
import math
import os
import re

DEFAULT_EMBED_MODEL = "minishlab/potion-retrieval-32M"
LLM_DEFAULT_MODEL = "claude-sonnet-4-6"
HIGH_THRESHOLD = 0.90
LOW_THRESHOLD = 0.78

Match = collections.namedtuple("Match", "action matter_id score vector")


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests and offline fallback.

    Hashes tokens into a fixed-dimension bag-of-words vector and L2-normalizes,
    using a stable hash so results are identical across processes. Texts that
    share most tokens land close; unrelated texts do not.
    """

    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, texts):
        return [self._one(text) for text in texts]

    def _one(self, text):
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        return _normalize(vec)


class LocalEmbedder:
    """model2vec static-embedding backend (lazy import, lazy model load)."""

    def __init__(self, model_name=None):
        self.model_name = (
            model_name or os.environ.get("MATTERS_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        )
        self._model = None

    def _load(self):
        if self._model is None:
            from model2vec import StaticModel  # lazy: only needed for real runs

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def embed(self, texts):
        vectors = self._load().encode(list(texts))
        return [_normalize([float(x) for x in vector]) for vector in vectors]


def get_embedder():
    """Return a LocalEmbedder if model2vec is importable, else None.

    The model itself is loaded lazily on first ``embed``; callers should treat a
    raised embedding error as "fall back to slug" (``ingest_candidates`` does).
    """
    import importlib.util

    if importlib.util.find_spec("model2vec") is None:
        return None
    return LocalEmbedder()


# --------------------------------------------------------------------------- #
# Embedding store (sidecar + brute-force nearest)
# --------------------------------------------------------------------------- #


class EmbeddingStore:
    """Persisted id -> (vector, text) store with brute-force cosine search.

    Vectors are kept L2-normalized so cosine similarity is a plain dot product.
    Backed by a NumPy ``.npz`` sidecar; swap ``nearest`` for an ANN index at
    scale without changing callers.
    """

    def __init__(self, path=None):
        self.path = path
        self._ids = []
        self._texts = []
        self._kinds = []
        self._statuses = []
        self._index = {}
        self._matrix = None  # np.ndarray (n, dim) float32, normalized rows

    @classmethod
    def load(cls, path):
        store = cls(path)
        if path and os.path.exists(path):
            import numpy as np

            data = np.load(path, allow_pickle=False)
            matrix = data["vectors"].astype("float32")
            store._ids = [str(x) for x in data["ids"]]
            store._texts = [str(x) for x in data["texts"]]
            store._kinds = (
                [str(x) for x in data["kinds"]]
                if "kinds" in data.files
                else [""] * len(store._ids)
            )
            store._statuses = (
                [str(x) for x in data["statuses"]]
                if "statuses" in data.files
                else [""] * len(store._ids)
            )
            store._matrix = matrix if matrix.size else None
            store._index = {mid: i for i, mid in enumerate(store._ids)}
        return store

    def __len__(self):
        return len(self._ids)

    def __contains__(self, matter_id):
        return matter_id in self._index

    def text_of(self, matter_id):
        idx = self._index.get(matter_id)
        return self._texts[idx] if idx is not None else ""

    def meta_of(self, matter_id):
        """Return (kind, status) recorded for a matter, or ("", "") if unknown."""
        idx = self._index.get(matter_id)
        if idx is None:
            return ("", "")
        return (self._kinds[idx], self._statuses[idx])

    def add(self, matter_id, vector, text="", kind="", status=""):
        import numpy as np

        row = _np_normalize(np.asarray(vector, dtype="float32"))
        if matter_id in self._index:
            i = self._index[matter_id]
            self._matrix[i] = row
            self._texts[i] = text
            self._kinds[i] = kind
            self._statuses[i] = status
            return
        self._index[matter_id] = len(self._ids)
        self._ids.append(matter_id)
        self._texts.append(text)
        self._kinds.append(kind)
        self._statuses.append(status)
        self._matrix = (
            row[None, :] if self._matrix is None else np.vstack([self._matrix, row])
        )

    def nearest(self, vector, top_k=10):
        if self._matrix is None or not self._ids:
            return []
        import numpy as np

        query = _np_normalize(np.asarray(vector, dtype="float32"))
        sims = self._matrix @ query
        k = min(top_k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._ids[i], float(sims[i])) for i in top]

    def save(self, path=None):
        import numpy as np

        target = path or self.path
        if not target:
            raise ValueError("EmbeddingStore.save requires a path")
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        matrix = (
            self._matrix
            if self._matrix is not None
            else np.zeros((0, 0), dtype="float32")
        )
        np.savez(
            target,
            vectors=matrix,
            ids=np.array(self._ids, dtype=str),
            texts=np.array(self._texts, dtype=str),
            kinds=np.array(self._kinds, dtype=str),
            statuses=np.array(self._statuses, dtype=str),
        )


# --------------------------------------------------------------------------- #
# Matching and ingestion
# --------------------------------------------------------------------------- #


def match_candidate(
    text,
    store,
    embedder,
    *,
    high=HIGH_THRESHOLD,
    low=LOW_THRESHOLD,
    llm_client=None,
    model=None,
    top_k=10,
):
    """Decide whether ``text`` is an existing matter.

    Returns a Match: ``("merge", id, score, vector)`` if it is the same as an
    existing matter, else ``("new", None, score, vector)``. ``vector`` is the
    computed embedding (or None on failure) so callers can reuse it.
    """
    if embedder is None or store is None:
        return Match("new", None, 0.0, None)
    try:
        vector = embedder.embed([text])[0]
    except Exception:  # noqa: BLE001 - embedding failure degrades to slug-only
        return Match("new", None, 0.0, None)

    neighbours = store.nearest(vector, top_k=top_k)
    if not neighbours:
        return Match("new", None, 0.0, vector)

    best_id, best_score = neighbours[0]
    if best_score >= high:
        return Match("merge", best_id, best_score, vector)
    if best_score < low or llm_client is None:
        return Match("new", None, best_score, vector)

    # Borderline band: ask the LLM whether any in-band neighbour is the same.
    for cand_id, score in neighbours:
        if score < low:
            break
        if _llm_same_matter(text, store.text_of(cand_id), llm_client, model):
            return Match("merge", cand_id, score, vector)
    return Match("new", None, best_score, vector)


def ingest_candidates(
    candidates,
    matters,
    conditions,
    store,
    embedder,
    *,
    llm_client=None,
    high=HIGH_THRESHOLD,
    low=LOW_THRESHOLD,
    model=None,
):
    """Merge candidate matters into the graph by identity, returning an id map.

    For each candidate: an exact slug collision or a semantic match maps it to
    the canonical id (no node added, conditions untouched); otherwise it becomes
    a new matter whose embedding is stored. The returned ``id_map`` (candidate id
    -> canonical id) lets callers repoint dependencies and merge provenance.
    """
    matters = set(matters)
    conditions = dict(conditions)
    id_map = {}

    for candidate in candidates:
        cid = candidate["id"]
        text = _candidate_text(candidate)

        if cid in matters:  # exact slug collision -> same matter
            id_map[cid] = cid
            continue

        match = match_candidate(
            text, store, embedder, high=high, low=low, llm_client=llm_client, model=model
        )
        if match.action == "merge" and match.matter_id in matters:
            id_map[cid] = match.matter_id
            continue

        matters.add(cid)
        conditions[cid] = candidate["conditions"]
        id_map[cid] = cid
        if match.vector is not None:
            try:
                store.add(cid, match.vector, text)
            except Exception:  # noqa: BLE001 - never let store growth break ingest
                pass

    return matters, conditions, id_map


SAME_MATTER_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
    "additionalProperties": False,
}

SAME_MATTER_SYSTEM = (
    "You decide whether two research 'matters' refer to the same thing. They are "
    "the same when they assert, ask, or aim at the same thing about the same "
    "subject, even if worded very differently. Different wording is fine. "
    "Opposite or distinct claims about the same topic (e.g. 'X works' vs 'X "
    "fails', or a method vs a finding that uses it) are NOT the same."
)


def _llm_same_matter(text_a, text_b, client, model=None):
    model = model or os.environ.get("MATTERS_EXTRACT_MODEL") or LLM_DEFAULT_MODEL
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=SAME_MATTER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Matter A: {text_a}\n\nMatter B: {text_b}\n\n"
                "Are A and B the same matter?",
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": SAME_MATTER_SCHEMA}
        },
    )
    text = next(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    return bool(json.loads(text).get("same"))


def _candidate_text(candidate):
    name = str(candidate.get("name") or candidate.get("id") or "").strip()
    description = str(candidate.get("description") or "").strip()
    return f"{name} — {description}" if description else name


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _np_normalize(vector):
    import numpy as np

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


# --------------------------------------------------------------------------- #
# Slice 2 — relationship-aware reconciliation
# --------------------------------------------------------------------------- #

CANDIDATE_FLOOR = 0.55  # widened net; the classifier is the precision gate

Relationship = collections.namedtuple("Relationship", "relation direction satisfied reason")

QUESTION_KINDS = {"question", "problem", "concern", "goal", "risk"}


def _derived_status(conditions):
    return "resolved" if conditions and all(c.get("truth") for c in conditions) else "open"


def _roles_conflict(new_kind, new_status, existing_kind, existing_status):
    """True when two matters play different roles and must not merge as 'same'.

    A deterministic backstop to the classifier: a resolved finding is never the
    same matter as an open question about it, and a problem is never the same as
    its solution, regardless of what the LLM proposes.
    """
    ns, es = (new_status or "").lower(), (existing_status or "").lower()
    if ns in ("resolved", "open") and es in ("resolved", "open") and ns != es:
        return True
    nk, ek = (new_kind or "").lower(), (existing_kind or "").lower()
    if nk and ek:
        if (nk in QUESTION_KINDS) != (ek in QUESTION_KINDS):
            return True
        if {nk, ek} == {"method", "finding"}:
            return True
    return False


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": ["same", "resolves", "link", "distinct"]},
        "direction": {
            "type": "string",
            "enum": ["new_before_existing", "existing_before_new", "none"],
        },
        "satisfied_condition_indices": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["relation", "direction", "satisfied_condition_indices", "reason"],
    "additionalProperties": False,
}

CLASSIFY_SYSTEM = (
    "You compare a NEW matter to an EXISTING matter in a research knowledge graph "
    "and classify their relationship. A 'matter' is a tracked claim, finding, "
    "method, problem, question, or goal with a resolution status. Choose exactly "
    "one relation:\n"
    "- same: the SAME matter — the same claim/question/method about the same "
    "subject, differing only in wording. Do NOT call them the same if they play "
    "different roles or have different status: a problem is NOT the same as its "
    "solution; a resolved finding is NOT the same as an open question about it; a "
    "method is NOT the same as a finding that uses it.\n"
    "- resolves: the NEW matter establishes or answers what the EXISTING matter "
    "(which must be OPEN) needs. Set satisfied_condition_indices to the 1-based "
    "indices of the EXISTING conditions the NEW matter now makes true. Only on "
    "clear, direct evidence.\n"
    "- link: complementary but distinct — e.g. a problem and its solution, a "
    "method and a finding that uses it, evidence and a claim that rests on it, a "
    "finding and an open question it raises. Set direction to which must be "
    "established first: 'new_before_existing' if NEW is the prerequisite, "
    "'existing_before_new' if EXISTING is, 'none' if neither truly is.\n"
    "- distinct: merely a similar topic.\n"
    "Be conservative: prefer link or distinct over same; prefer distinct over "
    "resolves unless evidence is clear. Use direction='none' and "
    "satisfied_condition_indices=[] when not applicable."
)


def classify_relationship(
    new_text, new_status, existing_text, existing_conditions, llm_client, model=None
):
    """Classify how a new matter relates to one existing matter (LLM-judged)."""
    model = model or os.environ.get("MATTERS_EXTRACT_MODEL") or LLM_DEFAULT_MODEL
    resolved = bool(existing_conditions) and all(
        c.get("truth") for c in existing_conditions
    )
    cond_lines = (
        "\n".join(
            f"  {i}. [{'met' if c.get('truth') else 'unmet'}] {c.get('label', '')}"
            for i, c in enumerate(existing_conditions, 1)
        )
        or "  (none)"
    )
    user = (
        f"NEW matter (status as extracted: {new_status or 'unknown'}):\n{new_text}\n\n"
        f"EXISTING matter (status: {'resolved' if resolved else 'open'}):\n{existing_text}\n"
        f"EXISTING conditions:\n{cond_lines}\n\n"
        "Classify the relationship of NEW to EXISTING."
    )
    response = llm_client.messages.create(
        model=model,
        max_tokens=400,
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": CLASSIFY_SCHEMA}},
    )
    text = next(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    )
    data = json.loads(text)
    return Relationship(
        data.get("relation", "distinct"),
        data.get("direction", "none"),
        [int(i) for i in data.get("satisfied_condition_indices", [])],
        data.get("reason", ""),
    )


def reconcile_candidates(
    candidates,
    matters,
    conditions,
    store,
    embedder,
    *,
    llm_client=None,
    floor=CANDIDATE_FLOOR,
    top_k=3,
    model=None,
):
    """Merge / resolve / link / keep new matters against the existing graph.

    Returns (matters, conditions, id_map, new_edges, flips):
      - SAME      -> id_map[new] = canonical (no node added)
      - RESOLVES  -> flip the existing OPEN matter's satisfied conditions to true,
                     add a prerequisite edge new -> existing
      - LINK      -> add a directed dependency edge between the two matters
      - DISTINCT  -> the new matter is added on its own
    new_edges is a list of (prerequisite, dependent); flips records what changed.
    Without an llm_client, only very-high-similarity SAME merges happen.
    """
    matters = set(matters)
    conditions = dict(conditions)
    id_map = {}
    new_edges = []
    flips = []

    for candidate in candidates:
        cid = candidate["id"]
        text = _candidate_text(candidate)
        if cid in matters:
            id_map[cid] = cid
            continue

        vector = None
        neighbours = []
        if embedder is not None and store is not None:
            try:
                vector = embedder.embed([text])[0]
            except Exception:  # noqa: BLE001 - embedding failure degrades to slug
                vector = None
            if vector is not None:
                neighbours = store.nearest(vector, top_k=top_k)
        considered = [(nid, sc) for nid, sc in neighbours if sc >= floor and nid in matters]

        results = []
        for nid, score in considered:
            if llm_client is None:
                relation = "same" if score >= HIGH_THRESHOLD else "distinct"
                results.append((nid, Relationship(relation, "none", [], "")))
            else:
                results.append(
                    (
                        nid,
                        classify_relationship(
                            text,
                            candidate.get("status", ""),
                            store.text_of(nid),
                            conditions.get(nid, []),
                            llm_client,
                            model,
                        ),
                    )
                )

        same = []
        for nid, rel in results:
            if rel.relation != "same":
                continue
            if _roles_conflict(
                candidate.get("kind"),
                candidate.get("status"),
                store.meta_of(nid)[0],
                _derived_status(conditions.get(nid, [])),
            ):
                continue  # role/status guard: mismatched roles are not the same matter
            same.append(nid)
        if same:
            id_map[cid] = same[0]
            continue

        matters.add(cid)
        conditions[cid] = candidate["conditions"]
        id_map[cid] = cid
        if vector is not None:
            try:
                store.add(
                    cid,
                    vector,
                    text,
                    candidate.get("kind", ""),
                    candidate.get("status", ""),
                )
            except Exception:  # noqa: BLE001
                pass

        for nid, rel in results:
            if rel.relation == "resolves":
                existing = conditions.get(nid)
                if existing and not all(c.get("truth") for c in existing):
                    conds = list(existing)
                    flipped = []
                    for idx in rel.satisfied:
                        if 1 <= idx <= len(conds) and not conds[idx - 1].get("truth"):
                            conds[idx - 1] = {**conds[idx - 1], "truth": True}
                            flipped.append(idx)
                    if flipped:
                        conditions[nid] = conds
                        new_edges.append((cid, nid))
                        flips.append({"matter": nid, "by": cid, "conditions": flipped})
            elif rel.relation == "link":
                if rel.direction == "new_before_existing":
                    new_edges.append((cid, nid))
                elif rel.direction == "existing_before_new":
                    new_edges.append((nid, cid))

    return matters, conditions, id_map, new_edges, flips
