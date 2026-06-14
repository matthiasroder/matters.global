import json
from types import SimpleNamespace

from matters import (
    EmbeddingStore,
    FakeEmbedder,
    ingest_candidates,
    match_candidate,
    reconcile_candidates,
)
from matters.identity import _candidate_text


class StubEmbedder:
    """Returns preset vectors per exact text, for precise threshold control."""

    def __init__(self, mapping):
        self.mapping = mapping

    def embed(self, texts):
        return [self.mapping[t] for t in texts]


def fake_llm(same):
    def create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({"same": same}))]
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def cand(cid, name, conditions=None):
    return {"id": cid, "name": name, "conditions": conditions or [{"label": "c", "truth": False}]}


# Vectors chosen so cosines land in known bands (high>=0.90, low<0.78):
A = cand("assoc_correlation_method", "Association-correlation method")
APRIME = cand("assoc_cloud_method", "Association-cloud method")  # reworded dup of A
B = cand("forward_flow_metric", "Forward flow metric")  # distinct
C = cand("assoc_overlap_method", "Association-overlap method")  # middle band vs A

VECS = {
    _candidate_text(A): [1.0, 0.0, 0.0],
    _candidate_text(APRIME): [0.98, 0.2, 0.0],   # cosine ~0.98 with A  -> high
    _candidate_text(B): [0.0, 1.0, 0.0],         # cosine 0 with A      -> low
    _candidate_text(C): [0.8, 0.6, 0.0],         # cosine 0.8 with A    -> middle
}


def new_store():
    return EmbeddingStore()


def test_reworded_duplicate_merges_within_batch():
    matters, conditions, id_map = ingest_candidates(
        [A, APRIME], set(), {}, new_store(), StubEmbedder(VECS)
    )
    assert matters == {"assoc_correlation_method"}
    assert id_map == {
        "assoc_correlation_method": "assoc_correlation_method",
        "assoc_cloud_method": "assoc_correlation_method",  # merged into canonical
    }
    assert "assoc_cloud_method" not in conditions


def test_distinct_matters_stay_separate():
    matters, _, id_map = ingest_candidates([A, B], set(), {}, new_store(), StubEmbedder(VECS))
    assert matters == {"assoc_correlation_method", "forward_flow_metric"}
    assert id_map["forward_flow_metric"] == "forward_flow_metric"


def test_middle_band_merges_only_when_llm_says_same():
    merged, _, id_map = ingest_candidates(
        [A, C], set(), {}, new_store(), StubEmbedder(VECS), llm_client=fake_llm(True)
    )
    assert id_map["assoc_overlap_method"] == "assoc_correlation_method"
    assert merged == {"assoc_correlation_method"}

    split, _, id_map2 = ingest_candidates(
        [A, C], set(), {}, new_store(), StubEmbedder(VECS), llm_client=fake_llm(False)
    )
    assert id_map2["assoc_overlap_method"] == "assoc_overlap_method"
    assert split == {"assoc_correlation_method", "assoc_overlap_method"}


def test_middle_band_without_llm_stays_separate():
    matters, _, _ = ingest_candidates(
        [A, C], set(), {}, new_store(), StubEmbedder(VECS), llm_client=None
    )
    assert matters == {"assoc_correlation_method", "assoc_overlap_method"}


def test_slug_collision_merges():
    _, _, id_map = ingest_candidates(
        [A], {"assoc_correlation_method"}, {}, new_store(), StubEmbedder(VECS)
    )
    assert id_map["assoc_correlation_method"] == "assoc_correlation_method"


def test_no_embedder_falls_back_to_slug():
    matters, _, id_map = ingest_candidates([A, APRIME], set(), {}, new_store(), None)
    # no semantic merge -> both kept (only slug collisions would merge)
    assert matters == {"assoc_correlation_method", "assoc_cloud_method"}
    assert id_map["assoc_cloud_method"] == "assoc_cloud_method"


def test_match_candidate_bands():
    store = new_store()
    emb = StubEmbedder(VECS)
    store.add(A["id"], emb.embed([_candidate_text(A)])[0], _candidate_text(A))
    assert match_candidate(_candidate_text(APRIME), store, emb).action == "merge"
    assert match_candidate(_candidate_text(B), store, emb).action == "new"
    assert match_candidate(_candidate_text(C), store, emb).action == "new"  # no llm
    assert (
        match_candidate(_candidate_text(C), store, emb, llm_client=fake_llm(True)).action
        == "merge"
    )


def test_embedding_store_roundtrip_and_nearest(tmp_path):
    path = str(tmp_path / "emb.npz")
    store = EmbeddingStore(path)
    emb = StubEmbedder(VECS)
    for c in (A, B):
        store.add(c["id"], emb.embed([_candidate_text(c)])[0], _candidate_text(c))
    store.save()

    reloaded = EmbeddingStore.load(path)
    assert len(reloaded) == 2
    assert reloaded.text_of("assoc_correlation_method") == _candidate_text(A)
    nearest = reloaded.nearest(emb.embed([_candidate_text(APRIME)])[0], top_k=2)
    assert nearest[0][0] == "assoc_correlation_method"  # closest to A's reworded dup
    assert nearest[0][1] > nearest[1][1]


def test_fake_embedder_separates_shared_from_disjoint():
    emb = FakeEmbedder()
    shared_a, shared_b, disjoint = emb.embed(
        [
            "semantic network rigidity in creative people",
            "rigidity of semantic networks in creative individuals",
            "default mode network openness to experience",
        ]
    )
    cos = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert cos(shared_a, shared_b) > cos(shared_a, disjoint)


# --- slice 2: relationship-aware reconciliation ---


def fake_classifier(relation, direction="none", satisfied=None, reason=""):
    payload = {
        "relation": relation,
        "direction": direction,
        "satisfied_condition_indices": satisfied or [],
        "reason": reason,
    }

    def create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))]
        )

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def prepared_store_with_A():
    emb = StubEmbedder(VECS)
    store = EmbeddingStore()
    store.add(A["id"], emb.embed([_candidate_text(A)])[0], _candidate_text(A))
    return emb, store


def test_reconcile_same_merges():
    emb, store = prepared_store_with_A()
    m, c, id_map, edges, flips = reconcile_candidates(
        [APRIME], {A["id"]}, {A["id"]: [{"label": "c", "truth": False}]}, store, emb,
        llm_client=fake_classifier("same"),
    )
    assert id_map[APRIME["id"]] == A["id"]
    assert APRIME["id"] not in m
    assert edges == []


def test_reconcile_resolves_flips_condition_and_adds_edge():
    emb, store = prepared_store_with_A()
    conditions = {A["id"]: [{"label": "needs X", "truth": False}, {"label": "needs Y", "truth": False}]}
    m, c, id_map, edges, flips = reconcile_candidates(
        [APRIME], {A["id"]}, conditions, store, emb,
        llm_client=fake_classifier("resolves", satisfied=[1]),
    )
    assert APRIME["id"] in m  # new node kept
    assert c[A["id"]][0]["truth"] is True   # condition 1 flipped
    assert c[A["id"]][1]["truth"] is False  # condition 2 untouched
    assert (APRIME["id"], A["id"]) in edges
    assert flips and flips[0]["matter"] == A["id"] and flips[0]["conditions"] == [1]


def test_reconcile_resolves_skips_already_resolved_matter():
    emb, store = prepared_store_with_A()
    _, c, _, edges, flips = reconcile_candidates(
        [APRIME], {A["id"]}, {A["id"]: [{"label": "done", "truth": True}]}, store, emb,
        llm_client=fake_classifier("resolves", satisfied=[1]),
    )
    assert flips == []
    assert edges == []


def test_reconcile_link_adds_directed_edge_both_ways():
    emb, store = prepared_store_with_A()
    _, _, _, edges, _ = reconcile_candidates(
        [APRIME], {A["id"]}, {A["id"]: [{"label": "c", "truth": False}]}, store, emb,
        llm_client=fake_classifier("link", direction="new_before_existing"),
    )
    assert (APRIME["id"], A["id"]) in edges

    emb2, store2 = prepared_store_with_A()
    _, _, _, edges2, _ = reconcile_candidates(
        [APRIME], {A["id"]}, {A["id"]: [{"label": "c", "truth": False}]}, store2, emb2,
        llm_client=fake_classifier("link", direction="existing_before_new"),
    )
    assert (A["id"], APRIME["id"]) in edges2


def test_reconcile_distinct_keeps_new_no_edge():
    emb, store = prepared_store_with_A()
    m, _, id_map, edges, flips = reconcile_candidates(
        [APRIME], {A["id"]}, {A["id"]: [{"label": "c", "truth": False}]}, store, emb,
        llm_client=fake_classifier("distinct"),
    )
    assert APRIME["id"] in m and id_map[APRIME["id"]] == APRIME["id"]
    assert edges == [] and flips == []


def test_reconcile_no_llm_merges_only_on_high_similarity():
    emb, store = prepared_store_with_A()
    _, _, id_map, _, _ = reconcile_candidates(
        [APRIME], {A["id"]}, {A["id"]: [{"label": "c", "truth": False}]}, store, emb,
        llm_client=None,
    )
    assert id_map[APRIME["id"]] == A["id"]  # cos ~0.98 >= high -> merge

    emb2, store2 = prepared_store_with_A()
    _, _, id_map2, _, _ = reconcile_candidates(
        [C], {A["id"]}, {A["id"]: [{"label": "c", "truth": False}]}, store2, emb2,
        llm_client=None,
    )
    assert id_map2[C["id"]] == C["id"]  # cos 0.8 < high -> new


def test_guard_blocks_same_across_role_status_even_when_llm_says_same():
    # existing: a RESOLVED contribution; new: an OPEN question worded the same.
    emb = StubEmbedder(VECS)
    store = EmbeddingStore()
    store.add(A["id"], emb.embed([_candidate_text(A)])[0], _candidate_text(A),
              "contribution", "resolved")
    new_q = {
        "id": "assoc_cloud_method", "name": "Association-cloud method",
        "kind": "question", "status": "open",
        "conditions": [{"label": "x", "truth": False}],
    }
    m, _, id_map, edges, flips = reconcile_candidates(
        [new_q], {A["id"]}, {A["id"]: [{"label": "done", "truth": True}]}, store, emb,
        llm_client=fake_classifier("same"),  # LLM wrongly says same
    )
    assert id_map["assoc_cloud_method"] == "assoc_cloud_method"  # guard kept it separate
    assert "assoc_cloud_method" in m


def test_guard_allows_same_role_merge():
    emb = StubEmbedder(VECS)
    store = EmbeddingStore()
    store.add(A["id"], emb.embed([_candidate_text(A)])[0], _candidate_text(A),
              "method", "resolved")
    new_m = {
        "id": "assoc_cloud_method", "name": "Association-cloud method",
        "kind": "method", "status": "resolved",
        "conditions": [{"label": "d", "truth": True}],
    }
    _, _, id_map, _, _ = reconcile_candidates(
        [new_m], {A["id"]}, {A["id"]: [{"label": "done", "truth": True}]}, store, emb,
        llm_client=fake_classifier("same"),
    )
    assert id_map["assoc_cloud_method"] == A["id"]  # same kind+status -> merges
