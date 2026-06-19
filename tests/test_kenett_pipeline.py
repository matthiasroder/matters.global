import importlib.util
import json
from pathlib import Path


def load_pipeline():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "kenett_creativity"
        / "pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("kenett_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_enrich_dependencies():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "kenett_creativity"
        / "enrich_dependencies.py"
    )
    spec = importlib.util.spec_from_file_location("kenett_enrich_dependencies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_is_current_requires_status_kind_and_condition_truth():
    pipeline = load_pipeline()

    assert pipeline.cache_is_current(
        {
            "engine": "llm",
            "candidates": [
                {
                    "kind": "finding",
                    "status": "resolved",
                    "conditions": [{"label": "Established", "truth": True}],
                }
            ],
        }
    )
    assert not pipeline.cache_is_current(
        {
            "engine": "llm",
            "candidates": [
                {
                    "kind": "finding",
                    "conditions": [{"label": "Established", "truth": False}],
                }
            ],
        }
    )
    assert not pipeline.cache_is_current(
        {
            "engine": "llm",
            "candidates": [
                {
                    "kind": "finding",
                    "status": "resolved",
                    "conditions": [{"label": "Established", "truth": False}],
                }
            ],
        }
    )
    assert not pipeline.cache_is_current(
        {
            "engine": "llm",
            "candidates": [
                {
                    "kind": "problem",
                    "status": "open",
                    "conditions": [{"label": "Open issue is solved", "truth": True}],
                }
            ],
        }
    )
    assert not pipeline.cache_is_current(
        {
            "engine": "llm",
            "candidates": [
                {
                    "kind": "finding",
                    "status": "resolved",
                    "conditions": [{"label": "Established"}],
                }
            ],
        }
    )


def test_build_graph_preserves_status_and_kind_in_provenance(monkeypatch):
    pipeline = load_pipeline()

    def fake_extract_paper(paper, client, existing_matters):
        return (
            {
                "engine": "llm",
                "candidates": [
                    {
                        "id": "established_finding",
                        "name": "Established finding",
                        "description": "The abstract establishes this finding.",
                        "kind": "finding",
                        "status": "resolved",
                        "source_type": "paper",
                        "conditions": [{"label": "Finding is established", "truth": True}],
                    }
                ],
                "dependency_candidates": [],
            },
            False,
        )

    monkeypatch.setattr(pipeline, "extract_paper", fake_extract_paper)
    matters, conditions, edges, provenance = pipeline.build_graph(
        [
            {
                "title": "A paper",
                "paperId": "paper-1",
                "year": 2026,
                "doi": "10.0000/example",
            }
        ],
        client=None,
    )

    assert matters == ["established_finding"]
    assert conditions["established_finding"] == [
        {"label": "Finding is established", "truth": True}
    ]
    assert edges == []
    assert provenance["established_finding"]["kind"] == "finding"
    assert provenance["established_finding"]["status"] == "resolved"
    assert provenance["established_finding"]["source_type"] == "paper"


def test_extract_paper_retries_invalid_llm_proposal(tmp_path, monkeypatch):
    pipeline = load_pipeline()
    calls = []
    invalid = {
        "engine": "llm",
        "candidates": [
            {
                "kind": "finding",
                "status": "resolved",
                "conditions": [{"label": "Open mechanistic gap", "truth": False}],
            }
        ],
        "dependency_candidates": [],
    }
    valid = {
        "engine": "llm",
        "candidates": [
            {
                "kind": "finding",
                "status": "resolved",
                "conditions": [{"label": "Finding is established", "truth": True}],
            }
        ],
        "dependency_candidates": [],
    }

    def fake_build_extraction_proposal(*args, **kwargs):
        calls.append((args, kwargs))
        return invalid if len(calls) == 1 else valid

    monkeypatch.setattr(pipeline, "EXTRACT_DIR", str(tmp_path))
    monkeypatch.setattr(
        pipeline, "build_extraction_proposal", fake_build_extraction_proposal
    )
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)

    proposal, was_cached = pipeline.extract_paper(
        {"paperId": "paper-1", "title": "A paper", "abstract": "An abstract."},
        client=None,
        existing_matters=[],
    )

    assert proposal == valid
    assert was_cached is False
    assert len(calls) == 2
    assert json.loads((tmp_path / "paper-1.json").read_text()) == valid


def test_extraction_manifest_records_inputs_caches_and_output_hashes(tmp_path, monkeypatch):
    pipeline = load_pipeline()
    corpus_path = tmp_path / "corpus.json"
    graph_path = tmp_path / "creativity.json"
    provenance_path = tmp_path / "matter_sources.json"
    manifest_path = tmp_path / "run_manifest.json"
    extract_dir = tmp_path / "extractions"
    extract_dir.mkdir()

    paper = {
        "paperId": "paper-1",
        "title": "A paper",
        "abstract": "A source abstract.",
        "year": 2026,
        "citationCount": 7,
        "doi": "10.0000/example",
    }
    corpus = [paper]
    matters = ["established_finding"]
    conditions = {
        "established_finding": [{"label": "Finding is established", "truth": True}]
    }
    edges = []
    provenance = {
        "established_finding": {
            "name": "Established finding",
            "description": "The abstract establishes this finding.",
            "kind": "finding",
            "status": "resolved",
            "source_type": "paper",
            "paper": "A paper",
            "paperId": "paper-1",
            "year": 2026,
            "doi": "10.0000/example",
        }
    }
    corpus_path.write_text(json.dumps(corpus))
    graph_path.write_text(json.dumps({"matters": matters, "conditions": conditions}))
    provenance_path.write_text(json.dumps(provenance))
    (extract_dir / "paper-1.json").write_text(
        json.dumps(
            {
                "engine": "llm",
                "model": "claude-sonnet-4-6",
                "candidates": [
                    {
                        "kind": "finding",
                        "status": "resolved",
                        "conditions": [
                            {"label": "Finding is established", "truth": True}
                        ],
                    }
                ],
                "dependency_candidates": [],
            }
        )
    )

    monkeypatch.setattr(pipeline, "CORPUS_PATH", str(corpus_path))
    monkeypatch.setattr(pipeline, "GRAPH_PATH", str(graph_path))
    monkeypatch.setattr(pipeline, "PROVENANCE_PATH", str(provenance_path))
    monkeypatch.setattr(pipeline, "MANIFEST_PATH", str(manifest_path))
    monkeypatch.setattr(pipeline, "EXTRACT_DIR", str(extract_dir))

    manifest = pipeline.write_manifest(corpus, corpus, matters, conditions, edges, provenance)

    assert manifest_path.exists()
    assert manifest["corpus"]["papers_total"] == 1
    assert manifest["selected_papers"][0]["paperId"] == "paper-1"
    assert manifest["extraction_cache"][0]["current"] is True
    assert manifest["outputs"]["condition_truth_counts"] == {"true": 1, "false": 0}
    assert manifest["outputs"]["graph_sha256"] == pipeline.file_hash(str(graph_path))


def test_enrichment_manifest_records_input_and_output_hashes(tmp_path, monkeypatch):
    enrich = load_enrich_dependencies()
    graph_path = tmp_path / "creativity.json"
    provenance_path = tmp_path / "matter_sources.json"
    added_path = tmp_path / "added_dependencies.json"
    manifest_path = tmp_path / "enrich_manifest.json"
    cache_path = tmp_path / "batch_000.json"

    graph_path.write_text(json.dumps({"matters": ["a", "b"], "dependencies": []}))
    provenance_path.write_text(json.dumps({"a": {"name": "A"}, "b": {"name": "B"}}))
    added = [{"prerequisite": "a", "dependent": "b", "reason": "A enables B"}]
    added_path.write_text(json.dumps(added))
    context_sha = enrich.cache_context("inventory-sha", ["b"], ["a -> b"])
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "context_sha256": context_sha,
                "proposals": added,
            }
        )
    )

    monkeypatch.setattr(enrich, "GRAPH", str(graph_path))
    monkeypatch.setattr(enrich, "PROV", str(provenance_path))
    monkeypatch.setattr(enrich, "ADDED_SIDECAR", str(added_path))
    monkeypatch.setattr(enrich, "MANIFEST", str(manifest_path))

    manifest = enrich.write_manifest(
        graph_input_sha=enrich.file_hash(str(graph_path)),
        prov_sha=enrich.file_hash(str(provenance_path)),
        matter_ids=["a", "b"],
        dependencies_before=[],
        final_deps=[["a", "b"]],
        added=added,
        rejected={"duplicate": 1},
        cache_paths=[str(cache_path)],
    )

    assert manifest_path.exists()
    assert manifest["inputs"]["matter_inventory_count"] == 2
    assert manifest["cache"][0]["proposals"] == 1
    assert manifest["outputs"]["added_dependencies"] == 1
    assert manifest["outputs"]["rejected"] == {"duplicate": 1}
    assert manifest["outputs"]["graph_output_sha256"] == enrich.file_hash(str(graph_path))


def test_enrichment_cache_reuse_requires_matching_context(tmp_path):
    enrich = load_enrich_dependencies()
    cache_path = tmp_path / "batch_000.json"
    proposals = [{"prerequisite": "a", "dependent": "b", "reason": "A enables B"}]
    context_sha = enrich.cache_context("inventory-sha", ["b"], ["a -> b"])

    enrich.write_cached_proposals(str(cache_path), context_sha, proposals)

    assert enrich.load_cached_proposals(str(cache_path), context_sha) == proposals
    assert enrich.load_cached_proposals(str(cache_path), "other-context") is None

    cache_path.write_text(json.dumps(proposals))
    assert enrich.load_cached_proposals(str(cache_path), context_sha) is None
