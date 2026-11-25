import json
from pathlib import Path

import pytest

from matters.identity import EmbeddingStore
from matters.portability import embedding_cache_path, export_bundle, import_bundle
import matters.portability as portability
from matters.rules import RuleError
from matters.storage import load_state, state_revision


EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture
def graph():
    return {
        "schema_version": 2,
        "matters": ["finish", "start"],
        "conditions": {
            "start": [{"label": "Plan is written", "truth": True}],
            "finish": [{"label": "Result is verified", "truth": False}],
        },
        "dependencies": [["start", "finish"]],
    }


@pytest.fixture
def source(tmp_path, graph):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(graph))
    return path


def citations():
    return {
        "schema_version": 1,
        "matter_sources": {"finish": [{
            "title": "An observable result", "url": "https://example.org/result.pdf",
            "authors": ["Léa Example"], "pdf_pages": [[2, 3]],
        }]},
    }


@pytest.mark.parametrize("relative", [
    "matters.example.json", "creativity_matters_graph/creativity_graph.json",
])
def test_round_trip_repository_examples_without_changing_original(relative, tmp_path):
    source = EXAMPLES / relative
    original = source.read_bytes()
    bundle = export_bundle(source)
    destination = tmp_path / "new" / "project.json"

    result = import_bundle(bundle, destination)

    assert source.read_bytes() == original
    assert load_state(destination) == load_state(source)
    assert result["revision"] == state_revision(*load_state(source))
    assert result["matters"] == len(load_state(source)[0])
    assert result["embeddings_rebuilt"] is False
    assert export_bundle(destination) == bundle


def test_export_is_deterministic_and_creates_no_lock_or_cache(source):
    before = set(source.parent.iterdir())

    first = export_bundle(source)
    second = export_bundle(source)

    assert first == second
    assert first["format"] == "matters-graph"
    assert first["schema_version"] == 1
    assert first["graph"]["matters"] == ["finish", "start"]
    assert set(source.parent.iterdir()) == before


def test_citations_round_trip_with_current_sidecar_format(source):
    source.with_suffix(".sources.json").write_text(json.dumps(citations()))
    destination = source.parent / "imported.json"
    bundle = export_bundle(source)

    result = import_bundle(bundle, destination)

    assert bundle["sources"] == citations()
    assert json.loads(destination.with_suffix(".sources.json").read_text()) == citations()
    assert result["sources"] == 1


def test_account_metadata_and_cached_vectors_are_not_portable(source):
    graph = json.loads(source.read_text())
    graph.update(owner="alice", editors=["bob"], api_key="not-portable")
    graph["conditions"]["finish"][0]["account_id"] = "alice"
    source.write_text(json.dumps(graph))
    sources = citations()
    sources["credentials"] = "not-portable"
    sources["matter_sources"]["finish"][0]["access_token"] = "not-portable"
    source.with_suffix(".sources.json").write_text(json.dumps(sources))
    embedding_cache_path(source).write_bytes(b"old vectors")
    bundle = export_bundle(source)
    bundle["permissions"] = {"alice": "owner"}

    destination = source.parent / "imported.json"
    import_bundle(bundle, destination)
    exported = export_bundle(destination)

    assert "not-portable" not in json.dumps(exported)
    assert "permissions" not in exported
    assert not embedding_cache_path(destination).exists()
    assert not destination.with_suffix(".npz").exists()
    assert embedding_cache_path(source).read_bytes() == b"old vectors"


@pytest.mark.parametrize("version", [None, 1, 2])
def test_native_legacy_condition_semantics_are_preserved(tmp_path, version):
    native = {"matters": ["a"], "conditions": {"a": {"Legacy condition": True}}, "dependencies": []}
    if version is not None:
        native["schema_version"] = version
    original = tmp_path / "legacy.json"
    original.write_text(json.dumps(native))
    destination = tmp_path / "imported.json"

    import_bundle(native, destination)

    assert load_state(original) == load_state(destination)
    assert json.loads(destination.read_text())["schema_version"] == 2


@pytest.mark.parametrize("field,value", [
    ("format", "some-other-format"), ("schema_version", 99), ("schema_version", True),
    ("graph", []), ("sources", {"schema_version": True, "matter_sources": {}}),
])
def test_invalid_package_does_not_create_destination(source, field, value):
    bundle = export_bundle(source)
    bundle[field] = value
    destination = source.parent / "invalid" / "graph.json"

    with pytest.raises(RuleError):
        import_bundle(bundle, destination)

    assert not destination.exists()
    assert not destination.parent.exists()


@pytest.mark.parametrize("change", ["unknown_endpoint", "cycle", "unknown_source", "bad_pages", "bad_native_version"])
def test_shared_validation_rejects_invalid_graphs_and_sources(source, change):
    bundle = export_bundle(source)
    if change == "unknown_endpoint":
        bundle["graph"]["dependencies"].append(["missing", "finish"])
    elif change == "cycle":
        bundle["graph"]["dependencies"].append(["finish", "start"])
    elif change == "unknown_source":
        bundle["sources"] = citations()
        bundle["sources"]["matter_sources"]["missing"] = []
    elif change == "bad_pages":
        bundle["sources"] = citations()
        bundle["sources"]["matter_sources"]["finish"][0]["pdf_pages"] = [[3, 2]]
    else:
        bundle["graph"]["schema_version"] = 999
    destination = source.parent / "imported.json"

    with pytest.raises(RuleError) as error:
        import_bundle(bundle, destination)

    assert error.value.code == ("state_cycle" if change == "cycle" else "invalid")
    assert not destination.exists()


@pytest.mark.parametrize("sidecar", ["graph", "sources", "embeddings", "legacy_cache", "dangling_symlink"])
def test_existing_destination_or_sidecar_is_never_overwritten(source, sidecar):
    destination = source.parent / "imported.json"
    targets = {
        "graph": destination,
        "sources": destination.with_suffix(".sources.json"),
        "embeddings": embedding_cache_path(destination),
        "legacy_cache": destination.with_suffix(".npz"),
        "dangling_symlink": destination,
    }
    existing = targets[sidecar]
    if sidecar == "dangling_symlink":
        existing.symlink_to(source.parent / "absent.json")
    else:
        existing.write_bytes(b"untouched")

    with pytest.raises(RuleError) as error:
        import_bundle(export_bundle(source), destination)

    assert error.value.code == "already_exists"
    if sidecar == "dangling_symlink":
        assert existing.is_symlink()
        assert not existing.exists()
    else:
        assert existing.read_bytes() == b"untouched"


class FakeEmbedder:
    cache_identity = {"model": "offline-example", "revision": "1"}

    def __init__(self):
        self.inputs = None

    def embed(self, texts):
        self.inputs = texts
        return [[1.0, float(index + 1)] for index in range(len(texts))]


def test_explicit_rebuild_regenerates_vectors_from_imported_facts(source):
    destination = source.parent / "imported.json"
    embedder = FakeEmbedder()

    result = import_bundle(export_bundle(source), destination, rebuild_embeddings=True, embedder=embedder)

    cache = EmbeddingStore.load(embedding_cache_path(destination))
    assert result["embeddings_rebuilt"] is True
    assert result["embedding_cache_path"] == str(embedding_cache_path(destination))
    assert len(cache) == 2
    assert cache.text_of("finish") == "finish. Result is verified"
    assert cache.meta_of("finish") == ("", "open")
    assert cache.meta_of("start") == ("", "resolved")
    assert cache._metadata["identity"] == embedder.cache_identity
    assert len(embedder.inputs) == 2


def test_default_import_never_selects_or_calls_an_embedding_model(source, monkeypatch):
    def forbidden():
        raise AssertionError("default imports must remain offline")

    monkeypatch.setattr(portability, "get_embedder", forbidden)
    import_bundle(export_bundle(source), source.parent / "imported.json")


def test_missing_rebuild_backend_leaves_no_graph_or_sidecars(source, monkeypatch):
    destination = source.parent / "imported.json"
    monkeypatch.setattr(portability, "get_embedder", lambda: None)

    with pytest.raises(RuleError) as error:
        import_bundle(export_bundle(source), destination, rebuild_embeddings=True)

    assert error.value.code == "unavailable"
    assert not destination.exists()
    assert not destination.with_suffix(".sources.json").exists()
    assert not embedding_cache_path(destination).exists()


def test_failed_publication_rolls_back_only_new_files(source, monkeypatch):
    destination = source.parent / "imported.json"
    publish = portability._publish_new

    def fail_graph(staged, target):
        if target == destination:
            raise OSError("simulated disk failure")
        publish(staged, target)

    monkeypatch.setattr(portability, "_publish_new", fail_graph)

    with pytest.raises(RuleError) as error:
        import_bundle(export_bundle(source), destination, rebuild_embeddings=True, embedder=FakeEmbedder())

    assert error.value.code == "io_error"
    assert not destination.exists()
    assert not destination.with_suffix(".sources.json").exists()
    assert not embedding_cache_path(destination).exists()
    assert not list(source.parent.glob(".matters-import-*"))


def test_racing_non_matters_writer_is_not_clobbered_or_removed(source, monkeypatch):
    destination = source.parent / "imported.json"
    publish = portability._publish_new

    def race_graph(staged, target):
        if target == destination:
            destination.write_bytes(b"racing writer owns this file")
        publish(staged, target)

    monkeypatch.setattr(portability, "_publish_new", race_graph)

    with pytest.raises(RuleError) as error:
        import_bundle(export_bundle(source), destination)

    assert error.value.code == "already_exists"
    assert destination.read_bytes() == b"racing writer owns this file"
    assert not destination.with_suffix(".sources.json").exists()


def test_bad_source_file_and_missing_graph_fail_read_only(source):
    sources = source.with_suffix(".sources.json")
    sources.write_text("{")
    before = {path: path.read_bytes() for path in source.parent.iterdir()}

    with pytest.raises(RuleError):
        export_bundle(source)
    with pytest.raises(RuleError) as error:
        export_bundle(source.parent / "missing.json")

    assert error.value.code == "not_found"
    assert before == {path: path.read_bytes() for path in source.parent.iterdir()}


def test_rebuild_option_requires_an_actual_boolean(source):
    with pytest.raises(RuleError, match="must be a boolean"):
        import_bundle(export_bundle(source), source.parent / "imported.json", rebuild_embeddings="false")


def test_interruption_before_graph_publication_removes_new_sidecars(source, monkeypatch):
    destination = source.parent / "imported.json"
    publish = portability._publish_new

    def interrupt_graph(staged, target):
        if target == destination:
            raise KeyboardInterrupt
        publish(staged, target)

    monkeypatch.setattr(portability, "_publish_new", interrupt_graph)

    with pytest.raises(KeyboardInterrupt):
        import_bundle(export_bundle(source), destination)

    assert not destination.exists()
    assert not destination.with_suffix(".sources.json").exists()
