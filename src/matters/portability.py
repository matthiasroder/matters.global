"""Portable graph packages, without installation-specific access or caches.

Exports are complete graph snapshots, not public/redacted exports. Imports
create a new graph only. Files are staged on the destination filesystem and
published with exclusive hard links, with the graph published last so readers
cannot discover it before its citations and optional rebuilt cache exist.
"""

import json
import os
import tempfile
from pathlib import Path

from .engine import normalize_conditions
from .graph_index import DependencyCycleError, GraphIndex
from .identity import EmbeddingStore, get_embedder
from .rules import RuleError, state_lock
from .sources import validate_source
from .storage import (
    normalize_dependency_records,
    resolve_state_path,
    save_state,
    state_revision,
    validate_state_data,
)


BUNDLE_FORMAT = "matters-graph"
BUNDLE_VERSION = 1


def embedding_cache_path(state_path):
    """Return the sidecar used for an explicitly rebuilt imported cache."""
    return Path(state_path).with_suffix(".embeddings.npz")


def _read_json(path, description):
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise RuleError(f"{description} does not exist", "not_found") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuleError(f"{description} is not readable JSON", "invalid") from error


def _canonical_graph(document):
    """Use the same normalization as native graph loading and saving."""
    try:
        validate_state_data(document)
        version = document.get("schema_version")
        if "schema_version" in document and (type(version) is not int or version not in (1, 2)):
            raise ValueError("unsupported native graph schema_version")
        matters = set(document["matters"])
        conditions = normalize_conditions(document["conditions"])
        dependencies = normalize_dependency_records(document["dependencies"], matters)
        GraphIndex(matters, conditions, dependencies)
    except DependencyCycleError as error:
        raise RuleError("graph contains a dependency cycle", "state_cycle", error.cycle) from error
    except (KeyError, TypeError, ValueError) as error:
        raise RuleError(f"invalid graph: {error}", "invalid") from error
    return {
        "schema_version": 2,
        "matters": sorted(matters),
        "conditions": {matter: conditions[matter] for matter in sorted(conditions)},
        "dependencies": [list(edge) for edge in sorted(dependencies)],
    }


def _canonical_sources(document, matters):
    try:
        if not isinstance(document, dict):
            raise ValueError("sources must be an object")
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("sources require schema_version 1")
        entries = document.get("matter_sources")
        if not isinstance(entries, dict):
            raise ValueError("sources require a matter_sources object")
        result = {}
        for matter in sorted(entries):
            if matter not in matters:
                raise ValueError(f"sources contain unknown matter: {matter}")
            citations = entries[matter]
            if not isinstance(citations, list):
                raise ValueError("matter citations must be a list")
            result[matter] = []
            for citation in citations:
                validate_source(citation)
                # Sharing, credentials and arbitrary installation metadata are
                # not part of a portable citation, even if present locally.
                result[matter].append({
                    key: citation[key]
                    for key in ("title", "url", "authors", "pdf_pages")
                    if key in citation
                })
        return {"schema_version": 1, "matter_sources": result}
    except (KeyError, TypeError, ValueError) as error:
        raise RuleError(f"invalid sources: {error}", "invalid") from error


def normalize_sources(document, matters):
    """Validate portable citations against the supplied graph's matter IDs."""
    return _canonical_sources(document, matters)


def export_bundle(state_path):
    """Read one graph snapshot and its canonical citation sidecar.

    This creates no lock, cache or output file. Citation files have no shared
    revision transaction today; edit them separately while graph revision
    continues to describe the captured graph facts.
    """
    path = resolve_state_path(state_path)
    graph = _read_json(path, "graph")
    sources_path = path.with_suffix(".sources.json")
    if os.path.lexists(sources_path):
        sources = _read_json(sources_path, "sources")
    else:
        sources = {"schema_version": 1, "matter_sources": {}}
    return build_bundle(graph, sources)


def build_bundle(graph, sources=None):
    """Build a portable package from facts, without reading or writing files.

    Hosted storage and local files share this codec so citations, legacy
    normalization and graph validation have the same meaning everywhere.
    Installation-specific graph identity and access records are never copied.
    """
    graph = _canonical_graph(graph)
    if sources is None:
        sources = {"schema_version": 1, "matter_sources": {}}
    sources = _canonical_sources(sources, graph["matters"])
    return {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_VERSION,
        "graph": graph,
        "sources": sources,
    }


def normalize_bundle(bundle):
    """Validate a package or native legacy graph and return canonical facts."""
    if not isinstance(bundle, dict):
        raise RuleError("import must be a graph package or native graph object", "invalid")
    if "format" in bundle or "graph" in bundle:
        if bundle.get("format") != BUNDLE_FORMAT:
            raise RuleError("unsupported graph package format", "invalid")
        if type(bundle.get("schema_version")) is not int or bundle["schema_version"] != BUNDLE_VERSION:
            raise RuleError("unsupported graph package schema_version", "invalid")
        graph = _canonical_graph(bundle.get("graph"))
        sources = _canonical_sources(bundle.get("sources"), graph["matters"])
    else:
        graph = _canonical_graph(bundle)
        sources = {"schema_version": 1, "matter_sources": {}}
    return graph, sources


# Kept for the file importer while exposing one pure codec to other storage
# adapters. No backend needs to implement its own package validation.
_unpack_bundle = normalize_bundle


def _write_json(path, document):
    with path.open("x", encoding="utf-8") as file:
        json.dump(document, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _rebuild_cache(path, graph, embedder):
    if embedder is None:
        embedder = get_embedder()
    if embedder is None:
        raise RuleError("embedding backend unavailable; install model2vec to rebuild", "unavailable")
    ids = graph["matters"]
    texts = [
        matter.replace("_", " ") + ". " + "; ".join(
            condition["label"] for condition in graph["conditions"].get(matter, [])
        )
        for matter in ids
    ]
    index = GraphIndex(set(ids), graph["conditions"], set(map(tuple, graph["dependencies"])))
    vectors = embedder.embed(texts) if texts else []
    if len(vectors) != len(ids):
        raise RuleError("embedding backend returned an unexpected vector count", "invalid")
    store = EmbeddingStore(path)
    for matter, text, vector in zip(ids, texts, vectors):
        store.ensure_compatible(embedder, vector)
        store.add(matter, vector, text, status="resolved" if index.resolved[matter] else "open")
    store.save()
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _publish_new(staged, destination):
    """Create a destination exclusively, including against non-Matters writers."""
    os.link(staged, destination)


def _remove_published(published):
    for destination, inode in reversed(published):
        try:
            stat = destination.lstat()
            if (stat.st_dev, stat.st_ino) == inode:
                destination.unlink()
        except FileNotFoundError:
            pass


def import_bundle(bundle, state_path, *, rebuild_embeddings=False, embedder=None):
    """Create a graph from a package or legacy/native graph, never overwrite.

    Optional embeddings are regenerated from graph names and condition labels,
    using the current local model, not imported vectors. Selecting rebuilding
    may download the configured model; ordinary imports make no model request.
    ``embedder`` supports application-provided/offline embedding backends.
    """
    if type(rebuild_embeddings) is not bool:
        raise RuleError("rebuild_embeddings must be a boolean", "invalid")
    graph, sources = _unpack_bundle(bundle)
    raw_path = resolve_state_path(state_path)
    # Canonicalize the directory, while retaining a final symlink so even a
    # dangling destination symlink is recognized as an existing target.
    path = raw_path.parent.resolve() / raw_path.name
    sources_path = path.with_suffix(".sources.json")
    cache_path = embedding_cache_path(path)
    protected = {path, sources_path, cache_path, path.with_suffix(".npz")}
    matters = set(graph["matters"])
    conditions = graph["conditions"]
    dependencies = set(map(tuple, graph["dependencies"]))
    summary = {
        "schema_version": BUNDLE_VERSION,
        "state_path": str(path),
        "revision": state_revision(matters, conditions, dependencies),
        "matters": len(matters),
        "conditions": sum(map(len, conditions.values())),
        "dependencies": len(dependencies),
        "sources": sum(map(len, sources["matter_sources"].values())),
        "embeddings_rebuilt": rebuild_embeddings,
    }
    published = []
    try:
        with state_lock(path):
            if any(os.path.lexists(candidate) for candidate in protected):
                raise RuleError("import destination graph or sidecar already exists", "already_exists")
            with tempfile.TemporaryDirectory(prefix=".matters-import-", dir=path.parent) as folder:
                staging = Path(folder)
                staged_graph = staging / "graph.json"
                staged_sources = staging / "sources.json"
                save_state(matters, conditions, dependencies, path=staged_graph)
                _write_json(staged_sources, sources)
                files = [(staged_sources, sources_path), (staged_graph, path)]
                if rebuild_embeddings:
                    staged_cache = staging / "embeddings.npz"
                    _rebuild_cache(staged_cache, graph, embedder)
                    files.insert(0, (staged_cache, cache_path))
                    summary["embedding_cache_path"] = str(cache_path)
                for staged, destination in files:
                    _publish_new(staged, destination)
                    stat = staged.stat()
                    published.append((destination, (stat.st_dev, stat.st_ino)))
    except BaseException as error:
        _remove_published(published)
        if not isinstance(error, Exception):
            raise
        if isinstance(error, RuleError):
            raise
        if isinstance(error, FileExistsError):
            raise RuleError("import destination graph or sidecar already exists", "already_exists") from error
        if isinstance(error, OSError):
            raise RuleError("could not write imported graph", "io_error") from error
        raise RuleError(f"could not rebuild imported graph cache: {type(error).__name__}", "invalid") from error
    return summary
