"""The same repository contract against SQLite and an opt-in real PostgreSQL."""

import copy
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import event, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateSchema, DropSchema

from matters.cloud.db import (
    CloudDatabaseError, Database, graph_events, graph_members, graphs,
    metadata, schema_versions, users,
)
from matters.cloud.repository import CloudGraphError, GraphRepository
from matters.local_service import LocalGraphService
from matters.storage import save_state, state_revision


@pytest.fixture(params=["sqlite", "postgresql"])
def database(request, tmp_path):
    if request.param == "sqlite":
        db = Database("sqlite:///" + str(tmp_path / "cloud.db")).initialize()
        yield db
        db.close()
        return
    configured = os.environ.get("MATTERS_TEST_DATABASE_URL")
    if not configured:
        pytest.skip("MATTERS_TEST_DATABASE_URL is required for real PostgreSQL checks")
    administrative = Database(configured)
    schema = "storage_" + uuid.uuid4().hex
    with administrative.engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    url = make_url(configured).update_query_dict({"options": "-csearch_path=" + schema})
    db = Database(url).initialize()
    try:
        yield db
    finally:
        db.close()
        with administrative.engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        administrative.close()


@pytest.fixture
def people(database):
    actors = {name: str(uuid.uuid4()) for name in ("alice", "bob", "carol")}
    with database.transaction(write=True) as connection:
        connection.execute(users.insert(), [
            {"id": identity, "email": name + "@example.org", "password_hash": "test-hash", "created_at": 1}
            for name, identity in actors.items()
        ])
    return actors


@pytest.fixture
def repository(database):
    return GraphRepository(database)


@pytest.fixture
def state():
    return {
        "schema_version": 2,
        "matters": ["define", "ship"],
        "conditions": {
            "define": [{"label": "Scope written", "truth": True}],
            "ship": [{"label": "Result verified", "truth": False}],
        },
        "dependencies": [["define", "ship"]],
    }


def event_count(database, graph_id):
    with database.transaction() as connection:
        return connection.execute(select(func.count()).select_from(graph_events).where(graph_events.c.graph_id == graph_id)).scalar_one()


def test_private_personal_and_project_graphs_have_stable_separate_ids(repository, people, state):
    personal = repository.create_graph(people["alice"], "Personal", "personal", state=state)
    project = repository.create_graph(people["alice"], "Project", state=state)
    other = repository.create_graph(people["bob"], "Private project", state=state)

    assert len({personal["graph_id"], project["graph_id"], other["graph_id"]}) == 3
    assert personal["kind"] == "personal"
    assert project["kind"] == "project"
    assert personal["role"] == "owner"
    assert personal["state"] == state
    listed = repository.list_graphs(people["alice"])
    assert {row["graph_id"] for row in listed["graphs"]} == {personal["graph_id"], project["graph_id"]}
    assert all("state" not in row and "password_hash" not in row for row in listed["graphs"])


def test_foreign_unknown_and_injection_graph_ids_have_same_refusal(repository, people):
    private = repository.create_graph(people["alice"], "Private")
    messages = []
    for identity in (private["graph_id"], str(uuid.uuid4()), "' OR 1=1; DROP TABLE users; --"):
        with pytest.raises(CloudGraphError) as error:
            repository.get_graph(people["bob"], identity)
        messages.append((error.value.code, str(error.value)))
    assert messages == [("not_found", "Graph not found.")] * 3
    assert len(repository.list_graphs(people["alice"])["graphs"]) == 1
    assert repository.list_graphs(people["bob"])["graphs"] == []


def test_viewer_editor_membership_revocation_and_owner_only_management(repository, people, state):
    graph = repository.create_graph(people["alice"], "Shared", state=state)
    identity = graph["graph_id"]
    repository.set_member(people["alice"], identity, people["bob"], "viewer")
    assert repository.get_graph(people["bob"], identity)["role"] == "viewer"
    with pytest.raises(CloudGraphError) as error:
        repository.mutate_graph(people["bob"], identity, "set_condition", {"matter_id": "ship", "index": 0, "truth": True}, expected_revision=graph["revision"])
    assert error.value.code == "forbidden"
    with pytest.raises(CloudGraphError) as error:
        repository.set_member(people["bob"], identity, people["carol"], "editor")
    assert error.value.code == "forbidden"
    repository.set_member(people["alice"], identity, people["bob"], "editor")
    updated = repository.mutate_graph(people["bob"], identity, "set_condition", {"matter_id": "ship", "index": 0, "truth": True}, expected_revision=graph["revision"])
    assert updated["state"]["conditions"]["ship"][0]["truth"] is True
    repository.remove_member(people["alice"], identity, people["bob"])
    with pytest.raises(CloudGraphError) as error:
        repository.get_graph(people["bob"], identity)
    assert error.value.code == "not_found"
    assert repository.list_graphs(people["bob"])["graphs"] == []


def test_all_eight_operations_match_file_engine(repository, people, tmp_path):
    path = tmp_path / "local.json"
    save_state(set(), {}, set(), path=path)
    local = LocalGraphService({"local": path})
    graph = repository.create_graph(people["alice"], "Parity")
    operations = [
        ("create_matter", {"id": "a", "conditions": [{"label": "Done", "truth": False}]}),
        ("add_condition", {"matter_id": "a", "label": "Second", "truth": False}),
        ("edit_condition", {"matter_id": "a", "index": 1, "label": "Validated"}),
        ("create_matter", {"id": "b", "conditions": [{"label": "Delivered", "truth": False}]}),
        ("add_dependency", {"source": "a", "target": "b"}),
        ("remove_dependency", {"source": "a", "target": "b"}),
        ("set_condition", {"matter_id": "a", "index": 0, "truth": True}),
        ("delete_condition", {"matter_id": "a", "index": 1, "confirmed": False}),
        ("delete_matter", {"matter_id": "b", "cascade": False, "confirmed": True}),
    ]
    for operation, payload in operations:
        local_revision = local.read("local", view="state")["revision"]
        local.mutate("local", operation, copy.deepcopy(payload), expected_revision=local_revision)
        graph = repository.mutate_graph(people["alice"], graph["graph_id"], operation, copy.deepcopy(payload), expected_revision=graph["revision"])
        expected = local.read("local", view="state")
        assert graph["state"] == expected["state"]
        assert graph["revision"] == expected["revision"]


def test_queries_match_file_engine(repository, people, state, tmp_path):
    path = tmp_path / "native.json"
    path.write_text(json.dumps(state))
    local = LocalGraphService({"native": path})
    graph = repository.create_graph(people["alice"], "Queries", state=state)
    for view, matter in (("state", None), ("matters", None), ("matter", "ship"), ("universe", None), ("frontier", "define"), ("horizon", "define"), ("unlock", None)):
        expected = local.read("native", view=view, matter_id=matter)
        actual = repository.query_graph(people["alice"], graph["graph_id"], view, matter_id=matter)
        for key, value in expected.items():
            if key not in {"graph_id", "offset", "limit"}:
                assert actual[key] == value


def test_rejected_cycle_and_payload_leave_state_revision_version_and_audit_unchanged(repository, database, people, state):
    graph = repository.create_graph(people["alice"], "Atomic", state=state)
    for operation, payload in [
        ("add_dependency", {"source": "ship", "target": "define"}),
        ("set_condition", {"matter_id": "ship", "index": 0, "truth": "false"}),
        ("set_condition", {"matter_id": "ship", "index": True, "truth": True}),
        ("create_matter", {"id": "stolen", "owner_id": people["bob"]}),
        ("not_an_operation", {}),
    ]:
        with pytest.raises(CloudGraphError):
            repository.mutate_graph(people["alice"], graph["graph_id"], operation, payload, expected_revision=graph["revision"])
        assert repository.get_graph(people["alice"], graph["graph_id"]) == graph
        assert event_count(database, graph["graph_id"]) == 1


def test_noop_keeps_revision_version_and_audit(repository, database, people, state):
    graph = repository.create_graph(people["alice"], "No-op", state=state)
    updated = repository.mutate_graph(people["alice"], graph["graph_id"], "set_condition", {"matter_id": "ship", "index": 0, "truth": False}, expected_revision=graph["revision"])
    assert updated["revision"] == graph["revision"]
    assert updated["version"] == graph["version"]
    assert updated["result"]["changed"] is False
    assert event_count(database, graph["graph_id"]) == 1


def test_stale_and_missing_revision_refuse_even_noop(repository, database, people, state):
    graph = repository.create_graph(people["alice"], "Stale", state=state)
    current = repository.mutate_graph(people["alice"], graph["graph_id"], "set_condition", {"matter_id": "ship", "index": 0, "truth": True}, expected_revision=graph["revision"])
    for revision in (graph["revision"], "", None):
        with pytest.raises(CloudGraphError) as error:
            repository.mutate_graph(people["alice"], graph["graph_id"], "set_condition", {"matter_id": "ship", "index": 0, "truth": True}, expected_revision=revision)
        assert error.value.code == "conflict"
    assert repository.get_graph(people["alice"], graph["graph_id"])["state"] == current["state"]
    assert event_count(database, graph["graph_id"]) == 2


def test_concurrent_writers_have_one_commit_and_one_conflict(repository, database, people):
    graph = repository.create_graph(people["alice"], "Concurrent")
    barrier = threading.Barrier(2)

    def write(matter):
        barrier.wait()
        try:
            return repository.mutate_graph(people["alice"], graph["graph_id"], "create_matter", {"id": matter}, expected_revision=graph["revision"])
        except CloudGraphError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, ("a", "b")))
    assert sum(isinstance(result, dict) for result in outcomes) == 1
    assert outcomes.count("conflict") == 1
    saved = repository.get_graph(people["alice"], graph["graph_id"])
    assert len(saved["state"]["matters"]) == 1
    assert saved["version"] == 2
    assert event_count(database, graph["graph_id"]) == 2


def test_revocation_is_rechecked_after_waiting_for_graph_write_lock(repository, database, people):
    graph = repository.create_graph(people["alice"], "Revocation")
    identity = graph["graph_id"]
    repository.set_member(people["alice"], identity, people["bob"], "editor")
    entered = threading.Event()

    def editor():
        entered.set()
        try:
            repository.mutate_graph(people["bob"], identity, "create_matter", {"id": "late_write"}, expected_revision=graph["revision"])
        except CloudGraphError as error:
            return error.code
        return "committed"

    with ThreadPoolExecutor(max_workers=1) as executor:
        with database.transaction(write=True) as connection:
            connection.execute(select(graphs).where(graphs.c.id == identity).with_for_update()).first()
            pending = executor.submit(editor)
            assert entered.wait(3)
            connection.execute(graph_members.delete().where(graph_members.c.graph_id == identity))
        assert pending.result(timeout=10) == "not_found"
    assert repository.get_graph(people["alice"], identity)["state"]["matters"] == []


def test_state_persists_through_close_and_reopen(repository, database, people, state):
    graph = repository.create_graph(people["alice"], "Persistent", state=state)
    url = database.engine.url
    database.close()
    reopened = Database(url).initialize()
    try:
        assert GraphRepository(reopened).get_graph(people["alice"], graph["graph_id"]) == graph
    finally:
        reopened.close()


def test_future_database_schema_fails_closed_without_modifying_graphs(repository, database, people, state):
    graph = repository.create_graph(people["alice"], "Keep", state=state)
    with database.transaction(write=True) as connection:
        connection.execute(schema_versions.update().values(version=99))
    with pytest.raises(CloudDatabaseError, match="Unsupported Cloud schema version"):
        database.initialize()
    with database.transaction() as connection:
        assert connection.execute(select(schema_versions.c.version)).scalar_one() == 99
        assert connection.execute(select(graphs.c.state)).scalar_one() == state


def test_future_and_invalid_graph_versions_rejected_before_insert(repository, people, state):
    for version in (99, True, None, "2"):
        invalid = {**state, "schema_version": version}
        with pytest.raises(CloudGraphError):
            repository.create_graph(people["alice"], "Unsupported", state=invalid)
    assert repository.list_graphs(people["alice"])["graphs"] == []


def test_legacy_graphs_normalize_and_cyclic_create_is_rejected(repository, people):
    native = {"matters": ["a"], "conditions": {"a": {"Legacy criterion": True}}, "dependencies": []}
    graph = repository.create_graph(people["alice"], "Legacy", state=native)
    assert graph["state"]["conditions"]["a"] == [{"label": "Legacy criterion", "truth": True}]
    cyclic = {**native, "dependencies": [["a", "a"]]}
    with pytest.raises(CloudGraphError) as error:
        repository.create_graph(people["alice"], "Cycle", state=cyclic)
    assert error.value.code == "state_cycle"
    assert len(repository.list_graphs(people["alice"])["graphs"]) == 1


def test_graph_catalog_does_not_load_large_graph_json(repository, database, people, state):
    repository.create_graph(people["alice"], "Metadata", state=state)
    statements = []

    def observe(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(database.engine, "before_cursor_execute", observe)
    try:
        assert len(repository.list_graphs(people["alice"])["graphs"]) == 1
    finally:
        event.remove(database.engine, "before_cursor_execute", observe)
    assert all("graphs.state" not in statement for statement in statements)


def test_mutation_returns_its_committed_snapshot_despite_immediate_second_writer(repository, database, people, monkeypatch):
    graph = repository.create_graph(people["alice"], "Commit snapshot")
    original = database.transaction
    injected = False

    @contextmanager
    def second_write_after_commit(write=False):
        nonlocal injected
        with original(write=write) as connection:
            yield connection
        if write and not injected:
            injected = True
            current = repository.get_graph(people["alice"], graph["graph_id"])
            repository.mutate_graph(people["alice"], graph["graph_id"], "create_matter", {"id": "second"}, expected_revision=current["revision"])

    monkeypatch.setattr(database, "transaction", second_write_after_commit)
    first = repository.mutate_graph(people["alice"], graph["graph_id"], "create_matter", {"id": "first"}, expected_revision=graph["revision"])
    persisted = repository.get_graph(people["alice"], graph["graph_id"])

    assert first["state"]["matters"] == ["first"]
    assert first["version"] == 2
    assert persisted["state"]["matters"] == ["first", "second"]
    assert persisted["version"] == 3
    assert first["revision"] != persisted["revision"]


def test_partial_cycle_repair_is_committed_without_a_failing_derived_read(repository, database, people):
    state = {"schema_version": 2, "matters": ["a", "b"], "conditions": {}, "dependencies": []}
    graph = repository.create_graph(people["alice"], "Repair", state=state)
    cyclic = {**state, "dependencies": [["a", "a"], ["b", "b"]]}
    revision = state_revision({"a", "b"}, {}, {("a", "a"), ("b", "b")})
    # Simulate stored legacy corruption; new graph creation refuses cycles.
    with database.transaction(write=True) as connection:
        connection.execute(graphs.update().where(graphs.c.id == graph["graph_id"]).values(state=cyclic, revision=revision))

    repaired = repository.mutate_graph(people["alice"], graph["graph_id"], "remove_dependency", {"source": "a", "target": "a"}, expected_revision=revision)

    assert repaired["state"]["dependencies"] == [["b", "b"]]
    assert repaired["version"] == 2
    assert event_count(database, graph["graph_id"]) == 2
    with pytest.raises(CloudGraphError) as error:
        repository.query_graph(people["alice"], graph["graph_id"], "universe")
    assert error.value.code == "state_cycle"


def test_sqlite_foreign_keys_and_memory_thread_access():
    db = Database("sqlite://").initialize()
    try:
        with pytest.raises(IntegrityError):
            with db.transaction(write=True) as connection:
                connection.execute(graph_members.insert().values(graph_id="missing", user_id="missing", role="viewer"))
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(lambda: db.initialize() is db).result(timeout=5)
    finally:
        db.close()


def test_unknown_database_driver_does_not_disclose_credentials():
    with pytest.raises(CloudDatabaseError) as error:
        Database("unknown://user:secret-sentinel@localhost/db")
    assert "secret-sentinel" not in str(error.value)


def test_unversioned_existing_cloud_tables_require_explicit_migration(tmp_path):
    db = Database("sqlite:///" + str(tmp_path / "unversioned.db"))
    try:
        users.create(db.engine)
        with pytest.raises(CloudDatabaseError, match="unversioned"):
            db.initialize()
    finally:
        db.close()
