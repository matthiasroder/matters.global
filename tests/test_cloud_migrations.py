"""Explicit schema upgrades preserve real version-1 accounts and graph facts."""

import json
import os
import subprocess
import sys
import time
import uuid

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")

from sqlalchemy import MetaData, create_engine, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from matters.cloud.accounts import CloudAccounts
from matters.cloud.db import (
    CloudDatabaseError, Database, VERSION_ONE_TABLES, VERSION_TWO_TABLES,
    graph_events, graph_sources, graphs, metadata, schema_versions,
)
from matters.cloud.repository import GraphRepository
from matters.storage import state_revision


BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def database(request, tmp_path):
    admin = None
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'version-one.db'}"
    else:
        pytest.importorskip("psycopg")
        configured = make_url(os.environ["MATTERS_TEST_DATABASE_URL"])
        assert configured.get_backend_name() == "postgresql"
        admin = create_engine(configured)
        name = "migration_" + uuid.uuid4().hex
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{name}"')
        url = configured.update_query_dict({"options": f"-csearch_path={name}"})
    db = Database(url)
    legacy = MetaData()
    for name in VERSION_ONE_TABLES:
        metadata.tables[name].to_metadata(legacy)
    with db.transaction(write=True) as connection:
        legacy.create_all(connection)
        connection.execute(schema_versions.insert().values(id=1, version=1))
    try:
        yield db
    finally:
        db.close()
        if admin:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{url.query["options"].split("=")[-1]}" CASCADE')
            admin.dispose()


def seed_legacy(db):
    accounts = CloudAccounts(db)
    owner = accounts.register("owner@example.com", "A legacy account password")
    session = accounts.login(owner["email"], "A legacy account password")
    graph_id = str(uuid.uuid4())
    state = {"schema_version": 2, "matters": ["keep_data"], "conditions": {
        "keep_data": [{"label": "Existing graph survives", "truth": True}],
    }, "dependencies": []}
    now = int(time.time())
    with db.transaction(write=True) as connection:
        connection.execute(graphs.insert().values(
            id=graph_id, owner_id=owner["id"], title="Legacy personal graph", kind="personal",
            state=state, revision=state_revision({"keep_data"}, state["conditions"], set()),
            version=3, created_at=now, updated_at=now,
        ))
        connection.execute(graph_events.insert().values(
            id=str(uuid.uuid4()), graph_id=graph_id, actor_id=owner["id"], operation="create_graph",
            revision="legacy-event", version=1, created_at=now,
        ))
    return owner, session, graph_id, state


def old_rows(db):
    with db.transaction() as connection:
        return {
            name: [dict(row) for row in connection.execute(select(metadata.tables[name])).mappings()]
            for name in sorted(VERSION_ONE_TABLES - {"schema_versions"})
        }


def test_explicit_upgrade_preserves_accounts_sessions_graphs_and_history(database):
    owner, session, graph_id, state = seed_legacy(database)
    before = old_rows(database)
    with pytest.raises(CloudDatabaseError, match="cloud-migrate"):
        database.initialize()
    assert database.migrate() == {"from_version": 1, "to_version": 2, "changed": True}
    assert database.initialize() is database
    assert old_rows(database) == before
    assert CloudAccounts(database).authenticate(session["token"]) == owner
    graph = GraphRepository(database).get_graph(owner["id"], graph_id)
    assert graph["state"] == state
    assert graph["role"] == "owner" and graph["version"] == 3
    assert database.migrate() == {"from_version": 2, "to_version": 2, "changed": False}
    with database.transaction() as connection:
        assert set(inspect(connection).get_table_names()) >= set(metadata.tables)


def test_failed_upgrade_rolls_back_tables_version_and_existing_rows(database, monkeypatch):
    seed_legacy(database)
    before = old_rows(database)

    def fail(*args, **kwargs):
        raise OperationalError("injected DDL failure", {}, Exception("synthetic"))

    monkeypatch.setattr(VERSION_TWO_TABLES[-1], "create", fail)
    with pytest.raises(CloudDatabaseError, match="rolled back"):
        database.migrate()
    assert old_rows(database) == before
    with database.transaction() as connection:
        assert connection.execute(select(schema_versions.c.version)).scalar_one() == 1
        assert not {t.name for t in VERSION_TWO_TABLES} & set(inspect(connection).get_table_names())


def test_partial_upgrade_is_refused_without_filling_or_rewriting(database):
    seed_legacy(database)
    with database.transaction(write=True) as connection:
        graph_sources.create(connection)
    before = old_rows(database)
    with pytest.raises(CloudDatabaseError, match="partially migrated"):
        database.migrate()
    assert old_rows(database) == before


def test_unknown_version_is_refused_without_changes(database):
    with database.transaction(write=True) as connection:
        connection.execute(schema_versions.update().values(version=99))
    with pytest.raises(CloudDatabaseError, match="Unsupported"):
        database.migrate()
    with database.transaction() as connection:
        assert connection.execute(select(schema_versions.c.version)).scalar_one() == 99


def test_cli_upgrade_is_explicit_and_reopens_same_database(database):
    owner, session, graph_id, state = seed_legacy(database)
    result = subprocess.run([
        sys.executable, "-m", "matters.cli", "cloud-migrate", "--database-url",
        database.engine.url.render_as_string(hide_password=False),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"from_version": 1, "to_version": 2, "changed": True}
    reopened = Database(database.engine.url).initialize()
    try:
        assert CloudAccounts(reopened).authenticate(session["token"]) == owner
        assert GraphRepository(reopened).get_graph(owner["id"], graph_id)["state"] == state
    finally:
        reopened.close()


def test_empty_database_requires_initialization_instead_of_migration(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'empty.db'}")
    try:
        with pytest.raises(CloudDatabaseError, match="existing versioned"):
            db.migrate()
        with db.transaction() as connection:
            assert not inspect(connection).get_table_names()
    finally:
        db.close()
