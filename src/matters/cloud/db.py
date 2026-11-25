"""Versioned SQL storage for the hosted Matters layers.

SQLAlchemy is an optional Cloud dependency. Nothing in the local engine imports
this module. PostgreSQL is the deployment database; SQLite supports local
development and the same repository contract.
"""

import threading
from contextlib import contextmanager, nullcontext

from sqlalchemy import (
    BigInteger, CheckConstraint, Column, ForeignKey, Integer, JSON, MetaData,
    String, Table, create_engine, event, inspect, select,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.pool import StaticPool


SCHEMA_VERSION = 1
metadata = MetaData()

schema_versions = Table(
    "schema_versions", metadata,
    Column("id", Integer, primary_key=True),
    Column("version", Integer, nullable=False),
    CheckConstraint("id = 1", name="ck_schema_version_singleton"),
)
users = Table(
    "users", metadata,
    Column("id", String(36), primary_key=True),
    Column("email", String(320), nullable=False, unique=True),
    Column("password_hash", String(512), nullable=False),
    Column("created_at", BigInteger, nullable=False),
    CheckConstraint("email = lower(email)", name="ck_users_email_lowercase"),
)
sessions = Table(
    "sessions", metadata,
    Column("token_hash", String(64), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("expires_at", BigInteger, nullable=False, index=True),
    Column("created_at", BigInteger, nullable=False),
)
graphs = Table(
    "graphs", metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), ForeignKey("users.id"), nullable=False, index=True),
    Column("title", String(200), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("state", JSON, nullable=False),
    Column("revision", String(64), nullable=False),
    Column("version", Integer, nullable=False),
    Column("created_at", BigInteger, nullable=False),
    Column("updated_at", BigInteger, nullable=False),
    CheckConstraint("kind IN ('personal', 'project')", name="ck_graphs_kind"),
    CheckConstraint("version >= 1", name="ck_graphs_version"),
)
graph_members = Table(
    "graph_members", metadata,
    Column("graph_id", String(36), ForeignKey("graphs.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role", String(16), nullable=False),
    CheckConstraint("role IN ('editor', 'viewer')", name="ck_graph_members_role"),
)
graph_events = Table(
    "graph_events", metadata,
    Column("id", String(36), primary_key=True),
    Column("graph_id", String(36), ForeignKey("graphs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("actor_id", String(36), ForeignKey("users.id"), nullable=False),
    Column("operation", String(64), nullable=False),
    Column("revision", String(64), nullable=False),
    Column("version", Integer, nullable=False),
    Column("created_at", BigInteger, nullable=False),
)


class CloudDatabaseError(RuntimeError):
    """A safe configuration/storage message without connection credentials."""


class Database:
    def __init__(self, url):
        try:
            parsed = make_url(url)
            if parsed.drivername == "postgresql":
                parsed = parsed.set(drivername="postgresql+psycopg")
            if parsed.drivername not in {"sqlite", "sqlite+pysqlite", "postgresql+psycopg"}:
                raise CloudDatabaseError("Cloud storage requires SQLite or PostgreSQL with psycopg.")
            self._sqlite = parsed.get_backend_name() == "sqlite"
            options = {"hide_parameters": True, "pool_pre_ping": True}
            if self._sqlite:
                options["connect_args"] = {"check_same_thread": False, "timeout": 10}
                if parsed.database in (None, "", ":memory:"):
                    options["poolclass"] = StaticPool
            self.engine = create_engine(parsed, **options)
        except (ArgumentError, ImportError, TypeError) as error:
            raise CloudDatabaseError("Cloud database configuration is invalid.") from None
        self._sqlite_lock = threading.RLock()
        if self._sqlite:
            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _record):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.close()

    @contextmanager
    def transaction(self, write=False):
        """Yield a connection and atomically commit or roll back its work.

        SQLite writes begin immediately so a concurrent writer cannot load a
        stale graph before obtaining ownership. PostgreSQL repositories lock
        the intended graph row inside their ordinary transaction.
        """
        if type(write) is not bool:
            raise ValueError("write must be a boolean")
        guard = self._sqlite_lock if self._sqlite else nullcontext()
        with guard:
            try:
                connection = self.engine.connect()
            except SQLAlchemyError:
                raise CloudDatabaseError("Cloud database is unavailable.") from None
            with connection:
                if self._sqlite and write:
                    connection.exec_driver_sql("BEGIN IMMEDIATE")
                else:
                    connection.begin()
                try:
                    yield connection
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise

    def initialize(self):
        """Create a new schema, or verify the exact supported schema version.

        Existing unversioned or unsupported schemas require an explicit
        migration. This method never upgrades, drops or resets stored data.
        """
        try:
            with self.transaction(write=True) as connection:
                existing = set(inspect(connection).get_table_names())
                required = set(metadata.tables)
                if schema_versions.name not in existing:
                    if existing & (required - {schema_versions.name}):
                        raise CloudDatabaseError("An unversioned Cloud schema requires an explicit migration.")
                    metadata.create_all(connection)
                    connection.execute(schema_versions.insert().values(id=1, version=SCHEMA_VERSION))
                else:
                    versions = connection.execute(select(schema_versions)).mappings().all()
                    if len(versions) != 1 or versions[0]["id"] != 1 or versions[0]["version"] != SCHEMA_VERSION:
                        raise CloudDatabaseError("Unsupported Cloud schema version; an explicit migration is required.")
                    if not required <= existing:
                        raise CloudDatabaseError("The Cloud schema is incomplete; recovery is required.")
        except CloudDatabaseError:
            raise
        except SQLAlchemyError:
            raise CloudDatabaseError("Cloud database initialization failed.") from None
        return self

    def close(self):
        self.engine.dispose()
