"""Authorized, transactional hosted graph operations using the shared engine."""

import copy
import time
import uuid
from contextlib import contextmanager

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError

from .. import rules
from ..engine import dependents, prerequisites
from ..reports import unlock_report
from ..portability import build_bundle, normalize_bundle, normalize_sources
from ..storage import state_data, state_from_data, state_revision
from .db import CloudDatabaseError, graph_events, graph_members, graph_sources, graphs, users


class CloudGraphError(ValueError):
    """A transport-neutral error that contains no database credentials."""

    def __init__(self, message, code="invalid", cycle=()):
        super().__init__(message)
        self.message = message
        self.code = code
        self.cycle = tuple(cycle)


class GraphRepository:
    def __init__(self, db):
        self.db = db

    @contextmanager
    def _transaction(self, *, write=False):
        try:
            with self.db.transaction(write=write) as connection:
                yield connection
        except rules.RuleError as error:
            raise CloudGraphError(str(error), error.code, error.cycle) from error
        except (SQLAlchemyError, CloudDatabaseError):
            raise CloudGraphError("Graph storage is unavailable.", "unavailable") from None

    @staticmethod
    def _not_found():
        # Unknown and inaccessible graph identifiers have the same response.
        return CloudGraphError("Graph not found.", "not_found")

    @classmethod
    def _require_user(cls, connection, actor_id):
        if not isinstance(actor_id, str) or not actor_id:
            raise cls._not_found()
        if connection.execute(select(users.c.id).where(users.c.id == actor_id)).first() is None:
            raise cls._not_found()

    @classmethod
    def _authorized_graph(cls, connection, actor_id, graph_id, *, write=False):
        if not isinstance(actor_id, str) or not actor_id or not isinstance(graph_id, str):
            raise cls._not_found()
        statement = select(graphs).where(graphs.c.id == graph_id)
        if write:
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().first()
        if row is None:
            raise cls._not_found()
        if row["owner_id"] == actor_id:
            role = "owner"
        else:
            role = connection.execute(select(graph_members.c.role).where(and_(
                graph_members.c.graph_id == graph_id,
                graph_members.c.user_id == actor_id,
            ))).scalar_one_or_none()
            if role is None:
                raise cls._not_found()
        return dict(row), role

    @staticmethod
    def _metadata(row, role):
        return {
            "schema_version": 1,
            "graph_id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
            "role": role,
            "revision": row["revision"],
            "version": row["version"],
        }

    @staticmethod
    def _snapshot(document, *, require_acyclic=False):
        try:
            if isinstance(document, dict) and "schema_version" in document:
                version = document["schema_version"]
                if type(version) is not int or version not in (1, 2):
                    raise CloudGraphError("Unsupported graph schema_version.")
            snapshot = state_from_data(document)
            if require_acyclic:
                rules.require_acyclic_index(*snapshot)
            return snapshot
        except CloudGraphError:
            raise
        except rules.RuleError as error:
            raise CloudGraphError(str(error), error.code, error.cycle) from error
        except (KeyError, TypeError, ValueError) as error:
            raise CloudGraphError(f"Invalid graph state: {error}") from error

    @classmethod
    def _with_state(cls, row, role):
        return {**cls._metadata(row, role), "state": state_data(*cls._snapshot(row["state"]))}

    @staticmethod
    def _event(connection, row, actor_id, operation):
        connection.execute(graph_events.insert().values(
            id=str(uuid.uuid4()), graph_id=row["id"], actor_id=actor_id,
            operation=operation, revision=row["revision"], version=row["version"],
            created_at=int(time.time()),
        ))

    def create_graph(self, actor_id, title, kind="project", *, state=None):
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 200:
            raise CloudGraphError("Graph title must contain 1 to 200 characters.")
        if kind not in ("personal", "project"):
            raise CloudGraphError("Graph kind must be personal or project.")
        if state is None:
            state = {"schema_version": 2, "matters": [], "conditions": {}, "dependencies": []}
        snapshot = self._snapshot(state, require_acyclic=True)
        now = int(time.time())
        row = {
            "id": str(uuid.uuid4()), "owner_id": actor_id, "title": title.strip(),
            "kind": kind, "state": state_data(*snapshot),
            "revision": state_revision(*snapshot), "version": 1,
            "created_at": now, "updated_at": now,
        }
        with self._transaction(write=True) as connection:
            self._require_user(connection, actor_id)
            connection.execute(graphs.insert().values(**row))
            self._event(connection, row, actor_id, "create_graph")
            result = self._with_state(row, "owner")
        return result

    def list_graphs(self, actor_id):
        with self._transaction() as connection:
            self._require_user(connection, actor_id)
            membership = graph_members.alias("actor_membership")
            statement = select(
                graphs.c.id, graphs.c.owner_id, graphs.c.title, graphs.c.kind,
                graphs.c.revision, graphs.c.version, membership.c.role.label("member_role"),
            ).select_from(
                graphs.outerjoin(membership, and_(
                    membership.c.graph_id == graphs.c.id,
                    membership.c.user_id == actor_id,
                ))
            ).where(or_(graphs.c.owner_id == actor_id, membership.c.user_id == actor_id))
            rows = connection.execute(statement.order_by(graphs.c.created_at, graphs.c.id)).mappings()
            entries = [
                self._metadata(row, "owner" if row["owner_id"] == actor_id else row["member_role"])
                for row in rows
            ]
        return {"schema_version": 1, "graphs": entries}

    def get_graph(self, actor_id, graph_id):
        with self._transaction() as connection:
            row, role = self._authorized_graph(connection, actor_id, graph_id)
            result = self._with_state(row, role)
        return result

    def mutate_graph(self, actor_id, graph_id, operation, payload, *, expected_revision):
        with self._transaction(write=True) as connection:
            # All membership changes use this same row lock. Access and
            # revision are checked after the lock, in the write transaction.
            row, role = self._authorized_graph(connection, actor_id, graph_id, write=True)
            if role not in ("owner", "editor"):
                raise CloudGraphError("This graph is read-only for your account.", "forbidden")
            if not isinstance(expected_revision, str) or not expected_revision.strip():
                raise CloudGraphError("A nonempty expected_revision is required.", "conflict")
            if row["revision"] != expected_revision:
                raise CloudGraphError("The graph changed. Refresh and review your edit.", "conflict")
            snapshot = self._snapshot(row["state"])
            before = copy.deepcopy(snapshot)
            draft = rules.StateDraft(None, *snapshot)
            operation_result = rules.apply_graph_operation(draft, operation, payload)
            final = draft.as_tuple()
            if final != before:
                row.update(
                    state=state_data(*final), revision=state_revision(*final),
                    version=row["version"] + 1, updated_at=int(time.time()),
                )
                connection.execute(graphs.update().where(graphs.c.id == graph_id).values(
                    state=row["state"], revision=row["revision"],
                    version=row["version"], updated_at=row["updated_at"],
                ))
                # A matter deletion also removes its citations in the same
                # commit. No export can retain orphaned source records.
                source_document = connection.execute(
                    select(graph_sources.c.document).where(graph_sources.c.graph_id == graph_id)
                ).scalar_one_or_none()
                if source_document is not None:
                    entries = source_document.get("matter_sources", {})
                    kept = {matter: sources for matter, sources in entries.items() if matter in final[0]}
                    if kept != entries:
                        connection.execute(graph_sources.update().where(
                            graph_sources.c.graph_id == graph_id
                        ).values(document={"schema_version": 1, "matter_sources": kept}))
                self._event(connection, row, actor_id, operation)
            result = {**self._with_state(row, role), "result": operation_result}
        # Return this transaction's snapshot, not a post-commit reread that
        # could describe a second writer's changes.
        return result

    @staticmethod
    def _source_document(connection, row):
        document = connection.execute(
            select(graph_sources.c.document).where(graph_sources.c.graph_id == row["id"])
        ).scalar_one_or_none()
        if document is None:
            document = {"schema_version": 1, "matter_sources": {}}
        return normalize_sources(document, row["state"]["matters"])

    def get_graph_with_sources(self, actor_id, graph_id):
        """Read graph and citations while holding the graph's mutation lock."""
        with self._transaction(write=True) as connection:
            row, role = self._authorized_graph(connection, actor_id, graph_id, write=True)
            result = {**self._with_state(row, role), "sources": self._source_document(connection, row)}
        return result

    def export_graph(self, actor_id, graph_id):
        """Export one authorized snapshot without account or sharing records."""
        with self._transaction(write=True) as connection:
            row, role = self._authorized_graph(connection, actor_id, graph_id, write=True)
            bundle = build_bundle(row["state"], self._source_document(connection, row))
            result = {**self._metadata(row, role), "bundle": bundle}
        return result

    def import_graph(self, actor_id, title, bundle, kind="project"):
        """Create a fresh owned graph and citations in one atomic transaction.

        A package is data, never a claim to its original installation's graph
        ID or permissions. Imported caches are not accepted or rebuilt here.
        """
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 200:
            raise CloudGraphError("Graph title must contain 1 to 200 characters.")
        if kind not in ("personal", "project"):
            raise CloudGraphError("Graph kind must be personal or project.")
        try:
            state, sources = normalize_bundle(bundle)
            snapshot = self._snapshot(state, require_acyclic=True)
        except rules.RuleError as error:
            raise CloudGraphError(str(error), error.code, error.cycle) from error
        now = int(time.time())
        row = {
            "id": str(uuid.uuid4()), "owner_id": actor_id, "title": title.strip(),
            "kind": kind, "state": state_data(*snapshot),
            "revision": state_revision(*snapshot), "version": 1,
            "created_at": now, "updated_at": now,
        }
        with self._transaction(write=True) as connection:
            self._require_user(connection, actor_id)
            connection.execute(graphs.insert().values(**row))
            connection.execute(graph_sources.insert().values(graph_id=row["id"], document=sources))
            self._event(connection, row, actor_id, "import_graph")
            result = {**self._with_state(row, "owner"), "imported": True}
        return result

    def query_graph(self, actor_id, graph_id, view, *, matter_id=None):
        graph = self.get_graph(actor_id, graph_id)
        if not isinstance(view, str):
            raise CloudGraphError("Unknown graph read view.")
        if view == "state":
            return graph
        snapshot = self._snapshot(graph["state"])
        matters, conditions, dependencies = snapshot
        result = {key: value for key, value in graph.items() if key != "state"}
        try:
            if view == "matters":
                return {**result, "matters": sorted(matters), "total": len(matters)}
            if view == "matter":
                if not isinstance(matter_id, str):
                    raise CloudGraphError("matter_id must be a string.")
                rules.require_matter_id_syntax(matter_id)
                rules.require_matter(matter_id, matters)
                return {
                    **result, "id": matter_id,
                    "conditions": [{"index": index, **condition} for index, condition in enumerate(conditions.get(matter_id, []))],
                    "prerequisites": sorted(prerequisites(matter_id, dependencies)),
                    "dependents": sorted(dependents(matter_id, dependencies)),
                }
            if view not in {"universe", "frontier", "horizon", "unlock"}:
                raise CloudGraphError("Unknown graph read view.")
            index = rules.build_index(*snapshot)
            if view == "unlock":
                return {**result, **unlock_report(*snapshot, index=index)}
            if view == "universe":
                return {**result, "matters": sorted(index.universe)}
            if not isinstance(matter_id, str):
                raise CloudGraphError("matter_id must be a string.")
            rules.require_matter_id_syntax(matter_id)
            rules.require_matter(matter_id, matters)
            return {**result, "root": matter_id, "matters": sorted(getattr(index, view)(matter_id))}
        except rules.RuleError as error:
            raise CloudGraphError(str(error), error.code, error.cycle) from error

    def set_member(self, actor_id, graph_id, user_id, role="viewer"):
        """Owner-only internal sharing primitive; invitations remain separate."""
        if role not in ("editor", "viewer"):
            raise CloudGraphError("Member role must be editor or viewer.")
        with self._transaction(write=True) as connection:
            row, actor_role = self._authorized_graph(connection, actor_id, graph_id, write=True)
            if actor_role != "owner":
                raise CloudGraphError("Only the owner can manage graph members.", "forbidden")
            if user_id == row["owner_id"]:
                raise CloudGraphError("Owner permissions cannot be replaced by membership.")
            self._require_user(connection, user_id)
            predicate = and_(graph_members.c.graph_id == graph_id, graph_members.c.user_id == user_id)
            previous = connection.execute(select(graph_members.c.role).where(predicate)).scalar_one_or_none()
            if previous is None:
                connection.execute(graph_members.insert().values(graph_id=graph_id, user_id=user_id, role=role))
            elif previous != role:
                connection.execute(graph_members.update().where(predicate).values(role=role))
            if previous != role:
                self._event(connection, row, actor_id, "set_member")
            result = {**self._metadata(row, actor_role), "member": {"user_id": user_id, "role": role}}
        return result

    def remove_member(self, actor_id, graph_id, user_id):
        with self._transaction(write=True) as connection:
            row, role = self._authorized_graph(connection, actor_id, graph_id, write=True)
            if role != "owner":
                raise CloudGraphError("Only the owner can manage graph members.", "forbidden")
            removed = connection.execute(graph_members.delete().where(and_(
                graph_members.c.graph_id == graph_id, graph_members.c.user_id == user_id,
            ))).rowcount > 0
            if removed:
                self._event(connection, row, actor_id, "remove_member")
            result = {**self._metadata(row, role), "removed": removed}
        return result
