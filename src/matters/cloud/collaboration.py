"""Email-bound, expiring whole-graph invitations and owner access controls.

Invitation links are returned for the owner to copy. This module sends no mail.
Only a token digest is stored, and accepting it also requires the invited
account's authenticated identity. All membership changes take the same graph
row lock used by graph writes, including acceptance and invitation revocation.
"""

import hashlib
import re
import secrets
import time
import uuid
from typing import Annotated, Literal

from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field, StrictStr
from sqlalchemy import and_, select

from .accounts import AccountError, _normalized_email
from .db import graph_invitations, graph_members, graphs, users
from .repository import CloudGraphError


DEFAULT_INVITATION_TTL = 7 * 86400
MAX_INVITATION_TTL = 30 * 86400
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")


class CloudCollaboration:
    def __init__(self, repository, *, now=None):
        self.repository = repository
        self.now = now or time.time

    @staticmethod
    def _owner(repository, connection, actor_id, graph_id):
        row, role = repository._authorized_graph(connection, actor_id, graph_id, write=True)
        if role != "owner":
            raise CloudGraphError("Only the owner can manage graph access.", "forbidden")
        return row

    @staticmethod
    def _role(role):
        if not isinstance(role, str) or role not in {"editor", "viewer"}:
            raise CloudGraphError("Invitation role must be editor or viewer.")
        return role

    @staticmethod
    def _token_hash(token):
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise CloudGraphError("Invitation not found.", "not_found")
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    @staticmethod
    def _public_invitation(row, now):
        status = (
            "revoked" if row["revoked_at"] is not None else
            "accepted" if row["accepted_at"] is not None else
            "expired" if row["expires_at"] <= now else "pending"
        )
        return {
            key: row[key] for key in (
                "id", "email", "role", "created_at", "expires_at", "accepted_at", "revoked_at"
            )
        } | {"status": status}

    def create_invitation(self, actor_id, graph_id, email, role="viewer", *, ttl_seconds=DEFAULT_INVITATION_TTL):
        role = self._role(role)
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_INVITATION_TTL:
            raise CloudGraphError("Invitation lifetime must be between 1 second and 30 days.")
        try:
            email = _normalized_email(email)
        except AccountError as error:
            raise CloudGraphError(error.message) from None
        now = int(self.now())
        token = secrets.token_urlsafe(32)
        invitation = {
            "id": str(uuid.uuid4()), "graph_id": graph_id, "email": email,
            "role": role, "token_hash": self._token_hash(token),
            "inviter_id": actor_id, "created_at": now,
            "expires_at": now + ttl_seconds, "accepted_at": None, "revoked_at": None,
        }
        repo = self.repository
        with repo._transaction(write=True) as connection:
            graph = self._owner(repo, connection, actor_id, graph_id)
            owner_email = connection.execute(select(users.c.email).where(users.c.id == graph["owner_id"])).scalar_one()
            if email == owner_email:
                raise CloudGraphError("The owner already has access to this graph.")
            # A replacement invite supersedes earlier unused links to this
            # graph/email. Accepted membership is unaffected until acceptance.
            connection.execute(graph_invitations.update().where(and_(
                graph_invitations.c.graph_id == graph_id,
                graph_invitations.c.email == email,
                graph_invitations.c.accepted_at.is_(None),
                graph_invitations.c.revoked_at.is_(None),
            )).values(revoked_at=now))
            connection.execute(graph_invitations.insert().values(**invitation))
            repo._event(connection, graph, actor_id, "create_invitation")
        return {
            "schema_version": 1, "graph_id": graph_id,
            "invitation": self._public_invitation(invitation, now), "token": token,
        }

    def list_access(self, actor_id, graph_id):
        repo = self.repository
        with repo._transaction(write=True) as connection:
            graph = self._owner(repo, connection, actor_id, graph_id)
            owner = connection.execute(select(users.c.id, users.c.email).where(users.c.id == graph["owner_id"])).mappings().one()
            members = connection.execute(
                select(graph_members.c.user_id, users.c.email, graph_members.c.role)
                .join(users, users.c.id == graph_members.c.user_id)
                .where(graph_members.c.graph_id == graph_id).order_by(users.c.email)
            ).mappings().all()
            invitations = connection.execute(select(graph_invitations).where(
                graph_invitations.c.graph_id == graph_id
            ).order_by(graph_invitations.c.created_at, graph_invitations.c.id)).mappings().all()
            now = int(self.now())
        return {
            **repo._metadata(graph, "owner"),
            "owner": {"user_id": owner["id"], "email": owner["email"], "role": "owner"},
            "members": [dict(member) for member in members],
            "invitations": [self._public_invitation(row, now) for row in invitations],
        }

    def accept_invitation(self, actor_id, token):
        token_hash = self._token_hash(token)
        repo = self.repository
        with repo._transaction(write=True) as connection:
            repo._require_user(connection, actor_id)
            first = connection.execute(select(graph_invitations.c.id, graph_invitations.c.graph_id).where(
                graph_invitations.c.token_hash == token_hash
            )).mappings().first()
            if first is None:
                raise CloudGraphError("Invitation not found.", "not_found")
            # Always graph first, invitation second: revoke and normal graph
            # edits follow this same lock order, avoiding stale grants/races.
            graph = connection.execute(select(graphs).where(
                graphs.c.id == first["graph_id"]
            ).with_for_update()).mappings().first()
            if graph is None:
                raise CloudGraphError("Invitation not found.", "not_found")
            invitation = connection.execute(select(graph_invitations).where(and_(
                graph_invitations.c.id == first["id"],
                graph_invitations.c.token_hash == token_hash,
            )).with_for_update()).mappings().one()
            now = int(self.now())
            email = connection.execute(select(users.c.email).where(users.c.id == actor_id)).scalar_one()
            if email != invitation["email"]:
                raise CloudGraphError("Invitation not found.", "not_found")
            if invitation["accepted_at"] is not None or invitation["revoked_at"] is not None or invitation["expires_at"] <= now:
                raise CloudGraphError("This invitation is no longer available. Ask the owner for a new link.", "conflict")
            if actor_id == graph["owner_id"]:
                raise CloudGraphError("The graph owner cannot accept a membership invitation.")
            predicate = and_(graph_members.c.graph_id == graph["id"], graph_members.c.user_id == actor_id)
            if connection.execute(select(graph_members.c.user_id).where(predicate)).first() is None:
                connection.execute(graph_members.insert().values(
                    graph_id=graph["id"], user_id=actor_id, role=invitation["role"],
                ))
            else:
                connection.execute(graph_members.update().where(predicate).values(role=invitation["role"]))
            connection.execute(graph_invitations.update().where(
                graph_invitations.c.id == invitation["id"]
            ).values(accepted_at=now))
            repo._event(connection, graph, actor_id, "accept_invitation")
            result = {
                "schema_version": 1, "graph_id": graph["id"], "role": invitation["role"],
                "invitation": self._public_invitation({**invitation, "accepted_at": now}, now),
            }
        return result

    def revoke_invitation(self, actor_id, graph_id, invitation_id):
        repo = self.repository
        now = int(self.now())
        with repo._transaction(write=True) as connection:
            graph = self._owner(repo, connection, actor_id, graph_id)
            invitation = connection.execute(select(graph_invitations).where(and_(
                graph_invitations.c.id == invitation_id,
                graph_invitations.c.graph_id == graph_id,
            )).with_for_update()).mappings().first()
            if invitation is None:
                raise CloudGraphError("Invitation not found.", "not_found")
            if invitation["accepted_at"] is not None:
                raise CloudGraphError("This invitation was accepted. Remove the member's graph access instead.", "conflict")
            if invitation["revoked_at"] is None:
                connection.execute(graph_invitations.update().where(
                    graph_invitations.c.id == invitation_id
                ).values(revoked_at=now))
                repo._event(connection, graph, actor_id, "revoke_invitation")
            invitation = {
                **invitation,
                "revoked_at": invitation["revoked_at"] if invitation["revoked_at"] is not None else now,
            }
        return {
            "schema_version": 1, "graph_id": graph_id,
            "invitation": self._public_invitation(invitation, now),
        }

    def set_member_role(self, actor_id, graph_id, user_id, role):
        role = self._role(role)
        repo = self.repository
        with repo._transaction(write=True) as connection:
            graph = self._owner(repo, connection, actor_id, graph_id)
            predicate = and_(graph_members.c.graph_id == graph_id, graph_members.c.user_id == user_id)
            previous = connection.execute(select(graph_members.c.role).where(predicate)).scalar_one_or_none()
            if previous is None:
                raise CloudGraphError("Graph member not found.", "not_found")
            email = connection.execute(select(users.c.email).where(users.c.id == user_id)).scalar_one()
            revoked = connection.execute(graph_invitations.update().where(and_(
                graph_invitations.c.graph_id == graph_id,
                graph_invitations.c.email == email,
                graph_invitations.c.accepted_at.is_(None),
                graph_invitations.c.revoked_at.is_(None),
            )).values(revoked_at=int(self.now()))).rowcount
            if previous != role:
                connection.execute(graph_members.update().where(predicate).values(role=role))
            if previous != role or revoked:
                repo._event(connection, graph, actor_id, "set_member_role")
            result = {
                **repo._metadata(graph, "owner"), "member": {"user_id": user_id, "role": role},
                "revoked_invitations": revoked,
            }
        return result

    def remove_member(self, actor_id, graph_id, user_id):
        repo = self.repository
        with repo._transaction(write=True) as connection:
            graph = self._owner(repo, connection, actor_id, graph_id)
            if user_id == graph["owner_id"]:
                raise CloudGraphError("The graph owner's access cannot be removed.")
            email = connection.execute(select(users.c.email).where(users.c.id == user_id)).scalar_one_or_none()
            if email is None:
                raise CloudGraphError("Graph member not found.", "not_found")
            removed = connection.execute(graph_members.delete().where(and_(
                graph_members.c.graph_id == graph_id, graph_members.c.user_id == user_id,
            ))).rowcount > 0
            # Removing access must not leave an older, unused role-change
            # invitation able to grant it straight back after revocation.
            revoked = connection.execute(graph_invitations.update().where(and_(
                graph_invitations.c.graph_id == graph_id,
                graph_invitations.c.email == email,
                graph_invitations.c.accepted_at.is_(None),
                graph_invitations.c.revoked_at.is_(None),
            )).values(revoked_at=int(self.now()))).rowcount
            if removed or revoked:
                repo._event(connection, graph, actor_id, "remove_member")
            result = {**repo._metadata(graph, "owner"), "removed": removed, "revoked_invitations": revoked}
        return result


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _NewInvitation(_Body):
    email: Annotated[StrictStr, Field(min_length=1, max_length=320)]
    role: Literal["editor", "viewer"] = "viewer"
    ttl_seconds: Annotated[int, Field(strict=True, ge=1, le=MAX_INVITATION_TTL)] = DEFAULT_INVITATION_TTL


class _AcceptInvitation(_Body):
    token: Annotated[StrictStr, Field(min_length=43, max_length=43)]


class _MemberRole(_Body):
    role: Literal["editor", "viewer"]


def register_collaboration_routes(app, repository, authenticated_user):
    """Mount cookie-authenticated routes without accepting a caller actor ID."""

    service = CloudCollaboration(repository)
    app.state.collaboration = service

    @app.get("/api/graphs/{graph_id}/sharing")
    def sharing(graph_id: str, user=Depends(authenticated_user)):
        return service.list_access(user["id"], graph_id)

    @app.post("/api/graphs/{graph_id}/invitations", status_code=201)
    def invite(graph_id: str, body: _NewInvitation, user=Depends(authenticated_user)):
        return service.create_invitation(user["id"], graph_id, body.email, body.role, ttl_seconds=body.ttl_seconds)

    @app.post("/api/invitations/accept")
    def accept(body: _AcceptInvitation, user=Depends(authenticated_user)):
        return service.accept_invitation(user["id"], body.token)

    @app.delete("/api/graphs/{graph_id}/invitations/{invitation_id}")
    def revoke(graph_id: str, invitation_id: str, user=Depends(authenticated_user)):
        return service.revoke_invitation(user["id"], graph_id, invitation_id)

    @app.put("/api/graphs/{graph_id}/members/{user_id}")
    def member_role(graph_id: str, user_id: str, body: _MemberRole, actor=Depends(authenticated_user)):
        return service.set_member_role(actor["id"], graph_id, user_id, body.role)

    @app.delete("/api/graphs/{graph_id}/members/{user_id}")
    def remove_member(graph_id: str, user_id: str, actor=Depends(authenticated_user)):
        return service.remove_member(actor["id"], graph_id, user_id)

    return service
