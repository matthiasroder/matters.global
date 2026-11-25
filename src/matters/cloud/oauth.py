"""Persistent, graph-scoped OAuth grants for hosted MCP.

This layer authenticates clients and records the account holder's consent. The
graph repository still checks current membership on each graph operation. No
client metadata URL is fetched and no installation permission enters a token.
"""

import base64
import copy
import hashlib
import hmac
import ipaddress
import re
import secrets
import time
from contextlib import contextmanager
from urllib.parse import urlencode, urlsplit

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from .db import CloudDatabaseError, oauth_records, sessions, users
from .repository import CloudGraphError


SCOPES = ("graphs:read", "graphs:write", "offline_access")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
VERIFIER_PATTERN = re.compile(r"[A-Za-z0-9._~-]{43,128}")
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 86400
REQUEST_TTL = 600
CODE_TTL = 120


class OAuthError(ValueError):
    """A public OAuth error, containing no token, credential or database text."""

    def __init__(self, message="The authorization request is invalid.", code="invalid_request", status=400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def token_hash(value):
    if not isinstance(value, str) or not TOKEN_PATTERN.fullmatch(value):
        raise OAuthError("The token is invalid.", "invalid_token", 401)
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _url(value, *, loopback_http=False, origin=False):
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or any(ord(c) <= 32 or ord(c) >= 127 for c in value):
        raise OAuthError("A valid HTTPS URL is required.")
    try:
        parts = urlsplit(value)
        if (
            not parts.hostname or parts.username is not None or parts.password is not None
            or parts.fragment or "*" in value or "\\" in value
            or (parts.scheme != "https" and not (loopback_http and parts.scheme == "http" and _loopback(parts.hostname)))
            or (origin and (parts.path not in ("", "/") or parts.query))
        ):
            raise ValueError()
        _ = parts.port
    except (ValueError, TypeError):
        raise OAuthError("A valid HTTPS URL is required.") from None
    if origin:
        host = parts.hostname.lower()
        if ":" in host:
            host = "[" + host + "]"
        port = parts.port
        if port and port != (443 if parts.scheme == "https" else 80):
            host += ":" + str(port)
        return parts.scheme + "://" + host
    return value


def _scopes(value, *, default=False):
    if value is None and default:
        return ["graphs:read"]
    if not isinstance(value, str) or len(value) > 200:
        raise OAuthError("The requested scopes are invalid.", "invalid_scope")
    scopes = value.split()
    if not scopes or len(scopes) != len(set(scopes)) or set(scopes) - set(SCOPES):
        raise OAuthError("The requested scopes are invalid.", "invalid_scope")
    if "graphs:write" in scopes and "graphs:read" not in scopes:
        scopes.append("graphs:read")
    if "graphs:read" not in scopes:
        raise OAuthError("Graph read access is required.", "invalid_scope")
    return [scope for scope in SCOPES if scope in scopes]


class CloudOAuth:
    """Use one database for clients, pending consent, codes and token families.

    Opaque random credentials are returned once. Only SHA-256 digests persist.
    Used refresh records are retained until their original expiry to detect
    replay and revoke the entire grant, including newly issued access tokens.
    """

    def __init__(self, db, *, issuer_url, resource_url=None, allow_loopback_http=False, now=None):
        if type(allow_loopback_http) is not bool:
            raise ValueError("allow_loopback_http must be a boolean")
        self.db = db
        self.allow_loopback_http = allow_loopback_http
        self.issuer_url = _url(issuer_url, loopback_http=allow_loopback_http, origin=True)
        self.resource_url = _url(resource_url or self.issuer_url + "/mcp", loopback_http=allow_loopback_http)
        if urlsplit(self.resource_url).query:
            raise ValueError("The MCP resource URL cannot contain a query")
        self._now = now or time.time

    @contextmanager
    def _transaction(self, *, write=False):
        try:
            with self.db.transaction(write=write) as connection:
                yield connection
        except (SQLAlchemyError, CloudDatabaseError):
            raise OAuthError("The authorization service is unavailable.", "temporarily_unavailable", 503) from None

    def _insert(self, connection, kind, data, ttl, *, actor_id=None, client_id=None, raw=None):
        raw = raw or secrets.token_urlsafe(32)
        identity = token_hash(raw)
        now = int(self._now())
        connection.execute(oauth_records.insert().values(
            id=identity, kind=kind, user_id=actor_id, client_id=client_id,
            data=copy.deepcopy(data), created_at=now, expires_at=now + ttl,
        ))
        return raw, identity

    def _record(self, connection, raw, kind, *, lock=False, identity=False, allow_expired=False):
        try:
            record_id = raw if identity else token_hash(raw)
        except OAuthError:
            raise OAuthError("The authorization is invalid or expired.", "invalid_grant") from None
        statement = select(oauth_records).where(oauth_records.c.id == record_id, oauth_records.c.kind == kind)
        if lock:
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().one_or_none()
        if row is None or (not allow_expired and row["expires_at"] <= int(self._now())):
            raise OAuthError("The authorization is invalid or expired.", "invalid_grant")
        return {**row, "data": copy.deepcopy(row["data"])}

    @staticmethod
    def _save(connection, record):
        connection.execute(oauth_records.update().where(oauth_records.c.id == record["id"]).values(data=record["data"]))

    def metadata(self):
        return {
            "issuer": self.issuer_url,
            "authorization_endpoint": self.issuer_url + "/oauth/authorize",
            "token_endpoint": self.issuer_url + "/oauth/token",
            "registration_endpoint": self.issuer_url + "/oauth/register",
            "revocation_endpoint": self.issuer_url + "/oauth/revoke",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none"],
            "revocation_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": list(SCOPES),
            "authorization_response_iss_parameter_supported": True,
        }

    def protected_resource_metadata(self):
        return {"resource": self.resource_url, "authorization_servers": [self.issuer_url],
                "scopes_supported": list(SCOPES), "bearer_methods_supported": ["header"]}

    def register_client(self, document):
        if not isinstance(document, dict) or set(document) - {"redirect_uris", "client_name", "grant_types", "response_types", "token_endpoint_auth_method", "scope"}:
            raise OAuthError("Client metadata does not match the supported schema.", "invalid_client_metadata")
        redirects = document.get("redirect_uris")
        if not isinstance(redirects, list) or not 1 <= len(redirects) <= 10 or any(not isinstance(uri, str) for uri in redirects) or len(set(redirects)) != len(redirects):
            raise OAuthError("Register one to ten exact redirect URIs.", "invalid_redirect_uri")
        try:
            for uri in redirects:
                # Native PKCE clients receive their callback on a loopback
                # listener even when this authorization server uses HTTPS.
                # The full callback (including port) is registered exactly.
                _url(uri, loopback_http=True)
        except OAuthError:
            raise OAuthError("Redirect URIs must use HTTPS.", "invalid_redirect_uri") from None
        name = document.get("client_name", "MCP client")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise OAuthError("The client name is invalid.", "invalid_client_metadata")
        if document.get("token_endpoint_auth_method", "none") != "none":
            raise OAuthError("Only public clients using PKCE are supported.", "invalid_client_metadata")
        grants = document.get("grant_types", ["authorization_code", "refresh_token"])
        if not isinstance(grants, list) or any(not isinstance(grant, str) for grant in grants) or "authorization_code" not in grants or len(grants) != len(set(grants)) or set(grants) - {"authorization_code", "refresh_token"}:
            raise OAuthError("Unsupported client grant types.", "invalid_client_metadata")
        if document.get("response_types", ["code"]) != ["code"]:
            raise OAuthError("Only authorization code responses are supported.", "invalid_client_metadata")
        scopes = _scopes(document.get("scope"), default=True)
        client_id = secrets.token_urlsafe(32)
        metadata = {"client_id": client_id, "client_name": name.strip(), "redirect_uris": redirects,
                    "grant_types": grants, "response_types": ["code"], "token_endpoint_auth_method": "none",
                    "scope": " ".join(scopes), "client_id_issued_at": int(self._now())}
        # Client identifiers are public; their database key is still a digest,
        # so the same lookup function cannot accidentally disclose raw secrets.
        with self._transaction(write=True) as connection:
            self._insert(connection, "client", metadata, 10 * 365 * 86400, client_id=client_id, raw=client_id)
        return metadata

    def _client(self, connection, client_id):
        try:
            return self._record(connection, client_id, "client")["data"]
        except OAuthError:
            raise OAuthError("The client is invalid.", "invalid_client", 401) from None

    def create_authorization_request(self, parameters):
        allowed = {"client_id", "redirect_uri", "response_type", "state", "scope", "resource", "code_challenge", "code_challenge_method"}
        if not isinstance(parameters, dict) or set(parameters) - allowed:
            raise OAuthError()
        if parameters.get("response_type") != "code":
            raise OAuthError("Only code responses are supported.", "unsupported_response_type")
        if parameters.get("resource") != self.resource_url:
            raise OAuthError("The requested resource is invalid.", "invalid_target")
        challenge = parameters.get("code_challenge")
        if parameters.get("code_challenge_method") != "S256" or not isinstance(challenge, str) or not TOKEN_PATTERN.fullmatch(challenge):
            raise OAuthError("PKCE using S256 is required.")
        state = parameters.get("state")
        if state is not None and (not isinstance(state, str) or not 1 <= len(state) <= 1024 or any(ord(c) < 32 for c in state)):
            raise OAuthError("The client state is invalid.")
        scopes = _scopes(parameters.get("scope"), default=True)
        with self._transaction(write=True) as connection:
            client = self._client(connection, parameters.get("client_id"))
            if parameters.get("redirect_uri") not in client["redirect_uris"]:
                raise OAuthError("The redirect URI was not registered.")
            if "offline_access" in scopes and "refresh_token" not in client["grant_types"]:
                raise OAuthError("This client does not support offline access.", "invalid_scope")
            data = {**parameters, "scopes": scopes, "client_name": client["client_name"], "used": False}
            raw, _ = self._insert(connection, "request", data, REQUEST_TTL, client_id=client["client_id"])
        return raw

    def inspect_request(self, request_token):
        with self._transaction() as connection:
            row = self._record(connection, request_token, "request")
            if row["data"].get("used"):
                raise OAuthError("This authorization request has already been completed.")
            return row["data"]

    def prepare_consent(self, request_token, actor_id, session_token, csrf_token):
        session_digest, csrf_digest = token_hash(session_token), token_hash(csrf_token)
        with self._transaction(write=True) as connection:
            request = self._record(connection, request_token, "request", lock=True)
            data = request["data"]
            if data.get("used") or (data.get("actor_id") and data["actor_id"] != actor_id):
                raise OAuthError("Start a new authorization request for this account.")
            if connection.execute(select(users.c.id).where(users.c.id == actor_id)).first() is None:
                raise OAuthError("Sign in to continue.", "access_denied", 401)
            data.update(actor_id=actor_id, session_hash=session_digest, csrf_hash=csrf_digest)
            self._save(connection, request)
            return copy.deepcopy(data)

    def consent(self, request_token, actor_id, session_token, csrf_token, graph_ids, *, repository, approve=True):
        if type(approve) is not bool or not isinstance(graph_ids, list) or any(not isinstance(g, str) for g in graph_ids) or len(graph_ids) > 100:
            raise OAuthError()
        with self._transaction(write=True) as connection:
            self._lock_user(connection, actor_id)
            active_session = connection.execute(select(sessions.c.user_id).where(
                sessions.c.token_hash == token_hash(session_token), sessions.c.user_id == actor_id,
                sessions.c.expires_at > int(self._now()),
            )).first()
            if active_session is None:
                raise OAuthError("Sign in again before authorizing a connection.", "access_denied", 401)
            request = self._record(connection, request_token, "request", lock=True)
            data = request["data"]
            if data.get("used") or data.get("actor_id") != actor_id or not hmac.compare_digest(data.get("session_hash", ""), token_hash(session_token)) or not hmac.compare_digest(data.get("csrf_hash", ""), token_hash(csrf_token)):
                raise OAuthError("Consent must come from the signed-in browser.", "access_denied", 403)
            data["used"] = True
            if not approve:
                response = {"error": "access_denied"}
            else:
                if not graph_ids or len(set(graph_ids)) != len(graph_ids):
                    raise OAuthError("Choose at least one graph.")
                # Use the same graph row locks as writes and membership changes.
                for graph_id in sorted(graph_ids):
                    try:
                        _, role = repository._authorized_graph(connection, actor_id, graph_id, write=True)
                    except CloudGraphError:
                        raise OAuthError("Graph not found.", "access_denied", 404) from None
                    if "graphs:write" in data["scopes"] and role not in ("owner", "editor"):
                        raise OAuthError("This graph does not permit editing.", "access_denied", 403)
                _, grant_id = self._insert(connection, "grant", {"graph_ids": graph_ids, "scopes": data["scopes"], "resource": self.resource_url, "revoked": False}, REFRESH_TTL, actor_id=actor_id, client_id=request["client_id"])
                code, _ = self._insert(connection, "code", {"grant_id": grant_id, "code_challenge": data["code_challenge"], "redirect_uri": data["redirect_uri"], "resource": self.resource_url, "used": False}, CODE_TTL, actor_id=actor_id, client_id=request["client_id"])
                response = {"code": code}
            self._save(connection, request)
            response["iss"] = self.issuer_url
            if data.get("state") is not None:
                response["state"] = data["state"]
            separator = "&" if urlsplit(data["redirect_uri"]).query else "?"
            redirect = data["redirect_uri"] + separator + urlencode(response)
        return redirect

    @staticmethod
    def _lock_user(connection, actor_id):
        if not actor_id or connection.execute(select(users.c.id).where(users.c.id == actor_id).with_for_update()).first() is None:
            raise OAuthError("The authorization is invalid or expired.", "invalid_grant")

    def _grant(self, connection, identity, *, lock=False):
        grant = self._record(connection, identity, "grant", identity=True, lock=lock)
        if grant["data"].get("revoked") or connection.execute(select(users.c.id).where(users.c.id == grant["user_id"])).first() is None:
            raise OAuthError("The authorization is invalid or expired.", "invalid_grant")
        return grant

    def _issue(self, connection, grant, scopes):
        common = {"grant_id": grant["id"], "resource": self.resource_url, "scopes": scopes}
        access_ttl = min(ACCESS_TTL, grant["expires_at"] - int(self._now()))
        access, _ = self._insert(connection, "access", common, access_ttl, actor_id=grant["user_id"], client_id=grant["client_id"])
        result = {"access_token": access, "token_type": "Bearer", "expires_in": access_ttl, "scope": " ".join(scopes)}
        if "offline_access" in scopes:
            refresh, _ = self._insert(connection, "refresh", {**common, "used": False}, max(1, grant["expires_at"] - int(self._now())), actor_id=grant["user_id"], client_id=grant["client_id"])
            result["refresh_token"] = refresh
        return result

    def exchange_code(self, *, client_id, code, redirect_uri, code_verifier, resource):
        if resource != self.resource_url:
            raise OAuthError("The requested resource is invalid.", "invalid_target")
        if not isinstance(code_verifier, str) or not VERIFIER_PATTERN.fullmatch(code_verifier):
            raise OAuthError("The authorization code is invalid.", "invalid_grant")
        challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        replay = False
        with self._transaction(write=True) as connection:
            self._client(connection, client_id)
            hint = self._record(connection, code, "code")
            self._lock_user(connection, hint["user_id"])
            row = self._record(connection, code, "code", lock=True)
            if row["client_id"] != client_id or row["data"]["redirect_uri"] != redirect_uri or row["data"]["resource"] != resource or not hmac.compare_digest(row["data"]["code_challenge"], challenge):
                raise OAuthError("The authorization code is invalid.", "invalid_grant")
            grant = self._grant(connection, row["data"]["grant_id"], lock=True)
            if row["data"]["used"]:
                grant["data"]["revoked"] = True
                self._save(connection, grant)
                replay = True
            else:
                row["data"]["used"] = True
                self._save(connection, row)
                result = self._issue(connection, grant, grant["data"]["scopes"])
        if replay:
            raise OAuthError("The authorization code has already been used.", "invalid_grant")
        return result

    def refresh(self, *, client_id, refresh_token, resource, scope=None):
        if resource != self.resource_url:
            raise OAuthError("The requested resource is invalid.", "invalid_target")
        replay = False
        with self._transaction(write=True) as connection:
            client = self._client(connection, client_id)
            if "refresh_token" not in client["grant_types"]:
                raise OAuthError("This client cannot refresh tokens.", "unauthorized_client")
            hint = self._record(connection, refresh_token, "refresh")
            self._lock_user(connection, hint["user_id"])
            row = self._record(connection, refresh_token, "refresh", lock=True)
            if row["client_id"] != client_id or row["data"]["resource"] != resource:
                raise OAuthError("The refresh token is invalid.", "invalid_grant")
            grant = self._grant(connection, row["data"]["grant_id"], lock=True)
            if row["data"]["used"]:
                grant["data"]["revoked"] = True
                self._save(connection, grant)
                replay = True
            else:
                scopes = _scopes(scope) if scope is not None else row["data"]["scopes"]
                if set(scopes) - set(row["data"]["scopes"]):
                    raise OAuthError("Refresh cannot increase access.", "invalid_scope")
                row["data"]["used"] = True
                self._save(connection, row)
                result = self._issue(connection, grant, scopes)
        if replay:
            raise OAuthError("The refresh token has already been used.", "invalid_grant")
        return result

    def validate_access_token(self, rawtoken, *, resource=None):
        expected_resource = self.resource_url if resource is None else resource
        try:
            with self._transaction() as connection:
                row = self._record(connection, rawtoken, "access")
                grant = self._grant(connection, row["data"]["grant_id"])
                if row["data"]["resource"] != expected_resource or expected_resource != self.resource_url or row["user_id"] != grant["user_id"] or row["client_id"] != grant["client_id"] or set(row["data"]["scopes"]) - set(grant["data"]["scopes"]):
                    raise OAuthError()
                return {"actor_id": row["user_id"], "graph_ids": list(grant["data"]["graph_ids"]), "scopes": list(row["data"]["scopes"]), "resource": expected_resource, "expires_at": min(row["expires_at"], grant["expires_at"]), "client_id": row["client_id"], "token_id": row["id"], "grant_id": grant["id"]}
        except OAuthError as error:
            if error.status == 503:
                raise
            raise OAuthError("The access token is invalid or expired.", "invalid_token", 401) from None

    def revoke(self, *, client_id, token):
        with self._transaction(write=True) as connection:
            self._client(connection, client_id)
            try:
                identity = token_hash(token)
            except OAuthError:
                return
            row = connection.execute(select(oauth_records).where(oauth_records.c.id == identity).with_for_update()).mappings().one_or_none()
            if row is None or row["kind"] not in ("access", "refresh") or row["client_id"] != client_id:
                return
            try:
                grant = self._record(connection, row["data"]["grant_id"], "grant", identity=True, lock=True, allow_expired=True)
            except OAuthError:
                return
            grant["data"]["revoked"] = True
            self._save(connection, grant)

    def list_grants(self, actor_id):
        with self._transaction() as connection:
            rows = connection.execute(select(oauth_records).where(
                oauth_records.c.kind == "grant", oauth_records.c.user_id == actor_id,
                oauth_records.c.expires_at > int(self._now()),
            ).order_by(oauth_records.c.created_at, oauth_records.c.id)).mappings()
            result = []
            for row in rows:
                if row["data"].get("revoked"):
                    continue
                client = self._client(connection, row["client_id"])
                result.append({"grant_id": row["id"], "client_name": client["client_name"],
                               "graph_ids": list(row["data"]["graph_ids"]), "scopes": list(row["data"]["scopes"]),
                               "created_at": row["created_at"], "expires_at": row["expires_at"]})
            return result

    def revoke_grant(self, actor_id, grant_id):
        with self._transaction(write=True) as connection:
            grant = self._record(connection, grant_id, "grant", identity=True, lock=True, allow_expired=True)
            if grant["user_id"] != actor_id:
                raise OAuthError("Connection not found.", "access_denied", 404)
            grant["data"]["revoked"] = True
            self._save(connection, grant)
