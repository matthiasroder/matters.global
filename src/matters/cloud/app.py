"""Authenticated HTTP boundary for the first Matters Cloud pilot.

This application deliberately has no local filesystem or terminal API. Account
identity comes from an opaque session cookie, and every graph request passes
that identity to the permission-checking repository.
"""

from contextlib import asynccontextmanager
import ipaddress
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, urlsplit

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from .accounts import AccountError, CloudAccounts
from .db import CloudDatabaseError, Database
from .repository import CloudGraphError, GraphRepository
from .rate_limits import DatabaseRateLimiter, RateLimitUnavailable
from .lifecycle import AccountLifecycle


SESSION_COOKIE = "matters_session"
MAX_REQUEST_BYTES = 256 * 1024
ASSETS = Path(__file__).with_name("assets")
ID = Annotated[StrictStr, Field(min_length=1, max_length=128)]
Label = Annotated[StrictStr, Field(min_length=1, max_length=2000)]
Index = Annotated[int, Field(strict=True, ge=0)]


def error_response(code, message, status_code):
    return JSONResponse(
        {"schema_version": 1, "error": {"code": code, "message": message}},
        status_code=status_code,
    )


def normalized_origin(value):
    """Return a comparable exact origin, rejecting credentials and paths."""

    try:
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path not in {"", "/"}
            or parts.query
            or parts.fragment
        ):
            return None
        return parts.scheme, parts.hostname.lower(), parts.port or (443 if parts.scheme == "https" else 80)
    except (ValueError, TypeError):
        return None


def loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class CloudBoundaryMiddleware:
    """Bound requests, enforce browser origins and keep errors free of secrets."""

    def __init__(self, app, *, public_origin=None, auth_limiter=None):
        self.app = app
        self.public_origin = normalized_origin(public_origin) if public_origin else None
        self.auth_limiter = auth_limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        started = False

        async def safe_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                response_headers = list(message.get("headers", []))
                response_headers.extend([
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"content-security-policy", b"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"),
                ])
                # OAuth HTML forms need an origin-preserving policy: browsers
                # send Origin:null for navigation POSTs under no-referrer.
                # Respect the route's origin-only policy without adding a
                # second, overriding value. Other responses retain no-referrer.
                if not any(key.lower() == b"referrer-policy" for key, _ in response_headers):
                    response_headers.append((b"referrer-policy", b"no-referrer"))
                message = {**message, "headers": response_headers}
            await send(message)

        async def refuse(code, message, status, response_headers=None):
            response = error_response(code, message, status)
            if response_headers:
                response.headers.update(response_headers)
            await response(scope, receive, safe_send)

        is_api = scope.get("path", "").startswith("/api/")
        unsafe = scope["method"] not in {"GET", "HEAD", "OPTIONS"}
        if is_api and scope.get("query_string"):
            keys = [key for key, _ in parse_qsl(scope["query_string"].decode("latin-1"), keep_blank_values=True)]
            allowed = {"matter_id"} if "/views/" in scope.get("path", "") else set()
            if len(keys) != len(set(keys)) or set(keys) - allowed:
                await refuse("invalid", "Query fields do not match the API schema.", 400)
                return
        if (
            scope["method"] == "POST"
            and scope.get("path") in {"/api/auth/register", "/api/auth/login"}
            and self.auth_limiter is not None
        ):
            peer = (scope.get("client") or ("unknown",))[0]
            try:
                allowed = await run_in_threadpool(self.auth_limiter.allow, peer)
            except RateLimitUnavailable:
                await refuse("unavailable", "The security limit service is temporarily unavailable.", 503)
                return
            if not allowed:
                await refuse("rate_limited", "Too many sign-in attempts. Please try again shortly.", 429, {"Retry-After": str(self.auth_limiter.window_seconds)})
                return
        if is_api and unsafe:
            host = headers.get(b"host", b"").decode("latin-1")
            expected = self.public_origin or normalized_origin(f"{scope.get('scheme', 'http')}://{host}")
            origin = headers.get(b"origin")
            fetch_site = headers.get(b"sec-fetch-site", b"").decode("latin-1").lower()
            if (
                expected is None
                or fetch_site not in {"", "same-origin", "none"}
                or (origin is not None and normalized_origin(origin.decode("latin-1")) != expected)
            ):
                await refuse("forbidden", "Cross-origin requests are not permitted.", 403)
                return

        body_parts = []
        if is_api and unsafe:
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    declared_length = int(length)
                except ValueError:
                    declared_length = -1
                if declared_length < 0:
                    await refuse("invalid", "Invalid request size.", 400)
                    return
                if declared_length > MAX_REQUEST_BYTES:
                    await refuse("too_large", "The request is too large.", 413)
                    return
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > MAX_REQUEST_BYTES:
                    await refuse("too_large", "The request is too large.", 413)
                    return
                body_parts.append(chunk)
                if not message.get("more_body", False):
                    break
            if total and headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != b"application/json":
                await refuse("invalid", "API requests must use application/json.", 415)
                return

        delivered = False

        async def replay_receive():
            nonlocal delivered
            if is_api and unsafe and not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(body_parts), "more_body": False}
            return await receive()

        try:
            await self.app(scope, replay_receive, safe_send)
        except Exception:
            # Do not put exception strings (including DB URLs, bound SQL values
            # or credentials) into either the response or request logs.
            if not started:
                await refuse("internal", "The request could not be completed.", 500)
            else:
                raise


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Credentials(APIModel):
    email: Annotated[StrictStr, Field(min_length=1, max_length=254)]
    password: Annotated[StrictStr, Field(min_length=1, max_length=1024)]


class NewGraph(APIModel):
    title: Annotated[StrictStr, Field(min_length=1, max_length=200)]
    kind: Literal["personal", "project"] = "project"


class Condition(APIModel):
    label: Label
    truth: StrictBool = False


class NewMatter(APIModel):
    id: ID
    conditions: Annotated[list[Condition], Field(min_length=1, max_length=1000)]


class SetCondition(APIModel):
    matter_id: ID
    index: Index
    truth: StrictBool


class AddCondition(APIModel):
    matter_id: ID
    label: Label
    truth: StrictBool = False


class EditCondition(APIModel):
    matter_id: ID
    index: Index
    label: Label


class DeleteCondition(APIModel):
    matter_id: ID
    index: Index
    confirmed: StrictBool = False


class DeleteMatter(APIModel):
    matter_id: ID
    confirmed: StrictBool = False
    cascade: StrictBool = False


class Dependency(APIModel):
    source: ID
    target: ID


OPERATION_PAYLOADS = {
    "create_matter": NewMatter,
    "set_condition": SetCondition,
    "add_condition": AddCondition,
    "edit_condition": EditCondition,
    "delete_condition": DeleteCondition,
    "delete_matter": DeleteMatter,
    "add_dependency": Dependency,
    "remove_dependency": Dependency,
}


class GraphOperation(APIModel):
    operation: Literal[
        "create_matter", "set_condition", "add_condition", "edit_condition",
        "delete_condition", "delete_matter", "add_dependency", "remove_dependency",
    ]
    payload: dict[str, Any]
    expected_revision: Annotated[StrictStr, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_operation_payload(self):
        self.payload = OPERATION_PAYLOADS[self.operation].model_validate(self.payload).model_dump()
        return self


def create_app(database_url, *, secure_cookies=True, public_origin=None, allowed_hosts=None, enable_mcp=False, mail_delivery=None, mail_event_sink=None):
    """Create a pilot ASGI application owning its database-engine lifecycle."""

    if type(secure_cookies) is not bool:
        raise ValueError("secure_cookies must be a boolean")
    if type(enable_mcp) is not bool:
        raise ValueError("enable_mcp must be a boolean")
    parsed_origin = normalized_origin(public_origin) if public_origin else None
    if public_origin and parsed_origin is None:
        raise ValueError("public_origin must be an exact HTTP or HTTPS origin")
    if secure_cookies and parsed_origin and parsed_origin[0] != "https":
        raise ValueError("secure cookies require an HTTPS public_origin")
    if allowed_hosts is None:
        allowed_hosts = [parsed_origin[1]] if parsed_origin else ["localhost", "127.0.0.1", "[::1]"]
    if isinstance(allowed_hosts, (str, bytes)) or not allowed_hosts or any(not isinstance(host, str) or not host or "*" in host or "/" in host for host in allowed_hosts):
        raise ValueError("allowed_hosts must contain explicit hostnames")
    if parsed_origin and parsed_origin[1] not in allowed_hosts:
        raise ValueError("public_origin host must be trusted")
    if not secure_cookies and any(not loopback_host(host) for host in allowed_hosts):
        raise ValueError("insecure development cookies require loopback hosts")
    if enable_mcp and parsed_origin is None:
        raise ValueError("hosted MCP requires an explicit public_origin")
    if enable_mcp:
        # Fail before touching a database when the optional SDK is absent.
        from .mcp import create_hosted_mcp
        from .oauth import CloudOAuth
        from .oauth_routes import build_oauth_router

    db = Database(database_url)
    try:
        db.initialize()
    except Exception:
        db.close()
        raise
    accounts = CloudAccounts(db)
    graphs = GraphRepository(db)
    try:
        lifecycle = AccountLifecycle(db, accounts, public_origin=public_origin, delivery=mail_delivery, allow_loopback_http=not secure_cookies, event_sink=mail_event_sink)
    except Exception:
        db.close()
        raise
    limiter = DatabaseRateLimiter(db)
    hosted_mcp = None

    @asynccontextmanager
    async def lifespan(_app):
        try:
            if hosted_mcp is None:
                yield
            else:
                async with hosted_mcp.lifespan():
                    yield
        finally:
            await run_in_threadpool(lifecycle.close)
            db.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db = db
    app.state.accounts = accounts
    app.state.graphs = graphs
    app.state.auth_limiter = limiter
    app.state.lifecycle = lifecycle
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.add_middleware(
        CloudBoundaryMiddleware, public_origin=public_origin,
        auth_limiter=app.state.auth_limiter,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _error):
        return error_response("invalid", "Request fields do not match the API schema.", 422)

    @app.exception_handler(AccountError)
    async def account_error(_request, error):
        status = {"invalid": 400, "conflict": 409, "unauthorized": 401, "forbidden": 403, "rate_limited": 429, "unavailable": 503}.get(error.code, 400)
        response = error_response(error.code, error.message, status)
        if status == 429:
            response.headers["Retry-After"] = str(getattr(error, "retry_after", 60))
        return response

    @app.exception_handler(RateLimitUnavailable)
    async def security_limit_unavailable(_request, _error):
        return error_response("unavailable", "The security limit service is temporarily unavailable.", 503)

    @app.exception_handler(CloudGraphError)
    async def graph_error(_request, error):
        status = {"invalid": 400, "not_found": 404, "forbidden": 403, "conflict": 409, "state_cycle": 422, "unavailable": 503}.get(error.code, 400)
        return error_response(error.code, error.message, status)

    @app.exception_handler(CloudDatabaseError)
    async def database_error(_request, _error):
        return error_response("unavailable", "The service is temporarily unavailable.", 503)

    def authenticated_user(request: Request):
        token = request.cookies.get(SESSION_COOKIE)
        if not token or len(token) > 512:
            raise AccountError("Sign in to continue.", "unauthorized")
        return accounts.authenticate(token)

    from .transfer import create_transfer_router
    from .collaboration import register_collaboration_routes
    from .visual import register_visual_routes
    from .lifecycle_routes import create_lifecycle_router

    app.include_router(create_transfer_router(graphs, authenticated_user))
    register_collaboration_routes(app, graphs, authenticated_user)
    register_visual_routes(app, graphs, authenticated_user)
    app.include_router(create_lifecycle_router(lifecycle, authenticated_user, public_origin=public_origin, secure_cookies=secure_cookies, cookie_name=SESSION_COOKIE, limiter=limiter))

    @app.get("/healthz")
    def health():
        with db.transaction() as connection:
            connection.execute(text("SELECT 1"))
        return {"schema_version": 1, "status": "ok"}

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    @app.post("/api/auth/register", status_code=201)
    def register(credentials: Credentials):
        user = accounts.register(credentials.email, credentials.password)
        return {"schema_version": 1, "user": user}

    @app.post("/api/auth/login")
    def login(credentials: Credentials):
        session = accounts.login(credentials.email, credentials.password)
        response = JSONResponse({"schema_version": 1, "user": session["user"]})
        response.set_cookie(
            SESSION_COOKIE, session["token"], max_age=accounts.session_ttl_seconds,
            httponly=True, secure=secure_cookies, samesite="strict", path="/",
        )
        return response

    @app.post("/api/auth/logout")
    def logout(request: Request):
        accounts.logout(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse({"schema_version": 1, "logged_out": True})
        response.delete_cookie(SESSION_COOKIE, path="/", secure=secure_cookies, httponly=True, samesite="strict")
        return response

    @app.get("/api/auth/me")
    def me(user=Depends(authenticated_user)):
        return {"schema_version": 1, "user": user}

    @app.get("/api/graphs")
    def list_graphs(user=Depends(authenticated_user)):
        return graphs.list_graphs(user["id"])

    @app.post("/api/graphs", status_code=201)
    def create_graph(body: NewGraph, user=Depends(authenticated_user)):
        return graphs.create_graph(user["id"], body.title, body.kind)

    @app.get("/api/graphs/{graph_id}")
    def get_graph(graph_id: str, user=Depends(authenticated_user)):
        return graphs.get_graph(user["id"], graph_id)

    @app.post("/api/graphs/{graph_id}/operations")
    def operate(graph_id: str, body: GraphOperation, user=Depends(authenticated_user)):
        return graphs.mutate_graph(
            user["id"], graph_id, body.operation, body.payload,
            expected_revision=body.expected_revision,
        )

    @app.get("/api/graphs/{graph_id}/views/{view}")
    def query(
        graph_id: str,
        view: Literal["state", "matters", "matter", "universe", "frontier", "horizon", "unlock"],
        matter_id: str | None = None,
        user=Depends(authenticated_user),
    ):
        return graphs.query_graph(user["id"], graph_id, view, matter_id=matter_id)

    @app.get("/assets/local/map-renderer.js")
    def shared_map_renderer():
        return FileResponse(ASSETS.parent.parent / "web_assets" / "map-renderer.js")

    app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")
    if enable_mcp:
        # MCP and its SDK stay optional for ordinary Cloud/browser operation.
        oauth = CloudOAuth(db, issuer_url=public_origin, allow_loopback_http=not secure_cookies)
        app.state.oauth = oauth
        app.include_router(build_oauth_router(
            oauth, accounts, graphs, secure_cookies=secure_cookies, cookie_name=SESSION_COOKIE, limiter=limiter,
        ))
        hosted_mcp = create_hosted_mcp(oauth, graphs)
        app.state.hosted_mcp = hosted_mcp
        app.mount("/", hosted_mcp.app)
    return app
