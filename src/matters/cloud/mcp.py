"""Protected Streamable HTTP transport for the shared Matters MCP tools.

OAuth owns account identity and selected-graph grants. The hosted graph service
revalidates that grant on each operation and the repository checks current graph
membership inside its transaction. No caller-provided identity is accepted.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import anyio
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.auth.routes import build_resource_metadata_url
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse
from starlette.authentication import AuthenticationError
from starlette.middleware.authentication import AuthenticationMiddleware
from pydantic import AnyHttpUrl

from ..local_service import LocalServiceError
from ..mcp_server import MattersMCP, _error, register_mcp_tools
from .oauth import OAuthError
from .service import HostedGraphService


MAX_REQUEST_BYTES = 256 * 1024
MAX_CONCURRENT_REQUESTS = 64


class AuthorizationUnavailable(AuthenticationError):
    """A safe authentication-storage failure before response/body processing."""


class CloudTokenVerifier:
    """MCP SDK bearer verifier, called before any MCP JSON parsing."""

    def __init__(self, oauth, resource_url):
        self.oauth = oauth
        self.resource_url = resource_url

    async def verify_token(self, token):
        try:
            principal = self.oauth.validate_access_token(token, resource=self.resource_url)
        except OAuthError as error:
            if error.status == 503:
                raise AuthorizationUnavailable() from None
            return None
        return AccessToken(
            token=token,
            client_id=principal["client_id"],
            scopes=list(principal["scopes"]),
            expires_at=principal["expires_at"],
            resource=principal["resource"],
            subject=principal["actor_id"],
            claims={"iss": self.oauth.issuer_url},
        )


class HostedMattersMCP(MattersMCP):
    """Revalidate every tool call, including unknown tools and bad arguments."""

    principal_provider = None
    resource_url = None

    async def list_tools(self):
        tools = await super().list_tools()
        for tool in tools:
            scopes = ["graphs:read"]
            if tool.annotations is not None and tool.annotations.readOnlyHint is False:
                scopes.append("graphs:write")
            schemes = [{"type": "oauth2", "scopes": scopes}]
            tool.meta = {**(tool.meta or {}), "securitySchemes": schemes}
            tool.securitySchemes = schemes
        return tools

    def authentication_challenge(self, code, message, *, error, scopes):
        result = _error(code, message)
        metadata_url = build_resource_metadata_url(AnyHttpUrl(self.resource_url))
        result.meta = {
            "mcp/www_authenticate": [
                f'Bearer resource_metadata="{metadata_url}", error="{error}", '
                f'error_description="{message}", scope="{" ".join(scopes)}"'
            ]
        }
        return result

    async def call_tool(self, name, arguments):
        tool = self._tool_manager.get_tool(name)
        needs_write = tool is not None and tool.annotations is not None and tool.annotations.readOnlyHint is False
        scopes = ["graphs:read", "graphs:write"] if needs_write else ["graphs:read"]
        try:
            principal = self.principal_provider()
        except LocalServiceError as error:
            if error.code == "unauthorized":
                return self.authentication_challenge(
                    error.code, error.message, error="invalid_token", scopes=scopes,
                )
            return _error(error.code, error.message, cycle=error.cycle)
        if (
            needs_write
            and "graphs:write" not in principal["scopes"]
        ):
            return self.authentication_challenge(
                "forbidden", "This operation requires graph write authorization.",
                error="insufficient_scope", scopes=["graphs:read", "graphs:write"],
            )
        return await super().call_tool(name, arguments)


class BoundedMCPRequests:
    """Fail fast at a finite process-local request bound.

    The transport is stateless, so it retains no MCP sessions or session-owned
    credentials between requests. This bound also limits concurrent stream
    connections; it is an operational guard, not a distributed usage quota.
    """

    def __init__(self, app, *, maximum):
        self.app = app
        self.limiter = anyio.Semaphore(maximum)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            self.limiter.acquire_nowait()
        except anyio.WouldBlock:
            response = JSONResponse(
                {"error": "temporarily_unavailable", "error_description": "Too many concurrent MCP requests."},
                status_code=503,
                headers={"Retry-After": "1", "Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return

        async def safe_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")])
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, safe_send)
        finally:
            self.limiter.release()


@dataclass(frozen=True)
class HostedMCP:
    server: HostedMattersMCP
    app: BoundedMCPRequests

    @asynccontextmanager
    async def lifespan(self):
        """Enter once from the parent ASGI app's lifespan when mounted."""

        async with self.server.session_manager.run():
            yield


def create_hosted_mcp(
    oauth,
    repository,
    *,
    resource_url=None,
    allowed_hosts=None,
    allowed_origins=None,
    max_request_bytes=MAX_REQUEST_BYTES,
    max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
):
    """Build the protected ``/mcp`` ASGI component without opening a socket.

    Include the OAuth discovery/authorization router before mounting ``app``
    last at ``/``. A parent app must explicitly enter ``lifespan()``; mounted
    Starlette applications do not receive their own lifespan automatically.
    A standalone TestClient can instead enter this component app's lifespan.
    """

    resource_url = resource_url or oauth.resource_url
    if resource_url != oauth.resource_url:
        raise ValueError("The MCP resource must match the OAuth resource configuration.")
    parsed = urlsplit(resource_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path != "/mcp" or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("The MCP resource must be an exact HTTP(S) /mcp URL.")
    if type(max_request_bytes) is not int or max_request_bytes < 1:
        raise ValueError("max_request_bytes must be a positive integer.")
    if type(max_concurrent_requests) is not int or max_concurrent_requests < 1:
        raise ValueError("max_concurrent_requests must be a positive integer.")
    if allowed_hosts is None:
        allowed_hosts = [parsed.netloc]
    if allowed_origins is None:
        allowed_origins = [f"{parsed.scheme}://{parsed.netloc}"]
    if isinstance(allowed_hosts, (str, bytes)) or not allowed_hosts or any(not isinstance(host, str) or not host or "*" in host for host in allowed_hosts):
        raise ValueError("MCP requires explicit allowed Host values.")
    if isinstance(allowed_origins, (str, bytes)) or any(not isinstance(origin, str) or not origin or "*" in origin for origin in allowed_origins):
        raise ValueError("MCP requires explicit allowed Origin values.")

    def principal_provider():
        access_token = get_access_token()
        if access_token is None:
            raise LocalServiceError("MCP authorization is required.", "unauthorized")
        try:
            return oauth.validate_access_token(access_token.token, resource=resource_url)
        except OAuthError as error:
            if error.status == 503:
                raise LocalServiceError("The authorization service is unavailable.", "unavailable") from None
            raise LocalServiceError("MCP authorization is no longer valid.", "unauthorized") from None

    service = HostedGraphService(repository, principal_provider)
    server = HostedMattersMCP(
        "Matters",
        instructions=(
            "Matters tool contract schema_version=1. Call list_graphs and select "
            "the intended graph. Every graph operation requires graph_id; project "
            "references select a graph but never grant access. Do not assume "
            "ChatGPT sends its project identity. Access is restricted to the "
            "authenticated account, selected graph grant and current membership. "
            "Reads return revision; every mutation requires expected_revision "
            "from the reviewed snapshot. On conflict, read and reassess before "
            "retrying. Condition indices are 0-based and truth is a JSON boolean. "
            "Verify completion evidence and obtain authorization for deletions. "
            "No tool accepts server file paths or shell commands."
        ),
        log_level="WARNING",
        json_response=True,
        stateless_http=True,
        streamable_http_path="/mcp",
        max_request_body_size=max_request_bytes,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed_hosts),
            allowed_origins=list(allowed_origins),
        ),
        token_verifier=CloudTokenVerifier(oauth, resource_url),
        auth=AuthSettings(
            issuer_url=oauth.issuer_url,
            resource_server_url=resource_url,
            validate_token_resource=True,
            required_scopes=["graphs:read"],
        ),
    )
    server.principal_provider = principal_provider
    server.resource_url = resource_url
    register_mcp_tools(server, service)
    http_app = server.streamable_http_app()

    def authentication_error(_connection, error):
        # AuthenticationMiddleware catches this before Starlette's generic
        # server-error middleware could start a 500 response. Keep the SDK's
        # bearer verifier and protected-resource challenge handling intact.
        if isinstance(error, AuthorizationUnavailable):
            return JSONResponse(
                {"error": "temporarily_unavailable", "error_description": "The authorization service is unavailable."},
                status_code=503, headers={"Retry-After": "1"},
            )
        return JSONResponse({"error": "invalid_token"}, status_code=401)

    for middleware in http_app.user_middleware:
        if middleware.cls is AuthenticationMiddleware:
            middleware.kwargs["on_error"] = authentication_error
            break
    else:  # Fail closed if a future SDK stops installing its auth boundary.
        raise RuntimeError("The MCP authentication boundary is unavailable.")
    app = BoundedMCPRequests(http_app, maximum=max_concurrent_requests)
    return HostedMCP(server, app)
