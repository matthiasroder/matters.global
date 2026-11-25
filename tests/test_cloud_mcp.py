"""Real MCP SDK HTTP calls with persistent OAuth and graph permissions."""

import asyncio
import base64
import hashlib
import json
import socket
import ssl
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp")
pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("httpx")

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from matters.cloud.accounts import CloudAccounts
from matters.cloud.db import Database
from matters.cloud.mcp import create_hosted_mcp
from matters.cloud.oauth import CloudOAuth, OAuthError
from matters.cloud.repository import GraphRepository
from matters.mcp_server import create_mcp_server


EXAMPLE = Path(__file__).parents[1] / "examples" / "matters.example.json"
PASSWORD = "Repository-only-example-password-42"
ORIGIN = "https://matters.example"


@pytest.fixture
def system(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'mcp.sqlite'}").initialize()
    accounts = CloudAccounts(db)
    alice = accounts.register("alice@example.com", PASSWORD)
    bob = accounts.register("bob@example.com", PASSWORD)
    repository = GraphRepository(db)
    state = json.loads(EXAMPLE.read_text())
    graph = repository.create_graph(alice["id"], "Alice project", "project", state=state)
    other = repository.create_graph(bob["id"], "Bob private", "personal", state=state)
    oauth = CloudOAuth(db, issuer_url=ORIGIN)
    yield {"db": db, "accounts": accounts, "alice": alice, "bob": bob,
           "repository": repository, "graph": graph, "other": other, "oauth": oauth}
    db.close()


def authorize(system, *, actor="alice", graph_ids=None, scope="graphs:read graphs:write offline_access", oauth=None):
    oauth = oauth or system["oauth"]
    user = system[actor]
    session = system["accounts"].login(user["email"], PASSWORD)
    redirect_uri = "https://client.example/callback"
    client = oauth.register_client({"redirect_uris": [redirect_uri], "client_name": "SDK HTTP checks", "scope": scope})
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    request = oauth.create_authorization_request({
        "client_id": client["client_id"], "redirect_uri": redirect_uri,
        "response_type": "code", "resource": oauth.resource_url, "scope": scope,
        "code_challenge": challenge, "code_challenge_method": "S256", "state": "test-state",
    })
    csrf = "c" * 43
    oauth.prepare_consent(request, user["id"], session["token"], csrf)
    redirect = oauth.consent(
        request, user["id"], session["token"], csrf,
        graph_ids or [system["graph"]["graph_id"]], repository=system["repository"],
    )
    code = parse_qs(urlsplit(redirect).query)["code"][0]
    tokens = oauth.exchange_code(
        client_id=client["client_id"], code=code, redirect_uri=redirect_uri,
        code_verifier=verifier, resource=oauth.resource_url,
    )
    return {**tokens, "client_id": client["client_id"]}


@asynccontextmanager
async def sdk_client(component, token, *, origin=ORIGIN):
    async with component.lifespan():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=component.app), base_url=origin,
            headers={"Authorization": f"Bearer {token}"},
        ) as http:
            async with streamable_http_client(origin + "/mcp", http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    await client.initialize()
                    yield client, http


def payload(result):
    assert result.isError is False, result
    assert result.structuredContent["schema_version"] == 1
    return result.structuredContent


def test_sdk_http_exposes_identical_tools_and_revision_checked_edits(system, tmp_path):
    token = authorize(system)["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"])
    local = create_mcp_server({"example": EXAMPLE})

    async def check():
        async with sdk_client(component, token) as (client, _http):
            hosted_tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            local_tools = {tool.name: tool for tool in await local.list_tools()}
            assert set(hosted_tools) == set(local_tools)
            assert len(hosted_tools) == 16
            for name in hosted_tools:
                assert hosted_tools[name].inputSchema == local_tools[name].inputSchema
                assert hosted_tools[name].outputSchema == local_tools[name].outputSchema
            assert hosted_tools["get_matter"].meta["securitySchemes"] == [{"type": "oauth2", "scopes": ["graphs:read"]}]
            assert hosted_tools["set_condition"].meta["securitySchemes"] == [{"type": "oauth2", "scopes": ["graphs:read", "graphs:write"]}]
            catalogue = payload(await client.call_tool("list_graphs", {}))
            assert [graph["graph_id"] for graph in catalogue["graphs"]] == [system["graph"]["graph_id"]]
            assert system["other"]["graph_id"] not in json.dumps(catalogue)
            graph_id = system["graph"]["graph_id"]
            before = payload(await client.call_tool("get_matter", {"graph_id": graph_id, "matter_id": "ship_first_version"}))
            result = payload(await client.call_tool("set_condition", {
                "graph_id": graph_id, "matter_id": "ship_first_version", "index": 0,
                "truth": True, "expected_revision": before["revision"],
            }))
            assert result["revision"] != before["revision"]
            reread = payload(await client.call_tool("get_matter", {"graph_id": graph_id, "matter_id": "ship_first_version"}))
            assert reread["conditions"][0]["truth"] is True
            stale = await client.call_tool("set_condition", {
                "graph_id": graph_id, "matter_id": "ship_first_version", "index": 0,
                "truth": False, "expected_revision": before["revision"],
            })
            assert stale.isError and stale.structuredContent["error"]["code"] == "conflict"
            assert system["repository"].get_graph(system["alice"]["id"], graph_id)["state"]["conditions"]["ship_first_version"][0]["truth"] is True

    asyncio.run(check())


def test_sdk_http_foreign_unknown_badtypes_cycle_and_shell_requests_are_safe(system):
    token = authorize(system)["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"])
    graph_id = system["graph"]["graph_id"]
    original = system["repository"].get_graph(system["alice"]["id"], graph_id)

    async def check():
        async with sdk_client(component, token) as (client, _http):
            for inaccessible in (system["other"]["graph_id"], "unknown-graph"):
                response = await client.call_tool("universe", {"graph_id": inaccessible})
                assert response.isError and response.structuredContent["error"]["code"] == "forbidden"
                assert "Bob private" not in response.content[0].text
            for field, value in (("truth", "false"), ("truth", 1), ("index", True), ("index", "0"), ("index", 0.0)):
                arguments = {"graph_id": graph_id, "matter_id": "ship_first_version", "index": 0, "truth": True, "expected_revision": original["revision"]}
                arguments[field] = value
                response = await client.call_tool("set_condition", arguments)
                assert response.isError and response.structuredContent["error"]["code"] == "invalid"
            response = await client.call_tool("add_dependency", {"graph_id": graph_id, "source": "ship_first_version", "target": "define_project_goal", "expected_revision": original["revision"]})
            assert response.isError
            response = await client.call_tool("run_shell", {"command": "whoami"})
            assert response.isError and response.structuredContent["error"]["code"] == "not_found"
            response = await client.call_tool("universe", {"graph_id": graph_id, "actor_id": system["bob"]["id"]})
            assert response.isError and response.structuredContent["error"]["code"] == "invalid"
            assert system["repository"].get_graph(system["alice"]["id"], graph_id) == original

    asyncio.run(check())


def test_sdk_http_read_scope_and_current_viewer_role_prevent_writes(system):
    graph_id = system["graph"]["graph_id"]
    readonly = authorize(system, scope="graphs:read")
    system["repository"].set_member(system["alice"]["id"], graph_id, system["bob"]["id"], "editor")
    downgraded = authorize(system, actor="bob")
    system["repository"].set_member(system["alice"]["id"], graph_id, system["bob"]["id"], "viewer")

    async def check(token):
        component = create_hosted_mcp(system["oauth"], system["repository"])
        async with sdk_client(component, token) as (client, _http):
            catalogue = payload(await client.call_tool("list_graphs", {}))
            assert catalogue["graphs"][0]["read_only"] is True
            response = await client.call_tool("set_condition", {"graph_id": graph_id, "matter_id": "ship_first_version", "index": 0, "truth": True, "expected_revision": system["graph"]["revision"]})
            assert response.isError and response.structuredContent["error"]["code"] == "forbidden"
            if "graphs:write" not in system["oauth"].validate_access_token(token)["scopes"]:
                challenge = response.meta["mcp/www_authenticate"][0]
                assert 'error="insufficient_scope"' in challenge
                assert 'scope="graphs:read graphs:write"' in challenge
                assert token not in challenge
            else:
                assert not response.meta or "mcp/www_authenticate" not in response.meta

    asyncio.run(check(readonly["access_token"]))
    asyncio.run(check(downgraded["access_token"]))


def test_sdk_http_membership_revocation_is_effective_without_reconnecting(system):
    graph_id = system["graph"]["graph_id"]
    system["repository"].set_member(system["alice"]["id"], graph_id, system["bob"]["id"], "editor")
    token = authorize(system, actor="bob")["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"])

    async def check():
        async with sdk_client(component, token) as (client, _http):
            assert payload(await client.call_tool("universe", {"graph_id": graph_id}))["matters"] == ["ship_first_version"]
            system["repository"].remove_member(system["alice"]["id"], graph_id, system["bob"]["id"])
            response = await client.call_tool("universe", {"graph_id": graph_id})
            assert response.isError and response.structuredContent["error"]["code"] == "not_found"
            assert payload(await client.call_tool("list_graphs", {}))["graphs"] == []

    asyncio.run(check())


def test_sdk_http_sources_and_export_preserve_data_without_permissions(system):
    sources = {"schema_version": 1, "matter_sources": {"ship_first_version": [{"title": "Pilot evidence", "url": "https://example.com/evidence", "authors": ["A. Example"]}]}}
    bundle = {"format": "matters-graph", "schema_version": 1, "graph": json.loads(EXAMPLE.read_text()), "sources": sources}
    imported = system["repository"].import_graph(system["alice"]["id"], "Imported evidence", bundle)
    token = authorize(system, graph_ids=[imported["graph_id"]])["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"])

    async def check():
        async with sdk_client(component, token) as (client, _http):
            described = payload(await client.call_tool("get_matter", {"graph_id": imported["graph_id"], "matter_id": "ship_first_version", "include_sources": True}))
            assert described["sources"] == sources["matter_sources"]["ship_first_version"]
            exported = payload(await client.call_tool("export_graph", {"graph_id": imported["graph_id"]}))
            assert exported["bundle"] == bundle
            assert system["alice"]["id"] not in json.dumps(exported["bundle"])
            assert "permissions" not in exported["bundle"]
            refused = await client.call_tool("get_matter", {"graph_id": system["graph"]["graph_id"], "matter_id": "ship_first_version"})
            assert refused.isError and refused.structuredContent["error"]["code"] == "forbidden"

    asyncio.run(check())


def test_sdk_concurrent_edits_preserve_one_winner_and_one_conflict(system):
    token = authorize(system)["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"])
    graph = system["graph"]

    async def check():
        async with sdk_client(component, token) as (client, _http):
            results = await asyncio.gather(
                client.call_tool("set_condition", {"graph_id": graph["graph_id"], "matter_id": "ship_first_version", "index": 0, "truth": True, "expected_revision": graph["revision"]}),
                client.call_tool("edit_condition", {"graph_id": graph["graph_id"], "matter_id": "ship_first_version", "index": 0, "label": "Pilot reviewed", "expected_revision": graph["revision"]}),
            )
            assert sum(result.isError is False for result in results) == 1
            assert [result.structuredContent["error"]["code"] for result in results if result.isError] == ["conflict"]
            current = system["repository"].get_graph(system["alice"]["id"], graph["graph_id"])
            assert current["version"] == graph["version"] + 1

    asyncio.run(check())


def test_authentication_precedes_body_parsing_and_resource_tokens_are_bound(system):
    token = authorize(system)["access_token"]
    other_oauth = CloudOAuth(system["db"], issuer_url="https://other.example")
    other_token = authorize(system, oauth=other_oauth)["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"], max_request_bytes=128)

    async def check():
        async with component.lifespan():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=component.app), base_url=ORIGIN) as client:
                for authorization in (None, "Bearer invalid", f"Bearer {other_token}"):
                    headers = {"Content-Type": "application/json"}
                    if authorization:
                        headers["Authorization"] = authorization
                    response = await client.post("/mcp", content=b"{" * 1000, headers=headers)
                    assert response.status_code == 401
                    assert "resource_metadata" in response.headers["www-authenticate"]
                    assert token not in response.text and other_token not in response.text
                response = await client.post("/mcp", content=b"{" * 1000, headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
                assert response.status_code == 413

    asyncio.run(check())


def test_authorization_storage_failure_is_safe_unavailable_not_invalid_credentials(system, monkeypatch):
    token = authorize(system)["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"])

    def unavailable(*_args, **_kwargs):
        raise OAuthError("The authorization service is unavailable.", "temporarily_unavailable", 503)

    monkeypatch.setattr(system["oauth"], "validate_access_token", unavailable)

    async def check():
        async with component.lifespan():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=component.app), base_url=ORIGIN) as client:
                response = await client.post("/mcp", headers={"Authorization": f"Bearer {token}"}, content="not-json")
                assert response.status_code == 503
                assert response.json()["error"] == "temporarily_unavailable"
                assert token not in response.text

    asyncio.run(check())


@pytest.mark.parametrize("tool,arguments", [("list_graphs", {}), ("run_shell", {"command": "whoami"}), ("universe", {})])
def test_token_is_revalidated_for_each_call_even_unknown_or_invalid_tools(system, monkeypatch, tool, arguments):
    grant = authorize(system)
    component = create_hosted_mcp(system["oauth"], system["repository"])
    original = system["oauth"].validate_access_token
    calls = []

    def revoke_after_authentication(rawtoken, *, resource=None):
        principal = original(rawtoken, resource=resource)
        calls.append(1)
        if len(calls) == 1:
            system["oauth"].revoke(client_id=grant["client_id"], token=rawtoken)
        return principal

    monkeypatch.setattr(system["oauth"], "validate_access_token", revoke_after_authentication)

    async def check():
        async with component.lifespan():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=component.app), base_url=ORIGIN) as client:
                response = await client.post("/mcp", headers={"Authorization": f"Bearer {grant['access_token']}", "Accept": "application/json, text/event-stream"}, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}})
                assert response.status_code == 200, response.text
                result = response.json()["result"]
                assert result["isError"] is True
                assert result["structuredContent"]["error"]["code"] == "unauthorized"
                assert 'error="invalid_token"' in result["_meta"]["mcp/www_authenticate"][0]

    asyncio.run(check())
    assert len(calls) == 1  # second validation raises after reading the revoked grant


def test_concurrent_request_bound_refuses_instead_of_accumulating_sessions(system):
    token = authorize(system)["access_token"]
    component = create_hosted_mcp(system["oauth"], system["repository"], max_concurrent_requests=1)

    async def check():
        async with component.lifespan():
            component.app.limiter.acquire_nowait()
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=component.app), base_url=ORIGIN) as client:
                    response = await client.post("/mcp", headers={"Authorization": f"Bearer {token}"}, json={})
                    assert response.status_code == 503
                    assert response.headers["retry-after"] == "1"
            finally:
                component.app.limiter.release()
            assert component.server.settings.stateless_http is True
            assert component.server.session_manager._server_instances == {}

    asyncio.run(check())


def test_real_tls_socket_sdk_round_trip(system, tmp_path):
    """Exercise actual HTTPS with a test certificate explicitly trusted by the SDK."""

    cryptography = pytest.importorskip("cryptography")
    import ipaddress
    import uvicorn
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "tls.crt", tmp_path / "tls.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key_path.chmod(0o600)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    origin = f"https://localhost:{listener.getsockname()[1]}"
    oauth = CloudOAuth(system["db"], issuer_url=origin)
    token = authorize(system, oauth=oauth)["access_token"]
    component = create_hosted_mcp(oauth, system["repository"])
    server = uvicorn.Server(uvicorn.Config(component.app, host="127.0.0.1", port=listener.getsockname()[1], ssl_certfile=str(cert_path), ssl_keyfile=str(key_path), access_log=False, log_level="warning"))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    async def check():
        context = ssl.create_default_context(cafile=str(cert_path))
        async with httpx.AsyncClient(verify=context, headers={"Authorization": f"Bearer {token}"}) as http:
            async with streamable_http_client(origin + "/mcp", http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    await client.initialize()
                    result = payload(await client.call_tool("universe", {"graph_id": system["graph"]["graph_id"]}))
                    assert result["matters"] == ["ship_first_version"]

    try:
        asyncio.run(check())
    finally:
        server.should_exit = True
        thread.join(5)
        listener.close()
    assert not thread.is_alive()
