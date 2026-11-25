"""Independent checks of consent, grant narrowing and portable graph isolation."""

import base64
import asyncio
import copy
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import runpy
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("mcp")
pytest.importorskip("fastapi")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata
from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url

from matters.cloud.db import Database, graph_events, users
from matters.cloud.app import create_app
from matters.cloud.collaboration import CloudCollaboration
from matters.cloud.oauth import CloudOAuth, OAuthError
from matters.cloud.repository import CloudGraphError, GraphRepository
from matters.cloud.service import HostedGraphService
from matters.local_service import LocalServiceError


ORIGIN = "https://matters-review.example"
REDIRECT = "https://client.example/complete"
BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def installation(request, tmp_path):
    admin = None
    schema = None
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'independent.db'}"
    else:
        pytest.importorskip("psycopg")
        configured = make_url(os.environ["MATTERS_TEST_DATABASE_URL"])
        assert configured.get_backend_name() == "postgresql"
        admin = create_engine(configured)
        schema = "hosted_review_" + uuid.uuid4().hex
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = configured.update_query_dict({"options": f"-csearch_path={schema}"})
    db = Database(url).initialize()
    actors = {name: str(uuid.uuid4()) for name in ("alice", "bob")}
    with db.transaction(write=True) as connection:
        connection.execute(users.insert(), [
            {"id": actor, "email": f"{name}@example.com", "password_hash": "unused-test-hash", "created_at": 1}
            for name, actor in actors.items()
        ])
    repository = GraphRepository(db)
    oauth = CloudOAuth(db, issuer_url=ORIGIN)
    try:
        yield db, actors, repository, oauth
    finally:
        db.close()
        if admin is not None:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
            admin.dispose()


def authorization(oauth, repository, actor, graph_ids, scope="graphs:read graphs:write offline_access"):
    client = oauth.register_client({"redirect_uris": [REDIRECT], "scope": scope})
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    parameters = {
        "response_type": "code", "client_id": client["client_id"], "redirect_uri": REDIRECT,
        "scope": scope, "resource": oauth.resource_url, "state": "client-state",
        "code_challenge": challenge, "code_challenge_method": "S256",
    }
    request = oauth.create_authorization_request(parameters)
    session, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    oauth.prepare_consent(request, actor, session, csrf)
    redirect = oauth.consent(request, actor, session, csrf, graph_ids, repository=repository)
    result = parse_qs(urlsplit(redirect).query)
    assert result["state"] == ["client-state"] and result["iss"] == [ORIGIN]
    exchange = {
        "client_id": client["client_id"], "code": result["code"][0], "redirect_uri": REDIRECT,
        "code_verifier": verifier, "resource": oauth.resource_url,
    }
    return oauth.exchange_code(**exchange), client, exchange, parameters


def test_granted_graphs_and_current_membership_both_limit_each_operation(installation):
    db, people, repository, oauth = installation
    alice_graph = repository.create_graph(people["alice"], "Selected Alice graph")
    alice_other = repository.create_graph(people["alice"], "Unselected Alice graph")
    bob_graph = repository.create_graph(people["bob"], "Bob shared graph")
    repository.set_member(people["bob"], bob_graph["graph_id"], people["alice"], "editor")
    tokens, _, _, _ = authorization(oauth, repository, people["alice"], [alice_graph["graph_id"], bob_graph["graph_id"]])
    service = HostedGraphService(repository, lambda: oauth.validate_access_token(tokens["access_token"]))
    assert {g["graph_id"] for g in service.catalog()["graphs"]} == {alice_graph["graph_id"], bob_graph["graph_id"]}
    with pytest.raises(LocalServiceError) as denied:
        service.read(alice_other["graph_id"])
    assert denied.value.code == "forbidden"
    repository.remove_member(people["bob"], bob_graph["graph_id"], people["alice"])
    assert [g["graph_id"] for g in service.catalog()["graphs"]] == [alice_graph["graph_id"]]
    before = repository.get_graph(people["bob"], bob_graph["graph_id"])
    for operation in (
        lambda: service.read(bob_graph["graph_id"]),
        lambda: service.read(bob_graph["graph_id"], view="export"),
        lambda: service.mutate(bob_graph["graph_id"], "create_matter", {"id": "intrusion", "conditions": [{"label": "Must not appear", "truth": False}]}, expected_revision=bob_graph["revision"]),
    ):
        with pytest.raises(LocalServiceError) as revoked:
            operation()
        assert revoked.value.code == "not_found"
    assert repository.get_graph(people["bob"], bob_graph["graph_id"]) == before


def test_public_client_code_is_bound_to_pkce_resource_redirect_and_client(installation):
    _, people, repository, oauth = installation
    graph = repository.create_graph(people["alice"], "Alice")
    tokens, client, exchange, parameters = authorization(oauth, repository, people["alice"], [graph["graph_id"]])
    other = oauth.register_client({"redirect_uris": [REDIRECT]})
    for patch in (
        {"resource": ORIGIN + "/other"}, {"redirect_uri": REDIRECT + "/extra"},
        {"client_id": other["client_id"]}, {"code_verifier": secrets.token_urlsafe(48)},
    ):
        with pytest.raises(OAuthError):
            oauth.exchange_code(**{**exchange, **patch})
        # A request with the wrong binding must not invalidate legitimate access.
        assert oauth.validate_access_token(tokens["access_token"])["actor_id"] == people["alice"]
    with pytest.raises(OAuthError) as wrong_resource:
        oauth.validate_access_token(tokens["access_token"], resource=ORIGIN + "/another")
    assert wrong_resource.value.code == "invalid_token"
    with pytest.raises(OAuthError):
        oauth.create_authorization_request({**parameters, "redirect_uri": "https://attacker.example/complete"})


def test_replayed_refresh_revokes_new_access_and_cannot_expand_scope(installation):
    _, people, repository, oauth = installation
    graph = repository.create_graph(people["alice"], "Alice")
    tokens, client, _, _ = authorization(oauth, repository, people["alice"], [graph["graph_id"]])
    refreshed = oauth.refresh(client_id=client["client_id"], refresh_token=tokens["refresh_token"], resource=oauth.resource_url, scope="graphs:read offline_access")
    assert "graphs:write" not in oauth.validate_access_token(refreshed["access_token"])["scopes"]
    with pytest.raises(OAuthError) as escalation:
        oauth.refresh(client_id=client["client_id"], refresh_token=refreshed["refresh_token"], resource=oauth.resource_url, scope="graphs:read graphs:write offline_access")
    assert escalation.value.code == "invalid_scope"
    assert oauth.validate_access_token(refreshed["access_token"])
    with pytest.raises(OAuthError) as replay:
        oauth.refresh(client_id=client["client_id"], refresh_token=tokens["refresh_token"], resource=oauth.resource_url)
    assert replay.value.code == "invalid_grant"
    for token in (tokens["access_token"], refreshed["access_token"]):
        with pytest.raises(OAuthError):
            oauth.validate_access_token(token)


def test_consent_refuses_account_session_csrf_and_foreign_graph_substitution(installation):
    _, people, repository, oauth = installation
    graph = repository.create_graph(people["alice"], "Alice")
    foreign = repository.create_graph(people["bob"], "Bob")
    _, _, _, parameters = authorization(oauth, repository, people["alice"], [graph["graph_id"]])
    request = oauth.create_authorization_request(parameters)
    session, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    oauth.prepare_consent(request, people["alice"], session, csrf)
    for actor, cookie, nonce, selected in (
        (people["bob"], session, csrf, [graph["graph_id"]]),
        (people["alice"], secrets.token_urlsafe(32), csrf, [graph["graph_id"]]),
        (people["alice"], session, secrets.token_urlsafe(32), [graph["graph_id"]]),
        (people["alice"], session, csrf, [foreign["graph_id"]]),
    ):
        with pytest.raises((OAuthError, CloudGraphError)):
            oauth.consent(request, actor, cookie, nonce, selected, repository=repository)
    # Rejected substitutions leave the original pending consent usable.
    assert "code=" in oauth.consent(request, people["alice"], session, csrf, [graph["graph_id"]], repository=repository)


def test_racing_writes_using_one_grant_produce_one_commit_and_one_conflict(installation):
    db, people, repository, oauth = installation
    graph = repository.create_graph(people["alice"], "Alice")
    tokens, _, _, _ = authorization(oauth, repository, people["alice"], [graph["graph_id"]])
    service = HostedGraphService(repository, lambda: oauth.validate_access_token(tokens["access_token"]))

    def write(index):
        try:
            result = service.mutate(graph["graph_id"], "create_matter", {"id": f"writer_{index}", "conditions": [{"label": "Result reviewed", "truth": False}]}, expected_revision=graph["revision"])
            return "ok", result
        except LocalServiceError as error:
            return error.code, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, (1, 2)))
    assert sorted(result[0] for result in results) == ["conflict", "ok"]
    saved = repository.get_graph(people["alice"], graph["graph_id"])
    assert saved["version"] == graph["version"] + 1 and len(saved["state"]["matters"]) == 1
    with db.transaction() as connection:
        history = connection.execute(select(graph_events).where(graph_events.c.graph_id == graph["graph_id"])).mappings().all()
    assert len(history) == 2


def test_export_import_moves_facts_and_citations_but_never_owner_or_members(installation):
    _, people, repository, oauth = installation
    bundle = {
        "format": "matters-graph", "schema_version": 1,
        "graph": {"schema_version": 2, "matters": ["review_evidence"], "conditions": {"review_evidence": [{"label": "Citation inspected", "truth": False}]}, "dependencies": []},
        "sources": {"schema_version": 1, "matter_sources": {"review_evidence": [{"title": "Repository source", "url": "https://example.com/evidence"}]}},
    }
    original = repository.import_graph(people["alice"], "Source", bundle)
    repository.set_member(people["alice"], original["graph_id"], people["bob"], "viewer")
    exported = repository.export_graph(people["bob"], original["graph_id"])["bundle"]
    assert exported == bundle
    copied = repository.import_graph(people["bob"], "Independent copy", exported)
    assert copied["graph_id"] != original["graph_id"] and copied["role"] == "owner"
    with pytest.raises(CloudGraphError) as no_access:
        repository.get_graph(people["alice"], copied["graph_id"])
    assert no_access.value.code == "not_found"
    assert repository.export_graph(people["bob"], copied["graph_id"])["bundle"] == bundle
    invalid = copy.deepcopy(bundle)
    invalid["graph"]["dependencies"] = [["review_evidence", "review_evidence"]]
    before = repository.list_graphs(people["bob"])
    with pytest.raises(CloudGraphError):
        repository.import_graph(people["bob"], "Rejected", invalid)
    assert repository.list_graphs(people["bob"]) == before


def test_integrated_http_consent_selected_graph_mcp_and_revocation(installation):
    db, people, repository, _ = installation
    app = create_app(db.engine.url.render_as_string(hide_password=False), public_origin=ORIGIN, enable_mcp=True)
    with TestClient(app, base_url=ORIGIN, follow_redirects=False) as client:
        credentials = {"email": "http-user@example.com", "password": "Independent HTTP review password"}
        assert client.post("/api/auth/register", json=credentials).status_code == 201
        assert client.post("/api/auth/login", json=credentials).status_code == 200
        selected = client.post("/api/graphs", json={"title": "Authorized HTTP graph", "kind": "project"}).json()
        excluded = client.post("/api/graphs", json={"title": "Unselected private title", "kind": "personal"}).json()
        foreign = repository.create_graph(people["bob"], "Foreign owner private title")
        no_auth = client.post("/mcp", json={"unparsed": True})
        assert no_auth.status_code == 401 and "resource_metadata=" in no_auth.headers["www-authenticate"]
        assert client.get("/.well-known/oauth-protected-resource/mcp").json()["resource"] == ORIGIN + "/mcp"
        metadata = client.get("/.well-known/oauth-authorization-server").json()
        assert metadata["code_challenge_methods_supported"] == ["S256"]
        scope = "graphs:read graphs:write offline_access"
        registration = client.post("/oauth/register", json={"redirect_uris": [REDIRECT], "scope": scope, "client_name": "Independent HTTP client"})
        assert registration.status_code == 201
        client_id = registration.json()["client_id"]
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        parameters = {"client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code", "resource": ORIGIN + "/mcp", "scope": scope, "state": "http-state", "code_challenge_method": "S256", "code_challenge": challenge}
        pending = client.get("/oauth/authorize", params=parameters)
        assert pending.status_code == 303
        consent_url = pending.headers["location"]
        page = client.get(consent_url)
        assert page.status_code == 200
        assert "Authorized HTTP graph" in page.text and "Unselected private title" in page.text
        assert "Foreign owner private title" not in page.text
        pending_token = parse_qs(urlsplit(consent_url).query)["request"][0]
        csrf = client.cookies.get("matters_oauth_csrf")
        form = {"request": pending_token, "csrf": csrf, "decision": "approve", "graph_id": selected["graph_id"]}
        cross_origin = client.post("/oauth/consent", data=form, headers={"Origin": "https://attacker.example"})
        assert cross_origin.status_code == 403
        wrong_graph = client.post("/oauth/consent", data={**form, "graph_id": foreign["graph_id"]}, headers={"Origin": ORIGIN})
        assert wrong_graph.status_code in (403, 404)
        accepted = client.post("/oauth/consent", data=form, headers={"Origin": ORIGIN})
        assert accepted.status_code == 303
        callback = parse_qs(urlsplit(accepted.headers["location"]).query)
        assert callback["state"] == ["http-state"] and callback["iss"] == [ORIGIN]
        tokens = client.post("/oauth/token", data={"grant_type": "authorization_code", "client_id": client_id, "code": callback["code"][0], "redirect_uri": REDIRECT, "code_verifier": verifier, "resource": ORIGIN + "/mcp"})
        assert tokens.status_code == 200
        token = tokens.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-11-25"}

        def call(name, arguments):
            response = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
            assert response.status_code == 200, response.text
            assert "no-store" in response.headers["cache-control"]
            return response.json()["result"]

        catalogue = call("list_graphs", {})
        assert catalogue.get("isError") is not True
        assert [g["graph_id"] for g in catalogue["structuredContent"]["graphs"]] == [selected["graph_id"]]
        assert "Unselected private title" not in json.dumps(catalogue)
        denied = call("universe", {"graph_id": excluded["graph_id"]})
        assert denied["isError"] is True and denied["structuredContent"]["error"]["code"] == "forbidden"
        added = call("create_matter", {"graph_id": selected["graph_id"], "matter_id": "http_review", "conditions": [{"label": "HTTP proof passes", "truth": False}], "expected_revision": selected["revision"]})
        assert added.get("isError") is not True
        stale = call("set_condition", {"graph_id": selected["graph_id"], "matter_id": "http_review", "index": 0, "truth": True, "expected_revision": selected["revision"]})
        assert stale["isError"] is True and stale["structuredContent"]["error"]["code"] == "conflict"
        assert client.post("/oauth/revoke", data={"client_id": client_id, "token": token}).status_code == 200
        revoked = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert revoked.status_code == 401
        # API browser sessions remain distinct from revoked external grants.
        assert client.get("/api/graphs").status_code == 200


def test_owner_role_change_invalidates_older_pending_role_invitations(installation):
    _, people, repository, _ = installation
    graph = repository.create_graph(people["alice"], "Shared graph")
    collaboration = CloudCollaboration(repository)
    repository.set_member(people["alice"], graph["graph_id"], people["bob"], "editor")
    invitation = collaboration.create_invitation(people["alice"], graph["graph_id"], "bob@example.com", "editor")
    collaboration.set_member_role(people["alice"], graph["graph_id"], people["bob"], "viewer")
    with pytest.raises(CloudGraphError) as obsolete:
        collaboration.accept_invitation(people["bob"], invitation["token"])
    assert obsolete.value.code == "conflict"
    assert repository.get_graph(people["bob"], graph["graph_id"])["role"] == "viewer"


def test_sdk_automatic_discovery_registration_pkce_and_client_reuse(installation):
    db, _, _, _ = installation
    app = create_app(db.engine.url.render_as_string(hide_password=False), public_origin=ORIGIN, enable_mcp=True)

    class MemoryStorage:
        tokens = None
        client_info = None

        async def get_tokens(self):
            return self.tokens

        async def set_tokens(self, tokens):
            self.tokens = tokens

        async def get_client_info(self):
            return self.client_info

        async def set_client_info(self, client_info):
            self.client_info = client_info

    async def check():
        storage = MemoryStorage()
        callbacks = []
        requests = []
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN) as browser:
                credentials = {"email": "sdk-oauth@example.com", "password": "Independent SDK review password"}
                assert (await browser.post("/api/auth/register", json=credentials)).status_code == 201
                assert (await browser.post("/api/auth/login", json=credentials)).status_code == 200
                graph = (await browser.post("/api/graphs", json={"title": "SDK selected graph", "kind": "project"})).json()
                other = (await browser.post("/api/graphs", json={"title": "SDK unselected graph", "kind": "personal"})).json()

                async def redirect_handler(url):
                    response = await browser.get(url)
                    assert response.status_code == 303
                    page = await browser.get(response.headers["location"])
                    assert page.status_code == 200 and "Review graph access" in page.text
                    pending = parse_qs(urlsplit(response.headers["location"]).query)["request"][0]
                    accepted = await browser.post("/oauth/consent", data={"request": pending, "csrf": browser.cookies.get("matters_oauth_csrf"), "decision": "approve", "graph_id": graph["graph_id"]}, headers={"Origin": ORIGIN})
                    assert accepted.status_code == 303
                    query = parse_qs(urlsplit(accepted.headers["location"]).query)
                    assert query["iss"] == [ORIGIN]
                    callbacks.append((query["code"][0], query["state"][0]))

                async def callback_handler():
                    return callbacks[-1]

                async def record_request(request):
                    requests.append(request.url.path)

                def provider():
                    return OAuthClientProvider(
                        ORIGIN + "/mcp",
                        OAuthClientMetadata(redirect_uris=[REDIRECT], client_name="Automatic SDK review", token_endpoint_auth_method="none", scope="graphs:read graphs:write offline_access"),
                        storage, redirect_handler=redirect_handler, callback_handler=callback_handler,
                    )

                async def connect(auth):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), auth=auth, event_hooks={"request": [record_request]}) as transport:
                        async with streamable_http_client(ORIGIN + "/mcp", http_client=transport) as streams:
                            async with ClientSession(streams[0], streams[1]) as session:
                                await session.initialize()
                                listed = await session.call_tool("list_graphs", {})
                                assert listed.isError is False
                                assert [g["graph_id"] for g in listed.structuredContent["graphs"]] == [graph["graph_id"]]
                                denied = await session.call_tool("universe", {"graph_id": other["graph_id"]})
                                assert denied.isError and denied.structuredContent["error"]["code"] == "forbidden"

                auth = provider()
                await connect(auth)
                assert callbacks and storage.tokens and storage.client_info
                assert "/.well-known/oauth-authorization-server" in requests
                assert any(path.startswith("/.well-known/oauth-protected-resource") for path in requests)
                assert requests.count("/oauth/register") == 1
                assert requests.count("/oauth/token") == 1
                original_refresh = storage.tokens.refresh_token
                auth.context.token_expiry_time = 1
                await connect(auth)
                assert storage.tokens.refresh_token != original_refresh
                assert requests.count("/oauth/token") == 2 and len(callbacks) == 1
                # A fresh SDK auth object loads the saved client/token state.
                await connect(provider())
                assert len(callbacks) == 1 and requests.count("/oauth/register") == 1

    asyncio.run(check())


def test_configured_origin_normalization_preserves_browser_consent(installation):
    db, _, _, _ = installation
    app = create_app(db.engine.url.render_as_string(hide_password=False), public_origin="HTTPS://MATTERS-REVIEW.EXAMPLE:443/", enable_mcp=True)
    with TestClient(app, base_url=ORIGIN, follow_redirects=False) as browser:
        credentials = {"email": "canonical-origin@example.com", "password": "Canonical browser origin password"}
        assert browser.post("/api/auth/register", json=credentials, headers={"Origin": ORIGIN}).status_code == 201
        assert browser.post("/api/auth/login", json=credentials, headers={"Origin": ORIGIN}).status_code == 200
        graph = browser.post("/api/graphs", json={"title": "Canonical origin"}, headers={"Origin": ORIGIN}).json()
        metadata = browser.get("/.well-known/oauth-authorization-server").json()
        assert metadata["issuer"] == ORIGIN
        client = browser.post("/oauth/register", json={"redirect_uris": [REDIRECT]}).json()
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        authorization = browser.get("/oauth/authorize", params={"response_type": "code", "client_id": client["client_id"], "redirect_uri": REDIRECT, "resource": ORIGIN + "/mcp", "code_challenge": challenge, "code_challenge_method": "S256"})
        assert authorization.status_code == 303
        page = browser.get(authorization.headers["location"])
        assert page.status_code == 200
        pending = parse_qs(urlsplit(authorization.headers["location"]).query)["request"][0]
        consent = browser.post("/oauth/consent", data={"request": pending, "csrf": browser.cookies.get("matters_oauth_csrf"), "decision": "approve", "graph_id": graph["graph_id"]}, headers={"Origin": ORIGIN})
        assert consent.status_code == 303
        assert parse_qs(urlsplit(consent.headers["location"]).query)["iss"] == [ORIGIN]


def test_http_invitation_visual_and_export_access_follow_current_member_role(installation):
    db, _, _, _ = installation
    app = create_app(db.engine.url.render_as_string(hide_password=False), public_origin=ORIGIN)
    password = "HTTP collaboration review password"
    for email in ("share-owner@example.com", "share-viewer@example.com"):
        app.state.accounts.register(email, password)
    owner = app.state.accounts.login("share-owner@example.com", password)
    member = app.state.accounts.login("share-viewer@example.com", password)
    owner_headers = {"Origin": ORIGIN, "Cookie": "matters_session=" + owner["token"]}
    member_headers = {"Origin": ORIGIN, "Cookie": "matters_session=" + member["token"]}
    with TestClient(app, base_url=ORIGIN) as client:
        graph = client.post("/api/graphs", headers=owner_headers, json={"title": "Shared HTTP example"}).json()
        graph_id = graph["graph_id"]
        invitation = client.post(f"/api/graphs/{graph_id}/invitations", headers=owner_headers, json={"email": "share-viewer@example.com", "role": "viewer"})
        assert invitation.status_code == 201
        raw_token = invitation.json()["token"]
        assert client.post("/api/invitations/accept", headers=member_headers, json={"token": raw_token}).status_code == 200
        visual = client.get(f"/api/graphs/{graph_id}/visual", headers=member_headers)
        assert visual.status_code == 200 and visual.json()["role"] == "viewer"
        assert "state_path" not in json.dumps(visual.json())
        hidden_sharing = client.get(f"/api/graphs/{graph_id}/sharing", headers=member_headers)
        assert hidden_sharing.status_code == 403 and "share-owner@example.com" not in hidden_sharing.text
        operation = {"operation": "create_matter", "payload": {"id": "collaborate", "conditions": [{"label": "Roles checked", "truth": False}]}, "expected_revision": graph["revision"]}
        assert client.post(f"/api/graphs/{graph_id}/operations", headers=member_headers, json=operation).status_code == 403
        assert client.put(f"/api/graphs/{graph_id}/members/{member['user']['id']}", headers=owner_headers, json={"role": "editor"}).status_code == 200
        assert client.post(f"/api/graphs/{graph_id}/operations", headers=member_headers, json=operation).status_code == 200
        assert client.delete(f"/api/graphs/{graph_id}/members/{member['user']['id']}", headers=owner_headers).status_code == 200
        for suffix in ("visual", "export"):
            response = client.get(f"/api/graphs/{graph_id}/{suffix}", headers=member_headers)
            assert response.status_code == 404 and "Shared HTTP example" not in response.text
        assert client.post(f"/api/graphs/{graph_id}/operations", headers=member_headers, json=operation).status_code == 404
        assert client.get("/api/terminal/sessions", headers=owner_headers).status_code == 404
        assert client.get("/api/state?path=/tmp/other.json", headers=owner_headers).status_code in (400, 404)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_late_invitation_acceptance_does_not_override_new_graph_navigation():
    review = runpy.run_path(str(Path(__file__).with_name("test_cloud_review.py")))
    harness = review["HARNESS"]
    harness = harness.replace("let releaseNavigation = null;", "let releaseNavigation = null; let releaseJoin = null;")
    harness = harness.replace(
        "  const match = path.match",
        "  if (path === '/api/invitations/accept') return new Promise(resolve => { releaseJoin = resolve; });\n  const match = path.match",
    )
    harness = harness.replace(
        "  if (scenario === 'private-form-drafts') {",
        """  if (scenario === 'pending-invitation') {
    document.getElementById('join-token').value = 't'.repeat(43);
    const joining = nodes.get('join-graph-form').events.submit({ preventDefault() {} });
    await flush();
    if (!releaseJoin) throw new Error('Invitation acceptance was not held');
    await nodes.get('graph-list').children[1].events.click();
    if (evaluate('state.graph.graph_id') !== 'alice-b') throw new Error('New graph did not load');
    releaseJoin(response({ graph_id: 'alice-a' }));
    await joining;
  } else if (scenario === 'private-form-drafts') {""",
        1,
    )
    result = subprocess.run(["node", "-e", harness, str(review["UI"]), "pending-invitation"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["user"] == "alice" and observed["graph"] == "alice-b"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("scenario", ["late-sharing", "late-invitation-link"])
def test_private_sharing_responses_cannot_cross_an_account_change(scenario):
    review = runpy.run_path(str(Path(__file__).with_name("test_cloud_review.py")))
    harness = review["HARNESS"].replace("let releaseNavigation = null;", "let releaseNavigation = null; let releasePrivateSharing = null;")
    harness = harness.replace(
        "  const match = path.match",
        "  if (path.endsWith('/sharing') || (path.endsWith('/invitations') && options.method === 'POST')) return new Promise(resolve => { releasePrivateSharing = resolve; });\n  const match = path.match",
    )
    harness = harness.replace(
        "  if (scenario === 'private-form-drafts') {",
        """  if (scenario === 'late-sharing' || scenario === 'late-invitation-link') {
    document.getElementById('invite-email').value = 'alice-private-invitee@example.com';
    document.getElementById('invite-role').value = 'editor';
    const pending = scenario === 'late-sharing'
      ? evaluate('loadSharing()').catch(() => {})
      : nodes.get('invitation-form').events.submit({ preventDefault() {} });
    await flush();
    if (!releasePrivateSharing) throw new Error('Private sharing request was not held');
    await loginBob();
    releasePrivateSharing(scenario === 'late-sharing'
      ? response({ owner: {email:'alice@example.com'}, members:[{email:'alice-private-invitee@example.com',user_id:'private',role:'editor'}], invitations:[] })
      : response({ token:'s'.repeat(43) }));
    await pending;
  } else if (scenario === 'private-form-drafts') {""",
        1,
    )
    harness = harness.replace(
        "  console.log(JSON.stringify(observed));",
        "  observed.sharingOwner = document.getElementById('sharing-owner').textContent; observed.invitationLink = document.getElementById('invitation-link').value; observed.members = document.getElementById('sharing-members').children.length;\n  console.log(JSON.stringify(observed));",
    )
    result = subprocess.run(["node", "-e", harness, str(review["UI"]), scenario], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["user"] == "bob" and observed["graph"] == "bob-a"
    assert observed["sharingOwner"] == "" and observed["invitationLink"] == "" and observed["members"] == 0
