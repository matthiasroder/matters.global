"""Persistent OAuth consent and adversarial token lifecycle checks."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import secrets
import uuid
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url

from matters.cloud.accounts import CloudAccounts
from matters.cloud.db import Database, oauth_records
from matters.cloud.oauth import CloudOAuth, OAuthError, ACCESS_TTL, CODE_TTL
from matters.cloud.oauth_routes import build_oauth_router, MAX_OAUTH_BODY
from matters.cloud.repository import GraphRepository


ORIGIN = "https://matters.example"
PASSWORD = "A separate temporary test password!"
BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def services(request, tmp_path):
    admin = None
    if request.param == "postgresql":
        schema = "oauth_test_" + uuid.uuid4().hex
        admin = create_engine(os.environ["MATTERS_TEST_DATABASE_URL"])
        with admin.begin() as connection:
            connection.execute(text('CREATE SCHEMA "' + schema + '"'))
        url = make_url(os.environ["MATTERS_TEST_DATABASE_URL"]).update_query_dict({"options": "-csearch_path=" + schema})
    else:
        url = "sqlite:///" + str(tmp_path / "oauth.db")
    db = Database(url).initialize()
    now = [1700000000]
    accounts = CloudAccounts(db, now=lambda: now[0])
    repo = GraphRepository(db)
    oauth = CloudOAuth(db, issuer_url=ORIGIN, now=lambda: now[0])
    users = [accounts.register(f"{name}@example.com", PASSWORD) for name in ("alice", "bob")]
    sessions = [accounts.login(user["email"], PASSWORD)["token"] for user in users]
    graphs = [repo.create_graph(user["id"], name, kind) for user, name, kind in zip(users, ("Alice graph", "Bob graph"), ("personal", "project"))]
    yield {"db": db, "accounts": accounts, "repo": repo, "oauth": oauth, "now": now, "users": users, "sessions": sessions, "graphs": graphs, "url": url}
    db.close()
    if admin:
        with admin.begin() as connection:
            connection.execute(text('DROP SCHEMA "' + schema + '" CASCADE'))
        admin.dispose()


def client_registration(oauth, **extra):
    return oauth.register_client({"redirect_uris": ["https://client.example/callback?existing=1"], "client_name": "Example client", "scope": "graphs:read graphs:write offline_access", **extra})


def request_parameters(oauth, client, verifier, scope="graphs:read graphs:write offline_access"):
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return {"client_id": client["client_id"], "redirect_uri": client["redirect_uris"][0], "response_type": "code", "state": "client-state", "resource": oauth.resource_url, "scope": scope, "code_challenge": challenge, "code_challenge_method": "S256"}


def authorized(services, *, scope="graphs:read graphs:write offline_access", graph_ids=None):
    oauth = services["oauth"]
    client = client_registration(oauth)
    verifier, csrf = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    parameters = request_parameters(oauth, client, verifier, scope)
    pending = oauth.create_authorization_request(parameters)
    actor, session = services["users"][0]["id"], services["sessions"][0]
    oauth.prepare_consent(pending, actor, session, csrf)
    redirect = oauth.consent(pending, actor, session, csrf, graph_ids or [services["graphs"][0]["graph_id"]], repository=services["repo"])
    query = parse_qs(urlsplit(redirect).query)
    assert query["state"] == ["client-state"] and query["iss"] == [ORIGIN] and query["existing"] == ["1"]
    arguments = {"client_id": client["client_id"], "code": query["code"][0], "redirect_uri": client["redirect_uris"][0], "code_verifier": verifier, "resource": oauth.resource_url}
    return client, arguments, pending


def test_persistent_hashed_graph_grant_and_expiry(services):
    oauth = services["oauth"]
    client, arguments, _ = authorized(services)
    tokens = oauth.exchange_code(**arguments)
    principal = oauth.validate_access_token(tokens["access_token"], resource=oauth.resource_url)
    assert principal["actor_id"] == services["users"][0]["id"]
    assert principal["graph_ids"] == [services["graphs"][0]["graph_id"]]
    assert principal["scopes"] == ["graphs:read", "graphs:write", "offline_access"]
    with services["db"].transaction() as connection:
        records = connection.execute(select(oauth_records)).mappings().all()
    stored = json.dumps([dict(record) for record in records])
    for raw in (tokens["access_token"], tokens["refresh_token"], arguments["code"], services["sessions"][0]):
        assert raw not in stored
    services["db"].close()
    reopened = Database(services["url"]).initialize()
    try:
        persisted = CloudOAuth(reopened, issuer_url=ORIGIN, now=lambda: services["now"][0])
        assert persisted.validate_access_token(tokens["access_token"])["graph_ids"] == principal["graph_ids"]
    finally:
        reopened.close()
    services["now"][0] += ACCESS_TTL
    with pytest.raises(OAuthError, match="invalid or expired"):
        oauth.validate_access_token(tokens["access_token"])


def test_refresh_rotation_replay_revokes_whole_grant(services):
    oauth = services["oauth"]
    client, arguments, _ = authorized(services)
    initial = oauth.exchange_code(**arguments)
    refreshed = oauth.refresh(client_id=client["client_id"], refresh_token=initial["refresh_token"], resource=oauth.resource_url)
    assert refreshed["access_token"] != initial["access_token"]
    assert refreshed["refresh_token"] != initial["refresh_token"]
    assert oauth.validate_access_token(refreshed["access_token"])["actor_id"] == services["users"][0]["id"]
    with pytest.raises(OAuthError, match="already been used"):
        oauth.refresh(client_id=client["client_id"], refresh_token=initial["refresh_token"], resource=oauth.resource_url)
    for token in (initial["access_token"], refreshed["access_token"]):
        with pytest.raises(OAuthError):
            oauth.validate_access_token(token)
    with pytest.raises(OAuthError):
        oauth.refresh(client_id=client["client_id"], refresh_token=refreshed["refresh_token"], resource=oauth.resource_url)


def test_code_replay_revokes_tokens_and_code_is_bound(services):
    oauth = services["oauth"]
    client, arguments, _ = authorized(services)
    other = client_registration(oauth)
    for changes in ({"client_id": other["client_id"]}, {"redirect_uri": "https://client.example/evil"}, {"resource": "https://elsewhere.example/mcp"}, {"code_verifier": secrets.token_urlsafe(48)}):
        with pytest.raises(OAuthError):
            oauth.exchange_code(**{**arguments, **changes})
    tokens = oauth.exchange_code(**arguments)
    with pytest.raises(OAuthError, match="already been used"):
        oauth.exchange_code(**arguments)
    with pytest.raises(OAuthError):
        oauth.validate_access_token(tokens["access_token"])


def test_refresh_cannot_escalate_scope_client_or_resource(services):
    oauth = services["oauth"]
    client, arguments, _ = authorized(services, scope="graphs:read offline_access")
    tokens = oauth.exchange_code(**arguments)
    other = client_registration(oauth)
    base = {"client_id": client["client_id"], "refresh_token": tokens["refresh_token"], "resource": oauth.resource_url}
    for changes in ({"scope": "graphs:read graphs:write offline_access"}, {"client_id": other["client_id"]}, {"resource": "https://foreign.example/mcp"}):
        with pytest.raises(OAuthError):
            oauth.refresh(**{**base, **changes})
    narrowed = oauth.refresh(**base, scope="graphs:read")
    assert "refresh_token" not in narrowed
    assert oauth.validate_access_token(narrowed["access_token"])["scopes"] == ["graphs:read"]


def test_explicit_graph_session_csrf_and_account_consent(services):
    oauth = services["oauth"]
    client = client_registration(oauth)
    pending = oauth.create_authorization_request(request_parameters(oauth, client, secrets.token_urlsafe(48)))
    actor, session = services["users"][0]["id"], services["sessions"][0]
    csrf = secrets.token_urlsafe(32)
    oauth.prepare_consent(pending, actor, session, csrf)
    for account, browser, nonce, selected in (
        (services["users"][1]["id"], services["sessions"][1], csrf, [services["graphs"][0]["graph_id"]]),
        (actor, session, secrets.token_urlsafe(32), [services["graphs"][0]["graph_id"]]),
        (actor, secrets.token_urlsafe(32), csrf, [services["graphs"][0]["graph_id"]]),
        (actor, session, csrf, []),
        (actor, session, csrf, [services["graphs"][1]["graph_id"]]),
    ):
        with pytest.raises(OAuthError):
            oauth.consent(pending, account, browser, nonce, selected, repository=services["repo"])
    assert not oauth.list_grants(actor)
    denied = oauth.consent(pending, actor, session, csrf, [], repository=services["repo"], approve=False)
    query = parse_qs(urlsplit(denied).query)
    assert query["error"] == ["access_denied"] and query["iss"] == [ORIGIN]
    with pytest.raises(OAuthError):
        oauth.consent(pending, actor, session, csrf, [], repository=services["repo"], approve=False)


def test_viewer_cannot_approve_write_and_owner_can_disconnect(services):
    oauth, repo = services["oauth"], services["repo"]
    alice, bob = services["users"]
    repo.set_member(bob["id"], services["graphs"][1]["graph_id"], alice["id"], "viewer")
    with pytest.raises(OAuthError, match="does not permit editing"):
        authorized(services, graph_ids=[services["graphs"][1]["graph_id"]])
    client, arguments, _ = authorized(services, scope="graphs:read offline_access", graph_ids=[services["graphs"][1]["graph_id"]])
    tokens = oauth.exchange_code(**arguments)
    grant = oauth.list_grants(alice["id"])[0]
    assert not oauth.list_grants(bob["id"])
    with pytest.raises(OAuthError):
        oauth.revoke_grant(bob["id"], grant["grant_id"])
    oauth.revoke_grant(alice["id"], grant["grant_id"])
    assert not oauth.list_grants(alice["id"])
    with pytest.raises(OAuthError):
        oauth.validate_access_token(tokens["access_token"])


def test_public_client_revocation_is_client_bound_and_idempotent(services):
    oauth = services["oauth"]
    client, arguments, _ = authorized(services)
    other = client_registration(oauth)
    tokens = oauth.exchange_code(**arguments)
    oauth.revoke(client_id=other["client_id"], token=tokens["access_token"])
    assert oauth.validate_access_token(tokens["access_token"])
    oauth.revoke(client_id=client["client_id"], token=tokens["refresh_token"])
    oauth.revoke(client_id=client["client_id"], token=tokens["refresh_token"])
    oauth.revoke(client_id=client["client_id"], token="not-a-token")
    with pytest.raises(OAuthError):
        oauth.validate_access_token(tokens["access_token"])


@pytest.mark.parametrize("metadata", [
    {"redirect_uris": [{}]}, {"redirect_uris": [True]}, {"redirect_uris": ["http://client.example/cb"]},
    {"redirect_uris": ["https://client.example/cb#fragment"]}, {"redirect_uris": ["https://name:password@client.example/cb"]},
    {"redirect_uris": ["https://*.example/cb"]}, {"grant_types": [{}]}, {"grant_types": ["password"]},
    {"token_endpoint_auth_method": "client_secret_post"}, {"client_name": ""}, {"scope": "graphs:read admin"},
])
def test_client_registration_rejects_untrusted_metadata(services, metadata):
    with pytest.raises(OAuthError):
        client_registration(services["oauth"], **metadata)


def test_authorization_requires_resource_exact_redirect_and_pkce(services):
    oauth = services["oauth"]
    client = client_registration(oauth)
    parameters = request_parameters(oauth, client, secrets.token_urlsafe(48))
    for changes in ({"redirect_uri": client["redirect_uris"][0] + "&extra=1"}, {"resource": "https://foreign.example/mcp"}, {"code_challenge_method": "plain"}, {"code_challenge": "short"}, {"response_type": "token"}, {"scope": "admin"}, {"arbitrary": "data"}):
        with pytest.raises(OAuthError):
            oauth.create_authorization_request({**parameters, **changes})
    pending = oauth.create_authorization_request(parameters)
    services["now"][0] += 600
    with pytest.raises(OAuthError):
        oauth.inspect_request(pending)


def test_code_expiry_and_concurrent_exchange(services):
    oauth = services["oauth"]
    _, expired, _ = authorized(services)
    services["now"][0] += CODE_TTL
    with pytest.raises(OAuthError):
        oauth.exchange_code(**expired)
    _, arguments, _ = authorized(services)
    def exchange():
        try:
            return oauth.exchange_code(**arguments)
        except OAuthError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: exchange(), range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert "invalid_grant" in results


def test_concurrent_refresh_replay_revokes_the_winning_token(services):
    oauth = services["oauth"]
    client, arguments, _ = authorized(services)
    issued = oauth.exchange_code(**arguments)
    def refresh():
        try:
            return oauth.refresh(client_id=client["client_id"], refresh_token=issued["refresh_token"], resource=oauth.resource_url)
        except OAuthError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: refresh(), range(2)))
    successful = [result for result in results if isinstance(result, dict)]
    assert len(successful) == 1 and "invalid_grant" in results
    with pytest.raises(OAuthError):
        oauth.validate_access_token(successful[0]["access_token"])


def route_client(services):
    app = FastAPI()
    app.include_router(build_oauth_router(services["oauth"], services["accounts"], services["repo"], secure_cookies=True))
    return TestClient(app, base_url=ORIGIN, follow_redirects=False)


def hidden(page, name):
    return re.search('name="' + name + '" value="([^"]+)"', page.text).group(1)


def test_real_browser_form_login_consent_exchange_and_disconnect(services):
    oauth = services["oauth"]
    with route_client(services) as client:
        metadata = client.get("/.well-known/oauth-authorization-server").json()
        assert metadata["code_challenge_methods_supported"] == ["S256"]
        assert metadata["token_endpoint_auth_methods_supported"] == ["none"]
        assert client.get("/.well-known/oauth-protected-resource/mcp").json()["resource"] == oauth.resource_url
        registration = client.post("/oauth/register", json={"redirect_uris": ["https://client.example/cb"], "client_name": "An example client", "scope": "graphs:read offline_access"})
        assert registration.status_code == 201
        verifier = secrets.token_urlsafe(48)
        parameters = request_parameters(oauth, registration.json(), verifier, "graphs:read offline_access")
        authorized_response = client.get("/oauth/authorize", params=parameters)
        assert authorized_response.status_code == 303
        page = client.get(authorized_response.headers["location"])
        assert "Sign in to connect" in page.text
        pending, csrf = hidden(page, "request"), hidden(page, "csrf")
        assert "httponly" in page.headers["set-cookie"].lower()
        login = client.post("/oauth/consent/login", data={"request": pending, "csrf": csrf, "email": services["users"][0]["email"], "password": PASSWORD}, headers={"Origin": ORIGIN})
        assert login.status_code == 303
        assert "httponly" in login.headers["set-cookie"].lower() and "secure" in login.headers["set-cookie"].lower()
        page = client.get(login.headers["location"])
        assert "Review graph access" in page.text and "Alice graph" in page.text and "Bob graph" not in page.text
        assert 'name="graph_id"' in page.text and ' checked' not in page.text
        approved = client.post("/oauth/consent", data={"request": pending, "csrf": hidden(page, "csrf"), "graph_id": services["graphs"][0]["graph_id"], "decision": "approve"}, headers={"Origin": ORIGIN})
        assert approved.status_code == 303
        code = parse_qs(urlsplit(approved.headers["location"]).query)["code"][0]
        exchanged = client.post("/oauth/token", data={"grant_type": "authorization_code", "client_id": registration.json()["client_id"], "code": code, "code_verifier": verifier, "redirect_uri": parameters["redirect_uri"], "resource": oauth.resource_url})
        assert exchanged.status_code == 200
        assert exchanged.headers["cache-control"] == "no-store"
        token = exchanged.json()["access_token"]
        assert oauth.validate_access_token(token)["graph_ids"] == [services["graphs"][0]["graph_id"]]
        page = client.get("/oauth/connections")
        assert "An example client" in page.text
        disconnected = client.post("/oauth/connections/revoke", data={"grant_id": hidden(page, "grant_id"), "csrf": hidden(page, "csrf")}, headers={"Origin": ORIGIN})
        assert disconnected.status_code == 303
        with pytest.raises(OAuthError):
            oauth.validate_access_token(token)


def test_routes_reject_csrf_duplicates_extra_fields_large_body_and_code_leak(services):
    oauth = services["oauth"]
    registration = client_registration(oauth)
    parameters = request_parameters(oauth, registration, secrets.token_urlsafe(48))
    with route_client(services) as client:
        bad = client.get("/oauth/authorize", params={**parameters, "redirect_uri": "https://attacker.example/steal"})
        assert bad.status_code == 400 and "location" not in bad.headers
        assert client.get("/oauth/authorize?" + urlencode(parameters) + "&state=duplicate").status_code == 400
        assert client.post("/oauth/register", content='{"redirect_uris":[],"redirect_uris":["https://client.example/cb"]}', headers={"Content-Type": "application/json"}).status_code == 400
        assert client.post("/oauth/register", content="x" * (MAX_OAUTH_BODY + 1), headers={"Content-Type": "application/json"}).status_code == 413
        assert client.post("/oauth/token", json={"grant_type": "password"}).status_code == 415
        assert client.post("/oauth/token", data={"grant_type": "password", "password": "should-not-echo"}).json()["error"] == "unsupported_grant_type"
        for origin in ("https://attacker.example", "null"):
            assert client.post("/oauth/consent", data={"csrf": "fake"}, headers={"Origin": origin}).status_code == 403
        assert client.post("/oauth/consent/login", data={"request": "fake", "csrf": "fake", "email": "x", "password": "should-not-echo"}, headers={"Origin": ORIGIN}).status_code == 403
        for attempt in range(31):
            limited = client.post("/oauth/register", json={"redirect_uris": ["https://client.example/cb"]})
        assert limited.status_code == 429


def test_loopback_http_requires_explicit_development_policy(services):
    with pytest.raises(OAuthError):
        CloudOAuth(services["db"], issuer_url="http://127.0.0.1:8766")
    local = CloudOAuth(services["db"], issuer_url="http://127.0.0.1:8766", allow_loopback_http=True)
    registration = local.register_client({"redirect_uris": ["http://127.0.0.1:12345/cb"]})
    assert registration["token_endpoint_auth_method"] == "none"
    assert services["oauth"].register_client({"redirect_uris": ["http://127.0.0.1:12345/cb"]})["redirect_uris"] == ["http://127.0.0.1:12345/cb"]
    with pytest.raises(OAuthError):
        local.register_client({"redirect_uris": ["http://remote.example/cb"]})


def test_issuer_origin_uses_browser_canonical_form(services):
    oauth = CloudOAuth(services["db"], issuer_url="https://MATTERS.EXAMPLE:443/")
    assert oauth.issuer_url == "https://matters.example"
    assert oauth.resource_url == "https://matters.example/mcp"
    ipv6 = CloudOAuth(services["db"], issuer_url="http://[::1]:8766/", allow_loopback_http=True)
    assert ipv6.issuer_url == "http://[::1]:8766"


def test_native_browser_consent_preserves_origin_and_csrf(tmp_path):
    """A native form must work without weakening the Origin:null refusal.

    HTTP clients do not apply document Referrer-Policy to form Origin headers.
    Exercise Chromium when the repository browser-check dependencies exist.
    """
    import asyncio
    from pathlib import Path
    import shutil
    import socket
    import subprocess
    import threading
    import time

    uvicorn = pytest.importorskip("uvicorn")
    node = shutil.which("node")
    executable = os.environ.get("MATTERS_BROWSER_EXECUTABLE", "/usr/bin/chromium")
    if node is None or not Path(executable).is_file():
        pytest.skip("Native browser verification requires Node and Chromium")
    probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, timeout=10)
    if probe.returncode:
        pytest.skip("Native browser verification requires Playwright")

    from matters.cloud.app import CloudBoundaryMiddleware

    owned_socket = socket.socket()
    owned_socket.bind(("127.0.0.1", 0))
    owned_socket.listen(128)
    origin = "http://127.0.0.1:" + str(owned_socket.getsockname()[1])
    db = Database("sqlite:///" + str(tmp_path / "browser-oauth.db")).initialize()
    accounts, repo = CloudAccounts(db), GraphRepository(db)
    user = accounts.register("native-browser@example.com", PASSWORD)
    session = accounts.login(user["email"], PASSWORD)["token"]
    graph = repo.create_graph(user["id"], "Browser consent graph")
    oauth = CloudOAuth(db, issuer_url=origin, allow_loopback_http=True)
    registered = oauth.register_client({"redirect_uris": [origin + "/callback"], "client_name": "Native browser test"})
    pending = oauth.create_authorization_request(request_parameters(oauth, registered, secrets.token_urlsafe(48), "graphs:read"))
    app = FastAPI()
    app.add_middleware(CloudBoundaryMiddleware, public_origin=origin)
    app.include_router(build_oauth_router(oauth, accounts, repo, secure_cookies=False))
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[owned_socket])), daemon=True)
    thread.start()
    script = r"""
const fs = require('node:fs');
const {chromium} = require('playwright');
const config = JSON.parse(fs.readFileSync(0,'utf8'));
(async()=>{
  const browser = await chromium.launch({executablePath:config.executable,headless:true,args:['--no-sandbox']});
  try {
    const context = await browser.newContext();
    await context.addCookies([{name:'matters_session',value:config.session,url:config.origin,httpOnly:true,sameSite:'Strict'}]);
    const page = await context.newPage();page.setDefaultTimeout(10000);
    await page.route(config.origin+'/callback?**',route=>route.fulfill({status:200,body:'Authorized'}));
    let formHeaders = null;
    page.on('request',request=>{
      if(request.method()==='POST' && new URL(request.url()).pathname==='/oauth/consent')
        formHeaders=request.allHeaders().then(headers=>({origin:headers['origin'],site:headers['sec-fetch-site']}));
    });
    const response = await page.goto(config.origin+'/oauth/consent?request='+config.pending);
    const policy = response.headers()['referrer-policy'];
    await page.locator('input[name=graph_id]').check();
    await Promise.all([page.waitForURL(config.origin+'/callback?**'),page.getByRole('button',{name:'Allow selected graphs',exact:true}).click()]);
    console.log(JSON.stringify({policy,headers:await formHeaders,callback:new URL(page.url()).pathname==='/callback'}));
  } finally {await browser.close();}
})().catch(()=>{console.error('Native browser consent failed');process.exitCode=1;});
"""
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "The owned browser-test listener failed to start"
        completed = subprocess.run([node, "-e", script], input=json.dumps({"origin": origin, "session": session, "pending": pending, "executable": executable}), text=True, capture_output=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["policy"] == "strict-origin" and result["callback"] is True
        assert result["headers"]["origin"] == origin
        # Some Chromium builds omit Fetch Metadata on plain local HTTP. The
        # separate HTTPS proof checks that header; a supplied value must agree.
        assert result["headers"].get("site", "") in {"", "same-origin"}
        assert len(oauth.list_grants(user["id"])) == 1
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        owned_socket.close()
        db.close()
        assert not thread.is_alive(), "The owned browser-test listener did not stop"
