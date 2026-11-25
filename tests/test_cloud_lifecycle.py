"""Account action persistence, recovery privacy, revocation and TLS delivery."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socketserver
import ssl
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url

from matters.cloud.accounts import AccountError, CloudAccounts
from matters.cloud.app import CloudBoundaryMiddleware
from matters.cloud.db import Database, account_actions, oauth_records, sessions, users
from matters.cloud.lifecycle import AccountLifecycle, RESET_TTL
from matters.cloud.lifecycle_routes import create_lifecycle_router
from matters.cloud.mail_delivery import CaptureMailDelivery, DisabledMailDelivery, MailDeliveryError, SMTPMailDelivery, mail_delivery_from_env
from matters.cloud.oauth import CloudOAuth, OAuthError
from matters.cloud.repository import GraphRepository


ORIGIN = "https://matters.example"
PASSWORD = "Temporary account password for tests!"
NEW_PASSWORD = "A different temporary password for tests!"
BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def state(request, tmp_path):
    admin = None
    if request.param == "postgresql":
        schema = "lifecycle_test_" + uuid.uuid4().hex
        admin = create_engine(os.environ["MATTERS_TEST_DATABASE_URL"])
        with admin.begin() as connection:
            connection.execute(text('CREATE SCHEMA "' + schema + '"'))
        url = make_url(os.environ["MATTERS_TEST_DATABASE_URL"]).update_query_dict({"options": "-csearch_path=" + schema})
    else:
        url = "sqlite:///" + str(tmp_path / "lifecycle.db")
    db = Database(url).initialize()
    now = [1700000000]
    accounts = CloudAccounts(db, now=lambda: now[0])
    people = [accounts.register(name + "@example.com", PASSWORD) for name in ("alice", "bob")]
    browser_sessions = [accounts.login(person["email"], PASSWORD)["token"] for person in people]
    repo = GraphRepository(db)
    graphs = [repo.create_graph(person["id"], name) for person, name in zip(people, ("Alice graph", "Bob graph"))]
    capture = CaptureMailDelivery()
    lifecycle = AccountLifecycle(db, accounts, public_origin=ORIGIN, delivery=capture, now=lambda: now[0])
    oauth = CloudOAuth(db, issuer_url=ORIGIN, now=lambda: now[0])
    yield {"db": db, "accounts": accounts, "people": people, "sessions": browser_sessions, "repo": repo, "graphs": graphs, "delivery": capture, "lifecycle": lifecycle, "oauth": oauth, "now": now, "url": url}
    lifecycle.close()
    db.close()
    if admin:
        with admin.begin() as connection:
            connection.execute(text('DROP SCHEMA "' + schema + '" CASCADE'))
        admin.dispose()


def delivered_token(state, kind="reset"):
    life = state["lifecycle"]
    if kind == "reset":
        life.request_password_reset(state["people"][0]["email"])
    else:
        life.request_verification(state["people"][0]["id"])
    assert life.drain_delivery()
    message = state["delivery"].messages[-1]
    link = next(line for line in message.body.splitlines() if line.startswith(ORIGIN))
    assert urlsplit(link).query == "" and urlsplit(link).path == "/account/" + kind
    return parse_qs(urlsplit(link).fragment)["token"][0]


def oauth_grant(state):
    oauth = state["oauth"]
    client = oauth.register_client({"redirect_uris": ["https://client.example/callback"], "scope": "graphs:read offline_access"})
    verifier, csrf = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    pending = oauth.create_authorization_request({"client_id": client["client_id"], "redirect_uri": client["redirect_uris"][0], "response_type": "code", "resource": oauth.resource_url, "scope": "graphs:read offline_access", "code_challenge": challenge, "code_challenge_method": "S256"})
    actor, session = state["people"][0]["id"], state["sessions"][0]
    oauth.prepare_consent(pending, actor, session, csrf)
    redirect = oauth.consent(pending, actor, session, csrf, [state["graphs"][0]["graph_id"]], repository=state["repo"])
    code = parse_qs(urlsplit(redirect).query)["code"][0]
    tokens = oauth.exchange_code(client_id=client["client_id"], code=code, redirect_uri=client["redirect_uris"][0], code_verifier=verifier, resource=oauth.resource_url)
    return client, tokens


def test_verification_is_hashed_single_use_and_does_not_gate_legacy_login(state):
    life, accounts = state["lifecycle"], state["accounts"]
    assert not life.status(state["people"][0]["id"])["email_verified"]
    assert accounts.login("alice@example.com", PASSWORD)["token"]
    token = delivered_token(state, "verify")
    with state["db"].transaction() as connection:
        action = connection.execute(select(account_actions)).mappings().one()
    assert action["id"] == hashlib.sha256(token.encode()).hexdigest()
    assert token not in json.dumps(dict(action))
    assert life.verify_email(token) == {"email_verified": True}
    assert life.status(state["people"][0]["id"])["email_verified_at"] == state["now"][0]
    with pytest.raises(AccountError, match="invalid or expired"):
        life.verify_email(token)
    assert accounts.authenticate(state["sessions"][0])["id"] == state["people"][0]["id"]


def test_public_reset_response_identical_unknown_disabled_and_failed_delivery(state):
    life = state["lifecycle"]
    expected = life.request_password_reset("alice@example.com")
    assert life.request_password_reset("unknown@example.com") == expected
    assert life.request_password_reset("invalid") == expected
    assert life.drain_delivery()
    assert len(state["delivery"].messages) == 1
    disabled = AccountLifecycle(state["db"], state["accounts"])
    assert not disabled.capabilities()["email_delivery"]
    assert disabled.request_password_reset("alice@example.com") == disabled.request_password_reset("unknown@example.com") == expected
    class FailedDelivery:
        configured = True
        def send(self, recipient, subject, body):
            raise RuntimeError(body)
    failed = AccountLifecycle(state["db"], state["accounts"], public_origin=ORIGIN, delivery=FailedDelivery(), now=lambda: state["now"][0])
    try:
        assert failed.request_password_reset("alice@example.com") == expected
        assert failed.request_password_reset("unknown@example.com") == expected
        assert failed.drain_delivery()
        with state["db"].transaction() as connection:
            actions = connection.execute(select(account_actions)).mappings().all()
        assert sum(row["used_at"] is not None for row in actions) == 1
    finally:
        failed.close()


def test_public_reset_does_not_wait_for_account_lookup_or_smtp(state):
    entered, release = threading.Event(), threading.Event()
    class SlowDelivery:
        configured = True
        def send(self, recipient, subject, body):
            entered.set()
            assert release.wait(timeout=5)
    life = AccountLifecycle(state["db"], state["accounts"], public_origin=ORIGIN, delivery=SlowDelivery(), now=lambda: state["now"][0])
    try:
        started = time.monotonic()
        first = life.request_password_reset("alice@example.com")
        assert time.monotonic() - started < 1
        assert entered.wait(timeout=3)
        started = time.monotonic()
        assert life.request_password_reset("unknown@example.com") == first
        assert time.monotonic() - started < 1
    finally:
        release.set()
        assert life.drain_delivery()
        life.close()


def test_reset_revokes_sessions_oauth_actions_without_graph_changes(state):
    life, accounts, oauth = state["lifecycle"], state["accounts"], state["oauth"]
    client, tokens = oauth_grant(state)
    verification = delivered_token(state, "verify")
    reset = delivered_token(state)
    alice, bob = state["people"]
    before = state["repo"].get_graph(alice["id"], state["graphs"][0]["graph_id"])
    assert life.reset_password(reset, NEW_PASSWORD) == {"password_changed": True, "sign_in_required": True}
    for raw in state["sessions"][:1]:
        with pytest.raises(AccountError):
            accounts.authenticate(raw)
    assert accounts.authenticate(state["sessions"][1])["id"] == bob["id"]
    with pytest.raises(OAuthError):
        oauth.validate_access_token(tokens["access_token"])
    with pytest.raises(OAuthError):
        oauth.refresh(client_id=client["client_id"], refresh_token=tokens["refresh_token"], resource=oauth.resource_url)
    with pytest.raises(AccountError):
        life.verify_email(verification)
    with pytest.raises(AccountError):
        life.reset_password(reset, NEW_PASSWORD)
    with pytest.raises(AccountError):
        accounts.login(alice["email"], PASSWORD)
    assert accounts.login(alice["email"], NEW_PASSWORD)["user"]["id"] == alice["id"]
    assert state["repo"].get_graph(alice["id"], before["graph_id"]) == before
    assert not life.status(alice["id"])["email_verified"]


def test_reset_expiry_credential_binding_and_invalid_password_leave_state_intact(state):
    life = state["lifecycle"]
    token = delivered_token(state)
    with pytest.raises(AccountError):
        life.reset_password(token, "short")
    assert state["accounts"].authenticate(state["sessions"][0])
    state["now"][0] += RESET_TTL
    with pytest.raises(AccountError):
        life.reset_password(token, NEW_PASSWORD)
    token = delivered_token(state)
    replacement = state["accounts"]._hasher.hash("An independently changed password!")
    with state["db"].transaction(write=True) as connection:
        connection.execute(users.update().where(users.c.id == state["people"][0]["id"]).values(password_hash=replacement))
    with pytest.raises(AccountError):
        life.reset_password(token, NEW_PASSWORD)


def test_concurrent_reset_is_single_use(state):
    token = delivered_token(state)
    def reset():
        try:
            return state["lifecycle"].reset_password(token, NEW_PASSWORD)
        except AccountError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reset(), range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1 and "rejected" in results


def test_password_change_requires_current_password_and_own_active_session(state):
    life = state["lifecycle"]
    actor = state["people"][0]["id"]
    client, tokens = oauth_grant(state)
    for session, current in ((state["sessions"][0], "Wrong temporary password!"), (state["sessions"][1], PASSWORD)):
        with pytest.raises(AccountError):
            life.change_password(actor, session, current, NEW_PASSWORD)
    assert state["accounts"].authenticate(state["sessions"][0])
    result = life.change_password(actor, state["sessions"][0], PASSWORD, NEW_PASSWORD)
    assert result["sign_in_required"]
    with pytest.raises(AccountError):
        state["accounts"].authenticate(state["sessions"][0])
    with pytest.raises(OAuthError):
        state["oauth"].validate_access_token(tokens["access_token"])
    assert state["accounts"].login("alice@example.com", NEW_PASSWORD)


def test_issued_action_survives_reopening_database(state):
    token = delivered_token(state)
    state["lifecycle"].close()
    state["db"].close()
    reopened = Database(state["url"]).initialize()
    accounts = CloudAccounts(reopened, now=lambda: state["now"][0])
    lifecycle = AccountLifecycle(reopened, accounts, now=lambda: state["now"][0])
    try:
        assert lifecycle.reset_password(token, NEW_PASSWORD)["password_changed"]
        assert accounts.login("alice@example.com", NEW_PASSWORD)
    finally:
        lifecycle.close()
        reopened.close()


def router_client(state):
    app = FastAPI()
    app.add_middleware(CloudBoundaryMiddleware, public_origin=ORIGIN)
    @app.exception_handler(AccountError)
    def error(request, error):
        status = {"unauthorized": 401, "forbidden": 403, "rate_limited": 429, "unavailable": 503}.get(error.code, 400)
        return JSONResponse({"schema_version": 1, "error": {"code": error.code, "message": error.message}}, status_code=status)
    def authenticated(request: Request):
        return state["accounts"].authenticate(request.cookies.get("matters_session"))
    app.include_router(create_lifecycle_router(state["lifecycle"], authenticated, public_origin=ORIGIN))
    client = TestClient(app, base_url=ORIGIN)
    client.cookies.set("matters_session", state["sessions"][0])
    return client


def test_http_lifecycle_csrf_no_token_echo_and_fragment_only_landings(state):
    with router_client(state) as client:
        capabilities = client.get("/api/auth/capabilities")
        csrf = capabilities.json()["csrf_token"]
        assert capabilities.json()["email_delivery"] and "httponly" in capabilities.headers["set-cookie"].lower()
        assert not client.get("/api/auth/security").json()["email_verified"]
        assert client.post("/api/auth/verification/request", json={"csrf_token": csrf}, headers={"Origin": ORIGIN}).status_code == 202
        assert state["lifecycle"].drain_delivery()
        link = next(line for line in state["delivery"].messages[-1].body.splitlines() if line.startswith(ORIGIN))
        token = parse_qs(urlsplit(link).fragment)["token"][0]
        landing = client.get(link)
        assert landing.status_code == 200 and token not in landing.text and "/assets/lifecycle.js" in landing.text
        assert '<form id="lifecycle-form" class="stack" method="post">' in landing.text
        assert not state["lifecycle"].status(state["people"][0]["id"])["email_verified"]
        assert client.get("/account/verify?token=" + token).status_code == 400
        for headers, data in (({"Origin": "null"}, {"token": token, "csrf_token": csrf}), ({"Origin": ORIGIN}, {"token": token, "csrf_token": "wrong"})):
            assert client.post("/api/auth/verification/confirm", json=data, headers=headers).status_code == 403
        verified = client.post("/api/auth/verification/confirm", json={"token": token, "csrf_token": csrf}, headers={"Origin": ORIGIN})
        assert verified.status_code == 200 and verified.json()["email_verified"] and token not in verified.text
        known = client.post("/api/auth/password/reset/request", json={"email": "alice@example.com"}, headers={"Origin": ORIGIN})
        unknown = client.post("/api/auth/password/reset/request", json={"email": "unknown@example.com"}, headers={"Origin": ORIGIN})
        assert known.status_code == unknown.status_code == 202 and known.json() == unknown.json()
        assert state["lifecycle"].drain_delivery()
        link = next(line for line in state["delivery"].messages[-1].body.splitlines() if line.startswith(ORIGIN))
        token = parse_qs(urlsplit(link).fragment)["token"][0]
        assert token not in known.text
        reset = client.post("/api/auth/password/reset/confirm", json={"token": token, "new_password": NEW_PASSWORD, "csrf_token": csrf}, headers={"Origin": ORIGIN})
        assert reset.status_code == 200 and reset.json()["password_changed"] and token not in reset.text
        assert client.get("/api/auth/security").status_code == 401


def test_lifecycle_limiting_rejects_peer_before_allocating_subject(state):
    from matters.cloud.lifecycle_routes import create_lifecycle_router
    calls = []
    class Limiter:
        def allow(self, subject, *, category):
            calls.append((category, subject))
            return False
    app = FastAPI()
    @app.exception_handler(AccountError)
    def error(request, error):
        return JSONResponse({"error": error.code}, status_code=429)
    app.include_router(create_lifecycle_router(state["lifecycle"], lambda: {}, public_origin=ORIGIN, limiter=Limiter()))
    with TestClient(app, base_url=ORIGIN) as client:
        result = client.post("/api/auth/password/reset/request", json={"email": "unknown@example.com"}, headers={"Origin": ORIGIN})
        assert result.status_code == 429
    assert calls == [("auth-recovery-peer", "testclient")]


def test_mail_configuration_requires_explicit_tls_private_secret_and_origin(tmp_path):
    assert isinstance(mail_delivery_from_env({}), DisabledMailDelivery)
    with pytest.raises(MailDeliveryError):
        mail_delivery_from_env({"MATTERS_SMTP_SENDER": "sender@example.com"})
    with pytest.raises(MailDeliveryError):
        SMTPMailDelivery("localhost", 25, "sender@example.com", tls="none")
    secret = tmp_path / "smtp-password"
    secret.write_text("Owned-test-password-only")
    secret.chmod(0o644)
    with pytest.raises(MailDeliveryError):
        SMTPMailDelivery("localhost", 465, "sender@example.com", username="test-user", password_file=secret, tls="implicit")
    secret.chmod(0o600)
    delivery = SMTPMailDelivery.from_env({"MATTERS_SMTP_HOST": "localhost", "MATTERS_SMTP_PORT": "465", "MATTERS_SMTP_SENDER": "sender@example.com", "MATTERS_SMTP_USERNAME": "test-user", "MATTERS_SMTP_PASSWORD_FILE": str(secret), "MATTERS_SMTP_TLS": "implicit"})
    assert delivery.configured
    with pytest.raises(ValueError, match="trusted public origin"):
        AccountLifecycle(None, None, delivery=delivery)


def test_smtp_delivery_uses_verified_tls_with_owned_sink_only(tmp_path):
    executable = shutil.which("openssl")
    if executable is None:
        pytest.skip("Owned TLS SMTP proof requires OpenSSL")
    certificate, key = tmp_path / "smtp-cert.pem", tmp_path / "smtp-key.pem"
    generated = subprocess.run([executable, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1", "-keyout", str(key), "-out", str(certificate)], capture_output=True, timeout=15)
    assert generated.returncode == 0
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    messages = []
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            with context.wrap_socket(self.request, server_side=True) as connection:
                connection.settimeout(5)
                stream = connection.makefile("rwb")
                stream.write(b"220 localhost owned-test-sink\r\n"); stream.flush()
                while True:
                    line = stream.readline(65536)
                    command = line.split(b" ", 1)[0].strip().upper()
                    if command == b"EHLO":
                        stream.write(b"250-localhost\r\n250 SIZE 65536\r\n")
                    elif command in {b"MAIL", b"RCPT", b"RSET"}:
                        stream.write(b"250 OK\r\n")
                    elif command == b"DATA":
                        stream.write(b"354 End with dot\r\n"); stream.flush()
                        content = bytearray()
                        while True:
                            part = stream.readline(65536)
                            if part in {b".\r\n", b""}:
                                break
                            content.extend(part)
                        messages.append(bytes(content))
                        stream.write(b"250 Accepted\r\n")
                    elif command == b"QUIT":
                        stream.write(b"221 Bye\r\n"); stream.flush(); break
                    else:
                        stream.write(b"500 Unsupported\r\n")
                    stream.flush()
    class Sink(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
        def handle_error(self, request, client_address):
            pass
    sink = Sink(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    try:
        port = sink.server_address[1]
        untrusted = SMTPMailDelivery("localhost", port, "sender@example.com", tls="implicit")
        with pytest.raises(MailDeliveryError):
            untrusted.send("alice@example.com", "Owned test", "Temporary test content")
        assert not messages
        trusted = SMTPMailDelivery("localhost", port, "sender@example.com", tls="implicit", ca_file=certificate)
        trusted.send("alice@example.com", "Owned test", "Temporary test content")
        assert len(messages) == 1 and b"Temporary test content" in messages[0]
    finally:
        sink.shutdown(); sink.server_close(); thread.join(timeout=5)
