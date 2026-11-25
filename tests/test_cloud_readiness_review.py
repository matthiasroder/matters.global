"""Independent adversarial checks for provider-independent Cloud readiness."""

import base64
import hashlib
import importlib.util
import io
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from matters.cloud.accounts import AccountError, CloudAccounts
from matters.cloud.db import CloudDatabaseError, Database, account_actions, rate_limit_buckets
from matters.cloud.oauth import CloudOAuth, OAuthError
from matters.cloud.repository import GraphRepository


ORIGIN = "https://readiness-review.example"
PASSWORD = "Initial readiness review password"
BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def installation(request, tmp_path):
    admin = None
    schema = None
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'readiness-review.db'}"
    else:
        pytest.importorskip("psycopg")
        configured = make_url(os.environ["MATTERS_TEST_DATABASE_URL"])
        assert configured.get_backend_name() == "postgresql"
        admin = create_engine(configured)
        schema = "readiness_review_" + uuid.uuid4().hex
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = configured.update_query_dict({"options": f"-csearch_path={schema}"})
    db = Database(url).initialize()
    try:
        yield db
    finally:
        db.close()
        if admin is not None:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
            admin.dispose()


def test_independent_limiter_instances_share_one_atomic_peer_budget(installation):
    from matters.cloud.rate_limits import DatabaseRateLimiter

    db = installation
    second_db = Database(db.engine.url).initialize()
    clock = [2000]
    limiters = [DatabaseRateLimiter(storage, attempts=3, capacity=8, now=lambda: clock[0]) for storage in (db, second_db)]
    try:
        with ThreadPoolExecutor(max_workers=12) as executor:
            allowed = list(executor.map(lambda index: limiters[index % 2].allow("peer-secret-address", category="login"), range(24)))
        assert sum(allowed) == 3
        assert not DatabaseRateLimiter(db, attempts=3, capacity=8, now=lambda: clock[0]).allow("peer-secret-address", category="login")
        with db.transaction() as connection:
            rows = [dict(row) for row in connection.execute(select(rate_limit_buckets)).mappings()]
        assert "peer-secret-address" not in str(rows)
    finally:
        second_db.close()


def test_limiter_capacity_cannot_evict_a_live_peer_and_reset_its_budget(installation):
    from matters.cloud.rate_limits import DatabaseRateLimiter

    db = installation
    clock = [3000]
    limiter = DatabaseRateLimiter(db, attempts=2, capacity=4, cleanup_batch_size=2, now=lambda: clock[0])
    assert limiter.allow("known-peer")
    assert limiter.allow("known-peer")
    for index in range(20):
        limiter.allow(f"attacker-peer-{index}")
    assert not limiter.allow("known-peer")
    with db.transaction() as connection:
        assert len(connection.execute(select(rate_limit_buckets)).all()) <= 5
    clock[0] += 61
    assert limiter.allow("new-peer-after-expiry")
    assert limiter.allow("known-peer")


def test_limiter_database_failure_does_not_fall_back_to_unlimited_access(installation, monkeypatch):
    from matters.cloud.rate_limits import DatabaseRateLimiter, RateLimitUnavailable

    limiter = DatabaseRateLimiter(installation)

    @contextmanager
    def unavailable(*args, **kwargs):
        raise CloudDatabaseError("Synthetic storage outage")
        yield

    monkeypatch.setattr(installation, "transaction", unavailable)
    with pytest.raises(RateLimitUnavailable):
        limiter.allow("peer")


def prepare_oauth(db):
    accounts = CloudAccounts(db)
    user = accounts.register("race-owner@example.com", PASSWORD)
    session = accounts.login(user["email"], PASSWORD)
    repository = GraphRepository(db)
    graph = repository.create_graph(user["id"], "Password change must preserve this graph")
    oauth = CloudOAuth(db, issuer_url=ORIGIN)
    client = oauth.register_client({"redirect_uris": ["https://client.example/callback"], "scope": "graphs:read graphs:write offline_access"})
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    request = oauth.create_authorization_request({"response_type": "code", "client_id": client["client_id"], "redirect_uri": "https://client.example/callback", "scope": "graphs:read graphs:write offline_access", "resource": oauth.resource_url, "code_challenge": challenge, "code_challenge_method": "S256"})
    csrf = secrets.token_urlsafe(32)
    oauth.prepare_consent(request, user["id"], session["token"], csrf)
    return accounts, user, session, repository, graph, oauth, client, verifier, request, csrf


@pytest.mark.parametrize("issuance", ["consent", "code", "refresh"])
@pytest.mark.parametrize("credential_operation", ["change", "reset"])
def test_credential_change_and_oauth_issuance_serialize_without_deadlock_or_surviving_tokens(installation, issuance, credential_operation):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    db = installation
    if db.engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row/FK locks are needed to exercise this interleaving")
    accounts, user, session, repository, graph, oauth, client, verifier, request, csrf = prepare_oauth(db)
    delivery = CaptureMailDelivery() if credential_operation == "reset" else None
    lifecycle = AccountLifecycle(db, accounts, public_origin=ORIGIN, delivery=delivery)
    reset_token = None
    if delivery is not None:
        lifecycle.request_password_reset(user["email"])
        assert lifecycle.drain_delivery()
        reset_token = delivered_token(delivery)
    exchange = None
    initial_tokens = None
    if issuance != "consent":
        callback = parse_qs(urlsplit(oauth.consent(request, user["id"], session["token"], csrf, [graph["graph_id"]], repository=repository)).query)
        exchange = {"client_id": client["client_id"], "code": callback["code"][0], "redirect_uri": "https://client.example/callback", "code_verifier": verifier, "resource": oauth.resource_url}
        if issuance == "refresh":
            initial_tokens = oauth.exchange_code(**exchange)

    local = threading.local()
    issue_paused = threading.Event()
    release_issue = threading.Event()
    change_attempted_user_lock = threading.Event()
    change_obtained_user_lock = threading.Event()

    def before_cursor(_connection, _cursor, statement, _parameters, context, _executemany):
        role = getattr(local, "role", None)
        if role == "issue" and getattr(context, "isinsert", False) and not issue_paused.is_set():
            expected = "grant" if issuance == "consent" else "access"
            if any(p.get("kind") == expected for p in context.compiled_parameters):
                issue_paused.set()
                assert release_issue.wait(8), "Password change test did not release issuance"
        if role == "change" and "users" in statement.lower() and "for update" in statement.lower():
            change_attempted_user_lock.set()

    def after_cursor(_connection, _cursor, statement, _parameters, _context, _executemany):
        if getattr(local, "role", None) == "change" and "users" in statement.lower() and "for update" in statement.lower():
            change_obtained_user_lock.set()

    event.listen(db.engine, "before_cursor_execute", before_cursor)
    event.listen(db.engine, "after_cursor_execute", after_cursor)

    def issue():
        local.role = "issue"
        if issuance == "consent":
            return oauth.consent(request, user["id"], session["token"], csrf, [graph["graph_id"]], repository=repository)
        if issuance == "code":
            return oauth.exchange_code(**exchange)
        return oauth.refresh(client_id=client["client_id"], refresh_token=initial_tokens["refresh_token"], resource=oauth.resource_url)

    def change():
        local.role = "change"
        if credential_operation == "reset":
            return lifecycle.reset_password(reset_token, "New readiness review password")
        return lifecycle.change_password(user["id"], session["token"], PASSWORD, "New readiness review password")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            issued = executor.submit(issue)
            assert issue_paused.wait(8), "Issuance did not reach its credential insert"
            changed = executor.submit(change)
            try:
                assert change_attempted_user_lock.wait(8), "Password change did not attempt its user lock"
                # Old lock order can obtain the user lock here and then block
                # deleting the issuance record. Correct order blocks earlier.
                change_obtained_user_lock.wait(0.2)
            finally:
                release_issue.set()
            try:
                result = issued.result(timeout=12)
            except OAuthError as error:
                assert error.code in {"invalid_grant", "access_denied"}
                result = None
            changed.result(timeout=12)
    finally:
        release_issue.set()
        event.remove(db.engine, "before_cursor_execute", before_cursor)
        event.remove(db.engine, "after_cursor_execute", after_cursor)
        lifecycle.close()

    with pytest.raises(AccountError):
        accounts.authenticate(session["token"])
    with pytest.raises(AccountError):
        accounts.login(user["email"], PASSWORD)
    assert accounts.login(user["email"], "New readiness review password")["user"]["id"] == user["id"]
    assert repository.get_graph(user["id"], graph["graph_id"]) == graph
    for tokens in (initial_tokens, result if isinstance(result, dict) else None):
        if tokens and "access_token" in tokens:
            with pytest.raises(OAuthError):
                oauth.validate_access_token(tokens["access_token"])
    if isinstance(result, str):
        code = parse_qs(urlsplit(result).query)["code"][0]
        with pytest.raises(OAuthError):
            oauth.exchange_code(client_id=client["client_id"], code=code, redirect_uri="https://client.example/callback", code_verifier=verifier, resource=oauth.resource_url)


@pytest.mark.parametrize("credential_operation", ["change", "reset"])
def test_login_verified_before_credential_change_cannot_create_an_old_credential_session(installation, credential_operation):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts, user, session, repository, graph, _, _, _, _, _ = prepare_oauth(installation)
    delivery = CaptureMailDelivery() if credential_operation == "reset" else None
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=delivery)
    reset_token = None
    if delivery is not None:
        lifecycle.request_password_reset(user["email"])
        assert lifecycle.drain_delivery()
        reset_token = delivered_token(delivery)
    verified = threading.Event()
    resume = threading.Event()
    actual = accounts._hasher

    class PausedLoginHasher:
        def verify(self, encoded, password):
            result = actual.verify(encoded, password)
            if threading.current_thread().name.startswith("old-login"):
                verified.set()
                assert resume.wait(8)
            return result

        def hash(self, password):
            return actual.hash(password)

        def check_needs_rehash(self, encoded):
            return actual.check_needs_rehash(encoded)

    accounts._hasher = PausedLoginHasher()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="old-login") as executor:
        pending = executor.submit(accounts.login, user["email"], PASSWORD)
        assert verified.wait(8)
        try:
            if credential_operation == "reset":
                lifecycle.reset_password(reset_token, "Changed while login was pending")
            else:
                lifecycle.change_password(user["id"], session["token"], PASSWORD, "Changed while login was pending")
        finally:
            resume.set()
        with pytest.raises(AccountError) as stale_login:
            pending.result(timeout=10)
        assert stale_login.value.code == "unauthorized"
    lifecycle.close()
    assert repository.get_graph(user["id"], graph["graph_id"]) == graph


def delivered_token(delivery):
    link = next(line for line in delivery.messages[-1].body.splitlines() if line.startswith(ORIGIN + "/account/"))
    assert not urlsplit(link).query
    return parse_qs(urlsplit(link).fragment)["token"][0]


def test_reset_request_returns_before_delivery_and_never_reveals_account_lookup(installation):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts = CloudAccounts(installation)
    user = accounts.register("async-mail@example.com", PASSWORD)
    started, release = threading.Event(), threading.Event()

    class SlowCapture(CaptureMailDelivery):
        def send(self, recipient, subject, body):
            started.set()
            assert release.wait(8)
            super().send(recipient, subject, body)

    capture = SlowCapture()
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=capture)
    disabled = AccountLifecycle(installation, accounts, public_origin=ORIGIN)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(lifecycle.request_password_reset, user["email"]).result(timeout=1)
        assert started.wait(5)
        assert lifecycle.request_password_reset("missing@example.com") == response
        assert lifecycle.request_password_reset("not an email") == response
        assert disabled.request_password_reset(user["email"]) == response
        assert disabled.request_password_reset("missing@example.com") == response
        assert disabled.capabilities()["email_delivery"] is False
    finally:
        release.set()
        lifecycle.close()
        disabled.close()


def test_delivery_failure_invalidates_its_link_without_public_secret_or_account_changes(installation, capsys):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts = CloudAccounts(installation)
    user = accounts.register("failed-mail@example.com", PASSWORD)
    session = accounts.login(user["email"], PASSWORD)

    class FailingCapture(CaptureMailDelivery):
        def send(self, recipient, subject, body):
            super().send(recipient, subject, body)
            raise RuntimeError("Synthetic secret-bearing transport failure: " + body)

    capture = FailingCapture()
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=capture)
    try:
        response = lifecycle.request_password_reset(user["email"])
        assert lifecycle.drain_delivery()
        assert response == lifecycle.request_password_reset("missing@example.com")
        token = delivered_token(capture)
        assert token not in repr(capture.messages[-1])
        with pytest.raises(AccountError):
            lifecycle.reset_password(token, "Rejected failed delivery password")
        assert accounts.authenticate(session["token"])["id"] == user["id"]
        with installation.transaction() as connection:
            action = connection.execute(select(account_actions)).mappings().one()
            assert action["used_at"] is not None and token not in str(action)
        output = capsys.readouterr()
        assert token not in output.out + output.err
    finally:
        lifecycle.close()


def test_expired_reset_and_verification_links_leave_credentials_and_status_unchanged(installation):
    from matters.cloud.lifecycle import AccountLifecycle, RESET_TTL, VERIFY_TTL
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts = CloudAccounts(installation)
    user = accounts.register("expired-mail@example.com", PASSWORD)
    capture = CaptureMailDelivery()
    clock = [1_900_000_000]
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=capture, now=lambda: clock[0])
    try:
        lifecycle.request_password_reset(user["email"])
        assert lifecycle.drain_delivery()
        reset_token = delivered_token(capture)
        lifecycle.request_verification(user["id"])
        assert lifecycle.drain_delivery()
        verify_token = delivered_token(capture)
        clock[0] += RESET_TTL
        with pytest.raises(AccountError):
            lifecycle.reset_password(reset_token, "Expired request must not change password")
        clock[0] += VERIFY_TTL
        with pytest.raises(AccountError):
            lifecycle.verify_email(verify_token)
        assert lifecycle.status(user["id"])["email_verified"] is False
        assert accounts.login(user["email"], PASSWORD)["user"]["id"] == user["id"]
    finally:
        lifecycle.close()


def test_reset_consumes_actions_and_revokes_old_oauth_and_sessions_without_graph_changes(installation):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts, user, session, repository, graph, oauth, client, verifier, request, csrf = prepare_oauth(installation)
    callback = parse_qs(urlsplit(oauth.consent(request, user["id"], session["token"], csrf, [graph["graph_id"]], repository=repository)).query)
    exchange = {"client_id": client["client_id"], "code": callback["code"][0], "redirect_uri": "https://client.example/callback", "code_verifier": verifier, "resource": oauth.resource_url}
    tokens = oauth.exchange_code(**exchange)
    capture = CaptureMailDelivery()
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=capture)
    try:
        lifecycle.request_verification(user["id"])
        assert lifecycle.drain_delivery()
        verification = delivered_token(capture)
        lifecycle.request_password_reset(user["email"])
        assert lifecycle.drain_delivery()
        reset = delivered_token(capture)
        assert lifecycle.reset_password(reset, "Recovered readiness account password")["sign_in_required"] is True
        for operation in (
            lambda: lifecycle.verify_email(verification),
            lambda: lifecycle.reset_password(reset, "Attempted replay password"),
            lambda: accounts.authenticate(session["token"]),
            lambda: accounts.login(user["email"], PASSWORD),
        ):
            with pytest.raises(AccountError):
                operation()
        for operation in (
            lambda: oauth.validate_access_token(tokens["access_token"]),
            lambda: oauth.refresh(client_id=client["client_id"], refresh_token=tokens["refresh_token"], resource=oauth.resource_url),
            lambda: oauth.exchange_code(**exchange),
        ):
            with pytest.raises(OAuthError):
                operation()
        assert accounts.login(user["email"], "Recovered readiness account password")["user"]["id"] == user["id"]
        assert repository.get_graph(user["id"], graph["graph_id"]) == graph
    finally:
        lifecycle.close()


def test_verification_is_one_use_and_bound_to_current_credentials(installation):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts = CloudAccounts(installation)
    user = accounts.register("verify-mail@example.com", PASSWORD)
    session = accounts.login(user["email"], PASSWORD)
    capture = CaptureMailDelivery()
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=capture)
    try:
        lifecycle.request_verification(user["id"])
        assert lifecycle.drain_delivery()
        old = delivered_token(capture)
        lifecycle.change_password(user["id"], session["token"], PASSWORD, "New verified account credential")
        with pytest.raises(AccountError):
            lifecycle.verify_email(old)
        assert lifecycle.status(user["id"])["email_verified"] is False
        lifecycle.request_verification(user["id"])
        assert lifecycle.drain_delivery()
        current = delivered_token(capture)
        assert lifecycle.verify_email(current) == {"email_verified": True}
        with pytest.raises(AccountError):
            lifecycle.verify_email(current)
        assert lifecycle.status(user["id"])["email_verified"] is True
    finally:
        lifecycle.close()


def test_http_reset_has_uniform_requests_and_requires_origin_csrf_and_the_action_token(installation):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from matters.cloud.app import create_app
    from matters.cloud.mail_delivery import CaptureMailDelivery

    capture = CaptureMailDelivery()
    app = create_app(installation.engine.url.render_as_string(hide_password=False), public_origin=ORIGIN, mail_delivery=capture)
    alice = app.state.accounts.register("http-alice@example.com", PASSWORD)
    bob = app.state.accounts.register("http-bob@example.com", PASSWORD)
    old_alice = app.state.accounts.login(alice["email"], PASSWORD)
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.post("/api/auth/login", json={"email": bob["email"], "password": PASSWORD}).status_code == 200
        bob_cookie = client.cookies.get("matters_session")
        capabilities = client.get("/api/auth/capabilities").json()
        private_fields = {"requested", "queued", "sent", "failed", "dropped", "skipped", "closed", "event_sink_failed", "queue_depth", "queue_capacity"}
        assert private_fields.isdisjoint(capabilities)
        assert private_fields.isdisjoint(client.get("/api/auth/security").json())
        nonce = capabilities["csrf_token"]
        first = client.post("/api/auth/password/reset/request", json={"email": alice["email"]}, headers={"Origin": ORIGIN})
        unknown = client.post("/api/auth/password/reset/request", json={"email": "missing@example.com"}, headers={"Origin": ORIGIN})
        assert first.status_code == unknown.status_code == 202 and first.json() == unknown.json()
        assert app.state.lifecycle.drain_delivery()
        token = delivered_token(capture)
        body = {"token": token, "new_password": "Recovered HTTP Alice password", "csrf_token": nonce}
        assert client.post("/api/auth/password/reset/confirm", json=body, headers={"Origin": "https://attacker.example"}).status_code == 403
        assert client.post("/api/auth/password/reset/confirm", json={**body, "csrf_token": "wrong"}, headers={"Origin": ORIGIN}).status_code == 403
        assert app.state.accounts.authenticate(old_alice["token"])["id"] == alice["id"]
        reset = client.post("/api/auth/password/reset/confirm", json=body, headers={"Origin": ORIGIN})
        assert reset.status_code == 200 and reset.json()["password_changed"] is True
        assert token not in reset.text and "no-store" in reset.headers["cache-control"]
        # A link targets its account, never whichever other account happens to
        # be signed in. Nor may a late response delete that other login cookie.
        assert client.cookies.get("matters_session") == bob_cookie
        assert client.get("/api/auth/me").json()["user"]["id"] == bob["id"]
        with pytest.raises(AccountError):
            app.state.accounts.authenticate(old_alice["token"])
        assert app.state.accounts.login(alice["email"], "Recovered HTTP Alice password")["user"]["id"] == alice["id"]
        assert client.post("/api/auth/password/reset/confirm", json=body, headers={"Origin": ORIGIN}).status_code == 400


def test_http_apps_share_login_budget_and_ignore_untrusted_forwarded_peer_headers(installation, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from matters.cloud.app import create_app

    first = create_app(installation.engine.url.render_as_string(hide_password=False), public_origin=ORIGIN)
    second = create_app(installation.engine.url.render_as_string(hide_password=False), public_origin=ORIGIN)
    first.state.auth_limiter.attempts = second.state.auth_limiter.attempts = 2
    credentials = {"email": "missing@example.com", "password": PASSWORD}
    with TestClient(first, base_url=ORIGIN) as one, TestClient(second, base_url=ORIGIN) as two:
        assert one.post("/api/auth/login", json=credentials, headers={"X-Forwarded-For": "198.51.100.1", "X-Real-IP": "198.51.100.1"}).status_code == 401
        assert two.post("/api/auth/login", json=credentials, headers={"X-Forwarded-For": "198.51.100.2", "X-Real-IP": "198.51.100.2"}).status_code == 401
        assert one.post("/api/auth/login", json=credentials, headers={"X-Forwarded-For": "198.51.100.3", "X-Real-IP": "198.51.100.3"}).status_code == 429

        @contextmanager
        def unavailable(*args, **kwargs):
            raise CloudDatabaseError("Synthetic limiter storage outage")
            yield

        monkeypatch.setattr(second.state.db, "transaction", unavailable)
        response = two.post("/api/auth/login", json=credentials)
        assert response.status_code == 503
        assert "missing@example.com" not in response.text and "Synthetic" not in response.text


def test_smtp_secret_named_pipe_is_refused_without_hanging_startup(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX named pipes are required")
    path = tmp_path / "smtp-password-pipe"
    os.mkfifo(path, 0o600)
    code = """
import sys
from matters.cloud.mail_delivery import SMTPMailDelivery, MailDeliveryError
try:
    SMTPMailDelivery('smtp.example.com', 587, 'sender@example.com', username='review', password_file=sys.argv[1])
except MailDeliveryError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, "Non-regular SMTP credential files must be refused safely"


@pytest.mark.parametrize("reader", ["backup-key", "database-password", "role-record"])
def test_operator_private_readers_refuse_named_pipes_without_hanging(tmp_path, reader):
    pytest.importorskip("cryptography")
    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX named pipes are required")
    path = tmp_path / "operator-pipe"
    os.mkfifo(path, 0o600)
    ops = Path(__file__).parents[1] / "ops" / "cloud"
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from recovery import OpsError, load_key
from database_roles import private_password, read_record
try:
    {'backup-key': load_key, 'database-password': private_password, 'role-record': read_record}[sys.argv[3]](sys.argv[2])
except OpsError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run([sys.executable, "-c", code, str(ops), str(path), reader], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, "Operator inputs must be regular files and must fail without blocking"


@pytest.fixture
def owned_operations_database(tmp_path):
    if os.environ.get("MATTERS_TEST_OPERATIONS") != "1":
        pytest.skip("Set MATTERS_TEST_OPERATIONS=1 for an owned disposable Docker/PostgreSQL drill")
    pytest.importorskip("cryptography")
    pytest.importorskip("psycopg")
    ops = Path(__file__).parents[1] / "ops" / "cloud"
    spec = importlib.util.spec_from_file_location("readiness_database_roles", ops / "database_roles.py")
    roles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roles)
    import recovery

    class OwnedDatabase:
        def __repr__(self):
            return "<owned disposable operations database; credentials redacted>"

        def url(self, kind, database=None):
            username = "review_admin" if kind == "admin" else self.plan["roles"][kind]
            password = self.admin_password if kind == "admin" else self.passwords[kind]
            return URL.create("postgresql+psycopg", username=username, password=password, host="127.0.0.1", port=self.port, database=database or self.database)

    owned = OwnedDatabase()
    owned.roles, owned.recovery, owned.directory = roles, recovery, tmp_path
    owned.owner = "review-" + uuid.uuid4().hex[:16]
    owned.database = "matters_review"
    owned.admin_password = secrets.token_urlsafe(32)
    owned.passwords = {kind: secrets.token_urlsafe(32) for kind in ("app", "migration", "backup")}
    password_file = tmp_path / "administrator-password"
    descriptor = os.open(password_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(owned.admin_password)
    docker = ["docker", "--host=unix:///var/run/docker.sock"]
    environment = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    created = subprocess.run([
        *docker, "run", "--rm", "-d", "--name", "matters-readiness-review-" + uuid.uuid4().hex[:10],
        "--label", recovery.OWNER_LABEL + "=" + owned.owner,
        "--publish", "127.0.0.1::5432", "--memory", "256m", "--pids-limit", "128",
        "--mount", f"type=bind,src={password_file},dst=/run/secrets/postgres-password,readonly",
        "--env", "POSTGRES_USER=review_admin", "--env", "POSTGRES_DB=postgres",
        "--env", "POSTGRES_PASSWORD_FILE=/run/secrets/postgres-password", "postgres:17-alpine",
    ], capture_output=True, text=True, timeout=30, env=environment, check=True)
    container = created.stdout.strip()
    try:
        ports = subprocess.run([*docker, "inspect", "--format", "{{json .NetworkSettings.Ports}}", container], capture_output=True, text=True, timeout=10, env=environment, check=True)
        owned.port = int(json.loads(ports.stdout)["5432/tcp"][0]["HostPort"])
        admin = create_engine(owned.url("admin", "postgres"), connect_args={"connect_timeout": 1})
        try:
            for _ in range(100):
                try:
                    with admin.connect() as connection:
                        connection.exec_driver_sql("SELECT 1")
                    break
                except SQLAlchemyError:
                    time.sleep(0.1)
            else:
                pytest.fail("Owned PostgreSQL server did not become ready")
        finally:
            admin.dispose()
        owned.admin = recovery.PostgreSQL(container, owned.owner, "review_admin", timeout=30)
        record = tmp_path / "source-roles.json"
        roles.provision(owned.admin, owned.database, "review_install", owned.passwords, record)
        owned.plan = roles.read_record(record)
        owned.backup = recovery.PostgreSQL(container, owned.owner, owned.plan["roles"]["backup"], timeout=30)
        yield owned
    finally:
        subprocess.run([*docker, "stop", container], capture_output=True, text=True, timeout=20, env=environment, check=True)
        password_file.unlink(missing_ok=True)


def test_real_runtime_roles_and_encrypted_schema3_restore_enforce_the_reviewed_boundaries(owned_operations_database):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery
    from matters.cloud.db import VERSION_THREE_TABLES, schema_versions

    owned = owned_operations_database
    for kind, wrong_password in (("app", owned.admin_password), ("migration", owned.passwords["app"]), ("backup", owned.passwords["migration"])):
        incorrect = create_engine(owned.url(kind).set(password=wrong_password), connect_args={"connect_timeout": 2})
        try:
            with pytest.raises(SQLAlchemyError):
                with incorrect.connect() as connection:
                    connection.exec_driver_sql("SELECT 1")
        finally:
            incorrect.dispose()

    # Exercise the real additive upgrade as the migration role, then reapply
    # the reviewed grants. The newly provisioned database is still empty.
    migration = Database(owned.url("migration"))
    try:
        with migration.transaction(write=True) as connection:
            for table in VERSION_THREE_TABLES:
                table.drop(connection)
            connection.execute(schema_versions.update().values(version=2))
        assert migration.migrate() == {"from_version": 2, "to_version": 3, "changed": True}
    finally:
        migration.close()
    owned.roles.apply_grants(owned.admin, owned.plan)
    runtime = Database(owned.url("app")).initialize()
    backup_engine = create_engine(owned.url("backup"))

    def denied(engine, statement):
        with pytest.raises(SQLAlchemyError):
            with engine.begin() as connection:
                connection.exec_driver_sql(statement)

    try:
        accounts, user, session, repository, graph, oauth, client, verifier, request, csrf = prepare_oauth(runtime)
        callback = parse_qs(urlsplit(oauth.consent(request, user["id"], session["token"], csrf, [graph["graph_id"]], repository=repository)).query)
        tokens = oauth.exchange_code(client_id=client["client_id"], code=callback["code"][0], redirect_uri="https://client.example/callback", code_verifier=verifier, resource=oauth.resource_url)
        delivery = CaptureMailDelivery()
        lifecycle = AccountLifecycle(runtime, accounts, public_origin=ORIGIN, delivery=delivery)
        try:
            lifecycle.request_verification(user["id"])
            assert lifecycle.drain_delivery()
            lifecycle.verify_email(delivered_token(delivery))
            lifecycle.request_password_reset(user["email"])
            assert lifecycle.drain_delivery()
            outstanding_reset = delivered_token(delivery)
            assert lifecycle.status(user["id"])["email_verified"] is True
        finally:
            lifecycle.close()

        for statement in (
            "CREATE TABLE public.unapproved_ddl(id integer)",
            "CREATE TEMP TABLE unapproved_temporary(id integer)",
            "UPDATE schema_versions SET version=99",
            "TRUNCATE graphs",
            "SET ROLE " + owned.roles.identifier(owned.plan["roles"]["migration"]),
        ):
            denied(runtime.engine, statement)
        for statement in (
            "DELETE FROM graphs", "CREATE TABLE public.unapproved_backup_ddl(id integer)",
            "CREATE TEMP TABLE backup_temporary(id integer)", "SELECT lo_create(0)",
            "SET ROLE " + owned.roles.identifier(owned.plan["roles"]["migration"]),
        ):
            denied(backup_engine, statement)

        # Reapplying reviewed grants must also remove direct privilege drift.
        app_role = owned.roles.identifier(owned.plan["roles"]["app"])
        database_name = owned.roles.identifier(owned.database)
        owned.admin.sql(owned.database, f"GRANT CREATE,TEMP ON DATABASE {database_name} TO {app_role}; GRANT CREATE ON SCHEMA public TO {app_role}")
        owned.roles.apply_grants(owned.admin, owned.plan)
        denied(runtime.engine, "CREATE TABLE public.drift_ddl(id integer)")
        denied(runtime.engine, "CREATE TEMP TABLE drift_temporary(id integer)")

        key = secrets.token_bytes(32)
        encrypted = owned.directory / "installation.aesgcm"
        owned.recovery.backup(owned.backup, owned.database, key, encrypted)
        destination = "restore_review_" + uuid.uuid4().hex[:12]
        restored = owned.recovery.restore(owned.admin, destination, destination, encrypted.read_bytes(), key)
        assert restored["account_actions_revoked"] is True
        owned.roles.grant_restored(owned.admin, owned.plan, destination, restored["restore_ownership_marker"], destination, owned.directory / "restored-roles.json")
        recovered = Database(owned.url("app", destination)).initialize()
        try:
            recovered_accounts = CloudAccounts(recovered)
            with pytest.raises(AccountError):
                recovered_accounts.authenticate(session["token"])
            assert recovered_accounts.login(user["email"], PASSWORD)["user"]["id"] == user["id"]
            assert GraphRepository(recovered).get_graph(user["id"], graph["graph_id"]) == graph
            recovered_lifecycle = AccountLifecycle(recovered, recovered_accounts, public_origin=ORIGIN)
            try:
                assert recovered_lifecycle.status(user["id"])["email_verified"] is True
                with pytest.raises(AccountError):
                    recovered_lifecycle.reset_password(outstanding_reset, "Must not revive a restored link")
            finally:
                recovered_lifecycle.close()
            with pytest.raises(OAuthError):
                CloudOAuth(recovered, issuer_url=ORIGIN).validate_access_token(tokens["access_token"])
            denied(recovered.engine, "CREATE TABLE public.restored_ddl(id integer)")
            denied(recovered.engine, "UPDATE schema_versions SET version=99")
            assert owned.recovery.PostgreSQL(owned.admin.container, owned.owner, owned.plan["roles"]["backup"], timeout=30).schema(destination) == 3
        finally:
            recovered.close()
    finally:
        runtime.close()
        backup_engine.dispose()


def test_private_delivery_observability_has_only_aggregate_fields_and_safe_sink_failures(installation, capsys):
    from matters.cloud.lifecycle import AccountLifecycle
    from matters.cloud.mail_delivery import CaptureMailDelivery

    accounts = CloudAccounts(installation)
    user = accounts.register("private-observer@example.com", PASSWORD)
    events = []

    class FailedDelivery(CaptureMailDelivery):
        def send(self, recipient, subject, body):
            super().send(recipient, subject, body)
            raise RuntimeError("Provider included message body: " + body)

    capture = FailedDelivery()
    lifecycle = AccountLifecycle(installation, accounts, public_origin=ORIGIN, delivery=capture, event_sink=events.append)
    try:
        response = lifecycle.request_password_reset(user["email"])
        assert lifecycle.request_password_reset("missing@example.com") == response
        assert lifecycle.request_password_reset("not an email") == response
        assert lifecycle.drain_delivery()
        snapshot = lifecycle.delivery_metrics()
        assert snapshot["requested"] == 3 and snapshot["queued"] == 3
        assert snapshot["failed"] == 1 and snapshot["skipped"] == 2
        assert snapshot["queue_depth"] == 0 and snapshot["queue_capacity"] == 128
        assert all(set(event) == {"event", "outcome", "count"} for event in events)
        token = delivered_token(capture)
        serialized = json.dumps({"events": events, "metrics": snapshot})
        assert user["email"] not in serialized and token not in serialized
        assert not set(snapshot) & set(lifecycle.capabilities())
        assert not set(snapshot) & set(lifecycle.status(user["id"]))
        output = capsys.readouterr()
        assert token not in output.out + output.err
    finally:
        lifecycle.close()

    def unavailable_sink(event):
        raise RuntimeError("Never publish sink exception details or an account address")

    failing_sink = AccountLifecycle(installation, accounts, public_origin=ORIGIN, event_sink=unavailable_sink)
    try:
        assert failing_sink.request_password_reset(user["email"]) == failing_sink.request_password_reset("missing@example.com")
        assert failing_sink.delivery_metrics()["event_sink_failed"] >= 2
    finally:
        failing_sink.close()


def test_operator_json_mail_sink_discards_provider_fields_and_rejects_unbounded_values():
    from matters.cloud.mail_delivery import JSONMailEventSink

    stream = io.StringIO()
    sink = JSONMailEventSink(stream)
    sink({"event": "forged-provider-name", "outcome": "failed", "count": 1, "recipient": "private@example.com", "token": "private-token", "provider_error": "private message body"})
    assert json.loads(stream.getvalue()) == {"event": "matters_mail_delivery", "outcome": "failed", "count": 1}
    for invalid in (
        {"outcome": "private provider exception", "count": 1},
        {"outcome": "sent", "count": True},
        {"outcome": "sent", "count": -1},
        {"outcome": "sent", "count": 2**63},
    ):
        with pytest.raises(ValueError):
            sink(invalid)
    assert "private" not in stream.getvalue()
