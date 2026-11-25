"""HTTP boundary and installation checks for the optional Cloud pilot."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from matters.cli import main
from matters.cloud.app import MAX_REQUEST_BYTES, create_app


PASSWORD = "A sufficiently long test password"


def test_oauth_navigation_forms_keep_one_origin_preserving_referrer_policy():
    """A duplicate no-referrer policy makes Chromium POST with Origin:null."""
    pytest.importorskip("mcp")
    app = create_app("sqlite://", secure_cookies=False, public_origin="http://localhost", enable_mcp=True)
    with TestClient(app, base_url="http://localhost") as client:
        registered = app.state.oauth.register_client({"redirect_uris": ["https://client.example/callback"]})
        pending = app.state.oauth.create_authorization_request({
            "client_id": registered["client_id"], "redirect_uri": registered["redirect_uris"][0],
            "response_type": "code", "resource": "http://localhost/mcp",
            "code_challenge": "a" * 43, "code_challenge_method": "S256",
        })
        response = client.get("/oauth/consent", params={"request": pending})
        assert response.status_code == 200
        assert response.headers.get_list("referrer-policy") == ["strict-origin"]
        assert 'action="/oauth/consent/login"' in response.text
        assert client.get("/").headers["referrer-policy"] == "no-referrer"


@pytest.fixture
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'cloud.db'}", secure_cookies=False)
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def sign_in(client, email="alice@example.com"):
    credentials = {"email": email, "password": PASSWORD}
    assert client.post("/api/auth/register", json=credentials).status_code == 201
    response = client.post("/api/auth/login", json=credentials)
    assert response.status_code == 200, response.text
    return response


def test_assets_health_and_authenticated_graph_boundary(client):
    assert client.get("/healthz").json() == {"schema_version": 1, "status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="auth-form"' in page.text
    assert 'autocomplete="username"' in page.text
    assert 'autocomplete="current-password"' in page.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/api/graphs").status_code == 401
    assert client.post("/api/terminal/sessions", json={}).status_code == 404
    assert client.post("/api/state", json={"path": "/tmp/private.json"}).status_code == 404
    assert client.get("/docs").status_code == 404
    assert page.headers["cache-control"] == "no-store"
    assert "script-src 'self'" in page.headers["content-security-policy"]


def test_login_cookie_is_http_only_and_never_returned_in_json(client):
    response = sign_in(client)
    cookie = response.headers["set-cookie"]
    token = response.cookies.get("matters_session")

    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/" in cookie
    assert token not in json.dumps(response.json())
    assert PASSWORD not in response.text
    assert client.get("/api/auth/me").json()["user"]["email"] == "alice@example.com"
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_secure_cookies_are_default_for_https(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'secure.db'}")
    with TestClient(app, base_url="https://localhost") as client:
        response = sign_in(client)
        assert "Secure" in response.headers["set-cookie"]
        assert client.get("/api/auth/me").status_code == 200


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"},
    {"Origin": "null"},
    {"Origin": "http://localhost.evil.example"},
    {"Origin": "http://user@localhost"},
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
])
def test_cross_origin_authentication_is_rejected_before_account_changes(client, headers):
    response = client.post("/api/auth/register", headers=headers, json={"email": "alice@example.com", "password": PASSWORD})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert PASSWORD not in response.text
    assert client.post("/api/auth/register", headers={"Origin": "http://localhost:80"}, json={"email": "alice@example.com", "password": PASSWORD}).status_code == 201


@pytest.mark.parametrize("host", ["attacker.example", "localhost@attacker.example", "localhost:80@attacker.example"])
def test_untrusted_or_malformed_host_headers_are_rejected(client, host):
    response = client.get("/healthz", headers={"Host": host})

    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"


def test_public_origin_proxy_configuration_requires_exact_https_origin(tmp_path):
    for origin in ("http://matters.example", "https://user:secret@matters.example", "https://matters.example/path", "https://matters.example?secret=x"):
        with pytest.raises(ValueError):
            create_app(f"sqlite:///{tmp_path / 'unused.db'}", public_origin=origin)
    with pytest.raises(ValueError):
        create_app(f"sqlite:///{tmp_path / 'unused.db'}", allowed_hosts=["*"])
    with pytest.raises(ValueError):
        create_app(f"sqlite:///{tmp_path / 'unused.db'}", allowed_hosts="localhost")
    with pytest.raises(ValueError):
        create_app(f"sqlite:///{tmp_path / 'unused.db'}", secure_cookies=False, allowed_hosts=["matters.example"])

    app = create_app(f"sqlite:///{tmp_path / 'public.db'}", public_origin="https://matters.example")
    with TestClient(app, base_url="http://matters.example") as client:
        response = client.post("/api/auth/register", headers={"Origin": "https://matters.example"}, json={"email": "alice@example.com", "password": PASSWORD})
        assert response.status_code == 201
        assert client.get("/healthz", headers={"Host": "localhost"}).status_code == 400


def test_request_limits_and_validation_errors_do_not_echo_credentials(client):
    oversized = json.dumps({"password": PASSWORD + "x" * MAX_REQUEST_BYTES})
    response = client.post("/api/auth/register", content=oversized, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert PASSWORD not in response.text

    response = client.post("/api/auth/register", content=PASSWORD, headers={"Content-Type": "text/plain"})
    assert response.status_code == 415
    assert PASSWORD not in response.text

    response = client.post("/api/auth/register", content='{"password":"' + PASSWORD, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert PASSWORD not in response.text

    response = client.post("/api/auth/register", json={"email": "alice@example.com", "password": PASSWORD, "actor_id": "forged-user"})
    assert response.status_code == 422
    assert PASSWORD not in response.text
    assert "forged-user" not in response.text


def test_auth_attempt_limiter_is_bounded_and_returns_retry_hint(client):
    client.app.state.auth_limiter.attempts = 2
    for _ in range(2):
        assert client.post("/api/auth/login", json={}).status_code == 422

    response = client.post("/api/auth/login", json={})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json()["error"]["code"] == "rate_limited"
    limiter = client.app.state.auth_limiter
    limiter.capacity = 3
    for peer in range(10):
        limiter.allow(f"peer-{peer}")
    from sqlalchemy import func, select
    from matters.cloud.db import rate_limit_buckets
    with client.app.state.db.transaction() as connection:
        # One durable capacity lock plus at most three subject buckets.
        assert connection.execute(select(func.count()).select_from(rate_limit_buckets)).scalar_one() <= 4


def test_unknown_identity_query_fields_are_rejected(client):
    sign_in(client)
    response = client.get("/api/graphs?actor_id=somebody-else")
    assert response.status_code == 400
    assert "somebody-else" not in response.text


def test_unexpected_backend_error_is_sanitized_without_log_leaks(client, monkeypatch, caplog):
    sign_in(client)
    secret = "postgresql://alice:SECRET-DB-PASSWORD@host/database"

    def fail(_actor):
        raise RuntimeError(secret)

    monkeypatch.setattr(client.app.state.graphs, "list_graphs", fail)
    response = client.get("/api/graphs")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal"
    assert secret not in response.text
    assert secret not in caplog.text


def test_optional_cloud_cli_configuration_and_loopback_development(tmp_path, monkeypatch, capsys):
    import uvicorn
    from matters.cloud import app as cloud_app

    invocations = []
    sentinel = object()
    monkeypatch.setattr(cloud_app, "create_app", lambda url, **options: invocations.append((url, options)) or sentinel)
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: invocations.append((app, options)))
    monkeypatch.setenv("MATTERS_CLOUD_DATABASE_URL", "sqlite:///pilot.db")

    assert main(["cloud", "--development", "--port", "8767"]) == 0
    assert invocations == [
        ("sqlite:///pilot.db", {"secure_cookies": False, "public_origin": None}),
        (sentinel, {"host": "127.0.0.1", "port": 8767, "access_log": False, "proxy_headers": False, "forwarded_allow_ips": ""}),
    ]
    assert capsys.readouterr().out == ""
    for arguments in (
        ["cloud", "--host", "0.0.0.0", "--development"],
        ["cloud", "--host", "0.0.0.0"],
        ["cloud", "--port", "0"],
        ["cloud", "--trusted-proxy", "*"],
        ["cloud", "--trusted-proxy", "0.0.0.0/0"],
        ["cloud", "--trusted-proxy", "not-a-proxy"],
        ["--state", str(tmp_path / "local.json"), "cloud"],
    ):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2

    invocations.clear()
    assert main(["cloud", "--development", "--trusted-proxy", "127.0.0.1", "--trusted-proxy", "172.30.1.0/24"]) == 0
    assert invocations[-1][1]["proxy_headers"] is True
    assert invocations[-1][1]["forwarded_allow_ips"] == "127.0.0.1,172.30.1.0/24"


def test_core_cli_remains_usable_without_optional_cloud_dependencies():
    example = Path(__file__).parents[1] / "examples" / "matters.example.json"
    script = """
import builtins,sys
original=builtins.__import__
def block_cloud(name,*args,**kwargs):
    if name in {'fastapi','sqlalchemy','argon2','uvicorn'} or name.startswith(('fastapi.','sqlalchemy.','argon2.','uvicorn.')):
        raise ModuleNotFoundError('Optional Cloud dependency is absent',name=name)
    return original(name,*args,**kwargs)
builtins.__import__=block_cloud
from matters.cli import main
assert main(['universe','--state',sys.argv[1]])==0
try:
    main(['cloud','--database-url','sqlite:///:memory:','--development'])
except SystemExit as error:
    assert error.code==2
else:
    raise AssertionError('Cloud SDK absence was not reported')
"""
    result = subprocess.run([sys.executable, "-c", script, str(example)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ship_first_version\n"
    assert "install matters-global[cloud]" in result.stderr
    assert "Traceback" not in result.stderr
