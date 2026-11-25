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
    assert "Make room for" in page.text
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
    assert len(limiter._buckets) <= 3


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
        (sentinel, {"host": "127.0.0.1", "port": 8767, "access_log": False}),
    ]
    assert capsys.readouterr().out == ""
    for arguments in (
        ["cloud", "--host", "0.0.0.0", "--development"],
        ["cloud", "--host", "0.0.0.0"],
        ["cloud", "--port", "0"],
        ["--state", str(tmp_path / "local.json"), "cloud"],
    ):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2


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
