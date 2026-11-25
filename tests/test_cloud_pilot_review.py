"""Independent checks for an existing-edge pilot; no remote server access."""

import importlib.util
import base64
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("independent_cloud_pilot", ROOT / "ops/cloud/pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def prepare(path, **updates):
    values = {
        "directory": path, "project": "review-pilot", "origin": "https://pilot-review.example:8443",
        "image": "matters-cloud:reviewed", "ingress_network": "review-pilot-edge",
        "ingress_subnet": "10.244.60.0/28", "backend_subnet": "10.244.61.0/28", "proxy_ip": "10.244.60.3",
    }
    return pilot.prepare(**{**values, **updates})


def docker_environment():
    environment = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    return environment


def test_preparation_exposes_no_credential_and_never_changes_an_existing_directory(tmp_path):
    directory = tmp_path / "pilot"
    metadata = prepare(directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir())
    secret_values = [(directory / (kind + ".password")).read_text().strip() for kind in ("admin", "app", "migration", "backup")]
    assert len(set(secret_values)) == 4
    assert all(secret not in json.dumps(metadata) for secret in secret_values)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    symlink = tmp_path / "symlink"
    symlink.symlink_to(directory, target_is_directory=True)
    with pytest.raises(FileExistsError):
        prepare(symlink)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize("updates", [
    {"origin": "https://pilot-review.example/{import /private/*}"},
    {"origin": "https://pilot-review.example\nINJECT=yes"},
    {"image": "review:tag\nINJECT=yes"}, {"project": "review;recreate-edge"},
    {"proxy_ip": "10.244.60.4"}, {"proxy_ip": "10.244.60.1"}, {"proxy_ip": "10.244.60.15"},
    {"proxy_ip": "10.244.61.3"}, {"backend_subnet": "10.244.60.0/28"},
    {"ingress_subnet": "203.0.113.0/28"}, {"app_cpus": float("inf")}, {"database_memory_mib": -1},
])
def test_invalid_proxy_or_configuration_boundaries_create_nothing(tmp_path, updates):
    directory = tmp_path / "must-not-exist"
    with pytest.raises(pilot.OpsError):
        prepare(directory, **updates)
    assert not directory.exists()


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker Compose is required for config inspection")
def test_rendered_compose_keeps_edge_and_database_trust_separate(tmp_path):
    directory = tmp_path / "pilot"
    metadata = prepare(directory)
    result = subprocess.run([
        "docker", "--host=unix:///var/run/docker.sock", "compose", "--project-name", metadata["project"],
        "--env-file", str(directory / "pilot.env"), "-f", str(directory / "pilot.compose.yml"),
        "config", "--format", "json",
    ], capture_output=True, text=True, timeout=20, env=docker_environment())
    assert result.returncode == 0, "Generated Compose configuration must validate"
    config = json.loads(result.stdout)
    app, database = config["services"]["app"], config["services"]["database"]
    assert not app.get("ports") and not database.get("ports")
    assert set(database["networks"]) == {"backend"}
    assert set(app["networks"]) == {"backend", "ingress"}
    assert config["networks"]["backend"]["internal"] is True
    assert config["networks"]["ingress"]["external"] is True
    assert app["command"][app["command"].index("--trusted-proxy") + 1] == metadata["trusted_proxy"]
    assert "--disable-signup" in app["command"]
    assert not app.get("secrets")
    assert set(app["environment"]) <= {"MATTERS_CLOUD_DATABASE_URL", "MATTERS_CLOUD_PUBLIC_ORIGIN"}
    assert "POSTGRES_PASSWORD" not in database["environment"]
    assert database["environment"]["POSTGRES_PASSWORD_FILE"] == "/run/secrets/admin_password"
    assert len(database["secrets"]) == 1
    assert app["user"] == "10001:10001" and app["read_only"] is True
    assert app["cap_drop"] == ["ALL"]
    assert not any("docker.sock" in str(volume) for service in (app, database) for volume in service.get("volumes", []))


def test_caddy_artifacts_preserve_canonical_authority_and_scope_error_suppression(tmp_path):
    directory = tmp_path / "pilot"
    metadata = prepare(directory)
    site = (directory / "Caddyfile.site").read_text()
    logging = (directory / "Caddyfile.logging.merge").read_text()
    assert "header_up Host pilot-review.example:8443" in site
    assert "header_up X-Forwarded-For {http.request.remote.host}" in site
    assert "output discard" in site and "log " + metadata["caddy_log_name"] in site
    assert "exclude http.log.error." + metadata["caddy_log_name"] in logging
    assert "http.log.error\n" not in logging
    assert "REVIEW/MERGE ONLY" in logging
    assert "reload" not in site and "network connect" not in site


@pytest.mark.skipif(os.environ.get("MATTERS_TEST_CADDY") != "1", reason="Set MATTERS_TEST_CADDY=1 for an owned local Caddy2.11.9 probe")
def test_actual_caddy_suppresses_matters_oauth_errors_without_changing_cotenant_routes_or_diagnostics(tmp_path):
    directory = tmp_path / "pilot"
    metadata = prepare(directory)
    site = (directory / "Caddyfile.site").read_text().replace(metadata["origin"], "http://pilot-review.example:8080", 1)
    site = site.replace(metadata["upstream"], "127.0.0.1:9")
    logging = (directory / "Caddyfile.logging.merge").read_text()
    cotenant = """
http://co-tenant.example:8080 {
    handle /ok {
        respond "existing route retained" 200
    }
    handle {
        reverse_proxy 127.0.0.1:9
    }
}
"""
    config = tmp_path / "Caddyfile"
    config.write_text(logging + "\n" + site + "\n" + cotenant)
    image = "caddy:2.11.9-alpine"
    docker = ["docker", "--host=unix:///var/run/docker.sock"]
    environment = docker_environment()
    adapted = subprocess.run([*docker, "run", "--rm", "--mount", f"type=bind,src={config},dst=/etc/caddy/Caddyfile,readonly", image, "caddy", "adapt", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"], capture_output=True, text=True, env=environment, timeout=30)
    assert adapted.returncode == 0, adapted.stderr
    data = json.loads(adapted.stdout)
    namespace = "http.log.error." + metadata["caddy_log_name"]
    assert namespace in data["logging"]["logs"]["default"]["exclude"]
    name = "matters-pilot-review-" + secrets.token_hex(6)
    started = subprocess.run([
        *docker, "run", "--rm", "-d", "--name", name, "--label", "global.matters.review-owner=" + name,
        "--publish", "127.0.0.1::8080", "--read-only", "--tmpfs", "/data", "--tmpfs", "/config", "--tmpfs", "/tmp",
        "--mount", f"type=bind,src={config},dst=/etc/caddy/Caddyfile,readonly", image,
        "caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile",
    ], capture_output=True, text=True, env=environment, timeout=30, check=True)
    container = started.stdout.strip()
    try:
        inspected = subprocess.run([*docker, "inspect", "--format", "{{json .NetworkSettings.Ports}}", container], capture_output=True, text=True, env=environment, timeout=10, check=True)
        port = int(json.loads(inspected.stdout)["8080/tcp"][0]["HostPort"])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(host, path):
            request = urllib.request.Request(f"http://127.0.0.1:{port}" + path, headers={"Host": host})
            try:
                with opener.open(request, timeout=5) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as error:
                return error.code, error.read().decode()

        for _ in range(80):
            try:
                status, body = request("co-tenant.example", "/ok")
                if status == 200:
                    break
            except urllib.error.URLError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("Owned Caddy did not become ready")
        assert body == "existing route retained"
        marker = "synthetic-matters-oauth-secret"
        assert request("pilot-review.example", "/oauth/consent?request=" + marker)[0] == 502
        assert request("co-tenant.example", "/failure?probe=cotenant-diagnostic")[0] == 502
        logs = subprocess.run([*docker, "logs", container], capture_output=True, text=True, env=environment, timeout=10, check=True)
        text = logs.stdout + logs.stderr
        assert marker not in text and namespace not in text
        assert "cotenant-diagnostic" in text and "http.log.error" in text
        assert request("co-tenant.example", "/ok") == (200, "existing route retained")
    finally:
        subprocess.run([*docker, "stop", container], capture_output=True, text=True, env=environment, timeout=15, check=True)


def test_closed_signup_is_enforced_inside_application_for_all_registration_spellings(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("argon2")
    pytest.importorskip("email_validator")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select
    from matters.cloud.app import create_app
    from matters.cloud.db import users
    from matters.cloud.mail_delivery import CaptureMailDelivery

    app = create_app(f"sqlite:///{tmp_path / 'closed.db'}", public_origin="https://pilot-review.example", disable_signup=True, mail_delivery=CaptureMailDelivery())
    password = "Existing operator provisioned password"
    app.state.accounts.register("operator@example.com", password)
    with TestClient(app, base_url="https://pilot-review.example") as client:
        capabilities = client.get("/api/auth/capabilities").json()
        assert capabilities["signup_enabled"] is False and capabilities["email_delivery"] is True
        for route in ("/api/auth/register", "/api/auth/register/", "/api/auth/%72egister"):
            response = client.post(route, json={"email": "new@example.com", "password": password})
            assert response.status_code == 403
        with app.state.db.transaction() as connection:
            assert connection.execute(select(func.count()).select_from(users)).scalar_one() == 1
        assert client.post("/api/auth/login", json={"email": "operator@example.com", "password": password}).status_code == 200
        assert client.post("/api/graphs", json={"title": "Existing account remains useful"}).status_code == 201


def test_oauth_login_reports_password_capacity_as_temporary_unavailability(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("argon2")
    pytest.importorskip("email_validator")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from matters.cloud.accounts import AccountError
    from matters.cloud.app import create_app

    origin = "https://pilot-review.example"
    app = create_app(f"sqlite:///{tmp_path / 'oauth-capacity.db'}", public_origin=origin, enable_mcp=True, disable_signup=True)
    client_metadata = app.state.oauth.register_client({"redirect_uris": ["https://client.example/callback"]})
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    pending = app.state.oauth.create_authorization_request({"client_id": client_metadata["client_id"], "redirect_uri": "https://client.example/callback", "response_type": "code", "resource": origin + "/mcp", "code_challenge": challenge, "code_challenge_method": "S256"})

    def busy(*args, **kwargs):
        raise AccountError("The password service is busy. Try again shortly.", "unavailable")

    monkeypatch.setattr(app.state.accounts, "login", busy)
    with TestClient(app, base_url=origin, follow_redirects=False) as client:
        assert client.get("/oauth/consent", params={"request": pending}).status_code == 200
        nonce = client.cookies.get("matters_oauth_csrf")
        response = client.post("/oauth/consent/login", headers={"Origin": origin}, data={"request": pending, "csrf": nonce, "email": "operator@example.com", "password": "An existing operator password"})
        assert response.status_code == 503
        assert response.json()["error"] == "temporarily_unavailable"
        assert "incorrect" not in response.text and "operator@example.com" not in response.text
