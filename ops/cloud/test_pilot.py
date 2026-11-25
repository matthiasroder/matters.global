"""Pilot preparation safety: no proxy/server changes or external credentials."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("cloud_pilot", Path(__file__).with_name("pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def prepare(directory, **changes):
    settings = dict(directory=directory, project="matters-fixture", origin="https://pilot.example.org", image="matters-cloud:reviewed", ingress_network="matters-fixture-edge", ingress_subnet="10.244.30.0/28", backend_subnet="10.244.31.0/28", proxy_ip="10.244.30.3")
    settings.update(changes)
    return pilot.prepare(**settings)


def test_fresh_preparation_keeps_secrets_private_and_out_of_metadata(tmp_path, capsys):
    directory = tmp_path / "new-pilot"
    result = prepare(directory)
    assert directory.stat().st_mode & 0o777 == 0o700
    passwords = [(directory / (kind + ".password")).read_text().strip() for kind in ("admin", "app", "migration", "backup")]
    assert len(set(passwords)) == 4 and all(len(value) >= 20 for value in passwords)
    for path in directory.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    assert len((directory / "backup.key").read_bytes()) == 32
    for secret in passwords:
        assert secret not in json.dumps(result)
    assert capsys.readouterr() == ("", "")
    environment = dict(line.split("=", 1) for line in (directory / "pilot.env").read_text().splitlines())
    assert passwords[0] not in environment.values()
    assert environment["MATTERS_CLOUD_DATABASE_URL"] == "postgresql+psycopg://matters_app:" + passwords[1] + "@database:5432/matters"
    assert result["trusted_proxy"] == "10.244.30.3/32"
    site = (directory / "Caddyfile.site").read_text()
    assert "https://pilot.example.org {" in site
    assert "reverse_proxy matters-fixture-app:3000" in site
    assert "header_up X-Forwarded-For {http.request.remote.host}" in site
    assert "header_up Authorization {http.request.header.Authorization}" in site
    assert "header_up Host pilot.example.org" in site
    compose = (directory / "pilot.compose.yml").read_text()
    assert '"--disable-signup"' in compose
    assert "output discard" in site
    logging = (directory / "Caddyfile.logging.merge").read_text()
    namespace = "http.log.error." + result["caddy_log_name"]
    assert "exclude " + namespace in logging
    assert "output stderr" not in logging and "include " + namespace not in logging
    assert "ports:" not in compose and "internal: true" in compose and "external: true" in compose
    assert "POSTGRES_PASSWORD_FILE:" in compose and "POSTGRES_PASSWORD:" not in compose
    assert "urlsplit(os.environ['MATTERS_CLOUD_PUBLIC_ORIGIN']).netloc" in compose


def test_prepare_never_adopts_existing_paths_or_checkout(tmp_path):
    marker = tmp_path / "retained"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare(tmp_path)
    assert marker.read_text() == "keep"
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        prepare(linked)
    with pytest.raises(pilot.OpsError, match="outside"):
        prepare(pilot.outside_repository.__globals__["REPOSITORY"] / "not-allowed")


@pytest.mark.parametrize("changes", [
    {"origin": "http://pilot.example.org"}, {"origin": "https://user:secret@pilot.example.org"},
    {"origin": "https://pilot.example.org/path"}, {"origin": "https://pilot.example.org?token=x"},
    {"origin": "https://pilot.example.org\nINJECT=yes"}, {"origin": "https://bad_host.example"},
    {"image": "example:tag\nINJECT=yes"}, {"image": "user:secret@registry.example/repository"},
    {"image": "https://registry.example/repository"}, {"project": "bad ${SECRET}"},
    {"ingress_network": "bad\nnetwork"}, {"ingress_network": "matters-fixture-backend"},
    {"proxy_ip": "10.244.30.4"},
    {"proxy_ip": "10.244.30.1"}, {"proxy_ip": "10.244.31.3"},
    {"backend_subnet": "10.244.30.0/28"}, {"ingress_subnet": "0.0.0.0/0"},
    {"ingress_subnet": "203.0.113.0/28"}, {"backend_subnet": "::/0"},
    {"app_memory_mib": 0}, {"app_cpus": float("nan")}, {"database_cpus": -1},
])
def test_invalid_boundaries_fail_before_any_file_is_created(tmp_path, changes):
    directory = tmp_path / "not-created"
    with pytest.raises(pilot.OpsError):
        prepare(directory, **changes)
    assert not directory.exists()


def test_provision_refuses_rerun_before_docker(tmp_path, monkeypatch):
    directory = tmp_path / "pilot"
    prepare(directory)
    (directory / "roles.json").write_text("existing inventory")
    monkeypatch.setattr(pilot.subprocess, "run", lambda *_a, **_k: pytest.fail("Existing inventory must not contact Docker"))
    with pytest.raises(pilot.OpsError, match="never reruns"):
        pilot.provision_database(directory)


@pytest.mark.parametrize("metadata", [[], None, {"format": "matters-pilot-preparation", "schema_version": 1, "project": 3}])
def test_corrupt_metadata_fails_before_docker(tmp_path, monkeypatch, metadata):
    directory = tmp_path / "pilot"
    prepare(directory)
    (directory / "operator.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(pilot.subprocess, "run", lambda *_a, **_k: pytest.fail("Corrupt inventory must not contact Docker"))
    with pytest.raises(pilot.OpsError):
        pilot.provision_database(directory)


def test_provision_uses_saved_project_exact_owner_and_only_runtime_role_secrets(tmp_path, monkeypatch):
    directory = tmp_path / "pilot"
    metadata = prepare(directory)
    calls = {}
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("MATTERS_CLOUD_DATABASE_URL", "postgresql://must-not-override-private-file")
    monkeypatch.setenv("DOCKER_HOST", "tcp://not-the-operator-daemon:2376")
    def execute(command, **kwargs):
        calls["command"] = command
        assert "DOCKER_TLS_VERIFY" not in kwargs["env"] and "DOCKER_HOST" not in kwargs["env"]
        assert "MATTERS_CLOUD_DATABASE_URL" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, b"owned-database\n")
    class PG:
        def __init__(self, container, owner, username):
            calls["pg"] = (container, owner, username)
    def provision(pg, database, prefix, passwords, record):
        calls["provision"] = (database, prefix, passwords, record)
        return {"schema_version": 1, "grants_applied": True}
    monkeypatch.setattr(pilot.subprocess, "run", execute)
    monkeypatch.setattr(pilot, "PostgreSQL", PG)
    monkeypatch.setattr(pilot, "provision", provision)
    assert pilot.provision_database(directory)["grants_applied"]
    assert calls["pg"] == ("owned-database", metadata["operations_owner"], "matters_ops")
    assert calls["command"][calls["command"].index("--project-name") + 1] == "matters-fixture"
    assert set(calls["provision"][2]) == {"app", "migration", "backup"}
    for secret in calls["provision"][2].values():
        assert secret not in repr(calls["command"])


def test_generated_compose_validates_quietly_without_printing_secrets(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose is required for this read-only configuration check")
    directory = tmp_path / "pilot"
    prepare(directory)
    environment = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        environment.pop(key, None)
    result = subprocess.run(["docker", "--host=unix:///var/run/docker.sock", "compose", "--project-name", "matters-fixture", "--env-file", str(directory / "pilot.env"), "-f", str(directory / "pilot.compose.yml"), "config", "--quiet"], env=environment, capture_output=True, timeout=20)
    assert result.returncode == 0 and result.stdout == b"" and result.stderr == b""


def test_nondefault_https_port_is_preserved_for_upstream_host_validation(tmp_path):
    directory = tmp_path / "pilot"
    prepare(directory, origin="https://pilot.example.org:8443")
    assert "header_up Host pilot.example.org:8443" in (directory / "Caddyfile.site").read_text()


def test_immutable_image_digest_is_kept_as_explicit_release_inventory(tmp_path):
    image = "registry.example/matters@sha256:" + "a" * 64
    assert prepare(tmp_path / "pilot", image=image)["image"] == image
