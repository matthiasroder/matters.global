import importlib.util
import os
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("cloud_staging", Path(__file__).with_name("staging.py"))
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


def test_prepare_refuses_repo_secrets_and_existing_paths(tmp_path):
    with pytest.raises(ValueError, match="outside the repository"):
        staging.prepare(staging.REPOSITORY / "unsafe-staging", "test:image")
    marker = tmp_path / "retained"
    marker.write_text("retain me")
    with pytest.raises(FileExistsError):
        staging.prepare(tmp_path, "test:image")
    assert marker.read_text() == "retain me"


def test_prepare_creates_private_disposable_configuration_without_printing_secrets(tmp_path, monkeypatch):
    def certificate(command, **_kwargs):
        Path(command[command.index("-keyout") + 1]).write_text("test-key")
        Path(command[command.index("-out") + 1]).write_text("test-cert")
    monkeypatch.setattr(staging.subprocess, "run", certificate)
    monkeypatch.setattr(staging.shutil, "which", lambda _name: "/usr/bin/openssl")
    target = tmp_path / "staging"
    result = staging.prepare(target, "test:image", 19444)
    assert target.stat().st_mode & 0o777 == 0o700
    assert (target / "backup.key").stat().st_mode & 0o777 == 0o600
    assert (target / "staging.env").stat().st_mode & 0o777 == 0o600
    assert len((target / "backup.key").read_bytes()) == 32
    assert result["origin"] == "https://localhost:19444"
    assert "PASSWORD" not in str(result) and "postgresql" not in str(result)
    values = dict(line.split("=", 1) for line in (target / "staging.env").read_text().splitlines())
    assert values["MATTERS_CLOUD_DATABASE_URL"].startswith("postgresql+psycopg://matters_app:")
    assert result["trusted_proxy"] == values["MATTERS_STAGING_PROXY_IP"] + "/32"
    assert result["subnet"] == "10.243.17.0/28"
    passwords = []
    for role in ("app", "migration", "backup"):
        secret = target / (role + ".password")
        assert secret.stat().st_mode & 0o777 == 0o600
        passwords.append(secret.read_text().strip())
    assert len(set(passwords)) == 3
    assert values["MATTERS_PG_PASSWORD"] not in passwords


@pytest.mark.parametrize("origin", ["http://localhost:19443", "https://example.com", "https://user:secret@localhost", "https://localhost/path", "https://localhost?key=secret"])
def test_probe_rejects_nonlocal_or_credential_bearing_origins(origin):
    with pytest.raises(ValueError):
        staging.probe(origin, "/does/not/exist")


@pytest.mark.parametrize("image", ["test:image\nINJECTED=yes", "test:${SECRET}", "test:image with-space"])
def test_prepare_refuses_environment_file_injection(tmp_path, image):
    target = tmp_path / "not-created"
    with pytest.raises(ValueError):
        staging.prepare(target, image)
    assert not target.exists()


@pytest.mark.parametrize("subnet", ["0.0.0.0/0", "10.0.0.0/8", "203.0.113.0/28", "::1/128", "10.0.0.0/28\nINJECTED=yes"])
def test_prepare_requires_narrow_private_network_before_creating_files(tmp_path, subnet):
    target = tmp_path / "not-created"
    with pytest.raises(ValueError):
        staging.prepare(target, "test:image", subnet=subnet)
    assert not target.exists()
