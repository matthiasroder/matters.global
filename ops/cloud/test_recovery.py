"""Safety checks for operator tools, with no Docker daemon required."""

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("cloud_recovery", Path(__file__).with_name("recovery.py"))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def package(version=2):
    dump = b"PGDMPexample-archive"
    key = os.urandom(32)
    metadata = {"format": "matters-postgresql-backup", "schema_version": 1, "database_schema": version, "source_database": "matters", "source_operations_owner": "our-owner", "dump_bytes": len(dump), "dump_sha256": hashlib.sha256(dump).hexdigest()}
    return recovery.seal(dump, key, metadata), key


def test_authenticated_encryption_rejects_wrong_key_and_modified_bytes():
    encrypted, key = package()
    assert b"PGDMP" not in encrypted and b"source_database" not in encrypted
    assert recovery.unseal(encrypted, key)[1] == b"PGDMPexample-archive"
    for candidate, candidate_key in ((encrypted, os.urandom(32)), (encrypted[:-1] + bytes([encrypted[-1] ^ 1]), key)):
        with pytest.raises(recovery.OpsError, match="authentication failed"):
            recovery.unseal(candidate, candidate_key)


def test_output_is_private_exclusive_and_never_follows_a_symlink(tmp_path):
    target = tmp_path / "backup.enc"
    recovery.write_exclusive(target, b"encrypted")
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(recovery.OpsError, match="already exists"):
        recovery.write_exclusive(target, b"replacement")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(recovery.OpsError):
        recovery.write_exclusive(link, b"replacement")
    assert target.read_bytes() == b"encrypted"
    assert not list(tmp_path.glob(".matters-backup-*"))


def test_key_permissions_and_repository_boundaries(tmp_path):
    key = tmp_path / "key"
    key.write_bytes(os.urandom(32))
    key.chmod(0o600)
    assert len(recovery.load_key(key)) == 32
    key.chmod(0o644)
    with pytest.raises(recovery.OpsError, match="private regular"):
        recovery.load_key(key)
    link = tmp_path / "linked-key"
    link.symlink_to(key)
    with pytest.raises(recovery.OpsError):
        recovery.load_key(link)
    with pytest.raises(recovery.OpsError, match="outside the repository"):
        recovery.outside_repository(recovery.REPOSITORY / "unsafe.key")


class FakePG:
    database_name = staticmethod(recovery.PostgreSQL.database_name)
    timeout = 1
    environment = {}
    owner = "our-owner"

    def __init__(self, existing=False, failed=False, changed_marker=False):
        self.statements = []
        self.existing, self.failed, self.changed_marker = existing, failed, changed_marker
        self.marker = None

    def sql(self, database, statement):
        self.statements.append((database, statement))
        if "SELECT count(*) FROM pg_database" in statement:
            return "1" if self.existing else "0"
        if statement.startswith("COMMENT ON DATABASE"):
            self.marker = statement.split("'")[1]
        if "obj_description" in statement:
            return "another-owner" if self.changed_marker else self.marker
        return ""

    def command(self, *_args):
        return ["fake-pg-restore"]

    def schema(self, _database):
        if self.failed:
            raise recovery.OpsError("schema verification failed")
        return 2

    def sample(self, _database):
        return {"accounts": 2, "graphs": 3}


@pytest.mark.parametrize("target,confirmation", [("matters", "matters"), ("postgres", "postgres"), ("restore_new", "different"), ("restore_x;DROP", "restore_x;DROP")])
def test_restore_rejects_wrong_or_source_destination_before_mutation(target, confirmation):
    encrypted, key = package()
    pg = FakePG()
    with pytest.raises(recovery.OpsError):
        recovery.restore(pg, target, confirmation, encrypted, key)
    assert pg.statements == []


def test_restore_authenticates_before_sql_and_refuses_existing_destination():
    encrypted, key = package()
    pg = FakePG()
    with pytest.raises(recovery.OpsError):
        recovery.restore(pg, "restore_new", "restore_new", encrypted, os.urandom(32))
    assert pg.statements == []
    pg = FakePG(existing=True)
    with pytest.raises(recovery.OpsError, match="already exists"):
        recovery.restore(pg, "restore_new", "restore_new", encrypted, key)
    assert not any("CREATE" in statement for _, statement in pg.statements)


def test_restore_refuses_cross_installation_owner_before_sql():
    encrypted, key = package()
    pg = FakePG()
    pg.owner = "another-company"
    with pytest.raises(recovery.OpsError, match="different installation owner"):
        recovery.restore(pg, "restore_new", "restore_new", encrypted, key)
    assert not pg.statements


def test_restore_revokes_sessions_and_issued_oauth_but_preserves_clients(monkeypatch):
    encrypted, key = package()
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0))
    pg = FakePG()
    result = recovery.restore(pg, "restore_new", "restore_new", encrypted, key)
    assert result["sessions_and_issued_grants_revoked"]
    assert ("restore_new", "DELETE FROM sessions") in pg.statements
    assert ("restore_new", "DELETE FROM oauth_records WHERE kind <> 'client'") in pg.statements
    assert not any("DROP DATABASE" in statement for _, statement in pg.statements)


@pytest.mark.parametrize("changed", [False, True])
def test_failure_cleanup_drops_only_the_database_with_our_unchanged_marker(monkeypatch, changed):
    encrypted, key = package()
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0))
    pg = FakePG(failed=True, changed_marker=changed)
    with pytest.raises(recovery.OpsError):
        recovery.restore(pg, "restore_new", "restore_new", encrypted, key)
    assert any("DROP DATABASE" in statement for _, statement in pg.statements) is not changed
    assert not any("FORCE" in statement for _, statement in pg.statements)


def test_capture_bounds_and_safe_errors():
    pg = object.__new__(recovery.PostgreSQL)
    pg.max_bytes, pg.timeout, pg.environment = 10, 1, dict(os.environ)
    with pytest.raises(recovery.OpsError, match="byte bound"):
        pg.capture([sys.executable, "-c", "print('x'*1000)"])
    with pytest.raises(recovery.OpsError) as error:
        pg.capture([sys.executable, "-c", "import sys; print('do-not-log-this-secret',file=sys.stderr); sys.exit(1)"])
    assert "do-not-log-this-secret" not in str(error.value)


def test_container_owner_must_match_before_database_command(monkeypatch):
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b'{"global.matters.operations-owner":"another-owner"}'))
    with pytest.raises(recovery.OpsError, match="operations-owner label"):
        recovery.PostgreSQL("staging-db", "our-owner", "matters_ops")
