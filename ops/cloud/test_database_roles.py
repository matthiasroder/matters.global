import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("database_roles", Path(__file__).with_name("database_roles.py"))
roles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roles)


@pytest.mark.parametrize("database,prefix", [("db;DROP ROLE x", "safe"), ("safe", "bad'role"), ("safe", "pg_reserved"), ("safe", "x" * 49)])
def test_names_cannot_inject_or_adopt_reserved_roles(database, prefix):
    with pytest.raises(roles.OpsError):
        roles.role_plan(database, prefix, "owner")


@pytest.mark.parametrize("value", ["x" * 20 + "\n\\connect postgres", "x" * 20 + "\x00", None])
def test_direct_password_values_cannot_escape_the_psql_prompt(value):
    with pytest.raises(roles.OpsError):
        roles.checked_password(value)


def test_password_files_private_and_never_in_repository(tmp_path):
    password = tmp_path / "password"
    password.write_text("private-password-value-with-'quotes\n")
    password.chmod(0o600)
    assert roles.private_password(password).endswith("'quotes")
    password.chmod(0o644)
    with pytest.raises(roles.OpsError, match="private regular"):
        roles.private_password(password)
    with pytest.raises(roles.OpsError, match="outside the repository"):
        roles.private_password(Path(__file__).resolve().parents[2] / "password")


def test_password_prompt_stays_on_stdin_not_argv_or_output(monkeypatch):
    class PG:
        docker, passfile, container, username, timeout, environment = ["docker"], None, "owned-db", "admin", 5, {}
    secret = "not-on-command-line-'$-secret"
    seen = {}
    def execute(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(roles.subprocess, "run", execute)
    roles.script(PG(), "postgres", "\\password test_app\n" + secret + "\n" + secret + "\n")
    assert secret not in repr(seen["command"])
    assert secret.encode() in seen["input"]
    assert seen["stdout"] == subprocess.DEVNULL and seen["stderr"] == subprocess.DEVNULL
    assert "log_statement=none" in " ".join(seen["command"])


def test_schema_allowlist_keeps_configuration_out_of_app_crud():
    assert {"account_actions", "account_security", "rate_limit_buckets"} <= roles.managed_tables(3)
    version, ddl = roles.schema_ddl()
    assert version == 3
    assert "CREATE TABLE account_security" in ddl
    with pytest.raises(roles.OpsError, match="reviewed role-policy"):
        roles.managed_tables(4)


def test_malformed_record_fails_before_sql():
    class PG:
        owner = "owner"
        def sql(self, *_args):
            raise AssertionError("Malformed records must not run SQL")
    for record in ([], {"schema_version": 1, "operations_owner": "owner", "database": "db", "roles": []}, {"schema_version": 1, "operations_owner": "owner", "database": "db", "roles": {"app": [], "migration": "m", "backup": "b"}}):
        with pytest.raises(roles.OpsError):
            roles.verify_ownership(PG(), record)


def test_cleanup_never_takes_over_changed_ownership():
    plan = roles.role_plan("owned_db", "owned", "owner")
    class PG:
        def sql(self, _database, statement):
            assert "DROP" not in statement
            return "someone-else"
    roles.cleanup_failed(PG(), plan)


@pytest.mark.parametrize("function", ["private_password", "read_record"])
def test_named_pipe_readers_fail_within_a_process_deadline(tmp_path, function):
    fifo = tmp_path / "not-a-secret-file"
    os.mkfifo(fifo, 0o600)
    code = "import sys;sys.path.insert(0,'ops/cloud');import database_roles as m;\ntry: m." + function + "(" + repr(str(fifo)) + ")\nexcept m.OpsError: pass\nelse: raise AssertionError('FIFO accepted')"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=3, capture_output=True)
