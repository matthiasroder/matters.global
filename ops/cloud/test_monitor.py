"""Owned TLS and encrypted archive fixtures for operator monitoring."""

import hashlib
import importlib.util
import json
import os
import ssl
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("cloud_monitor", Path(__file__).with_name("monitor.py"))
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
recovery = monitor._module("recovery")


def private(path, data):
    path.write_bytes(data)
    path.chmod(0o600)
    return path


@pytest.fixture
def archive(tmp_path):
    key = os.urandom(32)
    key_file = private(tmp_path / "backup.key", key)
    dump = b"PGDMP-only-generated-test-data"
    backup_file = tmp_path / "backup.enc"
    def write(age=0, corrupt=False):
        metadata = {
            "format": "matters-postgresql-backup", "schema_version": 1,
            "database_schema": 2, "source_database": "test_private_database",
            "source_operations_owner": "test-private-owner",
            "dump_bytes": len(dump), "dump_sha256": hashlib.sha256(dump).hexdigest(),
            "created_at_utc": datetime.fromtimestamp(time.time() - age, timezone.utc).isoformat(),
        }
        encrypted = recovery.seal(dump, key, metadata)
        if corrupt:
            encrypted = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
        private(backup_file, encrypted)
        return metadata
    write()
    return backup_file, key_file, write


@pytest.fixture
def tls(tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ], check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state = {"mode": "healthy", "requests": 0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            state["requests"] += 1
            mode = state["mode"]
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "https://must-not-be-requested.invalid/?secret=PRIVATE_SENTINEL")
                self.end_headers()
                return
            body = b'{"status":"ok"}' if mode in {"healthy", "slow"} else b"PRIVATE_SENTINEL" * 4096
            self.send_response(503 if mode == "failure" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if mode == "slow":
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.25)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://localhost:{server.server_port}", cert, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def check(tls, archive, *, previous=None, timeout=2):
    origin, certificate, _state = tls
    backup_file, key_file, _write = archive
    return monitor.sample(origin, certificate, backup_file, key_file, max_age=60, timeout=timeout, maximum=1024 * 1024, clock_skew=1, previous=previous)


def test_owned_tls_and_authenticated_backup_alert_transitions(tls, archive):
    first = check(tls, archive)
    assert first["state"] == first["event"] == "healthy"
    assert first["health"]["code"] == "https_ok"
    assert first["backup"]["authenticated"] is True
    assert check(tls, archive, previous="healthy")["event"] == "unchanged"
    tls[2]["mode"] = "failure"
    failed = check(tls, archive, previous="healthy")
    assert failed["state"] == failed["event"] == "failure"
    tls[2]["mode"] = "healthy"
    recovered = check(tls, archive, previous="failure")
    assert recovered["state"] == "healthy" and recovered["event"] == "recovered"
    archive[2](age=120)
    stale = check(tls, archive, previous="healthy")
    assert stale["state"] == stale["event"] == "stale"
    assert stale["backup"]["authenticated"] is True
    archive[2](age=0)
    assert check(tls, archive, previous="stale")["event"] == "recovered"
    rendered = json.dumps([first, failed, recovered, stale])
    for secret in ("PGDMP", "test_private_database", "test-private-owner", str(archive[0]), str(archive[1]), "PRIVATE_SENTINEL"):
        assert secret not in rendered


def test_wrong_ca_redirect_and_oversized_body_fail_without_logging_response(tls, archive, tmp_path):
    origin, _certificate, state = tls
    private(tmp_path / "not-a-ca", b"PRIVATE_SENTINEL")
    invalid = monitor.checked_health(origin, tmp_path / "not-a-ca", 2)
    assert invalid["state"] == "failure"
    for mode in ("redirect", "large"):
        state["mode"] = mode
        result = check(tls, archive)
        assert result["state"] == "failure"
        assert "PRIVATE_SENTINEL" not in json.dumps(result)
    assert state["requests"] == 2  # No redirect destination was requested.


def test_actual_untrusted_ca_is_rejected(tls, tmp_path):
    other = tmp_path / "other.crt"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-keyout", str(tmp_path / "other.key"), "-out", str(other), "-subj", "/CN=other",
    ], check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = monitor.checked_health(tls[0], other, 2)
    assert result["state"] == "failure"
    assert tls[2]["requests"] == 0


def test_explicit_remote_transport_uses_verified_tls_and_no_ambient_proxy(tls, monkeypatch):
    # Exercise the opt-in transport only against the owned loopback fixture.
    # No external hostname is resolved or contacted by this test.
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-be-contacted.invalid:9999")
    monkeypatch.setenv("HTTP_PROXY", "http://must-not-be-contacted.invalid:9999")
    assert monitor.checked_health(tls[0], tls[1], 2, allow_remote=True)["state"] == "healthy"
    for mode in ("redirect", "large"):
        tls[2]["mode"] = mode
        failed = monitor.checked_health(tls[0], tls[1], 2, allow_remote=True)
        assert failed["state"] == "failure"
        assert "PRIVATE_SENTINEL" not in json.dumps(failed)
    assert tls[2]["requests"] == 3


def test_remote_origin_requires_explicit_flag_and_keeps_strict_url_validation():
    for origin in ("https://monitor.example.org", "https://10.1.2.3:8443", "https://[::1]:8443"):
        with pytest.raises(monitor.MonitorError):
            monitor.validate_configuration(origin, 60, 1, 1024, 1, 0, 1)
        monitor.validate_configuration(origin, 60, 1, 1024, 1, 0, 1, allow_remote=True)
    for origin in ("https://private.example.org/path", "https://user:PRIVATE_SENTINEL@private.example.org", "https://private.example.org?secret=PRIVATE_SENTINEL", "https://localhost:0", "https://localhost\n"):
        with pytest.raises(monitor.MonitorError):
            monitor.validate_configuration(origin, 60, 1, 1024, 1, 0, 1, allow_remote=True)


def test_whole_check_deadline_stops_slow_body(tls, archive):
    tls[2]["mode"] = "slow"
    started = time.monotonic()
    result = check(tls, archive, timeout=0.4)
    assert time.monotonic() - started < 2
    assert result["state"] == "failure"
    assert result["health"]["code"] == "check_timeout"


def test_wrong_key_tampering_future_time_and_oversize_are_not_healthy(tls, archive):
    backup_file, key_file, write = archive
    original_key = key_file.read_bytes()
    private(key_file, os.urandom(32))
    assert check(tls, archive)["backup"]["state"] == "failure"
    private(key_file, original_key)
    write(corrupt=True)
    assert check(tls, archive)["backup"]["state"] == "failure"
    write(age=-120)
    assert check(tls, archive)["backup"]["state"] == "failure"
    private(backup_file, b"x" * (1024 * 1024 + 1))
    assert check(tls, archive)["backup"]["state"] == "failure"


@pytest.mark.parametrize("origin", ["http://localhost", "https://example.org", "https://user:PRIVATE_SENTINEL@localhost", "https://localhost?secret=PRIVATE_SENTINEL", "https://localhost/path", "https://localhost#PRIVATE_SENTINEL"])
def test_configuration_refuses_nonlocal_credentials_paths_queries(origin):
    with pytest.raises(monitor.MonitorError):
        monitor.validate_configuration(origin, 60, 1, 1024, 1, 0, 1)


@pytest.mark.parametrize("numbers", [(0, 1, 1024, 1, 0, 1), (60, 0, 1024, 1, 0, 1), (60, 31, 1024, 1, 0, 1), (60, 1, monitor.MAX_BACKUP_BYTES + 1, 1, 0, 1), (60, 1, 1024, 101, 1, 1), (60, 1, 1024, 2, 0, 1), (60, 30, 1024, 100, 60, 1), (float("nan"), 1, 1024, 1, 0, 1)])
def test_time_byte_iteration_and_total_run_bounds(numbers):
    with pytest.raises(monitor.MonitorError):
        monitor.validate_configuration("https://localhost", *numbers)


def test_private_append_only_state_reopens_with_recovery_and_refuses_other_files(tmp_path):
    directory = tmp_path / "monitor"
    with monitor.StateStore(directory, "a" * 64) as state:
        first = state.append({"state": "failure", "event": "failure"})
        assert first["sequence"] == 1
        with pytest.raises(monitor.MonitorError, match="state_busy"):
            with monitor.StateStore(directory, "a" * 64):
                pass
    retained = (directory / "sample-000001.json").read_bytes()
    with monitor.StateStore(directory, "a" * 64) as state:
        assert state.previous == "failure" and state.sequence == 1
        state.append({"state": "healthy", "event": "recovered"})
    assert (directory / "sample-000001.json").read_bytes() == retained
    assert directory.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in directory.iterdir())
    with pytest.raises(monitor.MonitorError, match="state_binding_mismatch"):
        with monitor.StateStore(directory, "b" * 64):
            pass
    private(directory / "operator-file", b"retain me")
    with pytest.raises(monitor.MonitorError):
        with monitor.StateStore(directory, "a" * 64):
            pass
    assert (directory / "operator-file").read_bytes() == b"retain me"


def test_state_and_inputs_reject_repo_symlinks_permissions_and_path_escape(tmp_path):
    with pytest.raises(monitor.MonitorError):
        monitor.StateStore(monitor.REPOSITORY / "unsafe-monitor", "a" * 64)
    with pytest.raises(monitor.MonitorError):
        monitor.outside_repository(tmp_path / ".." / "escape")
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    for target in (linked, linked / "child"):
        with pytest.raises(monitor.MonitorError):
            with monitor.StateStore(target, "a" * 64):
                pass
    assert not (real / "child").exists()
    artifact = private(tmp_path / "artifact", b"private data")
    link = tmp_path / "artifact-link"
    link.symlink_to(artifact)
    with pytest.raises(monitor.MonitorError):
        monitor.open_artifact(link)
    artifact.chmod(0o644)
    with pytest.raises(monitor.MonitorError):
        monitor.open_artifact(artifact)
    assert artifact.read_bytes() == b"private data"


def test_cli_finite_samples_exit_codes_and_safe_diagnostics(tls, archive, tmp_path, capsys):
    directory = tmp_path / "monitor-cli"
    args = ["--origin", tls[0], "--ca-file", str(tls[1]), "--backup-file", str(archive[0]), "--key-file", str(archive[1]), "--state-directory", str(directory), "--max-backup-age-seconds", "60", "--iterations", "2", "--interval-seconds", "0.01"]
    assert monitor.main(args) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2 and [line["event"] for line in lines] == ["healthy", "unchanged"]
    archive[2](age=120)
    assert monitor.main([*args[:-4], "--iterations", "1"]) == 1
    assert json.loads(capsys.readouterr().out)["event"] == "stale"
    archive[2]()
    assert monitor.main([*args[:-4], "--iterations", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "recovered"
    bad = args[:]
    bad[1] = "https://user:PRIVATE_SENTINEL@localhost?key=PRIVATE_SENTINEL"
    assert monitor.main(bad) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err
    with pytest.raises(SystemExit) as invalid:
        monitor.main(["--unexpected=PRIVATE_SENTINEL"])
    assert invalid.value.code == 2
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().err


def test_worker_stdout_bound_and_state_record_limit(tmp_path, monkeypatch):
    class FakeProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = (tmp_path / "output").open("rb")
            self.returncode = None
        def poll(self):
            return self.returncode
        def kill(self):
            self.returncode = -9
        def wait(self, **_):
            return 0
    private(tmp_path / "output", b"x" * (monitor.MAX_OUTPUT_BYTES + 1))
    # Regular files cannot be registered with epoll; select supports this
    # owned deterministic pipe-output simulation without a child process.
    monkeypatch.setattr(monitor.selectors, "DefaultSelector", monitor.selectors.SelectSelector)
    monkeypatch.setattr(monitor.subprocess, "Popen", FakeProcess)
    with pytest.raises(monitor.MonitorError, match="check_output_limit"):
        monitor.bounded_worker([], [], 1)
    with monitor.StateStore(tmp_path / "record-limit", "a" * 64) as state:
        monkeypatch.setattr(monitor, "MAX_RECORDS", 1)
        state.append({"state": "healthy", "event": "healthy"})
        with pytest.raises(monitor.MonitorError, match="state_limit"):
            state.append({"state": "healthy", "event": "unchanged"})
