import errno
import hashlib
import os
import shutil
import signal
import sys
import threading
import time
from http import HTTPStatus

import pytest

import matters.terminal as terminal
from matters.terminal import TerminalError, TerminalManager, TerminalSession


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "terminal condition did not become true"


def executable(tmp_path, body):
    path = tmp_path / "terminal-child"
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o700)
    return str(path)


def output(session):
    return "".join(chunk["data"] for chunk in session.output_since(0)["chunks"])


def test_natural_exit_drains_output_closes_pty_and_reaps_child(tmp_path):
    shell = executable(tmp_path, "import os\nos.write(1, b'final output\\n')\n")
    session = TerminalSession(workspace=tmp_path, shell=shell)
    try:
        wait_for(lambda: session.closed)
        session.reader.join(timeout=1)
        assert not session.reader.is_alive()
        assert session.master_fd is None
        assert session.process.returncode == 0
        assert "final output" in output(session)
        assert "[terminal exited]" in output(session)
        with pytest.raises(TerminalError) as error:
            session.write("late input")
        assert error.value.status == HTTPStatus.GONE
    finally:
        session.close()


def test_missing_shell_closes_both_descriptors_without_masking_error(tmp_path, monkeypatch):
    original_openpty = terminal.pty.openpty
    opened = []

    def openpty():
        pair = original_openpty()
        opened.extend(pair)
        return pair

    monkeypatch.setattr(terminal.pty, "openpty", openpty)
    with pytest.raises(TerminalError, match="terminal shell was not found") as error:
        TerminalSession(workspace=tmp_path, shell=str(tmp_path / "missing"))
    assert error.value.status == HTTPStatus.BAD_GATEWAY
    for fd in opened:
        with pytest.raises(OSError) as closed:
            os.fstat(fd)
        assert closed.value.errno == errno.EBADF


def test_partial_writes_and_eagain_deliver_all_utf8_input(tmp_path, monkeypatch):
    text = "λ-safe-paste-" * 3000
    expected = text.encode()
    shell = executable(
        tmp_path,
        "import hashlib, os, pathlib, tty\n"
        "tty.setraw(0)\n"
        "os.write(1, b'READY')\n"
        f"remaining = {len(expected)}\n"
        "received = bytearray()\n"
        "while remaining:\n"
        "    part = os.read(0, min(4096, remaining))\n"
        "    received.extend(part)\n"
        "    remaining -= len(part)\n"
        "pathlib.Path('received.sha256').write_text(hashlib.sha256(received).hexdigest())\n",
    )
    session = TerminalSession(workspace=tmp_path, shell=shell)
    fd = session.master_fd
    original_write = terminal.os.write
    attempts = []

    def short_write(target, data):
        if target == fd:
            attempts.append(len(data))
            if len(attempts) == 1:
                raise BlockingIOError(errno.EAGAIN, "backpressure")
            return original_write(target, data[:113])
        return original_write(target, data)

    try:
        wait_for(lambda: "READY" in output(session))
        monkeypatch.setattr(terminal.os, "write", short_write)
        accepted = session.write(text)
        assert accepted == {
            "accepted_bytes": len(expected),
            "queued_bytes": len(expected),
        }
        wait_for(lambda: session.closed)
        assert (tmp_path / "received.sha256").read_text() == hashlib.sha256(expected).hexdigest()
        assert len(attempts) > 2
        assert session.output_since(0)["undelivered_bytes"] == 0
    finally:
        session.close()


def test_concurrent_input_requests_do_not_interleave_their_bytes(tmp_path):
    blocks = [letter * 4000 for letter in "ABCDE"]
    shell = executable(
        tmp_path,
        "import os, pathlib, tty\ntty.setraw(0)\nos.write(1, b'READY')\n"
        "received = bytearray()\n"
        f"while len(received) < {sum(map(len, blocks))}:\n"
        "    received.extend(os.read(0, 4096))\n"
        "pathlib.Path('received.bin').write_bytes(received)\n",
    )
    session = TerminalSession(workspace=tmp_path, shell=shell)
    barrier = threading.Barrier(len(blocks))
    errors = []

    def send(block):
        try:
            barrier.wait(timeout=3)
            session.write(block)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=send, args=(block,)) for block in blocks]
    try:
        wait_for(lambda: "READY" in output(session))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        assert not errors
        wait_for(lambda: session.closed)
        received = (tmp_path / "received.bin").read_text()
        assert sorted(received[i:i + 4000] for i in range(0, len(received), 4000)) == blocks
    finally:
        session.close()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=3)


def test_utf8_output_is_decoded_across_read_boundaries(tmp_path):
    shell = executable(
        tmp_path,
        "import os, time\nos.write(1, b'\\xce')\ntime.sleep(0.05)\nos.write(1, b'\\xbb')\n",
    )
    session = TerminalSession(workspace=tmp_path, shell=shell)
    try:
        wait_for(lambda: session.closed)
        assert "λ" in output(session)
        assert "�" not in output(session)
    finally:
        session.close()


def test_full_input_queue_rejects_whole_request_and_reports_undelivered_bytes(
    tmp_path, monkeypatch
):
    shell = executable(
        tmp_path,
        "import os, time, tty\ntty.setraw(0)\nos.write(1, b'READY')\ntime.sleep(60)\n",
    )
    session = TerminalSession(workspace=tmp_path, shell=shell, max_input_bytes=16)
    fd = session.master_fd
    original_write = terminal.os.write

    def blocked_write(target, data):
        if target == fd:
            raise BlockingIOError(errno.EAGAIN, "backpressure")
        return original_write(target, data)

    try:
        wait_for(lambda: "READY" in output(session))
        monkeypatch.setattr(terminal.os, "write", blocked_write)
        session.write("0123456789")
        with pytest.raises(TerminalError) as full:
            session.write("1234567")
        assert full.value.status == HTTPStatus.TOO_MANY_REQUESTS
        with pytest.raises(TerminalError) as too_large:
            session.write("a" * 17)
        assert too_large.value.status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert session.output_since(0)["queued_bytes"] == 10
        session.close()
        assert session.undelivered_bytes == 10
        assert "10 queued input bytes were not delivered" in output(session)
        assert session.process.returncode is not None
        assert session.master_fd is None
    finally:
        session.close()


def test_cleanup_escalates_and_is_idempotent(tmp_path):
    shell = executable(
        tmp_path,
        "import os, signal, time\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "os.write(1, b'READY')\n"
        "time.sleep(60)\n",
    )
    session = TerminalSession(workspace=tmp_path, shell=shell)
    try:
        wait_for(lambda: "READY" in output(session))
        session.close()
        session.close()
        assert session.process.returncode is not None
        assert session.master_fd is None
        assert session.closed
        assert not session.reader.is_alive()
    finally:
        session.close()


def test_cleanup_signals_group_even_after_leader_exits(tmp_path, monkeypatch):
    original_killpg = terminal.os.killpg
    signals = []

    def record_killpg(pid, sig):
        signals.append((pid, sig))
        return original_killpg(pid, sig)

    monkeypatch.setattr(terminal.os, "killpg", record_killpg)
    session = TerminalSession(workspace=tmp_path, shell=shutil.which("true"))
    try:
        wait_for(lambda: session.closed)
        assert session.process.returncode == 0
        assert (session.process.pid, signal.SIGHUP) in signals
    finally:
        session.close()


def test_completed_sessions_expire_even_if_output_is_polled(tmp_path):
    now = [100.0]
    manager = TerminalManager(
        default_workspace=tmp_path,
        default_shell=shutil.which("true"),
        clock=lambda: now[0],
        completed_ttl=60,
        prune_interval=3600,
    )
    try:
        session = manager.create()
        wait_for(lambda: session.closed)
        assert session.master_fd is None
        now[0] += 59
        assert manager.get(session.id).output_since(0)["closed"] is True
        assert manager.prune() == 0
        now[0] += 1
        assert manager.prune() == 1
        assert session.id not in manager.sessions
        with pytest.raises(TerminalError) as missing:
            manager.get(session.id)
        assert missing.value.status == HTTPStatus.NOT_FOUND
    finally:
        manager.close_all()


def test_running_session_heartbeat_extends_expiry(tmp_path):
    now = [100.0]
    manager = TerminalManager(
        default_workspace=tmp_path,
        default_shell="/bin/sh",
        clock=lambda: now[0],
        idle_timeout=3600,
        prune_interval=3600,
    )
    try:
        session = manager.create()
        now[0] += 3599
        manager.get(session.id).output_since(0)
        now[0] += 2
        assert manager.prune() == 0
        assert session.process.poll() is None
        now[0] += 3599
        assert manager.prune() == 1
        assert session.closed
        assert session.process.returncode is not None
    finally:
        manager.close_all()


def test_reaper_runs_without_another_request_and_close_all_joins_it(tmp_path):
    manager = TerminalManager(
        default_workspace=tmp_path,
        default_shell=shutil.which("true"),
        completed_ttl=0.02,
        prune_interval=0.01,
    )
    assert manager._reaper is None
    try:
        session = manager.create()
        wait_for(lambda: session.id not in manager.sessions)
        assert session.closed
        assert session.master_fd is None
    finally:
        manager.close_all()
        manager.close_all()
    assert not manager._reaper.is_alive()
    assert manager.sessions == {}


def test_error_factory_preserves_adapter_error_type(tmp_path):
    class AdapterError(TerminalError):
        pass

    manager = TerminalManager(error_factory=AdapterError)
    with pytest.raises(AdapterError, match="workspace does not exist"):
        manager.create(workspace=tmp_path / "missing")
    assert manager._reaper is None
    manager.close_all()
