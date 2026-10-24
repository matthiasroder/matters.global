"""Owned local terminal sessions with bounded input and output buffers.

The web adapter supplies its error factory; this module owns processes, PTYs,
and session expiry without importing HTTP handlers or graph state.
"""

import codecs
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import termios
import threading
import time
import uuid
from http import HTTPStatus
from pathlib import Path


DEFAULT_TERMINAL_SHELL = os.environ.get("SHELL") or "/bin/sh"
MAX_TERMINAL_CHUNKS = 1000
MAX_TERMINAL_INPUT_BYTES = 1024 * 1024
DEFAULT_IDLE_TIMEOUT = 60 * 60
DEFAULT_COMPLETED_TTL = 60


class TerminalError(ValueError):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class TerminalManager:
    """Retain completed output briefly and expire abandoned running shells.

    Every get is a heartbeat, including output polling while a drawer is
    hidden. Output produced by a child is not a heartbeat: a detached noisy
    command must still expire. Completed output expires from completion time,
    regardless of continued polling. The reaper starts only when needed.
    """

    def __init__(
        self,
        default_workspace=None,
        default_shell=None,
        *,
        idle_timeout=DEFAULT_IDLE_TIMEOUT,
        completed_ttl=DEFAULT_COMPLETED_TTL,
        prune_interval=30,
        clock=time.monotonic,
        error_factory=TerminalError,
    ):
        self.sessions = {}
        self.lock = threading.Lock()
        self.default_workspace = default_workspace
        self.default_shell = default_shell or DEFAULT_TERMINAL_SHELL
        self.idle_timeout = idle_timeout
        self.completed_ttl = completed_ttl
        self.prune_interval = max(0.01, prune_interval)
        self.clock = clock
        self.error_factory = error_factory
        self._stopped = threading.Event()
        self._reaper = None

    def create(self, workspace=None, shell=None, rows=24, cols=100):
        with self.lock:
            if self._stopped.is_set():
                raise self.error_factory("terminal manager is closed", HTTPStatus.GONE)
            session = TerminalSession(
                workspace=workspace or self.default_workspace or Path.cwd(),
                shell=shell or self.default_shell,
                rows=rows,
                cols=cols,
                clock=self.clock,
                error_factory=self.error_factory,
            )
            self.sessions[session.id] = session
            if self._reaper is None:
                self._reaper = threading.Thread(
                    target=self._reap_loop, name="matters-terminal-reaper", daemon=True
                )
                try:
                    self._reaper.start()
                except Exception:
                    self._reaper = None
                    self.sessions.pop(session.id, None)
                    session.close()
                    raise
        return session

    def get(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                raise self.error_factory("terminal session not found", HTTPStatus.NOT_FOUND)
            session.touch()
        return session

    def close(self, session_id):
        with self.lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            raise self.error_factory("terminal session not found", HTTPStatus.NOT_FOUND)
        session.close()
        return {"closed": True, "undelivered_bytes": session.undelivered_bytes}

    def prune(self):
        now = self.clock()
        with self.lock:
            expired = [
                session
                for session in self.sessions.values()
                if session.expired(now, self.idle_timeout, self.completed_ttl)
            ]
            for session in expired:
                self.sessions.pop(session.id, None)
        for session in expired:
            session.close()
        return len(expired)

    def _reap_loop(self):
        while not self._stopped.wait(self.prune_interval):
            self.prune()

    def close_all(self):
        self._stopped.set()
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.close()
        if self._reaper is not None and self._reaper is not threading.current_thread():
            self._reaper.join()


class TerminalSession:
    def __init__(
        self,
        workspace=None,
        shell=DEFAULT_TERMINAL_SHELL,
        rows=24,
        cols=100,
        *,
        clock=time.monotonic,
        error_factory=TerminalError,
        max_input_bytes=MAX_TERMINAL_INPUT_BYTES,
    ):
        self.id = uuid.uuid4().hex
        self.workspace = Path(workspace or Path.cwd()).expanduser()
        self.shell = shell
        self.master_fd = None
        self.process = None
        self.reader = None
        self.lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self.chunks = []
        self.next_seq = 1
        self.closed = False
        self.started_at = time.time()
        self.clock = clock
        self.last_activity = clock()
        self.finished_at = None
        self.error_factory = error_factory
        self.max_input_bytes = max_input_bytes
        self._pending_input = bytearray()
        self.undelivered_bytes = 0

        if not self.workspace.is_dir():
            raise error_factory(
                f"terminal workspace does not exist: {self.workspace}", HTTPStatus.NOT_FOUND
            )

        master_fd, slave_fd = pty.openpty()
        try:
            os.set_blocking(master_fd, False)
            set_terminal_size(master_fd, rows, cols)
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["COLORTERM"] = "truecolor"
            self.process = subprocess.Popen(
                [shell],
                cwd=str(self.workspace),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                env=env,
                start_new_session=True,
            )
        except OSError as error:
            os.close(master_fd)
            if isinstance(error, FileNotFoundError):
                message = f"terminal shell was not found: {shell}"
            else:
                message = f"terminal shell could not start: {error.strerror}"
            raise error_factory(message, HTTPStatus.BAD_GATEWAY) from error
        except Exception:
            os.close(master_fd)
            raise
        finally:
            os.close(slave_fd)

        self.master_fd = master_fd
        self.reader = threading.Thread(
            target=self.read_loop, name=f"matters-terminal-{self.id}", daemon=True
        )
        try:
            self.reader.start()
        except Exception:
            self._finish()
            raise

    def to_payload(self):
        return {
            "id": self.id,
            "workspace": str(self.workspace),
            "shell": self.shell,
            "started_at": self.started_at,
        }

    def touch(self):
        with self.lock:
            self.last_activity = self.clock()

    def expired(self, now, idle_timeout, completed_ttl):
        with self.lock:
            if self.finished_at is not None:
                return now - self.finished_at >= completed_ttl
            return now - self.last_activity >= idle_timeout

    def write(self, data):
        """Atomically accept a bounded chunk; the I/O thread delivers every byte.

        Acknowledgement means queued, not already consumed by the child. A
        full queue rejects the entire request, allowing a client to retry it
        without duplicating a prefix. No request thread writes to the PTY.
        """
        if not isinstance(data, str):
            raise self.error_factory("terminal input must be text")
        encoded = data.encode(errors="replace")
        with self.lock:
            if self.closed or self._stop.is_set():
                raise self.error_factory("terminal session is closed", HTTPStatus.GONE)
            if len(encoded) > self.max_input_bytes:
                raise self.error_factory(
                    "terminal input exceeds the input buffer limit",
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            if len(self._pending_input) + len(encoded) > self.max_input_bytes:
                raise self.error_factory(
                    "terminal input buffer is full; retry after it drains",
                    HTTPStatus.TOO_MANY_REQUESTS,
                )
            self._pending_input.extend(encoded)
            self.last_activity = self.clock()
            return {
                "accepted_bytes": len(encoded),
                "queued_bytes": len(self._pending_input),
            }

    def resize(self, rows, cols):
        rows = min(65535, max(3, safe_int(rows, 24)))
        cols = min(65535, max(20, safe_int(cols, 100)))
        with self.lock:
            self.last_activity = self.clock()
            if self.closed or self.master_fd is None or self._stop.is_set():
                return {"resized": False}
            set_terminal_size(self.master_fd, rows, cols)
        return {"resized": True, "rows": rows, "cols": cols}

    def output_since(self, seq):
        seq = safe_int(seq, 0)
        with self.lock:
            self.last_activity = self.clock()
            return {
                "chunks": [chunk for chunk in self.chunks if chunk["seq"] > seq],
                "closed": self.closed,
                "queued_bytes": len(self._pending_input),
                "undelivered_bytes": self.undelivered_bytes,
            }

    def _flush_input(self, fd):
        with self.lock:
            if self._stop.is_set() or fd != self.master_fd or not self._pending_input:
                return
            try:
                count = os.write(fd, self._pending_input[:65536])
            except BlockingIOError:
                return
            if count <= 0:
                raise OSError("terminal write made no progress")
            del self._pending_input[:count]

    def read_loop(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while not self._stop.is_set():
                with self.lock:
                    fd = self.master_fd
                    has_input = bool(self._pending_input)
                readable, writable, _ = select.select(
                    [fd], [fd] if has_input else [], [], 0.1
                )
                if self._stop.is_set():
                    break
                if writable:
                    self._flush_input(fd)
                if readable:
                    try:
                        data = os.read(fd, 4096)
                    except BlockingIOError:
                        continue
                    if not data:
                        break
                    self.append_output(decoder.decode(data))
                elif self.process.poll() is not None:
                    # A readable PTY is drained first, preserving final output.
                    break
        except (OSError, ValueError):
            pass
        finally:
            self.append_output(decoder.decode(b"", final=True))
            self._finish()

    def append_output(self, data):
        if not data:
            return
        with self.lock:
            self.chunks.append({"seq": self.next_seq, "data": data})
            self.next_seq += 1
            if len(self.chunks) > MAX_TERMINAL_CHUNKS:
                del self.chunks[:-MAX_TERMINAL_CHUNKS]

    def _finish(self):
        """Release the process and fd exactly once, including natural exits."""
        with self._cleanup_lock:
            if self._finished.is_set():
                return
            self._stop.set()
            with self.lock:
                fd = self.master_fd
                self.master_fd = None
                self.undelivered_bytes = len(self._pending_input)
                self._pending_input.clear()
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if self.process is not None:
                # Give a naturally closing child a moment to finish before
                # terminating the process group owned by this session.
                try:
                    self.process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                # The shell may already have exited while background children
                # still hold its process group and PTY. Signal the group even
                # when the leader has been reaped.
                for sig, timeout in ((signal.SIGHUP, 0.3),
                                     (signal.SIGTERM, 0.3), (signal.SIGKILL, 1)):
                    try:
                        os.killpg(self.process.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        self.process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        continue
                    try:
                        os.killpg(self.process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                # A nonexistent group can race with wait(), so always collect
                # the direct child after signalling rather than relying on
                # another request eventually calling poll().
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if self.undelivered_bytes:
                self.append_output(
                    f"\r\n[terminal exited; {self.undelivered_bytes} queued input bytes "
                    "were not delivered]\r\n"
                )
            else:
                self.append_output("\r\n[terminal exited]\r\n")
            with self.lock:
                self.closed = True
                self.finished_at = self.clock()
            self._finished.set()

    def close(self):
        self._stop.set()
        if self.reader is not None and self.reader is not threading.current_thread():
            # The I/O loop has a bounded select wait and owns fd cleanup.
            if self.reader.ident is not None:
                self.reader.join()
        self._finish()


def set_terminal_size(fd, rows, cols):
    rows = min(65535, max(3, safe_int(rows, 24)))
    cols = min(65535, max(20, safe_int(cols, 100)))
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
