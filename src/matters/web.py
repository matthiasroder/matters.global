"""Local web UI server for matters graphs."""

import fcntl
import json
import mimetypes
import os
import pty
import re
import select
import signal
import subprocess
import struct
import termios
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .cli import create_matters_from_expression
from .engine import dependents, frontier, horizon, prerequisites, resolved, truth, universe
from .extraction import slugify
from .llm_extraction import build_extraction_proposal
from .reports import unlock_report
from .storage import load_state, resolve_state_path, save_state


DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
DEFAULT_TERMINAL_SHELL = os.environ.get("SHELL") or "/bin/sh"
TERMINAL_WORKSPACE_ENV = "MATTERS_TERMINAL_WORKSPACE"
MAX_TERMINAL_CHUNKS = 1000


class ApiError(ValueError):
    """Validation error that should be returned as an API response."""

    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


class StateMutationLocks:
    def __init__(self):
        self._locks = {}
        self._guard = threading.Lock()

    @contextmanager
    def lock(self, state_path):
        lock_key = str(resolve_state_path(state_path))
        with self._guard:
            lock = self._locks.setdefault(lock_key, threading.RLock())
        with lock:
            yield


state_mutation_locks = StateMutationLocks()


def graph_payload(state_path=None):
    matters, conditions, dependencies = load_state(state_path)
    actionable = universe(matters, conditions, dependencies)

    nodes = []
    for matter in sorted(matters):
        is_resolved = resolved(matter, conditions, dependencies)
        is_actionable = matter in actionable
        nodes.append(
            {
                "id": matter,
                "label": matter.replace("_", " "),
                "conditions": conditions.get(matter, []),
                "prerequisites": sorted(prerequisites(matter, dependencies)),
                "dependents": sorted(dependents(matter, dependencies)),
                "resolved": is_resolved,
                "actionable": is_actionable,
                "blocked": not is_resolved and not is_actionable,
            }
        )

    return {
        "state_path": str(resolve_state_path(state_path)),
        "nodes": nodes,
        "edges": [
            {"source": prerequisite, "target": dependent}
            for prerequisite, dependent in sorted(dependencies)
        ],
        "universe": sorted(actionable),
        "unlock": unlock_report(matters, conditions, dependencies),
    }


def create_matter(state_path, payload):
    with state_mutation_locks.lock(state_path):
        matters, conditions, dependencies = load_state(state_path)
        matter_id = normalized_matter_id(payload)
        if matter_id in matters:
            raise ApiError(f"matter already exists: {matter_id}", HTTPStatus.CONFLICT)

        condition_payloads = payload.get("conditions") or [
            {"label": f"Resolved: {matter_id.replace('_', ' ')}", "truth": False}
        ]
        normalized_conditions = [
            normalize_condition(condition, index)
            for index, condition in enumerate(condition_payloads, start=1)
        ]

        matters.add(matter_id)
        conditions[matter_id] = normalized_conditions
        save_state(matters, conditions, dependencies, path=state_path)
        return graph_payload(state_path)


def update_conditions(state_path, matter_id, payload):
    with state_mutation_locks.lock(state_path):
        matters, conditions, dependencies = load_state(state_path)
        if matter_id not in matters:
            raise ApiError(f"unknown matter: {matter_id}", HTTPStatus.NOT_FOUND)

        action = payload.get("action")
        current = list(conditions.get(matter_id, []))

        if "conditions" in payload:
            current = [
                normalize_condition(condition, index)
                for index, condition in enumerate(payload["conditions"], start=1)
            ]
        elif action == "toggle":
            index = require_condition_index(payload, current)
            current[index]["truth"] = not truth(current[index])
        elif action == "delete":
            index = require_condition_index(payload, current)
            del current[index]
        else:
            index = payload.get("index")
            if index is None:
                current.append(normalize_condition(payload, len(current) + 1))
            else:
                index = require_condition_index(payload, current)
                updated = dict(current[index])
                if "label" in payload:
                    updated["label"] = str(payload["label"]).strip()
                if "truth" in payload:
                    updated["truth"] = truth(payload["truth"])
                current[index] = normalize_condition(updated, index + 1)

        conditions[matter_id] = current
        save_state(matters, conditions, dependencies, path=state_path)
        return graph_payload(state_path)


def add_dependency(state_path, payload):
    with state_mutation_locks.lock(state_path):
        matters, conditions, dependencies = load_state(state_path)
        source, target = dependency_endpoints(payload, matters)
        next_dependencies = set(dependencies)
        next_dependencies.add((source, target))
        if has_dependency_cycle(next_dependencies):
            raise ApiError("dependency would create a cycle")

        save_state(matters, conditions, next_dependencies, path=state_path)
        return graph_payload(state_path)


def remove_dependency(state_path, payload):
    with state_mutation_locks.lock(state_path):
        matters, conditions, dependencies = load_state(state_path)
        source, target = dependency_endpoints(payload, matters)
        next_dependencies = set(dependencies)
        next_dependencies.discard((source, target))
        save_state(matters, conditions, next_dependencies, path=state_path)
        return graph_payload(state_path)


def run_command(state_path, payload):
    text = str(payload.get("text") or payload.get("command") or "").strip()
    if not text:
        raise ApiError("command is required")

    command, _, rest = text.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command == "create":
        if not rest:
            raise ApiError("create requires an expression")
        with state_mutation_locks.lock(state_path):
            matters, conditions, dependencies = load_state(state_path)
            try:
                created = create_matters_from_expression(
                    rest, matters, conditions, dependencies
                )
            except ValueError as error:
                raise ApiError(str(error)) from error
            if has_dependency_cycle(dependencies):
                raise ApiError("created expression would create a cycle")
            save_state(matters, conditions, dependencies, path=state_path)
            return {"type": "create", "created": created, "state": graph_payload(state_path)}

    matters, conditions, dependencies = load_state(state_path)

    if command == "universe":
        return {"type": "universe", "items": sorted(universe(matters, conditions, dependencies))}
    if command == "frontier":
        require_matter(rest, matters)
        return {"type": "frontier", "matter": rest, "items": sorted(frontier(rest, conditions, dependencies))}
    if command == "horizon":
        require_matter(rest, matters)
        return {"type": "horizon", "matter": rest, "items": sorted(horizon(rest, conditions, dependencies))}
    if command == "unlock":
        return {"type": "unlock", "report": unlock_report(matters, conditions, dependencies)}
    if command == "extract":
        if not rest:
            raise ApiError("extract requires source text")
        return {
            "type": "extract",
            "proposal": build_extraction_proposal(
                rest, source_type="text", existing_matters=matters
            ),
        }

    raise ApiError(f"unknown command: {command}")


class TerminalManager:
    def __init__(self, default_workspace=None, default_shell=None):
        self.sessions = {}
        self.lock = threading.Lock()
        self.default_workspace = default_workspace
        self.default_shell = default_shell or DEFAULT_TERMINAL_SHELL

    def create(self, workspace=None, shell=None, rows=24, cols=100):
        session = TerminalSession(
            workspace=workspace or self.default_workspace or Path.cwd(),
            shell=shell or self.default_shell,
            rows=rows,
            cols=cols,
        )
        with self.lock:
            self.sessions[session.id] = session
        return session

    def get(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
        if not session:
            raise ApiError("terminal session not found", HTTPStatus.NOT_FOUND)
        return session

    def close(self, session_id):
        session = self.get(session_id)
        session.close()
        with self.lock:
            self.sessions.pop(session_id, None)
        return {"closed": True}

    def close_all(self):
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.close()


class TerminalSession:
    def __init__(
        self,
        workspace=None,
        shell=DEFAULT_TERMINAL_SHELL,
        rows=24,
        cols=100,
    ):
        self.id = uuid.uuid4().hex
        self.workspace = Path(workspace or Path.cwd()).expanduser()
        self.shell = shell
        self.master_fd = None
        self.process = None
        self.lock = threading.Lock()
        self.chunks = []
        self.next_seq = 1
        self.closed = False
        self.started_at = time.time()

        if not self.workspace.exists():
            raise ApiError(f"terminal workspace does not exist: {self.workspace}", HTTPStatus.NOT_FOUND)

        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        os.set_blocking(master_fd, False)
        set_terminal_size(master_fd, rows, cols)

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"

        try:
            self.process = subprocess.Popen(
                [shell],
                cwd=str(self.workspace),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                env=env,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError as error:
            os.close(master_fd)
            os.close(slave_fd)
            raise ApiError(f"terminal shell was not found: {shell}", HTTPStatus.BAD_GATEWAY) from error
        finally:
            os.close(slave_fd)

        self.reader = threading.Thread(target=self.read_loop, daemon=True)
        self.reader.start()

    def to_payload(self):
        return {
            "id": self.id,
            "workspace": str(self.workspace),
            "shell": self.shell,
            "started_at": self.started_at,
        }

    def write(self, data):
        if self.closed:
            raise ApiError("terminal session is closed", HTTPStatus.GONE)
        if not isinstance(data, str):
            raise ApiError("terminal input must be text")
        os.write(self.master_fd, data.encode(errors="replace"))
        return {"written": len(data)}

    def resize(self, rows, cols):
        rows = max(3, safe_int(rows, 24))
        cols = max(20, safe_int(cols, 100))
        if self.closed:
            return {"resized": False}
        set_terminal_size(self.master_fd, rows, cols)
        return {"resized": True, "rows": rows, "cols": cols}

    def output_since(self, seq):
        seq = safe_int(seq, 0)
        with self.lock:
            chunks = [chunk for chunk in self.chunks if chunk["seq"] > seq]
            closed = self.closed
        return {"chunks": chunks, "closed": closed}

    def read_loop(self):
        while not self.closed:
            if self.process.poll() is not None:
                self.append_output("\r\n[terminal exited]\r\n")
                self.closed = True
                break
            try:
                readable, _, _ = select.select([self.master_fd], [], [], 0.1)
            except (OSError, ValueError):
                self.closed = True
                break
            if not readable:
                continue
            try:
                data = os.read(self.master_fd, 4096)
            except BlockingIOError:
                continue
            except OSError:
                self.closed = True
                break
            if not data:
                self.closed = True
                break
            self.append_output(data.decode(errors="replace"))

    def append_output(self, data):
        with self.lock:
            self.chunks.append({"seq": self.next_seq, "data": data})
            self.next_seq += 1
            if len(self.chunks) > MAX_TERMINAL_CHUNKS:
                self.chunks = self.chunks[-MAX_TERMINAL_CHUNKS:]

    def close(self):
        self.closed = True
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass


def set_terminal_size(fd, rows, cols):
    packed = struct.pack("HHHH", int(rows), int(cols), 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class StatePathStore:
    def __init__(self, state_path=None):
        self._path = resolve_state_path(state_path)
        self._lock = threading.Lock()

    def current(self):
        with self._lock:
            return self._path

    def switch(self, state_path):
        next_path = validate_switch_state_path(state_path)
        with self._lock:
            self._path = next_path
        return next_path


def validate_switch_state_path(state_path):
    raw_path = str(state_path or "").strip()
    if not raw_path:
        raise ApiError("state path is required")

    next_path = resolve_state_path(raw_path)
    if not next_path.exists():
        raise ApiError(f"state file does not exist: {next_path}", HTTPStatus.NOT_FOUND)

    try:
        load_state(next_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApiError(f"state file is not a valid matters graph: {next_path}") from error

    return next_path


def switch_state_path(state_paths, payload):
    next_path = state_paths.switch(payload.get("state_path") or payload.get("path"))
    return graph_payload(next_path)


def resolve_terminal_workspace(state_path=None, terminal_workspace=None):
    raw_workspace = terminal_workspace or os.environ.get(TERMINAL_WORKSPACE_ENV)
    if raw_workspace:
        return Path(raw_workspace).expanduser()

    if state_path is not None:
        state_parent = resolve_state_path(state_path).parent
        if state_parent.exists():
            return state_parent

    return Path.cwd()


def serve(
    state_path=None,
    host=DEFAULT_WEB_HOST,
    port=DEFAULT_WEB_PORT,
    open_browser=True,
    terminal_workspace=None,
    terminal_shell=None,
):
    resolved_state_path = resolve_state_path(state_path)
    state_paths = StatePathStore(resolved_state_path)
    terminal_workspace = resolve_terminal_workspace(
        resolved_state_path,
        terminal_workspace=terminal_workspace,
    )
    terminal_manager = TerminalManager(
        default_workspace=terminal_workspace,
        default_shell=terminal_shell,
    )
    handler = partial(
        MattersWebHandler,
        state_paths=state_paths,
        terminal_manager=terminal_manager,
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"Serving matters web UI at {url}")
    print(f"State file: {resolved_state_path}")
    print(f"Terminal workspace: {terminal_workspace}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping matters web UI")
    finally:
        terminal_manager.close_all()
        server.server_close()


class MattersWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state_paths=None, terminal_manager=None, **kwargs):
        self.state_paths = state_paths or StatePathStore()
        self.terminal_manager = terminal_manager or TerminalManager()
        super().__init__(*args, directory=str(web_assets_path()), **kwargs)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.write_json(graph_payload(self.current_state_path()))
            return
        match = re.fullmatch(r"/api/terminal/sessions/([^/]+)/output", parsed.path)
        if match:
            query = parse_qs(parsed.query)
            seq = query.get("seq", ["0"])[0]
            try:
                session = self.terminal_manager.get(unquote(match.group(1)))
                self.write_json(session.output_since(seq))
            except ApiError as error:
                self.write_error(error)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                self.require_api_mutation_request()
            if parsed.path == "/api/matters":
                self.write_json(create_matter(self.current_state_path(), self.read_json()), HTTPStatus.CREATED)
                return
            if parsed.path == "/api/dependencies":
                self.write_json(add_dependency(self.current_state_path(), self.read_json()), HTTPStatus.CREATED)
                return
            if parsed.path == "/api/state":
                self.write_json(switch_state_path(self.state_paths, self.read_json()))
                return
            if parsed.path == "/api/command":
                self.write_json(run_command(self.current_state_path(), self.read_json()))
                return
            if parsed.path == "/api/terminal/sessions":
                payload = self.read_json()
                session = self.terminal_manager.create(
                    rows=payload.get("rows", 24),
                    cols=payload.get("cols", 100),
                )
                self.write_json(session.to_payload(), HTTPStatus.CREATED)
                return
            match = re.fullmatch(r"/api/terminal/sessions/([^/]+)/(input|resize)", parsed.path)
            if match:
                session = self.terminal_manager.get(unquote(match.group(1)))
                payload = self.read_json()
                if match.group(2) == "input":
                    self.write_json(session.write(payload.get("data", "")))
                else:
                    self.write_json(session.resize(payload.get("rows"), payload.get("cols")))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except ApiError as error:
            self.write_error(error)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        match = re.fullmatch(r"/api/matters/([^/]+)/conditions", parsed.path)
        if not match:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        matter_id = unquote(match.group(1))
        try:
            self.require_api_mutation_request()
            self.write_json(update_conditions(self.current_state_path(), matter_id, self.read_json()))
        except ApiError as error:
            self.write_error(error)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                self.require_api_mutation_request()
            match = re.fullmatch(r"/api/terminal/sessions/([^/]+)", parsed.path)
            if match:
                self.write_json(self.terminal_manager.close(unquote(match.group(1))))
                return
            if parsed.path != "/api/dependencies":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.write_json(remove_dependency(self.current_state_path(), self.read_json()))
        except ApiError as error:
            self.write_error(error)

    def current_state_path(self):
        return self.state_paths.current()

    def guess_type(self, path):
        if path.endswith(".js"):
            return "text/javascript"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def require_api_mutation_request(self):
        self.require_same_origin_request()
        self.require_json_content_type()

    def require_same_origin_request(self):
        for header in ("Origin", "Referer"):
            value = self.headers.get(header)
            if value and not self.is_same_origin(value):
                raise ApiError("cross-origin API request rejected", HTTPStatus.FORBIDDEN)

    def is_same_origin(self, value):
        parsed = urlparse(value)
        host = (self.headers.get("Host") or "").lower()
        return parsed.scheme == "http" and parsed.netloc.lower() == host

    def require_json_content_type(self):
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise ApiError(
                "API mutation requests must use application/json",
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ApiError("invalid Content-Length") from error
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ApiError(f"invalid JSON: {error.msg}") from error

    def write_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, error):
        self.write_json({"error": str(error)}, error.status)


def web_assets_path():
    return resources.files("matters").joinpath("web_assets")


def normalized_matter_id(payload):
    raw_id = str(payload.get("id") or "").strip()
    title = str(payload.get("title") or payload.get("name") or "").strip()
    matter_id = raw_id or slugify(title)
    if not matter_id:
        raise ApiError("matter id or title is required")
    if not re.fullmatch(r"[a-z0-9_]+", matter_id):
        raise ApiError("matter id must contain lowercase letters, numbers, and underscores only")
    return matter_id


def normalize_condition(condition, index):
    if isinstance(condition, str):
        label = condition.strip()
        condition_truth = False
    else:
        label = str(condition.get("label") or "").strip()
        condition_truth = truth(condition.get("truth", False))
    if not label:
        label = f"Unlabeled condition {index}"
    return {"label": label, "truth": condition_truth}


def require_condition_index(payload, conditions):
    try:
        index = int(payload["index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError("valid condition index is required") from error
    if index < 0 or index >= len(conditions):
        raise ApiError("condition index is out of range", HTTPStatus.NOT_FOUND)
    return index


def dependency_endpoints(payload, matters):
    source = str(payload.get("source") or payload.get("prerequisite") or "").strip()
    target = str(payload.get("target") or payload.get("dependent") or "").strip()
    if not source or not target:
        raise ApiError("dependency source and target are required")
    if source not in matters:
        raise ApiError(f"unknown dependency source: {source}", HTTPStatus.NOT_FOUND)
    if target not in matters:
        raise ApiError(f"unknown dependency target: {target}", HTTPStatus.NOT_FOUND)
    return source, target


def require_matter(matter_id, matters):
    if not matter_id:
        raise ApiError("matter id is required")
    if matter_id not in matters:
        raise ApiError(f"unknown matter: {matter_id}", HTTPStatus.NOT_FOUND)


def has_dependency_cycle(dependencies):
    outgoing = {}
    for source, target in dependencies:
        outgoing.setdefault(source, set()).add(target)

    visiting = set()
    visited = set()

    def visit(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in outgoing.get(node, ()):
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in outgoing)
