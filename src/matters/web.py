"""Local web UI server for matters graphs."""

import fcntl
import ipaddress
import json
import mimetypes
import os
import pty
import re
import secrets
import select
import signal
import subprocess
import struct
import termios
import threading
import time
import uuid
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from . import rules
from .layout import build_overview_layout
from .llm_extraction import build_extraction_proposal
from .reports import unlock_report
from .rules import RuleError, create_matters_from_expression
from .storage import load_state, resolve_state_path


DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
DEFAULT_TERMINAL_SHELL = os.environ.get("SHELL") or "/bin/sh"
TERMINAL_WORKSPACE_ENV = "MATTERS_TERMINAL_WORKSPACE"
MAX_TERMINAL_CHUNKS = 1000
LOCAL_API_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
WILDCARD_API_HOSTS = frozenset({"0.0.0.0", "::", ""})

# The API grants an interactive shell, so it is authenticated. The Host and
# Origin checks below stay, but they only describe where a request claims to
# come from; anything that can open a socket to the port omits Origin and
# passes them. The token is what identifies the caller.
API_TOKEN_BYTES = 32
API_TOKEN_QUERY_PARAM = "token"
# Deliberately identical for a missing token and a wrong one: the response
# must not tell a prober which of the two it is.
API_UNAUTHORIZED_MESSAGE = "unauthorized"
# The only paths that may carry the token in the query string. This is the
# document request that ``serve`` opens; app.js moves the token into memory
# and strips it from the address bar before it makes an API call.
PAGE_LOAD_PATHS = frozenset({"/", "/index.html"})

# Every response carries this. The page imports four ES modules and one
# stylesheet from cdn.jsdelivr.net (app.js:1-4, index.html:7), so that exact
# origin is allowed for scripts and styles and nothing else is allowed at all.
#
# MITIGATION, NOT A FIX: this narrows what a compromise of those CDN packages
# can reach (no eval, no exfiltration to a third-party origin, no framing),
# but code from jsdelivr still runs same-origin with a server that hands out
# shells. Vendoring those assets into web_assets/ is the actual fix and is
# tracked separately -- it has licensing and packaging implications. Do not
# read this header as settling that question.
#
# 'unsafe-inline' appears for styles only, and it is load-bearing rather than
# cautious. Verified in a browser on 2026-08-03: dropping it makes cytoscape
# log "container has style position:static and so can not use UI extensions
# properly" and the graph renders with its node labels overlapping. Keep it.
# script-src stays strict -- that is the directive that would turn a CDN
# compromise into code execution, and it needs neither inline nor eval.
# 'unsafe-inline' appears for styles only: xterm.js and cytoscape both inject
# stylesheets at runtime, and a style-src that breaks the UI would be removed
# by the next person to touch this. script-src stays strict, which is the
# directive that matters for turning a CDN compromise into shell access.
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'",
        "font-src 'self' https://cdn.jsdelivr.net",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    )
)


class ApiError(ValueError):
    """Validation error that should be returned as an API response."""

    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


RULE_ERROR_STATUS = {
    "invalid": HTTPStatus.BAD_REQUEST,
    "not_found": HTTPStatus.NOT_FOUND,
    "conflict": HTTPStatus.CONFLICT,
    "state_cycle": HTTPStatus.UNPROCESSABLE_ENTITY,
    "locked": HTTPStatus.CONFLICT,
}


def api_error_for(error):
    """Translate a rules-layer ``RuleError`` into an ``ApiError``."""

    # ``.get`` on purpose: a code added to rules.py without a table entry here
    # degrades to 400, instead of raising KeyError past the ``except ApiError``
    # handler and turning a rejected write into a 500.
    return ApiError(
        str(error), RULE_ERROR_STATUS.get(error.code, HTTPStatus.BAD_REQUEST)
    )


def graph_payload(state_path=None):
    matters, conditions, dependencies = load_state(state_path)
    index = graph_index_or_api_error(matters, conditions, dependencies)
    overview_layout, overview_nodes = build_overview_layout(index)

    nodes = []
    for matter in sorted(matters):
        is_resolved = index.resolved[matter]
        is_actionable = matter in index.universe
        nodes.append(
            {
                "id": matter,
                "label": matter.replace("_", " "),
                "conditions": conditions.get(matter, []),
                "prerequisites": list(index.prerequisites[matter]),
                "dependents": list(index.dependents[matter]),
                "resolved": is_resolved,
                "actionable": is_actionable,
                "blocked": not is_resolved and not is_actionable,
                "overview": overview_nodes[matter],
            }
        )

    return {
        "state_path": str(resolve_state_path(state_path)),
        "nodes": nodes,
        "edges": [
            {"source": prerequisite, "target": dependent}
            for prerequisite, dependent in sorted(dependencies)
        ],
        "universe": sorted(index.universe),
        "unlock": unlock_report(
            matters, conditions, dependencies, index=index
        ),
        "overview_layout": overview_layout,
    }


def graph_index_or_api_error(matters, conditions, dependencies):
    try:
        return rules.build_index(matters, conditions, dependencies)
    except RuleError as error:
        raise api_error_for(error) from error


def create_matter(state_path, payload):
    try:
        rules.create_matter(state_path, payload, load=load_state)
    except RuleError as error:
        raise api_error_for(error) from error
    return graph_payload(state_path)


def update_conditions(state_path, matter_id, payload):
    try:
        rules.update_conditions(state_path, matter_id, payload, load=load_state)
    except RuleError as error:
        raise api_error_for(error) from error
    return graph_payload(state_path)


def add_dependency(state_path, payload):
    try:
        rules.add_dependency(state_path, payload, load=load_state)
    except RuleError as error:
        raise api_error_for(error) from error
    return graph_payload(state_path)


def remove_dependency(state_path, payload):
    """Remove one edge. Permitted on a state file that already has a cycle.

    The permission lives in ``rules.remove_dependency`` and is shared with
    the CLI's ``unlink``, so the two surfaces cannot drift apart on the one
    write that repairs a cyclic file.

    The removal is committed before ``graph_payload`` runs. If the file was
    holding more than one cycle, the removal still lands and this call still
    answers 422, because there is no graph to render yet -- the caller
    removes the next edge and asks again.
    """

    try:
        rules.remove_dependency(state_path, payload, load=load_state)
    except RuleError as error:
        raise api_error_for(error) from error
    return graph_payload(state_path)


def run_command(state_path, payload):
    try:
        return run_text_command(state_path, payload)
    except RuleError as error:
        raise api_error_for(error) from error


def run_text_command(state_path, payload):
    text = str(payload.get("text") or payload.get("command") or "").strip()
    if not text:
        raise ApiError("command is required")

    command, _, rest = text.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command == "create":
        if not rest:
            raise ApiError("create requires an expression")
        with rules.state_transaction(state_path, load=load_state) as draft:
            try:
                created = create_matters_from_expression(
                    rest, draft.matters, draft.conditions, draft.dependencies
                )
            except ValueError as error:
                raise ApiError(str(error)) from error
            if rules.has_cycle(draft.matters, draft.conditions, draft.dependencies):
                raise ApiError("created expression would create a cycle")
        return {"type": "create", "created": created, "state": graph_payload(state_path)}

    matters, conditions, dependencies = load_state(state_path)
    index = graph_index_or_api_error(matters, conditions, dependencies)

    if command == "universe":
        return {"type": "universe", "items": sorted(index.universe)}
    if command == "frontier":
        require_matter(rest, matters)
        return {"type": "frontier", "matter": rest, "items": sorted(index.frontier(rest))}
    if command == "horizon":
        require_matter(rest, matters)
        return {"type": "horizon", "matter": rest, "items": sorted(index.horizon(rest))}
    if command == "unlock":
        return {
            "type": "unlock",
            "report": unlock_report(
                matters, conditions, dependencies, index=index
            ),
        }
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
        candidate_payload = graph_payload(next_path)
        with self._lock:
            self._path = next_path
        return candidate_payload


def validate_switch_state_path(state_path):
    raw_path = str(state_path or "").strip()
    if not raw_path:
        raise ApiError("state path is required")

    next_path = resolve_state_path(raw_path)
    if not next_path.exists():
        raise ApiError(f"state file does not exist: {next_path}", HTTPStatus.NOT_FOUND)

    try:
        matters, conditions, dependencies = load_state(next_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ApiError(f"state file is not a valid matters graph: {next_path}") from error

    graph_index_or_api_error(matters, conditions, dependencies)

    return next_path


def switch_state_path(state_paths, payload):
    return state_paths.switch(payload.get("state_path") or payload.get("path"))


def resolve_terminal_workspace(state_path=None, terminal_workspace=None):
    raw_workspace = terminal_workspace or os.environ.get(TERMINAL_WORKSPACE_ENV)
    if raw_workspace:
        return Path(raw_workspace).expanduser()

    if state_path is not None:
        state_parent = resolve_state_path(state_path).parent
        if state_parent.exists():
            return state_parent

    return Path.cwd()


def api_host_allowlist(configured_host, bound_host):
    hosts = {
        normalized
        for host in (configured_host, bound_host)
        if (normalized := normalize_http_host(host)) not in WILDCARD_API_HOSTS
    }
    if any(is_local_api_host(host) for host in hosts) or not hosts:
        hosts.update(LOCAL_API_HOSTS)
    return frozenset(hosts)


def browser_host_for_bind(host):
    normalized = normalize_http_host(host)
    if normalized in WILDCARD_API_HOSTS:
        return "127.0.0.1"
    if ":" in normalized:
        return f"[{normalized}]"
    return normalized


def parse_http_host(value):
    if not value:
        return None, None
    parsed = urlparse(f"//{value.strip()}")
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None, None
    try:
        port = parsed.port
    except ValueError:
        return None, None
    return normalize_http_host(parsed.hostname), port


def normalize_http_host(value):
    if value is None:
        return ""
    host = str(value).strip().lower().rstrip(".")
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        return host


def is_local_api_host(host):
    if host in LOCAL_API_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_remote_bind_host(host):
    """True when binding ``host`` would put the UI on a non-loopback address.

    Wildcards answer False: they keep today's behaviour, and
    ``api_host_allowlist`` already collapses a wildcard bind to loopback-only
    for the Host check. A concrete LAN address does not -- it widens the
    allowlist to exactly that address, so every client on the network passes
    the origin check. That is the case this predicate exists to catch.
    """

    normalized = normalize_http_host(host)
    if normalized in WILDCARD_API_HOSTS:
        return False
    return not is_local_api_host(normalized)


def generate_api_token():
    """Mint the per-run API token.

    Lives only in this process: it is never written to disk, never logged,
    and never printed except inside the launch URL.
    """

    return secrets.token_urlsafe(API_TOKEN_BYTES)


def api_tokens_match(presented, expected):
    """Constant-time token comparison that is total over its inputs.

    ``secrets.compare_digest`` refuses non-ASCII ``str``, and request headers
    arrive latin-1 decoded, so both sides are encoded first: a header full of
    high bytes must answer False, not raise past the ApiError handler and
    turn a rejected request into a 500.
    """

    if not presented or not expected:
        return False
    return secrets.compare_digest(
        presented.encode("utf-8", "surrogatepass"),
        expected.encode("utf-8", "surrogatepass"),
    )


def request_api_token(authorization_header, path):
    """Read the token a request presents.

    ``Authorization: Bearer <token>`` is the only transport an ``/api/``
    request may use. The initial document request may also carry
    ``?token=``, because the launch URL is how the browser first receives
    it. Honouring the query parameter on any other path would put the token
    into ``/api/`` URLs, where it would leak through Referer to the CDN
    origins the page loads code from.
    """

    scheme, _, value = (authorization_header or "").partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()

    parsed = urlparse(path or "")
    if parsed.path in PAGE_LOAD_PATHS:
        return parse_qs(parsed.query).get(API_TOKEN_QUERY_PARAM, [None])[0] or None
    return None


def launch_url(host, port, api_token):
    """The URL ``serve`` prints and opens: the browser's only source of token."""

    query = urlencode({API_TOKEN_QUERY_PARAM: api_token})
    return f"http://{browser_host_for_bind(host)}:{port}/?{query}"


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
    api_token = generate_api_token()
    handler = partial(
        MattersWebHandler,
        state_paths=state_paths,
        terminal_manager=terminal_manager,
        api_token=api_token,
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.api_host_allowlist = api_host_allowlist(host, server.server_address[0])
    # The launch URL is the one place the token is ever rendered. Everything
    # else printed below stays token-free, and nothing writes it to disk.
    url = launch_url(host, server.server_port, api_token)
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
    def __init__(
        self,
        *args,
        state_paths=None,
        terminal_manager=None,
        api_token=None,
        **kwargs,
    ):
        self.state_paths = state_paths or StatePathStore()
        self.terminal_manager = terminal_manager or TerminalManager()
        # No fallback and no generated default: a handler wired without a
        # token answers 401 to every /api/ request. Failing closed is the
        # point -- the alternative is a server that quietly serves shells.
        self.api_token = api_token or None
        super().__init__(*args, directory=str(web_assets_path()), **kwargs)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                self.require_api_request()
            if parsed.path == "/api/state":
                self.write_json(graph_payload(self.current_state_path()))
                return
            match = re.fullmatch(r"/api/terminal/sessions/([^/]+)/output", parsed.path)
            if match:
                query = parse_qs(parsed.query)
                seq = query.get("seq", ["0"])[0]
                session = self.terminal_manager.get(unquote(match.group(1)))
                self.write_json(session.output_since(seq))
                return
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
        try:
            # Ahead of the 404 on purpose: an unauthenticated caller must not
            # be able to map which /api/ paths exist.
            if parsed.path.startswith("/api/"):
                self.require_api_token()
            match = re.fullmatch(r"/api/matters/([^/]+)/conditions", parsed.path)
            if not match:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            matter_id = unquote(match.group(1))
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

    def end_headers(self):
        # Every response: assets, JSON, and the errors http.server writes
        # itself all funnel through here, so none of them can ship without
        # the policy.
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def require_api_mutation_request(self):
        self.require_api_request()
        self.require_json_content_type()

    def require_api_request(self):
        # Token first: authentication decides whether the caller may be told
        # anything at all, including whether its Host header was acceptable.
        self.require_api_token()
        self.require_same_origin_request()

    def require_api_token(self):
        presented = request_api_token(self.headers.get("Authorization"), self.path)
        if not api_tokens_match(presented, self.api_token):
            raise ApiError(API_UNAUTHORIZED_MESSAGE, HTTPStatus.UNAUTHORIZED)

    def require_same_origin_request(self):
        request_origin = self.request_origin()
        if request_origin is None:
            raise ApiError("invalid API request host", HTTPStatus.FORBIDDEN)
        if request_origin[0] not in self.allowed_api_hosts():
            raise ApiError("invalid API request host", HTTPStatus.FORBIDDEN)

        for header in ("Origin", "Referer"):
            value = self.headers.get(header)
            if value and self.http_url_origin(value) != request_origin:
                raise ApiError("cross-origin API request rejected", HTTPStatus.FORBIDDEN)

    def request_origin(self):
        host, port = parse_http_host(self.headers.get("Host"))
        if host is None:
            return None
        port = port or 80
        if port != self.server.server_port:
            return None
        return host, port

    def http_url_origin(self, value):
        parsed = urlparse(value)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
            return None
        try:
            port = parsed.port or 80
        except ValueError:
            return None
        return normalize_http_host(parsed.hostname), port

    def allowed_api_hosts(self):
        configured_hosts = getattr(self.server, "api_host_allowlist", None)
        if configured_hosts is not None:
            return configured_hosts
        return api_host_allowlist(self.server.server_address[0], self.server.server_address[0])

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


# Backward-compatible aliases. These are the rules-layer functions
# themselves, not copies: `matters.web.normalize_condition is
# matters.rules.normalize_condition` must stay true (AC-13). They raise
# RuleError, which every caller in this module translates through
# `api_error_for`.
state_mutation_locks = rules.state_mutation_locks
normalized_matter_id = rules.normalized_matter_id
normalize_condition = rules.normalize_condition
require_condition_index = rules.require_condition_index
dependency_endpoints = rules.dependency_endpoints
require_matter = rules.require_matter
