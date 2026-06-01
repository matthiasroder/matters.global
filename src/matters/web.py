"""Local web UI server for matters graphs."""

import json
import mimetypes
import re
import subprocess
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote, urlparse

from .cli import create_matters_from_expression
from .engine import dependents, frontier, horizon, prerequisites, resolved, truth, universe
from .extraction import extraction_proposal, slugify
from .reports import unlock_report
from .storage import load_state, resolve_state_path, save_state


DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
CODEX_WORKSPACE = Path("/Users/matthias/.openclaw/workspace")
DEFAULT_CODEX_COMMAND = ("codex",)
CODEX_TIMEOUT_SECONDS = 300


class ApiError(ValueError):
    """Validation error that should be returned as an API response."""

    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


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
    matters, conditions, dependencies = load_state(state_path)
    source, target = dependency_endpoints(payload, matters)
    next_dependencies = set(dependencies)
    next_dependencies.add((source, target))
    if has_dependency_cycle(next_dependencies):
        raise ApiError("dependency would create a cycle")

    save_state(matters, conditions, next_dependencies, path=state_path)
    return graph_payload(state_path)


def remove_dependency(state_path, payload):
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

    matters, conditions, dependencies = load_state(state_path)
    command, _, rest = text.partition(" ")
    command = command.lower()
    rest = rest.strip()

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
    if command == "create":
        if not rest:
            raise ApiError("create requires an expression")
        try:
            created = create_matters_from_expression(rest, matters, conditions, dependencies)
        except ValueError as error:
            raise ApiError(str(error)) from error
        if has_dependency_cycle(dependencies):
            raise ApiError("created expression would create a cycle")
        save_state(matters, conditions, dependencies, path=state_path)
        return {"type": "create", "created": created, "state": graph_payload(state_path)}
    if command == "extract":
        if not rest:
            raise ApiError("extract requires source text")
        return {
            "type": "extract",
            "proposal": extraction_proposal(rest, source_type="text", existing_matters=matters),
        }

    raise ApiError(f"unknown command: {command}")


def run_codex_message(
    message,
    command=DEFAULT_CODEX_COMMAND,
    workspace=CODEX_WORKSPACE,
    timeout=CODEX_TIMEOUT_SECONDS,
    runner=subprocess.run,
):
    message = str(message or "").strip()
    if not message:
        raise ApiError("Codex message is required")

    args = [
        *command,
        "exec",
        "--cd",
        str(workspace),
        "--json",
        "-",
    ]

    try:
        completed = runner(
            args,
            input=message,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(workspace),
        )
    except FileNotFoundError as error:
        raise ApiError("Codex CLI was not found", HTTPStatus.BAD_GATEWAY) from error
    except subprocess.TimeoutExpired as error:
        raise ApiError("Codex timed out", HTTPStatus.GATEWAY_TIMEOUT) from error

    events = parse_codex_events(completed.stdout)
    final_response = codex_final_response(events) or completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or final_response or "Codex exited with an error"
        raise ApiError(detail, HTTPStatus.BAD_GATEWAY)

    return {
        "type": "codex",
        "workspace": str(workspace),
        "response": final_response,
        "events": summarize_codex_events(events),
        "stderr": completed.stderr.strip(),
    }


def parse_codex_events(output):
    events = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"type": "text", "text": line})
    return events


def codex_final_response(events):
    for event in reversed(events):
        value = codex_event_text(event)
        if value:
            return value
    return ""


def summarize_codex_events(events):
    summary = []
    for event in events[-20:]:
        item = {"type": event.get("type", "unknown")}
        value = codex_event_text(event)
        if value:
            item["text"] = value[:500]
        summary.append(item)
    return summary


def codex_event_text(event):
    for key in ("message", "text", "content", "last_message"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    payload = event.get("payload")
    if isinstance(payload, str) and payload.strip():
        return payload.strip()

    item = event.get("item")
    if isinstance(item, dict):
        for key in ("message", "text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def serve(state_path=None, host=DEFAULT_WEB_HOST, port=DEFAULT_WEB_PORT, open_browser=True):
    resolved_state_path = resolve_state_path(state_path)
    handler = partial(MattersWebHandler, state_path=resolved_state_path)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"Serving matters web UI at {url}")
    print(f"State file: {resolved_state_path}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping matters web UI")
    finally:
        server.server_close()


class MattersWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state_path=None, **kwargs):
        self.state_path = state_path
        super().__init__(*args, directory=str(web_assets_path()), **kwargs)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.write_json(graph_payload(self.state_path))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/matters":
                self.write_json(create_matter(self.state_path, self.read_json()), HTTPStatus.CREATED)
                return
            if parsed.path == "/api/dependencies":
                self.write_json(add_dependency(self.state_path, self.read_json()), HTTPStatus.CREATED)
                return
            if parsed.path == "/api/command":
                self.write_json(run_command(self.state_path, self.read_json()))
                return
            if parsed.path == "/api/codex":
                payload = self.read_json()
                self.write_json(run_codex_message(payload.get("message") or payload.get("text")))
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
            self.write_json(update_conditions(self.state_path, matter_id, self.read_json()))
        except ApiError as error:
            self.write_error(error)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/dependencies":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self.write_json(remove_dependency(self.state_path, self.read_json()))
        except ApiError as error:
            self.write_error(error)

    def guess_type(self, path):
        if path.endswith(".js"):
            return "text/javascript"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
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
