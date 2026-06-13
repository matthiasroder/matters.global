"""Small CLI wrapper around the matters engine."""

import argparse
import json
import sys

from .engine import frontier, horizon, universe
from .extraction import slugify
from .llm_extraction import build_extraction_proposal
from .reports import format_unlock_report, unlock_report
from .sharing import merge_public_state, public_state
from .storage import load_state, resolve_state_path, save_state


def main(argv=None):
    state_parent = argparse.ArgumentParser(add_help=False)
    state_parent.add_argument("--state", help="Path to matters JSON state file.")

    parser = argparse.ArgumentParser(prog="matters")
    parser.add_argument("--state", help="Path to matters JSON state file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "state-path", parents=[state_parent], help="Print the resolved state path."
    )
    subparsers.add_parser(
        "universe", parents=[state_parent], help="Print globally actionable matters."
    )
    create_parser = subparsers.add_parser(
        "create",
        parents=[state_parent],
        help="Create matters from a compact expression.",
    )
    create_parser.add_argument(
        "expression",
        nargs="*",
        help=(
            "Matter expression. Quote dependency chains that contain '>', "
            "for example: 'goal (condition) > prerequisite'."
        ),
    )
    unlock_parser = subparsers.add_parser(
        "unlock",
        parents=[state_parent],
        help="Print a short report of actionable matters and next actions.",
    )
    unlock_parser.add_argument(
        "--json", action="store_true", help="Print the unlock report as JSON."
    )

    extract_parser = subparsers.add_parser(
        "extract",
        parents=[state_parent],
        help="Extract candidate matters from a text source without saving them.",
    )
    extract_parser.add_argument(
        "source",
        nargs="?",
        default="-",
        help="Text file to read, or '-' for stdin.",
    )
    extract_parser.add_argument(
        "--source-type",
        default="text",
        help="Source label such as pdf, conversation, blog_post, notes, or text.",
    )
    extract_parser.add_argument(
        "--model",
        default=None,
        help="Override the extraction model id (default: claude-sonnet-4-6 or "
        "MATTERS_EXTRACT_MODEL).",
    )
    extract_parser.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Use only the deterministic marker engine; never call the LLM.",
    )

    export_public_parser = subparsers.add_parser(
        "export-public",
        parents=[state_parent],
        help="Print a sanitized public state from a visibility JSON file.",
    )
    export_public_parser.add_argument(
        "--visibility",
        required=True,
        help="JSON file mapping matter ids to public, shared, or private.",
    )

    merge_public_parser = subparsers.add_parser(
        "merge-public",
        parents=[state_parent],
        help="Merge an edited public state into a private state and print the result.",
    )
    merge_public_parser.add_argument(
        "--public-state",
        required=True,
        help="Edited public matters state to merge.",
    )
    merge_public_parser.add_argument(
        "--visibility",
        required=True,
        help="JSON file mapping matter ids to public, shared, or private.",
    )

    frontier_parser = subparsers.add_parser(
        "frontier", parents=[state_parent], help="Print a matter frontier."
    )
    frontier_parser.add_argument("matter")

    horizon_parser = subparsers.add_parser(
        "horizon", parents=[state_parent], help="Print a matter horizon."
    )
    horizon_parser.add_argument("matter")

    web_parser = subparsers.add_parser(
        "web", parents=[state_parent], help="Start the local browser graph UI."
    )
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    web_parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    web_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the browser automatically.",
    )

    args = parser.parse_args(argv)

    if args.command == "state-path":
        print(resolve_state_path(args.state))
        return 0

    if args.command == "web":
        from .web import serve

        serve(
            state_path=args.state,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
        return 0

    matters, conditions, dependencies = load_state(args.state)

    if args.command == "create":
        try:
            created = create_matters_from_expression(
                read_create_expression(args.expression),
                matters,
                conditions,
                dependencies,
            )
        except ValueError as error:
            parser.error(str(error))

        save_state(matters, conditions, dependencies, path=args.state)
        print_create_summary(created)
        return 0

    if args.command == "extract":
        source_text = read_source_text(args.source)
        print(
            json.dumps(
                build_extraction_proposal(
                    source_text,
                    source_type=args.source_type,
                    existing_matters=matters,
                    use_llm=args.use_llm,
                    model=args.model,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "export-public":
        with open(args.visibility) as f:
            visibility = json.load(f)
        print(
            json.dumps(
                public_state(matters, conditions, dependencies, visibility), indent=2
            )
        )
        return 0

    if args.command == "merge-public":
        with open(args.visibility) as f:
            visibility = json.load(f)
        with open(args.public_state) as f:
            incoming_state = json.load(f)
        print(
            json.dumps(
                merge_public_state(
                    matters, conditions, dependencies, visibility, incoming_state
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "universe":
        print_lines(universe(matters, conditions, dependencies))
        return 0

    if args.command == "unlock":
        report = unlock_report(matters, conditions, dependencies)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(format_unlock_report(report))
        return 0

    if args.command == "frontier":
        print_lines(frontier(args.matter, conditions, dependencies))
        return 0

    if args.command == "horizon":
        print_lines(horizon(args.matter, conditions, dependencies))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def print_lines(items):
    for item in sorted(items):
        print(item)


def read_source_text(source):
    if source == "-":
        return sys.stdin.read()
    with open(source) as f:
        return f.read()


def read_create_expression(parts):
    if parts:
        return " ".join(parts)
    if sys.stdin.isatty():
        raise ValueError("provide a matter expression or pipe one on stdin")
    return sys.stdin.read()


def create_matters_from_expression(expression, matters, conditions, dependencies):
    parsed_matters = parse_create_expression(expression)
    ids = [matter["id"] for matter in parsed_matters]
    duplicate_ids = sorted({matter_id for matter_id in ids if ids.count(matter_id) > 1})
    if duplicate_ids:
        raise ValueError("duplicate matter ids in expression: " + ", ".join(duplicate_ids))

    existing_ids = sorted(set(ids) & matters)
    if existing_ids:
        raise ValueError("matter already exists: " + ", ".join(existing_ids))

    for parsed_matter in parsed_matters:
        matter_id = parsed_matter["id"]
        matters.add(matter_id)
        conditions[matter_id] = [
            {"label": parsed_matter["condition"], "truth": False}
        ]

    for prerequisite, dependent in zip(parsed_matters[1:], parsed_matters):
        dependencies.add((prerequisite["id"], dependent["id"]))

    return parsed_matters


def parse_create_expression(expression):
    segments = [segment.strip() for segment in expression.split(">")]
    segments = [segment for segment in segments if segment]
    if not segments:
        raise ValueError("matter expression is empty")

    return [parse_create_segment(segment) for segment in segments]


def parse_create_segment(segment):
    name = segment
    condition = None

    if segment.endswith(")"):
        start = segment.rfind("(")
        if start > 0:
            name = segment[:start].strip()
            condition = segment[start + 1 : -1].strip()

    if not name:
        raise ValueError("matter name cannot be empty")

    if not condition:
        condition = f"Resolved: {name}"

    return {"id": slugify(name), "name": name, "condition": condition}


def print_create_summary(created):
    print("Created matters")
    for matter in created:
        print(f"- {matter['id']}: {matter['name']}")
        print(f"  - condition: {matter['condition']}")

    if len(created) > 1:
        print("")
        print("Dependencies")
        for prerequisite, dependent in zip(created[1:], created):
            print(f"- {prerequisite['id']} -> {dependent['id']}")


if __name__ == "__main__":
    raise SystemExit(main())
