"""Small CLI wrapper around the matters engine."""

import argparse
import json
import sys

from .engine import frontier, horizon, universe
from .extraction import extraction_proposal
from .reports import format_unlock_report, unlock_report
from .sharing import merge_public_state, public_state
from .storage import load_state, resolve_state_path


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

    args = parser.parse_args(argv)

    if args.command == "state-path":
        print(resolve_state_path(args.state))
        return 0

    matters, conditions, dependencies = load_state(args.state)

    if args.command == "extract":
        source_text = read_source_text(args.source)
        print(
            json.dumps(
                extraction_proposal(
                    source_text,
                    source_type=args.source_type,
                    existing_matters=matters,
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


if __name__ == "__main__":
    raise SystemExit(main())
