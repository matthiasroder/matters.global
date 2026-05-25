"""Small CLI wrapper around the matters engine."""

import argparse

from .engine import frontier, horizon, universe
from .storage import load_state, resolve_state_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="matters")
    parser.add_argument("--state", help="Path to matters JSON state file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("state-path", help="Print the resolved state path.")
    subparsers.add_parser("universe", help="Print globally actionable matters.")

    frontier_parser = subparsers.add_parser("frontier", help="Print a matter frontier.")
    frontier_parser.add_argument("matter")

    horizon_parser = subparsers.add_parser("horizon", help="Print a matter horizon.")
    horizon_parser.add_argument("matter")

    args = parser.parse_args(argv)

    if args.command == "state-path":
        print(resolve_state_path(args.state))
        return 0

    matters, conditions, dependencies = load_state(args.state)

    if args.command == "universe":
        print_lines(universe(matters, conditions, dependencies))
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


if __name__ == "__main__":
    raise SystemExit(main())
