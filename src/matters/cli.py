"""Small CLI wrapper around the matters engine."""

import argparse
import json
import sys

from . import rules
from .engine import frontier, horizon, universe
from .llm_extraction import build_extraction_proposal
from .llm import (
    ConfigError,
    GenerationError,
    config_diagnostics,
    load_config,
    resolve_config_path,
)
from .reports import format_unlock_report, unlock_report
from .rules import (
    RuleError,
    create_matters_from_expression,
    parse_create_expression,
    parse_create_segment,
)
from .sharing import merge_public_state, public_state
from .storage import load_state, resolve_state_path
from .tots import TotsError, build_tots_proposal


# One string, two parsers: the root parser and the per-subcommand parent must
# say the same thing about the same flag (R7).
STATE_HELP = (
    "Path to matters JSON state file. Write commands lock an empty sidecar "
    "file named .<state-file-name>.lock alongside it, which you may want to "
    "gitignore when the state file lives inside a repository."
)


def main(argv=None):
    state_parent = argparse.ArgumentParser(add_help=False)
    state_parent.add_argument(
        "--state", default=argparse.SUPPRESS, help=STATE_HELP
    )
    state_parent.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Path to Matters TOML configuration.",
    )

    parser = argparse.ArgumentParser(prog="matters")
    parser.add_argument("--state", help=STATE_HELP)
    parser.add_argument("--config", help="Path to Matters TOML configuration.")
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
        help="Override the extraction profile's model id.",
    )
    extract_parser.add_argument(
        "--llm-profile",
        default=None,
        help="Select a configured model profile for extraction.",
    )
    extract_parser.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Use only the deterministic marker engine; never call the LLM.",
    )

    tots_parser = subparsers.add_parser(
        "tots",
        parents=[state_parent],
        help="Explore a bounded hypothesis tree for an unresolved matter.",
    )
    tots_parser.add_argument("matter", help="Unresolved matter id to explore.")
    tots_parser.add_argument(
        "--context",
        default=None,
        help="Optional evidence text file, or '-' for stdin.",
    )
    tots_parser.add_argument(
        "--model",
        default=None,
        help="Override the ToTs profile's model id.",
    )
    tots_parser.add_argument(
        "--llm-profile",
        default=None,
        help="Select a configured model profile for ToTs.",
    )
    tots_parser.add_argument("--breadth", type=int, default=4)
    tots_parser.add_argument("--depth", type=int, default=2)
    tots_parser.add_argument("--max-candidates", type=int, default=8)
    tots_parser.add_argument("--max-comparisons", type=int, default=24)

    config_parser = subparsers.add_parser(
        "config",
        parents=[state_parent],
        help="Inspect model-provider configuration without generating content.",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )
    config_subparsers.add_parser(
        "path", parents=[state_parent], help="Print the resolved configuration path."
    )
    config_check_parser = config_subparsers.add_parser(
        "check", parents=[state_parent], help="Print sanitized provider readiness."
    )
    config_check_parser.add_argument(
        "--profile", default=None, help="Check only one named profile."
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
    web_parser.add_argument(
        "--terminal-workspace",
        default=None,
        help="Directory to use for the web terminal. Defaults to the state directory or cwd.",
    )
    web_parser.add_argument(
        "--terminal-shell",
        default=None,
        help="Shell executable to use for the web terminal. Defaults to $SHELL or /bin/sh.",
    )

    condition_ref_help = (
        "Condition number as shown by `matters show <matter>` (1-based), or "
        "an exact condition label. An all-digit argument is always a number."
    )
    label_help = (
        "Condition label. Leading and trailing whitespace is stripped and an "
        "empty label gets the same generated placeholder the web UI uses. "
        "Pass a label that looks like a flag after `--`, for example: "
        "matters add-condition a -- --force."
    )

    mark_parser = subparsers.add_parser(
        "mark",
        parents=[state_parent],
        help="Set one condition of a matter to true or false.",
    )
    mark_parser.add_argument("matter", help="Matter id that owns the condition.")
    mark_parser.add_argument("condition", help=condition_ref_help)
    mark_parser.add_argument(
        "truth",
        choices=("true", "false"),
        help=(
            "Exactly `true` or `false`. Other spellings such as `1`, `yes` or "
            "`True` are rejected."
        ),
    )

    add_condition_parser = subparsers.add_parser(
        "add-condition",
        parents=[state_parent],
        help="Append a condition to a matter.",
    )
    add_condition_parser.add_argument("matter", help="Matter id to append to.")
    add_condition_parser.add_argument("label", help=label_help)

    edit_condition_parser = subparsers.add_parser(
        "edit-condition",
        parents=[state_parent],
        help="Rename a condition, keeping its truth value and position.",
    )
    edit_condition_parser.add_argument(
        "matter", help="Matter id that owns the condition."
    )
    edit_condition_parser.add_argument("condition", help=condition_ref_help)
    edit_condition_parser.add_argument("label", help=label_help)

    delete_condition_parser = subparsers.add_parser(
        "delete-condition",
        parents=[state_parent],
        help="Delete a condition from a matter.",
    )
    delete_condition_parser.add_argument(
        "matter", help="Matter id that owns the condition."
    )
    delete_condition_parser.add_argument("condition", help=condition_ref_help)
    delete_condition_parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Confirm deleting the last condition of a matter, which makes it "
            "count as resolved and can unblock its dependents."
        ),
    )

    link_parser = subparsers.add_parser(
        "link",
        parents=[state_parent],
        help="Record that one matter needs another matter first.",
    )
    link_parser.add_argument(
        "matter", help="Matter id that needs the prerequisite."
    )
    link_parser.add_argument(
        "prerequisite",
        help="Matter id that must be resolved before the first one can be.",
    )

    unlink_parser = subparsers.add_parser(
        "unlink",
        parents=[state_parent],
        help="Remove the dependency of one matter on another.",
    )
    unlink_parser.add_argument(
        "matter", help="Matter id that currently needs the prerequisite."
    )
    unlink_parser.add_argument(
        "prerequisite", help="Matter id it should stop needing."
    )

    delete_matter_parser = subparsers.add_parser(
        "delete-matter",
        parents=[state_parent],
        help="Delete a matter together with its conditions.",
    )
    delete_matter_parser.add_argument("matter", help="Matter id to delete.")
    delete_matter_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the deletion. Always required by this command.",
    )
    delete_matter_parser.add_argument(
        "--cascade",
        action="store_true",
        help=(
            "Also delete every dependency the matter takes part in, including "
            "the ones that make other matters require it."
        ),
    )

    show_parser = subparsers.add_parser(
        "show",
        parents=[state_parent],
        help="Print the stored facts about one matter.",
    )
    show_parser.add_argument("matter", help="Matter id to describe.")
    show_parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print the matter as JSON, where conditions carry a 0-based "
            "`index` matching the web API and the stored order, while the "
            "text output numbers the same conditions from 1 because that is "
            "the number you pass back as a condition reference."
        ),
    )

    list_parser = subparsers.add_parser(
        "list",
        parents=[state_parent],
        help="Print every matter id, sorted, one per line.",
    )
    list_parser.add_argument(
        "--json", action="store_true", help="Print the matter ids as a JSON array."
    )

    args = parser.parse_args(argv)

    if args.command == "config":
        try:
            if args.config_command == "path":
                path, _ = resolve_config_path(args.config)
                print(path)
            else:
                config = load_config(args.config)
                print(
                    json.dumps(
                        config_diagnostics(config, profile_name=args.profile), indent=2
                    )
                )
        except ConfigError as error:
            parser.error(str(error))
        return 0

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
            terminal_workspace=args.terminal_workspace,
            terminal_shell=args.terminal_shell,
        )
        return 0

    # Every write verb dispatches ABOVE the unconditional load_state below.
    # That load raises json.JSONDecodeError on a malformed file before any
    # branch runs, which would leak a traceback (AC-16/F-11). Each branch
    # loads inside its own rules transaction instead and never touches the
    # pre-loaded triple.
    if args.command == "mark":
        # Pass a real bool, never the raw word: rules.set_condition_truth
        # funnels its argument through engine.truth, and bool("false") is
        # True. argparse's choices already guarantee one of the two words.
        try:
            result = rules.set_condition_truth(
                args.state, args.matter, args.condition, args.truth == "true"
            )
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        print(
            f"{result['matter']}: condition {result['position']} "
            f"{quoted(result['label'])} is now {truth_word(result['truth'])}"
        )
        return 0

    if args.command == "add-condition":
        try:
            result = rules.add_condition(args.state, args.matter, args.label)
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        print(
            f"{result['matter']}: added condition {result['position']} "
            f"{quoted(result['label'])} ({truth_word(result['truth'])})"
        )
        return 0

    if args.command == "edit-condition":
        try:
            result = rules.edit_condition_label(
                args.state, args.matter, args.condition, args.label
            )
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        print(
            f"{result['matter']}: condition {result['position']} renamed "
            f"{quoted(result['previous_label'])} -> {quoted(result['label'])}"
        )
        return 0

    if args.command == "delete-condition":
        try:
            result = rules.delete_condition(
                args.state, args.matter, args.condition, confirmed=args.yes
            )
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        print(
            f"{result['matter']}: deleted condition {result['position']} "
            f"{quoted(result['label'])}"
        )
        if result["emptied"]:
            print(
                f"{result['matter']} has no conditions left and now counts as "
                f"resolved; unblocked: "
                f"{rules.format_matter_list(result['unblocked'])}"
            )
        return 0

    if args.command == "link":
        # `matters link a b` reads "a needs b", so the CLI's `matter` is the
        # dependent and `prerequisite` is what it waits on. The edge is stored
        # source-first by rules.link; the CLI never builds a tuple itself.
        try:
            result = rules.link(args.state, args.matter, args.prerequisite)
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        verb = "now requires" if result["changed"] else "already requires"
        print(f"{result['dependent']} {verb} {result['prerequisite']}")
        return 0

    if args.command == "unlink":
        try:
            result = rules.unlink(args.state, args.matter, args.prerequisite)
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        verb = (
            "no longer requires" if result["changed"] else "already does not require"
        )
        print(f"{result['dependent']} {verb} {result['prerequisite']}")
        return 0

    if args.command == "delete-matter":
        # --yes is required whatever the shape of the graph; --cascade is the
        # separate answer to "other matters still require this one".
        try:
            result = rules.delete_matter(
                args.state,
                args.matter,
                cascade=args.cascade,
                confirmed=args.yes,
            )
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        removed = rules.format_dependency_count(len(result["removed_edges"]))
        print(f"deleted matter {result['matter']} (removed {removed})")
        if result["unblocked"]:
            print(f"unblocked: {rules.format_matter_list(result['unblocked'])}")
        return 0

    # The two read verbs dispatch here for the same reason the write verbs do:
    # the unconditional load below raises before any branch runs on a malformed
    # file (F-11). They take no lock and build no index, so they also leave no
    # sidecar behind (AC-10).
    if args.command == "show":
        try:
            described = rules.describe_matter(args.state, args.matter)
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        if args.json:
            print(json.dumps(described, indent=2))
        else:
            print_matter(described)
        return 0

    if args.command == "list":
        try:
            matter_ids = rules.list_matters(args.state)
        except (RuleError, ValueError, OSError) as error:
            parser.error(str(error))
        if args.json:
            print(json.dumps(matter_ids, indent=2))
        else:
            print_lines(matter_ids)
        return 0

    matters, conditions, dependencies = load_state(args.state)

    if args.command == "create":
        # The transaction owns the load, the advisory lock and the save.
        # require_acyclic=False and require_exists=False keep `create`
        # behaving exactly as it did on a cyclic or missing state file
        # (AC-21); the only new side effect is the sidecar lock file.
        try:
            with rules.state_transaction(
                args.state, require_exists=False, require_acyclic=False
            ) as draft:
                created = create_matters_from_expression(
                    read_create_expression(args.expression),
                    draft.matters,
                    draft.conditions,
                    draft.dependencies,
                )
        except ValueError as error:
            parser.error(str(error))

        print_create_summary(created)
        return 0

    if args.command == "extract":
        source_text = read_source_text(args.source)
        try:
            proposal = build_extraction_proposal(
                source_text,
                source_type=args.source_type,
                existing_matters=matters,
                use_llm=args.use_llm,
                model=args.model,
                config_path=args.config,
                llm_profile=args.llm_profile,
            )
        except (ConfigError, GenerationError, TypeError) as error:
            parser.error(str(error))
        print(json.dumps(proposal, indent=2))
        return 0

    if args.command == "tots":
        context_text = read_source_text(args.context) if args.context else ""
        try:
            proposal = build_tots_proposal(
                args.matter,
                matters,
                conditions,
                dependencies,
                context_text=context_text,
                breadth=args.breadth,
                depth=args.depth,
                max_candidates=args.max_candidates,
                max_comparisons=args.max_comparisons,
                model=args.model,
                config_path=args.config,
                llm_profile=args.llm_profile,
            )
        except TotsError as error:
            parser.error(str(error))
        print(json.dumps(proposal, indent=2))
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


def quoted(label):
    """Wrap a label in straight double quotes, printing it raw.

    Deliberately not ``json.dumps``: that escapes non-ASCII to ``\\uXXXX``
    and a label like ``提交申请`` must round-trip visibly unchanged (E-6).
    """

    return f'"{label}"'


def truth_word(value):
    return "true" if value else "false"


def print_matter(described):
    """Print the stored facts about one matter, for a person to read.

    Conditions are numbered from **1** here. This is the surface a human
    reads and then types back as a condition reference, so it uses the same
    base the reference does. ``show --json`` reports the very same conditions
    with a **0-based** ``index``, because that is a machine surface and agrees
    with the web API payload field and the stored list position. The number
    below is taken from the print position, never by arithmetic on the
    machine-facing index, so the two bases cannot drift into one another.

    Stored facts only: no resolved, actionable or blocked status (D6). Those
    route through the diverging engine and index code paths, and ``show`` must
    keep working on a state file that contains a cycle, which rules out
    building a graph index here at all.
    """

    print(described["id"])
    print("conditions:")
    if described["conditions"]:
        for number, condition in enumerate(described["conditions"], start=1):
            box = "[x]" if condition["truth"] else "[ ]"
            print(f"  {number}. {box} {condition['label']}")
    else:
        print("  none")
    print("requires:")
    print_matter_section(described["prerequisites"])
    print("required by:")
    print_matter_section(described["dependents"])


def print_matter_section(matter_ids):
    """Print one indented id per line, or ``  none`` for an empty section."""

    if not matter_ids:
        print("  none")
        return
    for matter_id in sorted(matter_ids):
        print(f"  {matter_id}")


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
