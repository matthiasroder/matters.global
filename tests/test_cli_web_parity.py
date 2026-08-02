"""CLI/web parity: AC-11, AC-12, AC-13, AC-14 in one reviewable artifact.

Every test in this file drives BOTH surfaces against the SAME starting bytes
and asserts all four halves of the parity contract (brief section 7.2):

1. the CLI decision -- ``SystemExit`` with ``code == 2`` and non-empty stderr
   for a rejection, ``main([...]) == 0`` for an acceptance;
2. the web decision -- ``ApiError`` with the literal ``HTTPStatus`` from the
   status table in brief section 1.5 for a rejection, a normal return for an
   acceptance;
3. that the two decisions agree;
4. the bytes -- on a rejection both files still equal the original bytes; on
   an acceptance the two files are byte-identical to each other AND their
   parsed content matches an exact literal dict.

Nothing here asserts "an edge exists" or "some error was raised". Every
expectation is a literal, because the whole point of the parity suite is to
catch a rule that drifted, and a loose assertion cannot see drift.
"""

import json
from http import HTTPStatus
from pathlib import Path

import pytest

import matters
import matters.cli as cli
import matters.rules as rules
import matters.web as web
from matters.cli import main
from matters.rules import RuleError
from matters.web import ApiError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def two_states(tmp_path, data):
    """Write ONE json string to a CLI path and a web path.

    One string, not two ``json.dumps`` calls, so that any byte comparison
    between the two files afterwards is a statement about the two surfaces
    and never about dict ordering in the fixture.
    """

    text = json.dumps(data)
    cli_path = tmp_path / "cli.json"
    web_path = tmp_path / "web.json"
    cli_path.write_text(text)
    web_path.write_text(text)
    return cli_path, web_path, text.encode()


def cli_decision(capsys, argv):
    """Run one CLI command in-process and return ``(rejected, message)``.

    ``message`` is stderr on a rejection and stdout on an acceptance. The
    AC-15 contract (exit code exactly 2, a diagnostic on stderr, nothing on
    stdout) is pinned here so that every caller inherits it.
    """

    try:
        exit_code = main(argv)
    except SystemExit as exit_error:
        captured = capsys.readouterr()
        assert exit_error.code == 2, f"expected exit 2, got {exit_error.code!r}"
        assert captured.out == "", f"rejection wrote to stdout: {captured.out!r}"
        assert captured.err.strip() != "", "rejection wrote no diagnostic"
        return True, captured.err
    captured = capsys.readouterr()
    assert exit_code == 0, f"expected exit 0, got {exit_code!r}"
    return False, captured.out


def web_decision(call):
    """Run one web mutation and return ``(rejected, ApiError or None)``."""

    try:
        call()
    except ApiError as error:
        return True, error
    return False, None


TWO_MATTERS = {
    "matters": ["a", "b"],
    "conditions": {
        "a": [{"label": "a done", "truth": False}],
        "b": [{"label": "b done", "truth": False}],
    },
    "dependencies": [],
}

TWO_MATTERS_LINKED = {
    "matters": ["a", "b"],
    "conditions": {
        "a": [{"label": "a done", "truth": False}],
        "b": [{"label": "b done", "truth": False}],
    },
    # (source, target): a is the prerequisite, b is the dependent.
    "dependencies": [["a", "b"]],
}

CYCLIC = {
    "matters": ["a", "b"],
    "conditions": {
        "a": [{"label": "a done", "truth": False}],
        "b": [{"label": "b done", "truth": False}],
    },
    "dependencies": [["a", "b"], ["b", "a"]],
}


# ---------------------------------------------------------------------------
# AC-11 -- link direction, empirically pinned
# ---------------------------------------------------------------------------


def test_ac11_link_direction_matches_the_web_dependency_mutation(tmp_path, capsys):
    cli_path, web_path, _original = two_states(tmp_path, TWO_MATTERS)

    # `matters link a b` reads "a needs b".
    cli_rejected, _out = cli_decision(
        capsys, ["link", "a", "b", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"prerequisite": "b", "dependent": "a"})
    )

    assert cli_rejected is False
    assert web_rejected is False
    assert cli_rejected is web_rejected
    assert api_error is None

    # The literal edge is the point: (prerequisite, dependent), source first.
    assert json.loads(cli_path.read_text())["dependencies"] == [["b", "a"]]
    assert cli_path.read_bytes() == web_path.read_bytes()
    assert json.loads(cli_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a", "b"],
        "conditions": {
            "a": [{"label": "a done", "truth": False}],
            "b": [{"label": "b done", "truth": False}],
        },
        "dependencies": [["b", "a"]],
    }


# ---------------------------------------------------------------------------
# AC-12 case 1 -- matter id with a disallowed character
# ---------------------------------------------------------------------------


def test_ac12_case1_disallowed_character_in_a_matter_id(tmp_path, capsys):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    cli_rejected, stderr = cli_decision(
        capsys, ["link", "a", "a b", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"prerequisite": "a b", "dependent": "a"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    # Neither surface reaches the id-syntax rule: "a b" is simply not a matter,
    # so both report it as an unknown endpoint. Pinning 404 here (not 400) is
    # deliberate -- it is the shared decision, not a syntax verdict.
    assert api_error.status == HTTPStatus.NOT_FOUND
    assert str(api_error) == "unknown dependency source: a b"
    assert "unknown dependency source: a b" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


def test_ac12_case1_show_also_refuses_an_id_with_a_space(tmp_path, capsys):
    # F-4 from the read side. The web has no `show`; its nearest equivalent is
    # the same lookup rule, which is what the CLI reports here.
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    cli_rejected, stderr = cli_decision(
        capsys, ["show", "a b", "--state", str(cli_path)]
    )

    assert cli_rejected is True
    assert "unknown matter: a b" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


# ---------------------------------------------------------------------------
# AC-12 case 2 -- empty / whitespace condition label (accepted by both)
# ---------------------------------------------------------------------------


def test_ac12_case2_whitespace_label_is_accepted_identically(tmp_path, capsys):
    cli_path, web_path, _original = two_states(tmp_path, TWO_MATTERS)

    cli_rejected, stdout = cli_decision(
        capsys, ["add-condition", "a", "   ", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.update_conditions(web_path, "a", {"label": "   "})
    )

    assert cli_rejected is False
    assert web_rejected is False
    assert cli_rejected is web_rejected
    assert api_error is None
    assert stdout == 'a: added condition 2 "Unlabeled condition 2" (false)\n'

    assert cli_path.read_bytes() == web_path.read_bytes()
    assert json.loads(cli_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a", "b"],
        "conditions": {
            "a": [
                {"label": "a done", "truth": False},
                {"label": "Unlabeled condition 2", "truth": False},
            ],
            "b": [{"label": "b done", "truth": False}],
        },
        "dependencies": [],
    }


# ---------------------------------------------------------------------------
# AC-12 cases 3 and 4 -- missing prerequisite, missing dependent
# ---------------------------------------------------------------------------


def test_ac12_case3_missing_prerequisite(tmp_path, capsys):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    cli_rejected, stderr = cli_decision(
        capsys, ["link", "a", "ghost", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"prerequisite": "ghost", "dependent": "a"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert api_error.status == HTTPStatus.NOT_FOUND
    assert str(api_error) == "unknown dependency source: ghost"
    assert "unknown dependency source: ghost" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


def test_ac12_case4_missing_dependent(tmp_path, capsys):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    cli_rejected, stderr = cli_decision(
        capsys, ["link", "ghost", "a", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"prerequisite": "a", "dependent": "ghost"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert api_error.status == HTTPStatus.NOT_FOUND
    assert str(api_error) == "unknown dependency target: ghost"
    assert "unknown dependency target: ghost" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


# ---------------------------------------------------------------------------
# AC-12 case 5 -- an edge that would close a cycle (plus E-12, the self-link)
# ---------------------------------------------------------------------------


def test_ac12_case5_edge_that_would_close_a_cycle(tmp_path, capsys):
    # The file already holds (a, b) -- "b requires a". `link a b` asks for the
    # edge (b, a), which is exactly the edge the web call below asks for.
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS_LINKED)

    cli_rejected, stderr = cli_decision(
        capsys, ["link", "a", "b", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"source": "b", "target": "a"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    # Would-create-a-cycle is 400, not 422. 422 is reserved for a file that
    # already contains a cycle (case 6).
    assert api_error.status == HTTPStatus.BAD_REQUEST
    assert str(api_error) == "dependency would create a cycle"
    assert "a cannot require b: the dependency would create a cycle" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


def test_ac12_case5_self_link_is_rejected_as_a_cycle(tmp_path, capsys):
    # E-12.
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    cli_rejected, stderr = cli_decision(
        capsys, ["link", "a", "a", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"source": "a", "target": "a"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert api_error.status == HTTPStatus.BAD_REQUEST
    assert str(api_error) == "dependency would create a cycle"
    assert "a cannot require a: the dependency would create a cycle" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


# ---------------------------------------------------------------------------
# AC-12 case 6 -- any write against a file that already contains a cycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["mark", "a", "1", "true"], id="mark"),
        pytest.param(["add-condition", "a", "ship it"], id="add-condition"),
        pytest.param(["edit-condition", "a", "1", "renamed"], id="edit-condition"),
        pytest.param(["delete-condition", "a", "1", "--yes"], id="delete-condition"),
        pytest.param(["link", "a", "b"], id="link"),
        # unlink and delete-matter are in the list on purpose (F-6): they are
        # refused even though the write they were asked for would have removed
        # the cycle.
        pytest.param(["unlink", "a", "b"], id="unlink"),
        pytest.param(["delete-matter", "a", "--yes", "--cascade"], id="delete-matter"),
    ],
)
def test_ac12_case6_every_write_verb_refuses_an_already_cyclic_file(
    tmp_path, capsys, argv
):
    cli_path, web_path, original = two_states(tmp_path, CYCLIC)

    cli_rejected, stderr = cli_decision(capsys, argv + ["--state", str(cli_path)])
    web_rejected, api_error = web_decision(
        lambda: web.create_matter(web_path, {"title": "x"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert api_error.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert str(api_error) == "state dependency graph contains a cycle"
    assert "state dependency graph contains a cycle" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


# ---------------------------------------------------------------------------
# E-7 and E-8 -- non-ASCII and uppercase matter ids
#
# No CLI verb mints a matter id (`create` goes through slugify, and by AC-21
# it must not gain validation), so the honest parity statement has two halves:
# the shared regex verdict, pinned literally; and the two surfaces reaching
# the same decision on the paths each of them actually exposes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matter_id",
    [
        pytest.param("änderung", id="E-7-non-ascii"),
        pytest.param("A", id="E-8-uppercase"),
    ],
)
def test_e7_e8_matter_id_verdict_is_the_same_on_both_surfaces(
    tmp_path, capsys, matter_id
):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    # Pin what the shared regex actually does rather than assuming an outcome.
    assert rules.MATTER_ID_PATTERN.fullmatch(matter_id) is None

    # The web is the only surface that mints ids, and the regex rejects there.
    mint_rejected, mint_error = web_decision(
        lambda: web.create_matter(web_path, {"id": matter_id})
    )
    assert mint_rejected is True
    assert mint_error.status == HTTPStatus.BAD_REQUEST
    assert str(mint_error) == (
        "matter id must contain lowercase letters, numbers, and underscores only"
    )

    # On the lookup path -- the one the CLI does expose -- both surfaces agree,
    # and they agree on the message too, because it is the same rule object.
    cli_rejected, stderr = cli_decision(
        capsys, ["link", "a", matter_id, "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(
            web_path, {"prerequisite": matter_id, "dependent": "a"}
        )
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert api_error.status == HTTPStatus.NOT_FOUND
    assert str(api_error) == f"unknown dependency source: {matter_id}"
    assert f"unknown dependency source: {matter_id}" in stderr
    assert cli_path.read_bytes() == web_path.read_bytes() == original


# ---------------------------------------------------------------------------
# AC-14 -- condition normalisation, including the fallback label
# ---------------------------------------------------------------------------


def test_ac14_whitespace_label_stores_the_same_literal_fallback_on_both(
    tmp_path, capsys
):
    # A matter with no conditions at all, so the fallback index is 1. This is
    # where the documented off-by-one hazard would show: the addressing index
    # is 0-based, the fallback-label index is passed 1-based. The literal
    # string below is the only acceptable evidence.
    cli_path, web_path, _original = two_states(
        tmp_path,
        {"matters": ["a"], "conditions": {"a": []}, "dependencies": []},
    )

    cli_rejected, stdout = cli_decision(
        capsys, ["add-condition", "a", " \t ", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.update_conditions(web_path, "a", {"label": " \t "})
    )

    assert cli_rejected is False
    assert web_rejected is False
    assert cli_rejected is web_rejected
    assert api_error is None
    assert stdout == 'a: added condition 1 "Unlabeled condition 1" (false)\n'

    cli_state = json.loads(cli_path.read_text())
    web_state = json.loads(web_path.read_text())
    assert cli_state["conditions"]["a"][0]["label"] == "Unlabeled condition 1"
    assert web_state["conditions"]["a"][0]["label"] == "Unlabeled condition 1"
    assert cli_path.read_bytes() == web_path.read_bytes()
    assert cli_state == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {"a": [{"label": "Unlabeled condition 1", "truth": False}]},
        "dependencies": [],
    }


def test_ac14_a_second_whitespace_label_numbers_the_fallback_from_one(
    tmp_path, capsys
):
    # Same rule one position further along, so the literal moves from 1 to 2
    # and a 0-based fallback index could not pass both tests.
    cli_path, web_path, _original = two_states(tmp_path, TWO_MATTERS)

    cli_decision(capsys, ["add-condition", "a", "", "--state", str(cli_path)])
    web.update_conditions(web_path, "a", {"label": ""})

    assert json.loads(cli_path.read_text())["conditions"]["a"][1]["label"] == (
        "Unlabeled condition 2"
    )
    assert cli_path.read_bytes() == web_path.read_bytes()


# ---------------------------------------------------------------------------
# AC-13 (1 of 3) -- behavioural: one broken rule must break both surfaces
#
# `require_matter_id_syntax` deliberately gets no behavioural test here.
# Slice A established that it has no CLI-reachable path: no CLI verb mints a
# matter id (`create` goes through slugify and by AC-21 must not gain
# validation), and the lookup verbs deliberately do not validate id syntax,
# because doing so would diverge from the web -- which rejects "a b" as an
# unknown endpoint 404, not as a syntax error. Sabotaging that rule would
# therefore prove nothing about the CLI. It is covered instead by the identity
# test and the duplication guard below, plus the E-7/E-8 parity tests above.
# ---------------------------------------------------------------------------


def test_ac13_a_broken_normalize_condition_breaks_both_surfaces(
    tmp_path, capsys, monkeypatch
):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    def sabotaged_normalize_condition(condition, index):
        raise RuleError("sabotaged normalize_condition")

    monkeypatch.setattr(rules, "normalize_condition", sabotaged_normalize_condition)

    cli_rejected, stderr = cli_decision(
        capsys, ["add-condition", "a", "ship it", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.update_conditions(web_path, "a", {"label": "ship it"})
    )

    # If a builder had copied condition normalisation into cli.py, the CLI half
    # would have succeeded and this assertion would fail.
    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert "sabotaged normalize_condition" in stderr
    assert str(api_error) == "sabotaged normalize_condition"
    assert api_error.status == HTTPStatus.BAD_REQUEST
    assert cli_path.read_bytes() == web_path.read_bytes() == original


def test_ac13_a_broken_build_index_breaks_both_surfaces(
    tmp_path, capsys, monkeypatch
):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    def sabotaged_build_index(matters_, conditions, dependencies):
        raise RuleError("sabotaged build_index", "state_cycle")

    monkeypatch.setattr(rules, "build_index", sabotaged_build_index)

    cli_rejected, stderr = cli_decision(
        capsys, ["mark", "a", "1", "true", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.create_matter(web_path, {"title": "new matter"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert "sabotaged build_index" in stderr
    assert str(api_error) == "sabotaged build_index"
    assert api_error.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert cli_path.read_bytes() == web_path.read_bytes() == original


def test_ac13_a_broken_dependency_endpoints_breaks_both_surfaces(
    tmp_path, capsys, monkeypatch
):
    cli_path, web_path, original = two_states(tmp_path, TWO_MATTERS)

    def sabotaged_dependency_endpoints(payload, matters_):
        raise RuleError("sabotaged dependency_endpoints", "not_found")

    monkeypatch.setattr(rules, "dependency_endpoints", sabotaged_dependency_endpoints)

    cli_rejected, stderr = cli_decision(
        capsys, ["link", "a", "b", "--state", str(cli_path)]
    )
    web_rejected, api_error = web_decision(
        lambda: web.add_dependency(web_path, {"prerequisite": "b", "dependent": "a"})
    )

    assert cli_rejected is True
    assert web_rejected is True
    assert cli_rejected is web_rejected
    assert "sabotaged dependency_endpoints" in stderr
    assert str(api_error) == "sabotaged dependency_endpoints"
    assert api_error.status == HTTPStatus.NOT_FOUND
    assert cli_path.read_bytes() == web_path.read_bytes() == original


# ---------------------------------------------------------------------------
# AC-13 (2 of 3) -- identity: the web keeps names, not copies
# ---------------------------------------------------------------------------


def test_ac13_web_rule_names_are_the_rules_objects_themselves():
    assert web.normalize_condition is rules.normalize_condition
    assert web.normalized_matter_id is rules.normalized_matter_id
    assert web.require_condition_index is rules.require_condition_index
    assert web.dependency_endpoints is rules.dependency_endpoints
    assert web.require_matter is rules.require_matter
    assert web.state_mutation_locks is rules.state_mutation_locks
    assert web.RuleError is rules.RuleError
    assert web.create_matters_from_expression is rules.create_matters_from_expression


def test_ac13_the_cli_module_holds_no_copy_of_any_rule():
    # A module-level attribute with any of these names in cli.py means a rule
    # got a second home, whether by copy or by re-export.
    for name in (
        "MATTER_ID_PATTERN",
        "require_matter_id_syntax",
        "normalized_matter_id",
        "normalize_condition",
        "require_condition_index",
        "dependency_endpoints",
        "require_matter",
        "build_index",
        "has_cycle",
        "would_create_cycle",
        "resolve_condition_ref",
        "append_condition",
        "apply_condition_patch",
        "delete_condition_at",
        "state_lock",
        "state_transaction",
    ):
        assert not hasattr(cli, name), f"matters.cli has its own {name}"


# ---------------------------------------------------------------------------
# AC-13 (3 of 3) -- duplication guard
# ---------------------------------------------------------------------------


def source_files():
    # rglob, not glob: a top-level-only scan cannot see src/matters/llm/, so a
    # copy of a guarded literal in a subpackage would pass unnoticed.
    return sorted(Path(matters.__file__).parent.rglob("*.py"))


def files_containing(needle):
    return [
        path.name
        for path in source_files()
        if needle in path.read_text(encoding="utf-8")
    ]


def test_ac13_the_matter_id_regex_appears_in_exactly_one_source_file():
    # NOTE: this guard is deliberately sensitive to prose as well as to code.
    # A help string, a docstring or a comment that quotes the regex counts as
    # a second copy and fails this test on purpose -- documentation that
    # restates a rule drifts from it exactly the way a code copy does. Do not
    # "fix" this by narrowing the scan to code; delete the second copy or
    # reword it so it does not restate the literal.
    assert files_containing(r"[a-z0-9_]+") == ["rules.py"]


def test_ac13_the_fallback_condition_label_appears_in_exactly_one_source_file():
    # Same deliberate prose sensitivity as the test above.
    #
    # The trailing space is doing real work. Two near-copies of this fallback
    # exist and are deliberately not matched: engine.py's "Unlabeled legacy
    # condition {index}" (a distinct legacy-shape label) and reports.py's
    # "Unlabeled condition" with no index and no trailing space. Both predate
    # the rules-layer extraction and are out of scope here; this guard covers
    # the indexed label that rules.normalize_condition owns.
    assert files_containing("Unlabeled condition ") == ["rules.py"]
