import json
import os
import threading
from http import HTTPStatus

import pytest

import matters.rules as rules
import matters.web as web
from matters.rules import RuleError


def write_state(path, data):
    path.write_text(json.dumps(data))


def simple_state(path):
    write_state(
        path,
        {
            "matters": ["a", "b"],
            "conditions": {
                "a": [{"label": "a done", "truth": False}],
                "b": [{"label": "b done", "truth": False}],
            },
            "dependencies": [["a", "b"]],
        },
    )


# ---------------------------------------------------------------------------
# resolve_condition_ref -- the five D1 outcomes
# ---------------------------------------------------------------------------


def test_resolve_condition_ref_index_form_is_one_based():
    conditions = [
        {"label": "first", "truth": False},
        {"label": "second", "truth": False},
    ]

    assert rules.resolve_condition_ref("a", conditions, "1") == 0
    assert rules.resolve_condition_ref("a", conditions, "2") == 1


def test_resolve_condition_ref_zero_is_an_index_and_always_out_of_range():
    conditions = [{"label": "first", "truth": False}]

    with pytest.raises(RuleError) as error:
        rules.resolve_condition_ref("a", conditions, "0")

    assert str(error.value) == "matter a has no condition 0 (it has 1)"
    assert error.value.code == "not_found"


def test_resolve_condition_ref_index_out_of_range_names_the_count():
    conditions = [{"label": "first", "truth": False}]

    with pytest.raises(RuleError) as error:
        rules.resolve_condition_ref("a", conditions, "2")

    assert str(error.value) == "matter a has no condition 2 (it has 1)"
    assert error.value.code == "not_found"


def test_resolve_condition_ref_rejects_non_ascii_digits_as_an_index():
    # "²".isdigit() is True and int("²") raises, which is why the
    # implementation uses re.fullmatch(r"[0-9]+", ...) instead.
    conditions = [{"label": "²", "truth": False}]

    assert rules.resolve_condition_ref("a", conditions, "²") == 0


def test_resolve_condition_ref_matches_an_exact_label():
    conditions = [
        {"label": "ship it", "truth": False},
        {"label": "review it", "truth": False},
    ]

    assert rules.resolve_condition_ref("a", conditions, "review it") == 1
    assert rules.resolve_condition_ref("a", conditions, "  ship it  ") == 0


def test_resolve_condition_ref_label_match_is_case_sensitive():
    conditions = [{"label": "Ship It", "truth": False}]

    with pytest.raises(RuleError) as error:
        rules.resolve_condition_ref("a", conditions, "ship it")

    assert error.value.code == "not_found"


def test_resolve_condition_ref_unknown_label_points_at_show():
    conditions = [{"label": "ship it", "truth": False}]

    with pytest.raises(RuleError) as error:
        rules.resolve_condition_ref("a", conditions, "missing")

    assert str(error.value) == (
        "matter a has no condition matching 'missing'; "
        "run `matters show a` to see condition numbers"
    )
    assert error.value.code == "not_found"


def test_resolve_condition_ref_refuses_duplicate_labels_and_lists_positions():
    conditions = [
        {"label": "ship it", "truth": False},
        {"label": "other", "truth": False},
        {"label": "ship it", "truth": True},
    ]

    with pytest.raises(RuleError) as error:
        rules.resolve_condition_ref("a", conditions, "ship it")

    assert str(error.value) == (
        "matter a has 2 conditions labelled 'ship it' (positions 1, 3); "
        "address it by number"
    )
    assert error.value.code == "invalid"


def test_resolve_condition_ref_treats_negative_and_decimal_refs_as_labels():
    conditions = [{"label": "-1", "truth": False}, {"label": "1.0", "truth": False}]

    assert rules.resolve_condition_ref("a", conditions, "-1") == 0
    assert rules.resolve_condition_ref("a", conditions, "1.0") == 1


# ---------------------------------------------------------------------------
# Condition editing primitives
# ---------------------------------------------------------------------------


def test_normalize_condition_fallback_label_is_one_based_literal():
    assert rules.normalize_condition("   ", 1) == {
        "label": "Unlabeled condition 1",
        "truth": False,
    }
    assert rules.normalize_condition({"label": "", "truth": True}, 3) == {
        "label": "Unlabeled condition 3",
        "truth": True,
    }


def test_append_condition_uses_a_one_based_fallback_index():
    current = [{"label": "first", "truth": False}]

    appended = rules.append_condition(current, "   ")

    assert appended == [
        {"label": "first", "truth": False},
        {"label": "Unlabeled condition 2", "truth": False},
    ]
    assert current == [{"label": "first", "truth": False}]


def test_apply_condition_patch_preserves_unset_fields():
    current = [
        {"label": "first", "truth": False},
        {"label": "second", "truth": True},
    ]

    assert rules.apply_condition_patch(current, 1, label="renamed") == {
        "label": "renamed",
        "truth": True,
    }
    assert rules.apply_condition_patch(current, 1, truth=False) == {
        "label": "second",
        "truth": False,
    }
    assert current[1] == {"label": "second", "truth": True}


def test_apply_condition_patch_strips_labels_and_falls_back_one_based():
    current = [{"label": "first", "truth": True}]

    assert rules.apply_condition_patch(current, 0, label="  spaced  ") == {
        "label": "spaced",
        "truth": True,
    }
    assert rules.apply_condition_patch(current, 0, label="   ") == {
        "label": "Unlabeled condition 1",
        "truth": True,
    }


def test_delete_condition_at_renumbers_by_position():
    current = [
        {"label": "first", "truth": False},
        {"label": "second", "truth": True},
        {"label": "third", "truth": False},
    ]

    removed = rules.delete_condition_at(current, 0)

    assert removed == {"label": "first", "truth": False}
    assert current == [
        {"label": "second", "truth": True},
        {"label": "third", "truth": False},
    ]
    assert rules.resolve_condition_ref("a", current, "1") == 0


# ---------------------------------------------------------------------------
# Cycle rules
# ---------------------------------------------------------------------------


def test_build_index_raises_the_pinned_cycle_message():
    with pytest.raises(RuleError) as error:
        rules.build_index(
            {"a", "b"}, {"a": [], "b": []}, {("a", "b"), ("b", "a")}
        )

    assert str(error.value) == "state dependency graph contains a cycle"
    assert error.value.code == "state_cycle"


def test_would_create_cycle_does_not_mutate_the_dependency_set():
    dependencies = {("a", "b")}

    assert rules.would_create_cycle(
        {"a", "b"}, {"a": [], "b": []}, dependencies, ("b", "a")
    )
    assert rules.would_create_cycle(
        {"a", "b"}, {"a": [], "b": []}, dependencies, ("a", "a")
    )
    assert dependencies == {("a", "b")}


# ---------------------------------------------------------------------------
# state_transaction
# ---------------------------------------------------------------------------


def test_state_transaction_skips_the_save_when_nothing_changed(tmp_path):
    state_path = tmp_path / "matters.json"
    # Deliberately non-canonical: no schema_version, unsorted matters, no
    # trailing newline. save_state would rewrite all of that.
    state_path.write_text(
        '{"matters": ["b", "a"], "conditions": {"b": [], "a": []}, '
        '"dependencies": []}'
    )
    original = state_path.read_bytes()

    with rules.state_transaction(state_path) as draft:
        assert draft.matters == {"a", "b"}

    assert state_path.read_bytes() == original


def test_state_transaction_saves_when_the_draft_changed(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    with rules.state_transaction(state_path) as draft:
        draft.matters.add("a")
        draft.conditions["a"] = [{"label": "done", "truth": False}]

    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {"a": [{"label": "done", "truth": False}]},
        "dependencies": [],
    }


def test_state_transaction_does_not_save_when_the_body_raises(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with pytest.raises(RuleError):
        with rules.state_transaction(state_path) as draft:
            draft.matters.add("c")
            draft.conditions["c"] = []
            raise RuleError("rejected after mutating the draft")

    assert state_path.read_bytes() == original


def test_state_transaction_refuses_a_preexisting_cycle_before_yielding(tmp_path):
    state_path = tmp_path / "cyclic.json"
    write_state(
        state_path,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [["a", "b"], ["b", "a"]],
        },
    )
    original = state_path.read_bytes()
    entered = []

    with pytest.raises(RuleError) as error:
        with rules.state_transaction(state_path) as draft:
            entered.append(draft)

    assert entered == []
    assert str(error.value) == "state dependency graph contains a cycle"
    assert error.value.code == "state_cycle"
    assert state_path.read_bytes() == original


def test_state_transaction_can_opt_out_of_the_cycle_guard(tmp_path):
    state_path = tmp_path / "cyclic.json"
    write_state(
        state_path,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [["a", "b"], ["b", "a"]],
        },
    )

    with rules.state_transaction(state_path, require_acyclic=False) as draft:
        assert draft.index is None


def test_state_transaction_require_exists_names_the_path(tmp_path):
    state_path = tmp_path / "missing.json"

    with pytest.raises(RuleError) as error:
        with rules.state_transaction(state_path, require_exists=True):
            pass

    assert str(error.value) == f"state file does not exist: {state_path}"
    assert error.value.code == "not_found"


def test_state_transaction_reports_malformed_json(tmp_path):
    state_path = tmp_path / "matters.json"
    state_path.write_text("{")
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        with rules.state_transaction(state_path):
            pass

    assert str(error.value).startswith(f"state file is not valid JSON: {state_path}: ")
    assert error.value.code == "invalid"
    assert state_path.read_bytes() == original


def test_state_transaction_reports_an_invalid_graph(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": ["a"], "conditions": {"ghost": []}, "dependencies": []})

    with pytest.raises(RuleError) as error:
        with rules.state_transaction(state_path):
            pass

    assert str(error.value) == (
        f"state file is not a valid matters graph: {state_path}: "
        "conditions contain unknown matter: ghost"
    )
    assert error.value.code == "invalid"


def test_state_transaction_reports_an_unreadable_file(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    state_path.chmod(0o000)
    if os.access(state_path, os.R_OK):  # pragma: no cover - running as root
        state_path.chmod(0o600)
        pytest.skip("state file is readable regardless of mode")

    try:
        with pytest.raises(RuleError) as error:
            with rules.state_transaction(state_path):
                pass
    finally:
        state_path.chmod(0o600)

    assert str(error.value).startswith(f"state file is not readable: {state_path}: ")
    assert error.value.code == "invalid"


def test_state_transaction_uses_the_injected_loader(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})
    seen = []

    def loader(path):
        seen.append(path)
        return {"injected"}, {"injected": []}, set()

    with rules.state_transaction(state_path, load=loader) as draft:
        assert draft.matters == {"injected"}

    assert seen == [state_path]


# ---------------------------------------------------------------------------
# Locking (brief section 7.4)
# ---------------------------------------------------------------------------


def test_state_lock_path_is_a_dot_prefixed_sidecar(tmp_path):
    state_path = tmp_path / "matters.json"

    assert rules.state_lock_path(state_path) == tmp_path / ".matters.json.lock"
    assert not (tmp_path / ".matters.json.lock").exists()


def test_a_successful_write_leaves_an_empty_sidecar_lock_file(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    rules.set_condition_truth(state_path, "a", "1", True)

    lock_path = rules.state_lock_path(state_path)
    assert lock_path.exists()
    assert lock_path.stat().st_size == 0
    assert lock_path.stat().st_mode & 0o777 == 0o600


def test_a_held_lock_rejects_a_cli_shaped_write_without_touching_the_file(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with rules.state_lock(state_path):
        with pytest.raises(RuleError) as error:
            rules.set_condition_truth(state_path, "a", "1", True)

    assert "locked by another matters process" in str(error.value)
    assert error.value.code == "locked"
    assert state_path.read_bytes() == original


def test_a_held_lock_rejects_a_web_mutation_with_409(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with rules.state_lock(state_path):
        with pytest.raises(web.ApiError) as error:
            web.create_matter(state_path, {"title": "blocked write"})

    assert error.value.status == HTTPStatus.CONFLICT
    assert state_path.read_bytes() == original


def test_the_lock_is_released_after_a_rejected_write(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    with pytest.raises(RuleError):
        rules.set_condition_truth(state_path, "ghost", "1", True)

    result = rules.set_condition_truth(state_path, "a", "1", True)

    assert result["truth"] is True
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": True}
    ]


def test_read_helpers_create_no_lock_file(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.list_matters(state_path) == ["a", "b"]
    assert rules.describe_matter(state_path, "a") == {
        "id": "a",
        "conditions": [{"index": 0, "label": "a done", "truth": False}],
        "prerequisites": [],
        "dependents": ["b"],
    }
    assert not rules.state_lock_path(state_path).exists()


def test_a_write_against_a_missing_state_path_creates_no_files(tmp_path):
    state_path = tmp_path / "missing.json"

    with pytest.raises(RuleError) as error:
        rules.set_condition_truth(state_path, "a", "1", True)

    assert error.value.code == "not_found"
    assert not state_path.exists()
    assert not rules.state_lock_path(state_path).exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# CLI-shaped operation contracts (the dicts cli.py prints from)
# ---------------------------------------------------------------------------


def test_set_condition_truth_reports_position_label_and_change(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.set_condition_truth(state_path, "a", "1", True) == {
        "matter": "a",
        "position": 1,
        "label": "a done",
        "truth": True,
        "changed": True,
    }
    assert rules.set_condition_truth(state_path, "a", "1", True)["changed"] is False


def test_set_condition_truth_is_byte_identical_when_already_set(tmp_path):
    state_path = tmp_path / "matters.json"
    # Non-canonical on purpose: an unconditional save would reformat it.
    state_path.write_text(
        '{"matters": ["a"], "conditions": {"a": [{"truth": true, "label": "done"}]},'
        ' "dependencies": []}'
    )
    original = state_path.read_bytes()

    result = rules.set_condition_truth(state_path, "a", "1", True)

    assert result["changed"] is False
    assert state_path.read_bytes() == original


def test_add_condition_appends_with_a_stripped_label_and_false_truth(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.add_condition(state_path, "a", "  ship it  ") == {
        "matter": "a",
        "position": 2,
        "label": "ship it",
        "truth": False,
        "changed": True,
    }
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": False},
        {"label": "ship it", "truth": False},
    ]


def test_add_condition_uses_the_shared_fallback_label(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.add_condition(state_path, "a", "   ")["label"] == (
        "Unlabeled condition 2"
    )


def test_edit_condition_label_preserves_truth_and_position(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {
            "matters": ["a"],
            "conditions": {
                "a": [
                    {"label": "first", "truth": True},
                    {"label": "second", "truth": True},
                ]
            },
            "dependencies": [],
        },
    )

    assert rules.edit_condition_label(state_path, "a", "2", "renamed") == {
        "matter": "a",
        "position": 2,
        "label": "renamed",
        "previous_label": "second",
        "truth": True,
        "changed": True,
    }
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "first", "truth": True},
        {"label": "renamed", "truth": True},
    ]


def test_delete_condition_refuses_to_empty_a_matter_without_confirmation(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        rules.delete_condition(state_path, "a", "1")

    assert str(error.value) == (
        "deleting the last condition of a will make it resolved, "
        "unblocking: b — rerun with --yes"
    )
    assert error.value.code == "conflict"
    assert state_path.read_bytes() == original


def test_delete_condition_reports_the_matters_it_unblocks(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.delete_condition(state_path, "a", "1", confirmed=True) == {
        "matter": "a",
        "position": 1,
        "label": "a done",
        "truth": False,
        "emptied": True,
        "unblocked": ["b"],
        "changed": True,
    }
    assert json.loads(state_path.read_text())["conditions"]["a"] == []


def test_link_stores_the_edge_as_prerequisite_then_dependent(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {"matters": ["a", "b"], "conditions": {"a": [], "b": []}, "dependencies": []},
    )

    assert rules.link(state_path, "a", "b") == {
        "dependent": "a",
        "prerequisite": "b",
        "changed": True,
    }
    assert json.loads(state_path.read_text())["dependencies"] == [["b", "a"]]


def test_link_is_idempotent_and_leaves_the_file_untouched(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    assert rules.link(state_path, "b", "a")["changed"] is False
    assert state_path.read_bytes() == original


def test_link_refuses_a_cycle_naming_both_endpoints(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        rules.link(state_path, "a", "b")

    assert str(error.value) == (
        "a cannot require b: the dependency would create a cycle"
    )
    assert error.value.code == "invalid"
    assert state_path.read_bytes() == original


def test_unlink_of_a_missing_edge_is_a_byte_identical_no_op(tmp_path):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        '{"matters": ["b", "a"], "conditions": {"a": [], "b": []}, '
        '"dependencies": []}'
    )
    original = state_path.read_bytes()

    assert rules.unlink(state_path, "a", "b")["changed"] is False
    assert state_path.read_bytes() == original


def test_delete_matter_refuses_dependents_without_cascade(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        rules.delete_matter(state_path, "a", confirmed=True)

    assert str(error.value) == (
        "matter a is required by: b — rerun with --cascade to delete it and "
        "those dependencies, or unlink them first"
    )
    assert error.value.code == "conflict"
    assert state_path.read_bytes() == original


def test_delete_matter_requires_confirmation_before_cascading(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        rules.delete_matter(state_path, "a", cascade=True)

    assert error.value.code == "conflict"
    assert "rerun with --yes" in str(error.value)
    assert state_path.read_bytes() == original


def test_delete_matter_cascade_removes_every_trace_of_the_id(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.delete_matter(
        state_path, "a", cascade=True, confirmed=True
    ) == {
        "matter": "a",
        "removed_edges": [["a", "b"]],
        "unblocked": ["b"],
        "changed": True,
    }
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["b"],
        "conditions": {"b": [{"label": "b done", "truth": False}]},
        "dependencies": [],
    }


def test_concurrent_create_matter_preserves_all_writes(tmp_path, monkeypatch):
    # Mirrors tests/test_web.py::test_concurrent_create_matter_preserves_all_writes
    # one layer down: this one patches the rules loader, so it keeps testing
    # the transaction even if the web wrappers stop passing load=.
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})
    original_load_state = rules.load_state
    barrier = threading.Barrier(2)
    barrier_threads = set()
    barrier_lock = threading.Lock()

    def racing_load_state(path):
        result = original_load_state(path)
        thread_name = threading.current_thread().name
        with barrier_lock:
            should_wait = (
                thread_name.startswith("writer-") and thread_name not in barrier_threads
            )
            if should_wait:
                barrier_threads.add(thread_name)
        if should_wait:
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
        return result

    monkeypatch.setattr(rules, "load_state", racing_load_state)
    errors = []

    def write(title):
        try:
            rules.create_matter(state_path, {"title": title, "conditions": ["done"]})
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [
        threading.Thread(target=write, name="writer-a", args=("First write",)),
        threading.Thread(target=write, name="writer-b", args=("Second write",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert set(json.loads(state_path.read_text())["matters"]) == {
        "first_write",
        "second_write",
    }


# ---------------------------------------------------------------------------
# set_condition_truth takes a real bool, never a truthy string
# ---------------------------------------------------------------------------


def test_set_condition_truth_rejects_a_string_instead_of_coercing_it(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        rules.set_condition_truth(state_path, "a", "1", "false")

    assert str(error.value) == (
        "condition truth must be true or false, not str: 'false'"
    )
    assert error.value.code == "invalid"
    # engine.truth would have coerced "false" to True and written the opposite
    # of what the caller asked for. Nothing is written, locked, or created.
    assert state_path.read_bytes() == original
    assert not rules.state_lock_path(state_path).exists()


def test_set_condition_truth_writes_the_bool_it_is_given(tmp_path):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert rules.set_condition_truth(state_path, "a", "1", True)["truth"] is True
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": True}
    ]

    assert rules.set_condition_truth(state_path, "a", "1", False)["truth"] is False
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": False}
    ]


# ---------------------------------------------------------------------------
# Locking, from the CLI side (brief section 7.4)
#
# The block above exercises the lock through the rules functions directly.
# These go through `matters.cli.main`, because that is the surface a user
# actually holds: a lock that works in `rules` but that a CLI branch bypasses
# would pass every test above and still lose a write.
# ---------------------------------------------------------------------------

from matters.cli import main  # noqa: E402  (appended section, see comment above)


def test_a_held_lock_rejects_the_cli_mark_verb(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    original = state_path.read_bytes()

    with rules.state_lock(state_path):
        with pytest.raises(SystemExit) as error:
            main(["mark", "a", "1", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "locked by another matters process" in captured.err
    assert str(rules.state_lock_path(state_path)) in captured.err
    assert captured.out == ""
    assert state_path.read_bytes() == original


def test_a_cli_write_in_flight_excludes_a_web_mutation(tmp_path, monkeypatch):
    # The cross-surface proof in its strongest form. Rather than holding the
    # lock by hand, this runs a real CLI write and, from inside that write's
    # transaction while the file lock is held, attempts a web mutation on the
    # same path. flock is owned by the open file description, so the web
    # attempt opens a second descriptor and is refused even though both halves
    # live in one process -- which is what makes the two surfaces exclude each
    # other rather than only their own kind.
    state_path = tmp_path / "matters.json"
    simple_state(state_path)
    observed = []
    real_save_state = rules.save_state

    def save_state_while_the_web_tries(*args, **kwargs):
        try:
            web.create_matter(state_path, {"title": "web write"})
            observed.append(None)
        except web.ApiError as error:
            observed.append(error)
        return real_save_state(*args, **kwargs)

    monkeypatch.setattr(rules, "save_state", save_state_while_the_web_tries)

    assert main(["mark", "a", "1", "true", "--state", str(state_path)]) == 0

    assert len(observed) == 1
    assert isinstance(observed[0], web.ApiError)
    assert observed[0].status == HTTPStatus.CONFLICT
    assert "locked by another matters process" in str(observed[0])
    # The CLI write itself still lands, and the web write never did.
    assert json.loads(state_path.read_text())["matters"] == ["a", "b"]
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": True}
    ]


def test_a_successful_cli_write_leaves_an_empty_sidecar_lock_file(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert main(["mark", "a", "1", "true", "--state", str(state_path)]) == 0

    lock_path = rules.state_lock_path(state_path)
    assert lock_path == tmp_path / ".matters.json.lock"
    assert lock_path.exists()
    assert lock_path.stat().st_size == 0
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        ".matters.json.lock",
        "matters.json",
    ]


def test_cli_show_and_list_create_no_lock_file(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    assert main(["show", "a", "--state", str(state_path)]) == 0
    assert main(["list", "--state", str(state_path)]) == 0
    capsys.readouterr()

    assert not rules.state_lock_path(state_path).exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["matters.json"]


def test_a_cli_write_against_a_missing_state_creates_no_state_and_no_lock(
    tmp_path, capsys
):
    # F-10: the existence check runs before any lock file can be created.
    state_path = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "1", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert f"state file does not exist: {state_path}" in captured.err
    assert captured.out == ""
    assert not state_path.exists()
    assert not rules.state_lock_path(state_path).exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == []


def test_the_lock_is_released_after_a_rejected_cli_write(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    simple_state(state_path)

    with pytest.raises(SystemExit) as error:
        main(["mark", "ghost", "1", "true", "--state", str(state_path)])
    assert error.value.code == 2
    assert "unknown matter: ghost" in capsys.readouterr().err

    # The lock file now exists; if the rejection had leaked the flock, this
    # second write would fail with "locked by another matters process".
    assert rules.state_lock_path(state_path).exists()
    assert main(["mark", "a", "1", "true", "--state", str(state_path)]) == 0
    assert capsys.readouterr().out == 'a: condition 1 "a done" is now true\n'
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": True}
    ]


# ---------------------------------------------------------------------------
# Loader translation of structurally invalid state (AC-15/AC-16, F-11)
#
# Valid JSON whose shape the engine cannot walk. load_state ->
# engine.normalize_conditions -> engine.as_condition_list calls list() on the
# value, so a scalar raises TypeError, not ValueError. The catch set here
# matches web.validate_switch_state_path's against the same load_state call.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scalar,type_name",
    [(5, "int"), (True, "bool"), (1.5, "float")],
)
def test_the_loader_rejects_a_scalar_conditions_value(tmp_path, scalar, type_name):
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {"matters": ["a"], "conditions": {"a": scalar}, "dependencies": []},
    )
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        rules.load_state_or_rule_error(state_path)

    assert str(error.value) == (
        f"state file is not a valid matters graph: {state_path}: "
        f"'{type_name}' object is not iterable"
    )
    assert error.value.code == "invalid"
    assert state_path.read_bytes() == original


def test_state_transaction_rejects_a_scalar_conditions_value(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {"matters": ["a"], "conditions": {"a": 5}, "dependencies": []},
    )
    original = state_path.read_bytes()

    with pytest.raises(RuleError) as error:
        with rules.state_transaction(state_path):
            pass

    assert str(error.value) == (
        f"state file is not a valid matters graph: {state_path}: "
        "'int' object is not iterable"
    )
    assert error.value.code == "invalid"
    assert state_path.read_bytes() == original


def test_read_helpers_reject_a_scalar_conditions_value(tmp_path):
    # describe_matter and list_matters share the loader but take no lock, so
    # they need their own coverage: they are the two verbs that never open a
    # transaction.
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {"matters": ["a"], "conditions": {"a": 5}, "dependencies": []},
    )

    for call in (
        lambda: rules.describe_matter(state_path, "a"),
        lambda: rules.list_matters(state_path),
    ):
        with pytest.raises(RuleError) as error:
            call()
        assert str(error.value) == (
            f"state file is not a valid matters graph: {state_path}: "
            "'int' object is not iterable"
        )
        assert error.value.code == "invalid"


def test_a_rejected_structural_load_maps_to_the_same_status_as_bad_json(tmp_path):
    # "invalid" is the code both branches raise, so the web surface answers
    # 400 for a scalar conditions value exactly as it does for a truncated
    # file. Guards the rules/web contract, not the HTTP layer's table.
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {"matters": ["a"], "conditions": {"a": 5}, "dependencies": []},
    )

    with pytest.raises(RuleError) as error:
        rules.load_state_or_rule_error(state_path)

    assert web.api_error_for(error.value).status == HTTPStatus.BAD_REQUEST


def test_api_error_for_degrades_an_unmapped_rule_code_to_bad_request():
    # A code added to rules.py without a RULE_ERROR_STATUS entry must not
    # raise KeyError past the handler's `except ApiError` and become a 500.
    unmapped = RuleError("something new went wrong", "not_in_the_table")

    api_error = web.api_error_for(unmapped)

    assert api_error.status == HTTPStatus.BAD_REQUEST
    assert str(api_error) == "something new went wrong"
    assert "not_in_the_table" not in web.RULE_ERROR_STATUS
