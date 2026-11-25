"""The same primitive graph rules serve files and database-owned drafts."""

import copy
import json

import pytest

from matters import rules, storage
from matters.graph_index import GraphIndex


@pytest.fixture
def graph_data():
    return {
        "schema_version": 2,
        "matters": ["root", "child", "leaf"],
        "conditions": {
            "root": [{"label": "Root verified", "truth": True}],
            "child": [
                {"label": "First check", "truth": False},
                {"label": "Second check", "truth": False},
            ],
            "leaf": [{"label": "Leaf delivered", "truth": False}],
        },
        "dependencies": [["root", "child"], ["child", "leaf"]],
    }


def memory_draft(data):
    return rules.StateDraft(None, *storage.state_from_data(copy.deepcopy(data)))


def file_operation(path, operation, payload):
    revision = storage.state_revision(*storage.load_state(path))
    if operation == "create_matter":
        return rules.create_matter(path, payload, expected_revision=revision)
    if operation in {"add_dependency", "remove_dependency"}:
        return getattr(rules, operation)(path, payload, expected_revision=revision)
    matter_id = payload["matter_id"]
    if operation == "delete_matter":
        return rules.delete_matter(
            path, matter_id, cascade=payload.get("cascade", False),
            confirmed=payload.get("confirmed", False), expected_revision=revision,
        )
    if operation == "delete_condition":
        return rules.delete_condition(
            path, matter_id, str(payload["index"] + 1),
            confirmed=payload.get("confirmed", False), expected_revision=revision,
        )
    return rules.update_conditions(
        path, matter_id, {key: value for key, value in payload.items() if key != "matter_id"},
        expected_revision=revision,
    )


ALL_OPERATIONS = [
    ("create_matter", {"id": "new", "conditions": [{"label": "Ship new", "truth": False}]}),
    ("set_condition", {"matter_id": "child", "index": 0, "truth": True}),
    ("add_condition", {"matter_id": "child", "label": "Evidence checked", "truth": True}),
    ("edit_condition", {"matter_id": "child", "index": 0, "label": "Check renamed"}),
    ("delete_condition", {"matter_id": "child", "index": 0}),
    ("delete_matter", {"matter_id": "child", "cascade": True, "confirmed": True}),
    ("add_dependency", {"source": "root", "target": "leaf"}),
    ("remove_dependency", {"source": "root", "target": "child"}),
]


@pytest.mark.parametrize("operation,payload", ALL_OPERATIONS)
def test_each_memory_operation_matches_file_rules(tmp_path, graph_data, operation, payload):
    path = tmp_path / "graph.json"
    storage.save_state(*storage.state_from_data(graph_data), path=path)
    draft = memory_draft(graph_data)
    original_containers = draft.as_tuple()

    result = rules.apply_graph_operation(draft, operation, payload)
    file_result = file_operation(path, operation, payload)

    assert storage.state_data(*draft.as_tuple()) == storage.state_data(*storage.load_state(path))
    assert result["operation"] == operation
    assert result["changed"] is True
    assert all(after is before for after, before in zip(draft.as_tuple(), original_containers))
    assert draft.index is None
    if isinstance(file_result, dict):
        comparable = dict(file_result)
        if "position" in comparable:
            comparable["index"] = comparable.pop("position") - 1
        assert result == {"operation": operation, **comparable}


def test_multiple_memory_operations_refresh_derived_context(graph_data):
    draft = memory_draft(graph_data)
    assert GraphIndex(*draft.as_tuple()).universe == {"child"}
    rules.apply_graph_operation(draft, "set_condition", {"matter_id": "child", "index": 0, "truth": True})
    rules.apply_graph_operation(draft, "set_condition", {"matter_id": "child", "index": 1, "truth": True})
    assert GraphIndex(*draft.as_tuple()).universe == {"leaf"}
    result = rules.apply_graph_operation(draft, "delete_condition", {
        "matter_id": "leaf", "index": 0, "confirmed": True,
    })
    assert result["index"] == 0
    assert result["emptied"] is True
    assert GraphIndex(*draft.as_tuple()).resolved["leaf"] is True


@pytest.mark.parametrize("operation,payload", [
    ("set_condition", {"matter_id": "child", "index": 0, "truth": False}),
    ("edit_condition", {"matter_id": "child", "index": 0, "label": "First check"}),
    ("add_dependency", {"source": "root", "target": "child"}),
    ("remove_dependency", {"source": "root", "target": "leaf"}),
])
def test_noop_memory_and_file_operations_preserve_facts_and_file_bytes(tmp_path, graph_data, operation, payload):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph_data, separators=(",", ":")))
    before_bytes = path.read_bytes()
    draft = memory_draft(graph_data)
    before = copy.deepcopy(draft.as_tuple())
    result = rules.apply_graph_operation(draft, operation, payload)
    file_operation(path, operation, payload)
    assert result["changed"] is False
    assert draft.as_tuple() == before
    assert path.read_bytes() == before_bytes


@pytest.mark.parametrize("operation,payload", [
    ("set_condition", {"matter_id": "child", "index": True, "truth": True}),
    ("set_condition", {"matter_id": "child", "index": "0", "truth": True}),
    ("set_condition", {"matter_id": "child", "index": -1, "truth": True}),
    ("set_condition", {"matter_id": "child", "index": 9, "truth": True}),
    ("set_condition", {"matter_id": "child", "index": 0, "truth": "false"}),
    ("set_condition", {"matter_id": "child", "index": 0, "truth": 1}),
    ("set_condition", {"matter_id": "child", "truth": True}),
    ("set_condition", {"matter_id": "child", "index": 0, "truth": True, "path": "/other.json"}),
    ("set_condition", {"matter_id": "bad id", "index": 0, "truth": True}),
    ("set_condition", {"matter_id": 1, "index": 0, "truth": True}),
    ("add_condition", {"matter_id": "child", "label": 2}),
    ("add_condition", {"matter_id": "child", "label": "Check", "truth": None}),
    ("edit_condition", {"matter_id": "child", "index": 0}),
    ("create_matter", {"id": "new", "conditions": "not a list"}),
    ("create_matter", {"id": "new", "conditions": ["not an object"]}),
    ("create_matter", {"id": "new", "conditions": [{"label": "Done", "truth": "false"}]}),
    ("create_matter", {"id": "new", "conditions": [{"label": "Done", "owner": "someone"}]}),
    ("create_matter", {"id": "new", "conditions": [{"label": 1}]}),
    ("delete_condition", {"matter_id": "child", "index": 0, "confirmed": "yes"}),
    ("delete_matter", {"matter_id": "leaf", "confirmed": True, "cascade": 1}),
    ("add_dependency", {"source": "root", "target": None}),
    ("remove_dependency", {"source": "root", "target": "leaf", "confirmed": True}),
    ("unknown", {}),
    (None, {}),
    ([], {}),
    ("set_condition", None),
])
def test_strict_operation_inputs_leave_draft_unchanged(graph_data, operation, payload):
    draft = memory_draft(graph_data)
    before = copy.deepcopy(draft.as_tuple())
    with pytest.raises(rules.RuleError):
        rules.apply_graph_operation(draft, operation, payload)
    assert draft.as_tuple() == before


@pytest.mark.parametrize("operation,payload", [
    ("delete_condition", {"matter_id": "leaf", "index": 0}),
    ("delete_matter", {"matter_id": "leaf"}),
    ("delete_matter", {"matter_id": "child", "cascade": True}),
    ("delete_matter", {"matter_id": "child", "confirmed": True}),
    ("add_dependency", {"source": "leaf", "target": "root"}),
    ("create_matter", {"id": "root"}),
    ("set_condition", {"matter_id": "missing", "index": 0, "truth": True}),
])
def test_rejected_graph_rules_leave_memory_draft_unchanged(graph_data, operation, payload):
    draft = memory_draft(graph_data)
    before = copy.deepcopy(draft.as_tuple())
    with pytest.raises(rules.RuleError):
        rules.apply_graph_operation(draft, operation, payload)
    assert draft.as_tuple() == before


def test_memory_operations_need_no_file_or_transaction_functions(graph_data, monkeypatch):
    draft = memory_draft(graph_data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Pure graph operation attempted I/O")

    for name in ("state_transaction", "state_lock", "load_state", "save_state", "load_state_or_rule_error"):
        monkeypatch.setattr(rules, name, forbidden)
    result = rules.apply_graph_operation(draft, "set_condition", {
        "matter_id": "child", "index": 0, "truth": True,
    })
    assert result["changed"] is True
    assert draft.conditions["child"][0]["truth"] is True


@pytest.mark.parametrize("helper,operation,payload", [
    ("_create_matter_in_draft", "create_matter", {"id": "new"}),
    ("_update_conditions_in_draft", "set_condition", {"matter_id": "child", "index": 0, "truth": True}),
    ("_add_dependency_in_draft", "add_dependency", {"source": "root", "target": "leaf"}),
    ("_remove_dependency_in_draft", "remove_dependency", {"source": "root", "target": "child"}),
    ("_delete_condition_in_draft", "delete_condition", {"matter_id": "child", "index": 0}),
    ("_delete_matter_in_draft", "delete_matter", {"matter_id": "leaf", "confirmed": True}),
])
def test_memory_and_file_routes_call_the_same_helper(tmp_path, graph_data, monkeypatch, helper, operation, payload):
    path = tmp_path / "graph.json"
    storage.save_state(*storage.state_from_data(graph_data), path=path)
    before = path.read_bytes()

    def replacement(*args, **kwargs):
        raise rules.RuleError("shared helper replaced")

    monkeypatch.setattr(rules, helper, replacement)
    with pytest.raises(rules.RuleError, match="shared helper replaced"):
        rules.apply_graph_operation(memory_draft(graph_data), operation, payload)
    with pytest.raises(rules.RuleError, match="shared helper replaced"):
        file_operation(path, operation, payload)
    assert path.read_bytes() == before


def test_partial_cycle_repair_works_without_a_derived_index():
    data = {
        "matters": ["a", "b", "c", "d"],
        "conditions": {matter: [False] for matter in ("a", "b", "c", "d")},
        "dependencies": [["a", "b"], ["b", "a"], ["c", "d"], ["d", "c"]],
    }
    draft = memory_draft(data)
    before = copy.deepcopy(draft.as_tuple())
    with pytest.raises(rules.RuleError) as error:
        rules.apply_graph_operation(draft, "set_condition", {"matter_id": "a", "index": 0, "truth": True})
    assert error.value.code == "state_cycle"
    assert error.value.cycle
    assert draft.as_tuple() == before
    result = rules.apply_graph_operation(draft, "remove_dependency", {"source": "b", "target": "a"})
    assert result["changed"] is True
    assert ("b", "a") not in draft.dependencies
    with pytest.raises(rules.RuleError) as error:
        rules.require_acyclic_index(*draft.as_tuple())
    assert error.value.code == "state_cycle"
    rules.apply_graph_operation(draft, "remove_dependency", {"source": "d", "target": "c"})
    assert rules.require_acyclic_index(*draft.as_tuple()).matters == ("a", "b", "c", "d")


def test_legacy_web_toggle_and_condition_replacement_keep_existing_behavior(tmp_path, graph_data):
    path = tmp_path / "graph.json"
    storage.save_state(*storage.state_from_data(graph_data), path=path)
    assert rules.update_conditions(path, "child", {"action": "toggle", "index": 0}) is None
    assert storage.load_state(path)[1]["child"][0]["truth"] is True
    assert rules.update_conditions(path, "child", {"conditions": ["Legacy text condition"]}) is None
    assert storage.load_state(path)[1]["child"] == [{"label": "Legacy text condition", "truth": False}]
    assert rules.update_conditions(path, "leaf", {"action": "delete", "index": 0}) is None
    assert storage.load_state(path)[1]["leaf"] == []


def test_state_codecs_preserve_legacy_normalization_and_file_parity(tmp_path):
    legacy = {
        "schema_version": 1,
        "matters": ["b", "a"],
        "conditions": {"a": [True, False], "b": {"Named check": True}},
        "dependencies": [["a", "b"], ["a", "b"]],
    }
    snapshot = storage.state_from_data(legacy)
    encoded = storage.state_data(*snapshot)
    assert encoded == {
        "schema_version": 2,
        "matters": ["a", "b"],
        "conditions": {
            "a": [
                {"label": "Unlabeled legacy condition 1", "truth": True},
                {"label": "Unlabeled legacy condition 2", "truth": False},
            ],
            "b": [{"label": "Named check", "truth": True}],
        },
        "dependencies": [["a", "b"]],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    assert storage.load_state(path) == snapshot
    storage.save_state(*snapshot, path=path)
    assert json.loads(path.read_text()) == encoded
    assert storage.state_from_data(encoded) == snapshot


def test_state_codecs_are_pure_and_do_not_mutate_input(graph_data, monkeypatch):
    original = copy.deepcopy(graph_data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Codec attempted file I/O")

    monkeypatch.setattr(storage.Path, "open", forbidden)
    snapshot = storage.state_from_data(graph_data)
    assert storage.state_data(*snapshot)["schema_version"] == 2
    assert graph_data == original


@pytest.mark.parametrize("data", [
    None,
    {"matters": [], "conditions": [], "dependencies": []},
    {"matters": ["a"], "conditions": {"ghost": []}, "dependencies": []},
    {"matters": ["a"], "conditions": {}, "dependencies": [["a", "ghost"]]},
])
def test_state_decoder_rejects_structurally_invalid_documents(data):
    with pytest.raises(ValueError):
        storage.state_from_data(data)
