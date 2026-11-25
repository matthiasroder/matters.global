import json

import pytest

from matters import rules
from matters.local_service import LocalGraphService, LocalServiceError
from matters.storage import load_state, save_state, state_revision
from matters.web import graph_payload


@pytest.fixture
def local_graphs(tmp_path):
    project = tmp_path / "project.json"
    personal = tmp_path / "personal.json"
    save_state(
        {"root", "child", "leaf"},
        {
            "root": [{"label": "Root verified", "truth": True}],
            "child": [
                {"label": "First check", "truth": False},
                {"label": "Second check", "truth": False},
            ],
            "leaf": [{"label": "Leaf delivered", "truth": False}],
        },
        {("root", "child"), ("child", "leaf")},
        path=project,
    )
    save_state(
        {"private"}, {"private": [{"label": "Personal check", "truth": False}]},
        set(), path=personal,
    )
    return LocalGraphService({"project": project, "personal": personal}), project, personal


def revision(service, graph_id="project"):
    return service.read(graph_id, view="state")["revision"]


def change(service, operation, payload, graph_id="project", expected_revision=None):
    return service.mutate(
        graph_id, operation, payload,
        expected_revision=expected_revision or revision(service, graph_id),
    )


def test_catalog_publishes_only_configured_ids_not_paths(local_graphs):
    service, project, _ = local_graphs
    catalog = service.catalog()
    assert catalog["schema_version"] == 1
    assert [entry["graph_id"] for entry in catalog["graphs"]] == ["personal", "project"]
    assert all(entry["read_only"] is False for entry in catalog["graphs"])
    assert str(project.parent) not in json.dumps(catalog)


def test_configured_graph_id_syntax_is_shared_with_transport(local_graphs):
    _, project, _ = local_graphs
    service = LocalGraphService({"project.v1-test_2": project})
    assert service.catalog()["graphs"][0]["graph_id"] == "project.v1-test_2"
    for graph_id in ("_hidden", "-leading", ".dot", "a" * 129):
        with pytest.raises(LocalServiceError):
            LocalGraphService({graph_id: project})


@pytest.mark.parametrize("graph_id", ["missing", "../personal.json", "", None, [], 1])
def test_unconfigured_or_invalid_graph_id_is_forbidden(local_graphs, graph_id):
    service, project, personal = local_graphs
    original = (project.read_bytes(), personal.read_bytes())
    with pytest.raises(LocalServiceError) as error:
        service.read(graph_id)
    assert error.value.code == "forbidden"
    assert str(project.parent) not in str(error.value)
    with pytest.raises(LocalServiceError) as error:
        service.mutate(graph_id, "set_condition", {}, expected_revision="revision")
    assert error.value.code == "forbidden"
    assert (project.read_bytes(), personal.read_bytes()) == original


def test_readonly_rejects_changes_without_creating_locks(local_graphs):
    _, project, personal = local_graphs
    service = LocalGraphService({"project": project}, read_only=True)
    original = project.read_bytes()
    with pytest.raises(LocalServiceError) as error:
        change(service, "set_condition", {"matter_id": "child", "index": 0, "truth": True})
    assert error.value.code == "forbidden"
    assert service.catalog()["graphs"][0]["read_only"] is True
    assert project.read_bytes() == original
    assert not rules.state_lock_path(project).exists()
    assert personal.exists()


def test_reads_share_snapshot_revision_and_engine_views(local_graphs):
    service, project, _ = local_graphs
    before = project.read_bytes()
    responses = [
        service.read("project", view="matter", matter_id="child"),
        service.read("project", view="universe"),
        service.read("project", view="frontier", matter_id="root"),
        service.read("project", view="horizon", matter_id="root"),
        service.read("project", view="unlock"),
        service.read("project", view="state"),
    ]
    assert {response["revision"] for response in responses} == {revision(service)}
    assert responses[0]["conditions"][0] == {
        "index": 0, "label": "First check", "truth": False,
    }
    assert responses[0]["prerequisites"] == ["root"]
    assert responses[0]["dependents"] == ["leaf"]
    assert responses[1]["matters"] == responses[2]["matters"] == ["child"]
    assert responses[3]["matters"] == ["leaf"]
    assert responses[4]["universe"] == ["child"]
    assert responses[5]["state"]["schema_version"] == 2
    assert project.read_bytes() == before
    assert not rules.state_lock_path(project).exists()


def test_paginated_matter_read_and_strict_bounds(local_graphs):
    service, _, _ = local_graphs
    page = service.read("project", view="matters", offset=1, limit=1)
    assert page["matters"] == ["leaf"]
    assert (page["total"], page["offset"], page["limit"]) == (3, 1, 1)
    for kwargs in ({"offset": True}, {"offset": -1}, {"limit": 0}, {"limit": 501}, {"limit": "2"}):
        with pytest.raises(LocalServiceError):
            service.read("project", view="matters", **kwargs)


def test_sources_are_scoped_to_the_selected_graph(local_graphs):
    service, project, _ = local_graphs
    citation = {"title": "Verification", "url": "https://example.org/evidence"}
    project.with_suffix(".sources.json").write_text(json.dumps({
        "schema_version": 1, "matter_sources": {"child": [citation]},
    }))
    result = service.read("project", view="matter", matter_id="child", include_sources=True)
    assert result["sources"] == [citation]
    assert "sources" not in service.read("project", view="matter", matter_id="child")
    assert service.read("personal", view="matter", matter_id="private", include_sources=True)["sources"] == []


def test_condition_updates_match_cli_rules_and_browser_reads(local_graphs):
    service, project, personal = local_graphs
    untouched = personal.read_bytes()
    result = change(service, "set_condition", {"matter_id": "child", "index": 0, "truth": True})
    assert result["graph_id"] == "project"
    assert result["revision"] == rules.describe_matter(project, "child")["revision"]
    assert rules.describe_matter(project, "child")["conditions"][0]["truth"] is True
    web = graph_payload(project)
    assert web["revision"] == result["revision"]
    child = next(node for node in web["nodes"] if node["id"] == "child")
    assert child["conditions"][0]["truth"] is True
    assert personal.read_bytes() == untouched


@pytest.mark.parametrize("index,truth", [(True, True), (-1, True), ("0", True), (0, "false"), (0, 1)])
def test_condition_write_rejects_coerced_indexes_and_booleans(local_graphs, index, truth):
    service, project, _ = local_graphs
    before = project.read_bytes()
    with pytest.raises(LocalServiceError) as error:
        change(service, "set_condition", {"matter_id": "child", "index": index, "truth": truth})
    assert error.value.code == "invalid"
    assert project.read_bytes() == before


@pytest.mark.parametrize("expected", [None, "", "  ", 1, False])
def test_mutations_require_nonempty_revision(local_graphs, expected):
    service, project, _ = local_graphs
    before = project.read_bytes()
    with pytest.raises(LocalServiceError) as error:
        service.mutate("project", "set_condition", {
            "matter_id": "child", "index": 0, "truth": True,
        }, expected_revision=expected)
    assert error.value.code == "conflict"
    assert project.read_bytes() == before


def test_stale_condition_position_is_rejected_after_deletion(local_graphs):
    service, project, _ = local_graphs
    stale = revision(service)
    result = change(service, "delete_condition", {"matter_id": "child", "index": 0})
    current = project.read_bytes()
    with pytest.raises(LocalServiceError) as error:
        change(service, "set_condition", {"matter_id": "child", "index": 0, "truth": True}, expected_revision=stale)
    assert error.value.code == "conflict"
    assert project.read_bytes() == current
    assert service.read("project", view="matter", matter_id="child")["conditions"][0]["label"] == "Second check"
    assert result["revision"] != stale


@pytest.mark.parametrize("operation,payload", [
    ("delete_condition", {"matter_id": "child", "index": 0}),
    ("delete_matter", {"matter_id": "leaf", "confirmed": True}),
])
def test_destructive_mutations_check_revision_inside_rules(local_graphs, operation, payload):
    service, project, _ = local_graphs
    before = project.read_bytes()
    with pytest.raises(LocalServiceError) as error:
        change(service, operation, payload, expected_revision="stale")
    assert error.value.code == "conflict"
    assert project.read_bytes() == before


def test_creation_and_condition_edits_use_existing_rules(local_graphs):
    service, _, _ = local_graphs
    change(service, "create_matter", {"id": "new", "conditions": [{"label": "New check", "truth": False}]})
    change(service, "add_condition", {"matter_id": "new", "label": "Evidence reviewed", "truth": True})
    change(service, "edit_condition", {"matter_id": "new", "index": 0, "label": "Deliverable accepted"})
    matter = service.read("project", view="matter", matter_id="new")
    assert matter["conditions"] == [
        {"index": 0, "label": "Deliverable accepted", "truth": False},
        {"index": 1, "label": "Evidence reviewed", "truth": True},
    ]
    change(service, "add_dependency", {"source": "leaf", "target": "new"})
    assert service.read("project", view="matter", matter_id="new")["prerequisites"] == ["leaf"]
    change(service, "remove_dependency", {"source": "leaf", "target": "new"})
    assert service.read("project", view="matter", matter_id="new")["prerequisites"] == []


def test_creation_rejects_invalid_condition_truth_without_writing(local_graphs):
    service, project, _ = local_graphs
    before = project.read_bytes()
    with pytest.raises(LocalServiceError) as error:
        change(service, "create_matter", {
            "id": "new", "conditions": [{"label": "New check", "truth": "false"}],
        })
    assert error.value.code == "invalid"
    assert project.read_bytes() == before


def test_cycle_creation_and_unconfirmed_deletion_preserve_graph(local_graphs):
    service, project, _ = local_graphs
    for operation, payload in [
        ("add_dependency", {"source": "leaf", "target": "root"}),
        ("delete_condition", {"matter_id": "leaf", "index": 0}),
        ("delete_matter", {"matter_id": "leaf"}),
        ("delete_matter", {"matter_id": "root", "confirmed": True}),
    ]:
        before = project.read_bytes()
        with pytest.raises(LocalServiceError):
            change(service, operation, payload)
        assert project.read_bytes() == before
    change(service, "delete_condition", {"matter_id": "leaf", "index": 0, "confirmed": True})
    assert service.read("project", view="matter", matter_id="leaf")["conditions"] == []
    change(service, "delete_matter", {"matter_id": "leaf", "confirmed": True})
    assert "leaf" not in service.read("project", view="matters")["matters"]


def test_unknown_fields_cannot_select_a_different_file(local_graphs):
    service, project, personal = local_graphs
    before = (project.read_bytes(), personal.read_bytes())
    with pytest.raises(LocalServiceError):
        change(service, "set_condition", {
            "matter_id": "child", "index": 0, "truth": True,
            "state_path": str(personal),
        })
    assert (project.read_bytes(), personal.read_bytes()) == before


def test_returned_revision_describes_own_write_even_if_another_writer_follows(local_graphs, monkeypatch):
    service, project, _ = local_graphs
    original = rules.update_conditions
    committed = []

    def write_then_another_write(*args, **kwargs):
        original(*args, **kwargs)
        committed.append(state_revision(*load_state(project)))
        rules.set_condition_truth(project, "child", "2", True)

    monkeypatch.setattr(rules, "update_conditions", write_then_another_write)
    response = change(service, "set_condition", {"matter_id": "child", "index": 0, "truth": True})
    assert response["revision"] == committed[0]
    assert response["revision"] != revision(service)


def test_export_revision_describes_exported_snapshot(local_graphs, monkeypatch):
    from matters import portability

    service, project, _ = local_graphs
    original = portability.export_bundle
    exported_revision = revision(service)

    def export_then_another_write(path):
        bundle = original(path)
        rules.set_condition_truth(project, "child", "1", True)
        return bundle

    monkeypatch.setattr(portability, "export_bundle", export_then_another_write)
    response = service.read("project", view="export")
    assert response["revision"] == exported_revision
    assert response["revision"] != revision(service)
    assert response["bundle"]["graph"]["conditions"]["child"][0]["truth"] is False


def test_cycle_repairs_do_not_report_failure_after_partial_success(tmp_path):
    path = tmp_path / "broken.json"
    save_state(
        {"a", "b", "c", "d"},
        {matter: [{"label": "Done", "truth": False}] for matter in ("a", "b", "c", "d")},
        {("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")}, path=path,
    )
    service = LocalGraphService({"broken": path})
    with pytest.raises(LocalServiceError) as error:
        service.read("broken")
    assert error.value.code == "state_cycle"
    assert error.value.cycle
    result = change(service, "remove_dependency", {"source": "b", "target": "a"}, graph_id="broken")
    assert result["revision"] == revision(service, "broken")
    assert ("b", "a") not in load_state(path)[2]
    with pytest.raises(LocalServiceError) as error:
        service.read("broken")
    assert error.value.code == "state_cycle"


def test_missing_graph_is_not_recreated_after_configuration(local_graphs):
    service, project, _ = local_graphs
    project.unlink()
    with pytest.raises(LocalServiceError) as error:
        service.read("project", view="state")
    assert error.value.code == "not_found"
    with pytest.raises(LocalServiceError):
        service.mutate("project", "create_matter", {"id": "new"}, expected_revision="stale")
    assert not project.exists()


def test_configuration_requires_explicit_valid_existing_graphs(tmp_path):
    for graphs in ({}, {"bad/path": tmp_path / "a.json"}, {"valid": ""}, {"valid": tmp_path / "missing.json"}):
        with pytest.raises(LocalServiceError):
            LocalGraphService(graphs)
