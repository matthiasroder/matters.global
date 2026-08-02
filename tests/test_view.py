"""`matters view`: one matter, one self-contained HTML file.

Two properties carry this feature and both are pinned here. The **slice** is
everything connected to the matter unless ``--depth`` narrows it, and status
always comes from the whole graph even when the slice is narrowed. The
**file** stands alone: no stylesheet link, no script source, no fetch, no
absolute URL of any kind, so it opens from disk and survives being mailed.
"""

import json
import webbrowser
from importlib import resources

import pytest

from matters import rules
from matters.cli import main
from matters.view import (
    RENDERER_ASSET,
    RENDERER_EXPORT,
    build_view_payload,
    read_renderer_source,
    render_view_html,
    write_view,
)


# a -> b -> c -> d, plus one matter connected to nothing.
CHAIN = {("a", "b"), ("b", "c"), ("c", "d")}
MATTERS = {"a", "b", "c", "d", "lonely"}


def conditions_for(matters, resolved=()):
    return {
        matter: [{"label": f"{matter} done", "truth": matter in resolved}]
        for matter in matters
    }


def payload_for(
    matter,
    *,
    matters=MATTERS,
    dependencies=CHAIN,
    resolved=("a",),
    depth=None,
):
    return build_view_payload(
        matter,
        set(matters),
        conditions_for(matters, resolved),
        set(dependencies),
        depth=depth,
    )


def sliced(payload):
    return sorted(node["id"] for node in payload["nodes"])


def write_state(path, matters=MATTERS, dependencies=CHAIN, resolved=("a",)):
    path.write_text(
        json.dumps(
            {
                "matters": sorted(matters),
                "conditions": conditions_for(matters, resolved),
                "dependencies": sorted([list(edge) for edge in dependencies]),
            }
        )
    )
    return path


# ---------------------------------------------------------------------------
# The slice
# ---------------------------------------------------------------------------


def test_the_default_slice_is_everything_connected_to_the_matter():
    """D3: the whole ancestor chain and the whole dependent chain.

    Not a neighbourhood: the question `view` answers is what this matter's
    predicament is, which reaches as far as the graph does. The matter
    connected to nothing is the proof that the slice is connectivity and not
    the whole file.
    """

    payload = payload_for("c")

    assert sliced(payload) == ["a", "b", "c", "d"]
    assert payload["edges"] == [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
        {"source": "c", "target": "d"},
    ]
    assert payload["depth"] is None
    assert payload["ancestors"] == ["a", "b"]
    assert payload["direct_dependents"] == ["d"]


def test_depth_narrows_the_slice_in_both_directions():
    assert sliced(payload_for("c", depth=0)) == ["c"]
    assert sliced(payload_for("c", depth=1)) == ["b", "c", "d"]
    assert sliced(payload_for("c", depth=2)) == ["a", "b", "c", "d"]
    assert payload_for("c", depth=0)["edges"] == []
    assert payload_for("c", depth=1)["edges"] == [
        {"source": "b", "target": "c"},
        {"source": "c", "target": "d"},
    ]


def test_depth_counts_hops_in_each_direction_not_across_the_matter():
    """A sibling is two hops away, up and back down, so it is not depth 1."""

    matters = {"prerequisite", "target", "sibling"}
    dependencies = {("prerequisite", "target"), ("prerequisite", "sibling")}

    assert sliced(
        payload_for(
            "target", matters=matters, dependencies=dependencies, depth=1
        )
    ) == ["prerequisite", "target"]
    # ...and it is not in the unlimited slice either: nothing connects it to
    # the target except through their shared prerequisite.
    assert sliced(
        payload_for("target", matters=matters, dependencies=dependencies)
    ) == ["prerequisite", "target"]


def test_a_matter_connected_to_nothing_is_its_own_slice():
    payload = payload_for("lonely")

    assert sliced(payload) == ["lonely"]
    assert payload["edges"] == []
    assert payload["ancestors"] == []
    assert payload["direct_dependents"] == []
    assert payload["blocking"] == []


def test_an_edge_is_kept_only_when_both_of_its_endpoints_are():
    """The induced subgraph rule, seen from the page.

    `d`'s other prerequisite is outside the slice, so the edge into `d` from
    it must not be in the payload: the renderer would draw an arrow from
    nowhere.
    """

    matters = MATTERS | {"outsider"}
    dependencies = CHAIN | {("outsider", "d")}
    payload = payload_for(
        "c", matters=matters, dependencies=dependencies, depth=1
    )

    assert sliced(payload) == ["b", "c", "d"]
    assert {"source": "outsider", "target": "d"} not in payload["edges"]
    node = next(node for node in payload["nodes"] if node["id"] == "d")
    assert node["prerequisites"] == ["c"]


def test_a_negative_depth_is_refused():
    with pytest.raises(ValueError, match="depth must not be negative: -1"):
        payload_for("c", depth=-1)


def test_an_unknown_matter_is_refused_by_name():
    with pytest.raises(rules.RuleError, match="unknown matter: nope") as error:
        payload_for("nope")

    assert error.value.code == "not_found"


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


def test_the_node_and_edge_shape_is_the_one_the_renderer_reads():
    """map-renderer's `normalizeNode` reads exactly these coordinate keys.

    Asserted as literal key sets rather than by sampling: the renderer falls
    back to a placeholder ring for a coordinate it cannot find, so a renamed
    key produces a plausible wrong picture instead of an error.
    """

    payload = payload_for("c")
    node = next(node for node in payload["nodes"] if node["id"] == "c")

    assert set(node) == {
        "id",
        "label",
        "conditions",
        "prerequisites",
        "dependents",
        "resolved",
        "actionable",
        "blocked",
        "overview",
    }
    assert set(node["overview"]) == {
        "x",
        "y",
        "z",
        "depth",
        "orbit_level",
        "downstream_impact",
        "system",
        "system_count",
        "system_population",
        "system_radius",
        "orbit_radius",
    }
    assert all(set(edge) == {"source", "target"} for edge in payload["edges"])
    assert node["label"] == "c"
    assert payload_for("a")["nodes"][0]["label"] == "a"


def test_the_label_opens_out_underscores_the_way_the_web_payload_does():
    matters = {"go_to_mars", "build_ship"}
    payload = payload_for(
        "go_to_mars",
        matters=matters,
        dependencies={("build_ship", "go_to_mars")},
    )

    assert [node["label"] for node in payload["nodes"]] == [
        "build ship",
        "go to mars",
    ]


def test_status_comes_from_the_whole_graph_not_from_the_slice():
    """The one thing a slice may not be allowed to change.

    `c` waits on `b`, which is unresolved. Under `--depth 0` the slice is
    `c` alone, and an index built over that slice would call it actionable,
    which is the opposite of true. Only the layout is slice-local.
    """

    payload = payload_for("c", depth=0)
    node = payload["nodes"][0]

    assert node["id"] == "c"
    assert node["resolved"] is False
    assert node["actionable"] is False
    assert node["blocked"] is True


def test_the_blocking_set_is_the_unresolved_part_of_the_ancestry():
    payload = payload_for("d", resolved=("a",))

    assert payload["ancestors"] == ["a", "b", "c"]
    assert payload["blocking"] == ["b", "c"]


def test_the_layout_is_built_over_the_slice_alone():
    """D4: an independent coordinate space, not a window onto the big one."""

    whole = payload_for("c")
    narrowed = payload_for("c", depth=1)

    assert whole["overview_layout"]["algorithm"] == "solar-systems-v1"
    # The depth the layout sees is the slice's own: `a` is gone from the
    # narrowed slice, so the longest chain in it is one edge shorter.
    assert whole["overview_layout"]["max_depth"] == 3
    assert narrowed["overview_layout"]["max_depth"] == 2
    positions = {
        node["id"]: (node["overview"]["x"], node["overview"]["z"])
        for node in narrowed["nodes"]
    }
    assert len(set(positions.values())) == len(positions)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_the_page_reaches_the_network_nowhere():
    """The whole promise of the file, asserted the blunt way."""

    html = render_view_html(payload_for("c"))

    assert "http://" not in html
    assert "https://" not in html
    assert "fetch(" not in html
    assert "<link" not in html
    assert "src=" not in html
    assert "@import" not in html
    assert "import " not in html


def test_the_renderer_is_inlined_verbatim_apart_from_its_export_keyword():
    asset = (
        resources.files("matters")
        .joinpath("web_assets")
        .joinpath(RENDERER_ASSET)
        .read_text(encoding="utf-8")
    )
    inlined = read_renderer_source()
    html = render_view_html(payload_for("c"))

    assert RENDERER_EXPORT in asset
    assert len(asset) - len(inlined) == len("export ")
    assert inlined == asset.replace(
        RENDERER_EXPORT, "function createOverviewRenderer(", 1
    )
    assert inlined in html
    assert "export " not in html


def test_a_condition_label_cannot_close_the_script_element():
    """Free text reaches the page as JSON inside a script element."""

    matters = {"a"}
    conditions = {"a": [{"label": "</script><img> & <b>", "truth": False}]}
    payload = build_view_payload("a", matters, conditions, set())
    html = render_view_html(payload)

    assert html.count("</script>") == 2
    assert "\\u003c/script\\u003e" in html
    assert "<img>" not in html


def test_the_page_names_the_matter_it_is_about():
    html = render_view_html(payload_for("c"))

    assert "<title>matters view: c</title>" in html
    assert '"matter": "c"' in html


def test_the_payload_is_embedded_as_data_the_page_can_parse():
    """The graph travels as a literal, so the page needs no server."""

    html = render_view_html(payload_for("c"))
    embedded = html.split("const VIEW = ", 1)[1].split("\nconst byId", 1)[0]

    assert json.loads(embedded.rstrip(";"))["matter"] == "c"


# ---------------------------------------------------------------------------
# A state file that contains a cycle
# ---------------------------------------------------------------------------


CYCLIC = {("a", "b"), ("b", "a"), ("b", "c"), ("c", "d")}


def test_a_cyclic_state_file_still_gets_a_page():
    """`view` must not become a fourth read verb that refuses.

    `show` and `list` already guarantee a broken file stays inspectable, and
    a picture is what someone holding a tangled graph came for. What goes is
    the derived status, which genuinely cannot be computed, and it goes
    visibly rather than quietly.
    """

    payload = payload_for("c", dependencies=CYCLIC)

    assert payload["status_available"] is False
    assert payload["cycle"] == ["a", "b"]
    assert sliced(payload) == ["a", "b", "c", "d"]
    assert all(node["resolved"] is None for node in payload["nodes"])
    assert all(node["actionable"] is None for node in payload["nodes"])
    assert all(node["blocked"] is None for node in payload["nodes"])
    assert payload["blocking"] == []
    assert payload["ancestors"] == ["a", "b"]
    assert payload["direct_dependents"] == ["d"]


def test_the_cyclic_page_says_so_where_a_reader_will_see_it():
    page = render_view_html(payload_for("c", dependencies=CYCLIC))

    assert "Structure only." in page
    # `rules.format_cycle` is the one renderer of a cycle; the arrows are
    # HTML-escaped on the way into the banner and read back as `a -> b -> a`.
    assert rules.format_cycle(("a", "b")) == "a -> b -> a"
    assert "a -&gt; b -&gt; a" in page
    assert "http://" not in page
    assert "https://" not in page


def test_the_cyclic_slice_keeps_the_same_node_shape():
    node = next(
        node
        for node in payload_for("c", dependencies=CYCLIC)["nodes"]
        if node["id"] == "c"
    )

    assert set(node["overview"]) == {
        "x",
        "y",
        "z",
        "depth",
        "orbit_level",
        "downstream_impact",
        "system",
        "system_count",
        "system_population",
        "system_radius",
        "orbit_radius",
    }


def test_the_cyclic_fallback_puts_the_matter_at_the_centre_of_hop_rings():
    payload = payload_for("c", dependencies=CYCLIC)
    overview = {node["id"]: node["overview"] for node in payload["nodes"]}

    assert payload["overview_layout"] == {
        "version": 1,
        "algorithm": "hop-rings-v1",
        "max_depth": 2,
        "system_count": 1,
    }
    assert (overview["c"]["x"], overview["c"]["y"], overview["c"]["z"]) == (
        0.0,
        0.0,
        0.0,
    )
    # One hop up (`b`) and one hop down (`d`) share a ring; `a` is two hops
    # up, behind the loop, and sits further out.
    assert overview["b"]["orbit_level"] == overview["d"]["orbit_level"] == 1
    assert overview["a"]["orbit_level"] == 2
    assert overview["a"]["orbit_radius"] > overview["b"]["orbit_radius"] > 0
    assert overview["b"]["orbit_radius"] == overview["d"]["orbit_radius"]
    assert {node["overview"]["system"] for node in payload["nodes"]} == {"c"}


def test_depth_still_narrows_a_cyclic_slice():
    assert sliced(payload_for("c", dependencies=CYCLIC, depth=1)) == [
        "b",
        "c",
        "d",
    ]
    assert sliced(payload_for("c", dependencies=CYCLIC, depth=0)) == ["c"]


def test_a_self_loop_does_not_hang_the_walk():
    payload = payload_for("c", dependencies=CHAIN | {("c", "c")})

    assert payload["status_available"] is False
    assert payload["cycle"] == ["c"]
    assert sliced(payload) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# Writing and opening
# ---------------------------------------------------------------------------


def test_write_view_writes_one_file_and_returns_its_payload(tmp_path):
    state_path = write_state(tmp_path / "matters.json")
    output = tmp_path / "pages" / "c.html"

    path, payload = write_view("c", state_path=state_path, output=output)

    assert path == output
    assert payload["matter"] == "c"
    assert "<title>matters view: c</title>" in output.read_text(encoding="utf-8")
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "matters.json",
        "pages",
    ]


def test_view_writes_where_output_says(tmp_path, capsys):
    state_path = write_state(tmp_path / "matters.json")
    output = tmp_path / "elsewhere.html"

    assert (
        main(
            [
                "view",
                "c",
                "--state",
                str(state_path),
                "--output",
                str(output),
                "--no-open",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == (
        "c: 4 matters, 3 dependencies\n" f"wrote {output}\n"
    )
    assert output.exists()


def test_view_defaults_to_the_working_directory(tmp_path, monkeypatch, capsys):
    state_path = write_state(tmp_path / "matters.json")
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    assert main(["view", "c", "--state", str(state_path), "--no-open"]) == 0

    capsys.readouterr()
    assert [item.name for item in working.iterdir()] == ["matters-view-c.html"]


def test_view_opens_the_written_file_and_no_open_stops_it(
    tmp_path, monkeypatch, capsys
):
    state_path = write_state(tmp_path / "matters.json")
    output = tmp_path / "c.html"
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    assert (
        main(
            [
                "view",
                "c",
                "--state",
                str(state_path),
                "--output",
                str(output),
                "--no-open",
            ]
        )
        == 0
    )
    assert opened == []

    assert (
        main(
            ["view", "c", "--state", str(state_path), "--output", str(output)]
        )
        == 0
    )

    capsys.readouterr()
    assert opened == [output.resolve().as_uri()]
    assert opened[0].startswith("file://")


def test_view_takes_no_lock_and_leaves_the_state_file_alone(
    tmp_path, monkeypatch, capsys
):
    """AC-10: `view` is a read. It writes its page and nothing else."""

    state_path = tmp_path / "matters.json"
    # Non-canonical on purpose: a read must not rewrite the file into
    # save_state's canonical form either.
    state_path.write_text(
        '{"matters": ["b", "a"], "conditions": {"a": [], "b": []},'
        ' "dependencies": [["b", "a"]]}'
    )
    original = state_path.read_bytes()
    output = tmp_path / "pages" / "a.html"

    def refuse(*args, **kwargs):
        raise AssertionError("view must not open a state transaction")

    monkeypatch.setattr(rules, "state_transaction", refuse)

    assert (
        main(
            [
                "view",
                "a",
                "--state",
                str(state_path),
                "--output",
                str(output),
                "--no-open",
            ]
        )
        == 0
    )

    capsys.readouterr()
    assert state_path.read_bytes() == original
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "matters.json",
        "pages",
    ]
