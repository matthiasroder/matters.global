import json

import pytest

from matters.cli import main


@pytest.fixture
def state_path(tmp_path):
    path = tmp_path / "research-2026.json"
    path.write_text(json.dumps({
        "matters": ["research_question"],
        "conditions": {},
        "dependencies": [],
    }))
    return path


def test_show_sources_are_opt_in_and_read_only(state_path, capsys):
    source_path = state_path.with_suffix(".sources.json")
    sources = [
        {
            "title": "An empirical study",
            "authors": ["Léa Example"],
            "url": "https://example.org/study.pdf",
            "pdf_pages": [[2, 2], [5, 7]],
        },
        {"title": "Review", "url": "https://example.org/review"},
    ]
    source_path.write_text(json.dumps({
        "schema_version": 1,
        "matter_sources": {"research_question": sources},
    }), encoding="utf-8")
    original_state = state_path.read_bytes()
    original_source = source_path.read_bytes()
    command = ["show", "research_question", "--state", str(state_path)]

    assert main(command + ["--json"]) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert "sources" not in baseline
    assert main(command + ["--sources", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {**baseline, "sources": sources}
    assert main(command + ["--sources"]) == 0
    assert capsys.readouterr().out.endswith(
        "sources:\n"
        "  An empirical study — Léa Example\n"
        "    https://example.org/study.pdf\n"
        "    PDF pages 2, 5–7\n"
        "  Review\n"
        "    https://example.org/review\n"
    )
    assert state_path.read_bytes() == original_state
    assert source_path.read_bytes() == original_source
    assert set(state_path.parent.iterdir()) == {state_path, source_path}


def test_show_sources_can_be_missing(state_path, capsys):
    command = ["show", "research_question", "--sources", "--json", "--state", str(state_path)]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["sources"] == []
    state_path.with_suffix(".sources.json").write_text(
        '{"schema_version": 1, "matter_sources": {}}'
    )
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["sources"] == []


@pytest.mark.parametrize(
    ("source_contents", "message"),
    [
        ("{", "Invalid sources file"),
        ('{"schema_version": true, "matter_sources": {}}', "Expected schema_version 1"),
        (
            json.dumps({"schema_version": 1, "matter_sources": {
                "research_question": [
                    {"title": "Study", "url": "https://example.org", "pdf_pages": [[7, 5]]}
                ]
            }}),
            "Expected each PDF page range",
        ),
    ],
)
def test_show_reports_invalid_sources_only_when_requested(
    state_path, capsys, source_contents, message
):
    source_path = state_path.with_suffix(".sources.json")
    source_path.write_text(source_contents)
    command = ["show", "research_question", "--state", str(state_path)]

    assert main(command) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        main(command + ["--sources"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(source_path) in captured.err
    assert message in captured.err
