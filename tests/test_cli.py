import json

from matters.cli import main


def test_cli_accepts_state_after_command(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "done", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["universe", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "a\n"


def test_cli_unlock_can_emit_json(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "done", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["unlock", "--json", "--state", str(state_path)]) == 0

    assert json.loads(capsys.readouterr().out)["universe"] == ["a"]


def test_cli_extract_reads_text_file_without_saving(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    source_path = tmp_path / "notes.txt"
    state_path.write_text(
        json.dumps({"matters": [], "conditions": {}, "dependencies": []})
    )
    source_path.write_text("Goal: Build shared matter map\n")

    assert (
        main(
            [
                "extract",
                str(source_path),
                "--source-type",
                "notes",
                "--state",
                str(state_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["candidates"][0]["id"] == "build_shared_matter_map"
    assert output["requires_confirmation"] is True
    assert json.loads(state_path.read_text()) == {
        "matters": [],
        "conditions": {},
        "dependencies": [],
    }


def test_cli_export_public_uses_visibility_file(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    visibility_path = tmp_path / "visibility.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["public_goal", "private_goal"],
                "conditions": {
                    "public_goal": [{"label": "ready", "truth": False}],
                    "private_goal": [{"label": "secret", "truth": False}],
                },
                "dependencies": [["private_goal", "public_goal"]],
            }
        )
    )
    visibility_path.write_text(
        json.dumps({"public_goal": "public", "private_goal": "private"})
    )

    assert (
        main(
            [
                "export-public",
                "--state",
                str(state_path),
                "--visibility",
                str(visibility_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["matters"] == ["public_goal"]
    assert output["dependencies"] == []


def test_cli_merge_public_prints_merged_state(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    public_path = tmp_path / "public.json"
    visibility_path = tmp_path / "visibility.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["public_goal"],
                "conditions": {
                    "public_goal": [{"label": "ready", "truth": False}],
                },
                "dependencies": [],
            }
        )
    )
    public_path.write_text(
        json.dumps(
            {
                "matters": ["public_goal"],
                "conditions": {
                    "public_goal": [{"label": "ready", "truth": True}],
                },
                "dependencies": [],
            }
        )
    )
    visibility_path.write_text(json.dumps({"public_goal": "public"}))

    assert (
        main(
            [
                "merge-public",
                "--state",
                str(state_path),
                "--public-state",
                str(public_path),
                "--visibility",
                str(visibility_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["conditions"]["public_goal"][0]["truth"] is True
