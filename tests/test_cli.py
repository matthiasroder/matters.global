import json

import pytest

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


def test_cli_create_writes_shorthand_dependency_chain(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps({"matters": [], "conditions": {}, "dependencies": []})
    )

    assert (
        main(
            [
                "create",
                (
                    "go to Mars "
                    "(human lands and stays on Mars for at least one year) > "
                    "build spaceship that can fly to Mars > "
                    "assemble spaceship in earth orbit"
                ),
                "--state",
                str(state_path),
            ]
        )
        == 0
    )

    assert "Created matters" in capsys.readouterr().out
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": [
            "assemble_spaceship_in_earth_orbit",
            "build_spaceship_that_can_fly_to_mars",
            "go_to_mars",
        ],
        "conditions": {
            "assemble_spaceship_in_earth_orbit": [
                {
                    "label": "Resolved: assemble spaceship in earth orbit",
                    "truth": False,
                }
            ],
            "build_spaceship_that_can_fly_to_mars": [
                {
                    "label": "Resolved: build spaceship that can fly to Mars",
                    "truth": False,
                }
            ],
            "go_to_mars": [
                {
                    "label": "human lands and stays on Mars for at least one year",
                    "truth": False,
                }
            ],
        },
        "dependencies": [
            [
                "assemble_spaceship_in_earth_orbit",
                "build_spaceship_that_can_fly_to_mars",
            ],
            ["build_spaceship_that_can_fly_to_mars", "go_to_mars"],
        ],
    }


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
                "--no-llm",
                "--state",
                str(state_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["candidates"][0]["id"] == "build_shared_matter_map"
    assert output["requires_confirmation"] is True
    assert output["engine"] == "marker"
    assert json.loads(state_path.read_text()) == {
        "matters": [],
        "conditions": {},
        "dependencies": [],
    }


def test_cli_tots_passes_options_and_never_writes_state(
    tmp_path, capsys, monkeypatch
):
    state_path = tmp_path / "matters.json"
    context_path = tmp_path / "evidence.txt"
    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    original = json.dumps(
        {
            "matters": ["open_question"],
            "conditions": {
                "open_question": [{"label": "answered", "truth": False}]
            },
            "dependencies": [],
        },
        indent=2,
    )
    state_path.write_text(original)
    context_path.write_text("Evidence line")
    received = {}

    def fake_build(matter, matters, conditions, dependencies, **kwargs):
        received.update(
            matter=matter,
            matters=matters,
            conditions=conditions,
            dependencies=dependencies,
            kwargs=kwargs,
        )
        return {
            "target": matter,
            "requires_confirmation": True,
            "state_modified": False,
        }

    monkeypatch.setattr("matters.cli.build_tots_proposal", fake_build)

    assert (
        main(
            [
                "tots",
                "open_question",
                "--context",
                str(context_path),
                "--breadth",
                "3",
                "--depth",
                "1",
                "--max-candidates",
                "5",
                "--max-comparisons",
                "8",
                "--model",
                "test-model",
                "--llm-profile",
                "personal",
                "--config",
                str(config_path),
                "--state",
                str(state_path),
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["state_modified"] is False
    assert received["matter"] == "open_question"
    assert received["kwargs"] == {
        "context_text": "Evidence line",
        "breadth": 3,
        "depth": 1,
        "max_candidates": 5,
        "max_comparisons": 8,
        "model": "test-model",
        "config_path": str(config_path),
        "llm_profile": "personal",
    }
    assert state_path.read_text() == original


def test_cli_config_path_accepts_global_option_before_command(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text("")

    assert main(["--config", str(config_path), "config", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(config_path)


def test_cli_config_check_is_sanitized_and_does_not_load_state(
    tmp_path, capsys, monkeypatch
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    received = {}

    def fake_diagnostics(config, profile_name=None):
        received["path"] = str(config.path)
        received["profile"] = profile_name
        return {"ready": True, "credential_available": True}

    monkeypatch.setattr("matters.cli.config_diagnostics", fake_diagnostics)

    assert (
        main(
            [
                "config",
                "check",
                "--profile",
                "personal",
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ready"] is True
    assert received == {"path": str(config_path), "profile": "personal"}


def test_cli_tots_surfaces_actionable_error_without_writing_state(
    tmp_path, capsys, monkeypatch
):
    state_path = tmp_path / "matters.json"
    original = json.dumps(
        {
            "matters": ["open_question"],
            "conditions": {"open_question": []},
            "dependencies": [],
        }
    )
    state_path.write_text(original)

    def fail(*_args, **_kwargs):
        from matters import TotsError

        raise TotsError("model credential is unavailable")

    monkeypatch.setattr("matters.cli.build_tots_proposal", fail)

    with pytest.raises(SystemExit) as error:
        main(["tots", "open_question", "--state", str(state_path)])

    assert error.value.code == 2
    assert "model credential is unavailable" in capsys.readouterr().err
    assert state_path.read_text() == original


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
