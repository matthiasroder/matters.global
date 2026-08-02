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


def test_cli_mark_sets_one_condition_and_prints_one_line(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "a done", "truth": False},
                        {"label": "ship it", "truth": False},
                    ]
                },
                "dependencies": [],
            }
        )
    )

    assert main(["mark", "a", "2", "true", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: condition 2 "ship it" is now true\n'
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {
            "a": [
                {"label": "a done", "truth": False},
                {"label": "ship it", "truth": True},
            ]
        },
        "dependencies": [],
    }


def test_cli_mark_is_idempotent_and_leaves_the_file_byte_identical(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # Deliberately non-canonical: no schema_version, keys in another order.
    # An unconditional save would rewrite these bytes.
    state_path.write_text(
        '{"matters": ["a"], "conditions": {"a": [{"truth": true, "label": "done"}]},'
        ' "dependencies": []}'
    )
    original = state_path.read_bytes()

    assert main(["mark", "a", "1", "true", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: condition 1 "done" is now true\n'
    assert state_path.read_bytes() == original


def test_cli_mark_false_sets_a_true_condition_back(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "a done", "truth": True},
                        {"label": "ship it", "truth": True},
                    ]
                },
                "dependencies": [],
            }
        )
    )

    assert main(["mark", "a", "1", "false", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: condition 1 "a done" is now false\n'
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {
            "a": [
                {"label": "a done", "truth": False},
                {"label": "ship it", "truth": True},
            ]
        },
        "dependencies": [],
    }


def test_cli_mark_addresses_a_condition_by_exact_label(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "a done", "truth": False},
                        {"label": "ship it", "truth": False},
                    ]
                },
                "dependencies": [],
            }
        )
    )

    assert main(["mark", "a", "  ship it  ", "true", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: condition 2 "ship it" is now true\n'
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": False},
        {"label": "ship it", "truth": True},
    ]


def test_cli_mark_rejects_an_unknown_matter_without_writing(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["mark", "ghost", "1", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "unknown matter: ghost" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_mark_rejects_an_unresolvable_condition_reference(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "nope", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert (
        "matter a has no condition matching 'nope'; "
        "run `matters show a` to see condition numbers"
    ) in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_mark_rejects_an_out_of_range_condition_number(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "0", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "matter a has no condition 0 (it has 1)" in captured.err
    assert state_path.read_bytes() == original


def test_cli_mark_rejects_a_truth_value_that_is_not_true_or_false(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "1", "yes", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "invalid choice: 'yes'" in captured.err
    assert "'true', 'false'" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_mark_requires_all_three_positional_arguments(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "1", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "the following arguments are required: truth" in captured.err
    assert state_path.read_bytes() == original


def test_cli_mark_refuses_a_duplicate_label_and_lists_the_positions(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "ship it", "truth": False},
                        {"label": "other", "truth": False},
                        {"label": "ship it", "truth": True},
                    ]
                },
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "ship it", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert (
        "matter a has 2 conditions labelled 'ship it' (positions 1, 3); "
        "address it by number"
    ) in captured.err
    assert state_path.read_bytes() == original


def test_cli_add_condition_appends_a_stripped_label_that_starts_false(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": True}]},
                "dependencies": [],
            }
        )
    )

    assert main(["add-condition", "a", "  ship it  ", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: added condition 2 "ship it" (false)\n'
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {
            "a": [
                {"label": "a done", "truth": True},
                {"label": "ship it", "truth": False},
            ]
        },
        "dependencies": [],
    }


def test_cli_add_condition_accepts_a_duplicate_label(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "ship it", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["add-condition", "a", "ship it", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: added condition 2 "ship it" (false)\n'
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "ship it", "truth": False},
        {"label": "ship it", "truth": False},
    ]


def test_cli_add_condition_falls_back_to_the_generated_label(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["add-condition", "a", "   ", "--state", str(state_path)]) == 0

    assert (
        capsys.readouterr().out
        == 'a: added condition 2 "Unlabeled condition 2" (false)\n'
    )
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": False},
        {"label": "Unlabeled condition 2", "truth": False},
    ]


def test_cli_add_condition_round_trips_non_ascii_labels(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps({"matters": ["a"], "conditions": {"a": []}, "dependencies": []})
    )

    assert (
        main(["add-condition", "a", "Anträge einreichen", "--state", str(state_path)])
        == 0
    )
    assert main(["add-condition", "a", "提交申请", "--state", str(state_path)]) == 0
    assert main(["add-condition", "a", "ship it 🚀", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == (
        'a: added condition 1 "Anträge einreichen" (false)\n'
        'a: added condition 2 "提交申请" (false)\n'
        'a: added condition 3 "ship it 🚀" (false)\n'
    )
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "Anträge einreichen", "truth": False},
        {"label": "提交申请", "truth": False},
        {"label": "ship it 🚀", "truth": False},
    ]


def test_cli_add_condition_stores_a_numeric_label_that_is_only_addressable_by_number(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["add-condition", "a", "42", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: added condition 2 "42" (false)\n'
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "a done", "truth": False},
        {"label": "42", "truth": False},
    ]

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "42", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "matter a has no condition 42 (it has 2)" in captured.err


def test_cli_add_condition_accepts_a_flag_shaped_label_after_a_double_dash(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps({"matters": ["a"], "conditions": {"a": []}, "dependencies": []})
    )

    assert (
        main(["add-condition", "a", "--state", str(state_path), "--", "--force"]) == 0
    )

    assert capsys.readouterr().out == 'a: added condition 1 "--force" (false)\n'
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "--force", "truth": False}
    ]


def test_cli_add_condition_rejects_an_unknown_matter_without_writing(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps({"matters": ["a"], "conditions": {"a": []}, "dependencies": []})
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["add-condition", "ghost", "ship it", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "unknown matter: ghost" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_edit_condition_preserves_truth_and_position(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "a done", "truth": False},
                        {"label": "ship it", "truth": True},
                        {"label": "tell them", "truth": False},
                    ]
                },
                "dependencies": [],
            }
        )
    )

    assert (
        main(["edit-condition", "a", "2", "shipped", "--state", str(state_path)]) == 0
    )

    assert capsys.readouterr().out == 'a: condition 2 renamed "ship it" -> "shipped"\n'
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {
            "a": [
                {"label": "a done", "truth": False},
                {"label": "shipped", "truth": True},
                {"label": "tell them", "truth": False},
            ]
        },
        "dependencies": [],
    }


def test_cli_edit_condition_rejects_an_unknown_reference_without_writing(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": True}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["edit-condition", "a", "7", "shipped", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "matter a has no condition 7 (it has 1)" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_delete_condition_removes_only_that_condition_and_renumbers(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "first", "truth": False},
                        {"label": "second", "truth": True},
                        {"label": "third", "truth": False},
                    ]
                },
                "dependencies": [],
            }
        )
    )

    assert main(["delete-condition", "a", "1", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: deleted condition 1 "first"\n'
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {
            "a": [
                {"label": "second", "truth": True},
                {"label": "third", "truth": False},
            ]
        },
        "dependencies": [],
    }

    # E-11: the delete renumbered the survivors, so 1 now names "second".
    assert main(["mark", "a", "1", "false", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: condition 1 "second" is now false\n'
    assert json.loads(state_path.read_text())["conditions"]["a"] == [
        {"label": "second", "truth": False},
        {"label": "third", "truth": False},
    ]


def test_cli_delete_condition_refuses_to_empty_a_matter_without_yes(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [["a", "b"]],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["delete-condition", "a", "1", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert (
        "deleting the last condition of a will make it resolved, "
        "unblocking: b — rerun with --yes"
    ) in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_delete_condition_with_yes_empties_a_matter_and_reports_unblocking(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [["a", "b"]],
            }
        )
    )

    assert main(["delete-condition", "a", "1", "--yes", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == (
        'a: deleted condition 1 "a done"\n'
        "a has no conditions left and now counts as resolved; unblocked: b\n"
    )
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a", "b"],
        "conditions": {"a": [], "b": [{"label": "b done", "truth": False}]},
        "dependencies": [["a", "b"]],
    }

    # E-2: the emptied matter really does count as resolved now.
    assert main(["universe", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "b\n"


def test_cli_delete_condition_reports_no_unblocked_matters_as_none(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["delete-condition", "a", "1", "--yes", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == (
        'a: deleted condition 1 "a done"\n'
        "a has no conditions left and now counts as resolved; unblocked: none\n"
    )
    assert json.loads(state_path.read_text())["conditions"]["a"] == []


def test_cli_condition_verbs_report_malformed_json_without_a_traceback(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text("{")
    original = state_path.read_bytes()
    state = ["--state", str(state_path)]

    for argv in (
        ["mark", "a", "1", "true"] + state,
        ["add-condition", "a", "ship it"] + state,
        ["edit-condition", "a", "1", "shipped"] + state,
        ["delete-condition", "a", "1", "--yes"] + state,
    ):
        with pytest.raises(SystemExit) as error:
            main(argv)

        captured = capsys.readouterr()
        assert error.value.code == 2
        assert captured.out == ""
        assert f"state file is not valid JSON: {state_path}: " in captured.err
        assert "Traceback" not in captured.err
        assert state_path.read_bytes() == original


def test_cli_condition_verbs_name_a_missing_state_file_without_creating_one(
    tmp_path, capsys
):
    state_path = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as error:
        main(["mark", "a", "1", "true", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert f"state file does not exist: {state_path}" in captured.err
    assert "Traceback" not in captured.err
    assert sorted(item.name for item in tmp_path.iterdir()) == []


def reloaded_state(state_path):
    """Reload the file exactly as every read verb does.

    AC-7 is an invariant about the file, not about the return value: after a
    delete the file must still load without error, so every delete-matter
    test round-trips through the real loader instead of trusting json.loads.
    """

    from matters.storage import load_state

    return load_state(state_path)


def assert_no_trace_of(matter_id, state_path):
    matters, conditions, dependencies = reloaded_state(state_path)

    assert matter_id not in matters
    assert matter_id not in conditions
    assert not any(matter_id in edge for edge in dependencies)


def test_cli_link_adds_exactly_one_edge_and_prints_one_line(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [],
            }
        )
    )

    assert main(["link", "a", "b", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "a now requires b\n"
    # AC-11 direction: `link a b` means "a needs b", stored prerequisite first.
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a", "b"],
        "conditions": {
            "a": [{"label": "a done", "truth": False}],
            "b": [{"label": "b done", "truth": False}],
        },
        "dependencies": [["b", "a"]],
    }


def test_cli_link_is_idempotent_and_leaves_the_file_byte_identical(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # Deliberately non-canonical: no schema_version, ids out of order. An
    # unconditional save would rewrite these bytes even though nothing changed.
    state_path.write_text(
        '{"matters": ["b", "a"], "conditions": {"a": [], "b": []},'
        ' "dependencies": [["b", "a"]]}'
    )
    original = state_path.read_bytes()

    assert main(["link", "a", "b", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "a already requires b\n"
    assert state_path.read_bytes() == original


def test_cli_unlink_removes_the_edge_and_changes_nothing_else(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b", "c"],
                "conditions": {
                    "a": [{"label": "a done", "truth": True}],
                    "b": [{"label": "b done", "truth": False}],
                    "c": [{"label": "c done", "truth": False}],
                },
                "dependencies": [["b", "a"], ["c", "a"]],
            }
        )
    )

    assert main(["unlink", "a", "b", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "a no longer requires b\n"
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a", "b", "c"],
        "conditions": {
            "a": [{"label": "a done", "truth": True}],
            "b": [{"label": "b done", "truth": False}],
            "c": [{"label": "c done", "truth": False}],
        },
        "dependencies": [["c", "a"]],
    }


def test_cli_unlink_of_an_edge_that_does_not_exist_is_a_byte_identical_no_op(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    # Non-canonical on purpose (F-7): this is what proves the skip-save path
    # rather than a save that happens to produce the same bytes.
    state_path.write_text(
        '{"matters": ["b", "a"], "conditions": {"a": [], "b": []},'
        ' "dependencies": []}'
    )
    original = state_path.read_bytes()

    assert main(["unlink", "a", "b", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "a already does not require b\n"
    assert state_path.read_bytes() == original


def test_cli_link_refuses_an_edge_that_would_close_a_cycle(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [["a", "b"]],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["link", "a", "b", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "a cannot require b: the dependency would create a cycle" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_link_rejects_a_self_link_as_a_cycle(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["link", "a", "a", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "a cannot require a: the dependency would create a cycle" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_link_names_a_missing_endpoint_and_creates_no_matter(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["link", "a", "ghost", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "unknown dependency source: ghost" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original

    with pytest.raises(SystemExit) as error:
        main(["link", "ghost", "a", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "unknown dependency target: ghost" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original

    matters, _conditions, _dependencies = reloaded_state(state_path)
    assert matters == {"a"}


def test_cli_delete_matter_removes_an_isolated_matter_and_reloads_clean(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [],
            }
        )
    )

    assert main(["delete-matter", "a", "--yes", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == (
        "deleted matter a (removed no dependencies)\n"
    )
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["b"],
        "conditions": {"b": [{"label": "b done", "truth": False}]},
        "dependencies": [],
    }
    # AC-7 first branch: the file loads and holds no trace of the id.
    assert_no_trace_of("a", state_path)


def test_cli_delete_matter_also_drops_the_edges_it_depends_on(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # b is a's prerequisite, so a has no dependents and needs no --cascade,
    # yet the edge naming a as its target must still disappear (AC-7).
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [["b", "a"]],
            }
        )
    )

    assert main(["delete-matter", "a", "--yes", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == (
        "deleted matter a (removed 1 dependency)\n"
    )
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["b"],
        "conditions": {"b": [{"label": "b done", "truth": False}]},
        "dependencies": [],
    }
    assert_no_trace_of("a", state_path)


def test_cli_delete_matter_refuses_while_other_matters_depend_on_it(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b", "c"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                    "c": [{"label": "c done", "truth": False}],
                },
                "dependencies": [["a", "b"], ["a", "c"]],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["delete-matter", "a", "--yes", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert (
        "matter a is required by: b, c — rerun with --cascade to delete it "
        "and those dependencies, or unlink them first"
    ) in captured.err
    assert "Traceback" not in captured.err
    # AC-7 second branch: byte-identical, and still a file that loads.
    assert state_path.read_bytes() == original
    matters, conditions, dependencies = reloaded_state(state_path)
    assert matters == {"a", "b", "c"}
    assert conditions["a"] == [{"label": "a done", "truth": False}]
    assert dependencies == {("a", "b"), ("a", "c")}


def test_cli_delete_matter_cascade_removes_the_matter_and_unblocks_dependents(
    tmp_path, capsys
):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b", "c"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                    "c": [{"label": "c done", "truth": False}],
                },
                "dependencies": [["a", "b"], ["a", "c"]],
            }
        )
    )

    assert (
        main(["delete-matter", "a", "--yes", "--cascade", "--state", str(state_path)])
        == 0
    )

    assert capsys.readouterr().out == (
        "deleted matter a (removed 2 dependencies)\n"
        "unblocked: b, c\n"
    )
    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["b", "c"],
        "conditions": {
            "b": [{"label": "b done", "truth": False}],
            "c": [{"label": "c done", "truth": False}],
        },
        "dependencies": [],
    }
    assert_no_trace_of("a", state_path)

    # E-3: the dependents really are actionable now, not just reported as such.
    assert main(["universe", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "b\nc\n"


def test_cli_delete_matter_requires_yes_even_with_no_dependencies(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["delete-matter", "a", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert (
        "deleting matter a removes no dependencies; unblocked: none "
        "— rerun with --yes"
    ) in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original
    assert_no_trace_of("ghost", state_path)


def test_cli_graph_verbs_report_malformed_json_without_a_traceback(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text("{")
    original = state_path.read_bytes()
    state = ["--state", str(state_path)]

    for argv in (
        ["link", "a", "b"] + state,
        ["unlink", "a", "b"] + state,
        ["delete-matter", "a", "--yes", "--cascade"] + state,
    ):
        with pytest.raises(SystemExit) as error:
            main(argv)

        captured = capsys.readouterr()
        assert error.value.code == 2
        assert captured.out == ""
        assert f"state file is not valid JSON: {state_path}: " in captured.err
        assert "Traceback" not in captured.err
        assert state_path.read_bytes() == original


def test_cli_delete_matter_names_an_unknown_matter_without_writing(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["delete-matter", "ghost", "--yes", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "unknown matter: ghost" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_show_prints_the_stored_facts_about_one_matter(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # Prerequisites are stored out of order on purpose: show sorts them.
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b", "c", "d"],
                "conditions": {
                    "a": [
                        {"label": "a done", "truth": True},
                        {"label": "ship it", "truth": False},
                    ],
                    "b": [],
                    "c": [],
                    "d": [],
                },
                "dependencies": [["d", "a"], ["b", "a"], ["a", "c"]],
            }
        )
    )
    original = state_path.read_bytes()

    assert main(["show", "a", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == (
        "a\n"
        "conditions:\n"
        "  1. [x] a done\n"
        "  2. [ ] ship it\n"
        "requires:\n"
        "  b\n"
        "  d\n"
        "required by:\n"
        "  c\n"
    )
    assert captured.err == ""
    # D6: stored facts only. No derived status may appear in this output.
    for derived in ("resolved", "actionable", "blocked"):
        assert derived not in captured.out
    # AC-8: show writes nothing.
    assert state_path.read_bytes() == original


def test_cli_show_numbers_conditions_from_one_in_text_and_from_zero_in_json(
    tmp_path, capsys
):
    """The guard against the two index bases drifting into each other.

    One condition, both surfaces, one test: the number a person reads and
    types back as a condition reference is 1, and the machine-facing ``index``
    for that same condition is 0, matching the web API payload field and the
    stored list position.
    """

    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )

    assert main(["show", "a", "--state", str(state_path)]) == 0

    text = capsys.readouterr().out
    assert "  1. [ ] a done\n" in text
    assert "  0. " not in text

    assert main(["show", "a", "--json", "--state", str(state_path)]) == 0

    payload = capsys.readouterr().out
    assert '"index": 0' in payload
    assert '"index": 1' not in payload
    assert json.loads(payload) == {
        "id": "a",
        "conditions": [{"index": 0, "label": "a done", "truth": False}],
        "prerequisites": [],
        "dependents": [],
    }

    # And the 1 the text printed really is the reference the write verbs take.
    assert main(["mark", "a", "1", "true", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == 'a: condition 1 "a done" is now true\n'


def test_cli_show_prints_none_for_every_empty_section(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # E-13: `lonely` has no entry in the conditions map at all.
    state_path.write_text(
        json.dumps({"matters": ["lonely"], "conditions": {}, "dependencies": []})
    )
    original = state_path.read_bytes()

    assert main(["show", "lonely", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == (
        "lonely\n"
        "conditions:\n"
        "  none\n"
        "requires:\n"
        "  none\n"
        "required by:\n"
        "  none\n"
    )
    assert captured.err == ""

    assert main(["show", "lonely", "--json", "--state", str(state_path)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "id": "lonely",
        "conditions": [],
        "prerequisites": [],
        "dependents": [],
    }
    assert state_path.read_bytes() == original


def test_cli_show_round_trips_non_ascii_labels(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {
                    "a": [
                        {"label": "Anträge einreichen", "truth": False},
                        {"label": "提交申请", "truth": True},
                    ]
                },
                "dependencies": [],
            }
        )
    )

    assert main(["show", "a", "--state", str(state_path)]) == 0

    text = capsys.readouterr().out
    assert text == (
        "a\n"
        "conditions:\n"
        "  1. [ ] Anträge einreichen\n"
        "  2. [x] 提交申请\n"
        "requires:\n"
        "  none\n"
        "required by:\n"
        "  none\n"
    )
    # E-6: no mojibake and no visible escaping on the human surface.
    assert "\\u" not in text

    assert main(["show", "a", "--json", "--state", str(state_path)]) == 0

    # The JSON surface escapes non-ASCII, as json.dumps has always done here;
    # what matters is that the labels come back out identical.
    assert [
        condition["label"]
        for condition in json.loads(capsys.readouterr().out)["conditions"]
    ] == ["Anträge einreichen", "提交申请"]


def test_cli_show_names_an_unknown_matter_without_writing(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "a done", "truth": False}]},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    with pytest.raises(SystemExit) as error:
        main(["show", "ghost", "--state", str(state_path)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert captured.out == ""
    assert "unknown matter: ghost" in captured.err
    assert "Traceback" not in captured.err
    assert state_path.read_bytes() == original


def test_cli_list_prints_every_matter_id_sorted_one_per_line(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # `lonely` has no conditions entry and no edges: it is still a matter.
    state_path.write_text(
        json.dumps(
            {
                "matters": ["c", "a", "lonely"],
                "conditions": {"a": [{"label": "a done", "truth": False}], "c": []},
                "dependencies": [["a", "c"]],
            }
        )
    )
    original = state_path.read_bytes()

    assert main(["list", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "a\nc\nlonely\n"
    assert captured.err == ""

    assert main(["list", "--json", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == '[\n  "a",\n  "c",\n  "lonely"\n]\n'
    # AC-9: list writes nothing.
    assert state_path.read_bytes() == original


def test_cli_list_on_an_empty_state_prints_nothing_and_exits_zero(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps({"matters": [], "conditions": {}, "dependencies": []})
    )

    assert main(["list", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == ""

    assert main(["list", "--json", "--state", str(state_path)]) == 0

    assert capsys.readouterr().out == "[]\n"


def test_cli_read_verbs_survive_a_state_file_that_contains_a_cycle(tmp_path, capsys):
    """E-14: neither read verb may build a graph index.

    Today ``universe`` exits 0 silently on this file, ``unlock`` dumps a
    traceback and the web API answers 422. ``show``/``list`` must not add a
    fourth behaviour: they report stored facts, which is exactly why ``show``
    reports no derived status (D6).
    """

    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": False}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [["a", "b"], ["b", "a"]],
            }
        )
    )
    original = state_path.read_bytes()

    assert main(["show", "a", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == (
        "a\n"
        "conditions:\n"
        "  1. [ ] a done\n"
        "requires:\n"
        "  b\n"
        "required by:\n"
        "  b\n"
    )
    assert captured.err == ""
    assert "Traceback" not in captured.err

    assert main(["show", "a", "--json", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["prerequisites"] == ["b"]
    assert captured.err == ""

    assert main(["list", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "a\nb\n"
    assert captured.err == ""

    assert main(["list", "--json", "--state", str(state_path)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == ["a", "b"]
    assert captured.err == ""
    assert state_path.read_bytes() == original


def test_cli_read_verbs_report_malformed_json_without_a_traceback(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    state_path.write_text("{")
    original = state_path.read_bytes()
    state = ["--state", str(state_path)]

    for argv in (
        ["show", "a"] + state,
        ["show", "a", "--json"] + state,
        ["list"] + state,
        ["list", "--json"] + state,
    ):
        with pytest.raises(SystemExit) as error:
            main(argv)

        captured = capsys.readouterr()
        assert error.value.code == 2
        assert captured.out == ""
        assert f"state file is not valid JSON: {state_path}: " in captured.err
        assert "Traceback" not in captured.err
        assert state_path.read_bytes() == original


def test_cli_read_verbs_write_nothing_and_leave_no_lock_file(tmp_path, capsys):
    state_path = tmp_path / "matters.json"
    # Non-canonical on purpose: a read must not rewrite the file into
    # save_state's canonical form either (AC-10, "no reformatting").
    state_path.write_text(
        '{"matters": ["b", "a"], "conditions": {"a": [], "b": []},'
        ' "dependencies": [["b", "a"]]}'
    )
    original = state_path.read_bytes()

    assert main(["show", "a", "--state", str(state_path)]) == 0
    assert main(["show", "a", "--json", "--state", str(state_path)]) == 0
    assert main(["list", "--state", str(state_path)]) == 0
    assert main(["list", "--json", "--state", str(state_path)]) == 0

    capsys.readouterr()
    assert state_path.read_bytes() == original
    # A read takes no lock, so it must not leave the sidecar behind either.
    assert sorted(item.name for item in tmp_path.iterdir()) == ["matters.json"]


# ---------------------------------------------------------------------------
# AC-15/AC-16, F-11: structurally invalid state, all nine verbs
#
# The malformed-JSON loops above cover a file that json.loads itself rejects.
# This one covers the other half: JSON that parses and then fails inside the
# engine, where the exception type is TypeError rather than ValueError.
# ---------------------------------------------------------------------------


NINE_VERBS_AGAINST_A_BROKEN_STATE = (
    ["mark", "a", "1", "true"],
    ["add-condition", "a", "ship it"],
    ["edit-condition", "a", "1", "shipped"],
    ["delete-condition", "a", "1", "--yes"],
    ["link", "a", "b"],
    ["unlink", "a", "b"],
    ["delete-matter", "a", "--yes", "--cascade"],
    ["show", "a"],
    ["list"],
)


@pytest.mark.parametrize(
    "conditions_value", [5, True, 1.5], ids=["int", "bool", "float"]
)
def test_cli_all_verbs_report_a_structurally_invalid_state_without_a_traceback(
    tmp_path, capsys, conditions_value
):
    state_path = tmp_path / "matters.json"
    # Valid JSON, invalid graph: a scalar where a condition list belongs.
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": conditions_value},
                "dependencies": [],
            }
        )
    )
    original = state_path.read_bytes()

    for argv in NINE_VERBS_AGAINST_A_BROKEN_STATE:
        with pytest.raises(SystemExit) as error:
            main(argv + ["--state", str(state_path)])

        captured = capsys.readouterr()
        assert error.value.code == 2, argv
        assert captured.out == "", argv
        assert (
            f"state file is not a valid matters graph: {state_path}: "
        ) in captured.err, argv
        assert "Traceback" not in captured.err, argv
        # One line of diagnostic, after argparse's own usage preamble.
        assert captured.err.splitlines()[-1].endswith(
            "object is not iterable"
        ), argv
        assert state_path.read_bytes() == original, argv


def test_cli_json_read_verbs_also_survive_a_structurally_invalid_state(
    tmp_path, capsys
):
    # --json takes a different print path; it must not emit a partial document
    # on stdout before the loader rejects the file.
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {"matters": ["a"], "conditions": {"a": 5}, "dependencies": []}
        )
    )

    for argv in (["show", "a", "--json"], ["list", "--json"]):
        with pytest.raises(SystemExit) as error:
            main(argv + ["--state", str(state_path)])

        captured = capsys.readouterr()
        assert error.value.code == 2, argv
        assert captured.out == "", argv
        assert "Traceback" not in captured.err, argv
