import json
import subprocess
import traceback
from pathlib import Path

import pytest

from matters.llm import Readiness, register_provider, resolve_generator
from matters.llm.config import (
    ConfigError,
    config_diagnostics,
    load_config,
    resolve_config_path,
    resolve_workflow,
)


CONFIG = """
[llm]
default_profile = "personal"

[llm.profiles.personal]
provider = "codex-cli"
model = "gpt-test"
auth = "chatgpt"
timeout_seconds = 45

[llm.profiles.openai]
provider = "openai-api"
model = "gpt-api-test"
api_key_env = "TEST_OPENAI_KEY"

[llm.workflows.extraction]
profile = "openai"
on_unavailable = "marker"

[llm.workflows.reconciliation]
on_unavailable = "skip"

[llm.workflows.tots]
profile = "personal"
on_unavailable = "error"
"""


def write_config(tmp_path, text=CONFIG):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def formatted_exception(error):
    return "".join(traceback.format_exception(error))


def test_missing_config_does_not_discover_api_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("matters.llm.config.user_config_dir", lambda _name: str(tmp_path))
    config = load_config(environ={"OPENAI_API_KEY": "must-not-select-provider"})

    assert config.exists is False
    assert resolve_workflow(config, "extraction", environ={}) is None


def test_repository_local_config_is_not_implicitly_loaded(tmp_path, monkeypatch):
    repository_config = tmp_path / "config.toml"
    repository_config.write_text(CONFIG)
    user_config = tmp_path / "user-config"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "matters.llm.config.user_config_dir", lambda _name: str(user_config)
    )

    config = load_config(environ={"OPENAI_API_KEY": "must-not-select-provider"})

    assert config.exists is False
    assert config.path == user_config / "config.toml"
    assert resolve_workflow(config, "extraction", environ={}) is None


def test_config_path_precedence_and_explicit_missing_error(tmp_path, monkeypatch):
    explicit = write_config(tmp_path)
    env_path = tmp_path / "environment.toml"
    env_path.write_text(CONFIG)

    assert resolve_config_path(explicit, environ={"MATTERS_CONFIG": str(env_path)}) == (
        explicit,
        True,
    )
    assert resolve_config_path(environ={"MATTERS_CONFIG": str(env_path)}) == (
        env_path,
        True,
    )
    with pytest.raises(ConfigError, match="file does not exist"):
        resolve_config_path(tmp_path / "missing.toml", environ={})


def test_profile_workflow_and_model_override_precedence(tmp_path):
    config = load_config(write_config(tmp_path), environ={})

    extracted = resolve_workflow(
        config, "extraction", environ={"MATTERS_EXTRACT_MODEL": "env-model"}
    )
    selected = resolve_workflow(
        config,
        "extraction",
        profile_override="personal",
        model_override="cli-model",
        environ={"MATTERS_EXTRACT_MODEL": "env-model"},
    )
    reconciled = resolve_workflow(config, "reconciliation", environ={})

    assert extracted.profile.name == "openai"
    assert extracted.profile.model == "env-model"
    assert selected.profile.name == "personal"
    assert selected.profile.model == "cli-model"
    assert reconciled.profile.name == "personal"
    assert reconciled.workflow.on_unavailable == "skip"


@pytest.mark.parametrize(
    "fragment, expected",
    [
        ('api_key = "secret"', "secret values are forbidden"),
        ('unknown = "value"', "unknown field"),
        ('auth = "api-key"', "unsupported auth mode"),
        ('timeout_seconds = 0', "positive number"),
    ],
)
def test_strict_profile_validation_never_echoes_values(tmp_path, fragment, expected):
    secret = "credential-material-must-not-leak"
    text = f"""
[llm]
default_profile = "bad"
[llm.profiles.bad]
provider = "codex-cli"
model = "gpt-test"
{fragment.replace('secret', secret)}
"""
    with pytest.raises(ConfigError, match=expected) as error:
        load_config(write_config(tmp_path, text), environ={})
    assert secret not in str(error.value)
    assert secret not in formatted_exception(error.value)


@pytest.mark.parametrize("timeout", ["nan", "inf"])
def test_profile_timeout_must_be_finite(tmp_path, timeout):
    text = f"""
[llm.profiles.bad]
provider = "codex-cli"
model = "gpt-test"
timeout_seconds = {timeout}
"""
    with pytest.raises(ConfigError, match="positive number"):
        load_config(write_config(tmp_path, text), environ={})


def test_malformed_config_traceback_does_not_expose_source(tmp_path):
    sentinel = "RAW_TOML_SECRET_SENTINEL"
    text = f'''[llm]\ndefault_profile = "{sentinel}\n'''

    with pytest.raises(ConfigError, match="invalid TOML") as error:
        load_config(write_config(tmp_path, text), environ={})

    assert sentinel not in str(error.value)
    assert sentinel not in formatted_exception(error.value)


def test_api_profile_requires_environment_variable_name(tmp_path):
    text = """
[llm.profiles.bad]
provider = "openai-api"
model = "gpt-test"
api_key_env = "not an env name"
"""
    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(write_config(tmp_path, text), environ={})


def test_unregistered_custom_provider_remains_invalid(tmp_path):
    text = """
[llm.profiles.bad]
provider = "unregistered-test-provider"
model = "test-model"
"""
    with pytest.raises(ConfigError, match="unknown provider"):
        load_config(write_config(tmp_path, text), environ={})


def test_registered_custom_provider_resolves_from_toml(tmp_path):
    built = []

    class CustomGenerator:
        provider = "registered-config-provider"
        model = "custom-model"

        def check(self):
            return Readiness(self.provider, self.model, True, "custom", True, True)

        def generate(self, _request):  # pragma: no cover
            raise AssertionError("not used")

    def builder(profile, *, environ=None):
        built.append((profile, environ))
        return CustomGenerator()

    register_provider("registered-config-provider", builder)
    text = """
[llm]
default_profile = "custom"

[llm.profiles.custom]
provider = "registered-config-provider"
model = "custom-model"
auth = "managed"

[llm.workflows.extraction]
profile = "custom"
on_unavailable = "marker"
"""
    path = write_config(tmp_path, text)
    config = load_config(path, environ={})
    selection = resolve_generator("extraction", config_path=path, environ={})

    assert config.profiles["custom"].provider == "registered-config-provider"
    assert selection.provider == "registered-config-provider"
    assert selection.model == "custom-model"
    assert built[0][0].name == "custom"


def test_diagnostics_are_sanitized_and_non_generating(tmp_path, monkeypatch):
    config = load_config(write_config(tmp_path), environ={})
    calls = []

    class FakeGenerator:
        def check(self):
            calls.append("check")
            from matters.llm import Readiness

            return Readiness("fake", "model", True, "managed", True, True)

        def generate(self, _request):  # pragma: no cover
            raise AssertionError("diagnostics must not generate")

    monkeypatch.setattr(
        "matters.llm.factory.create_generator",
        lambda _profile, environ=None: FakeGenerator(),
    )
    result = config_diagnostics(config, environ={"TEST_OPENAI_KEY": "secret-value"})

    assert calls == ["check", "check"]
    assert "secret-value" not in json.dumps(result)
    assert result["workflows"]["tots"]["profile"] == "personal"


def test_diagnostics_require_codex_authentication_and_isolation_preflight(
    tmp_path, monkeypatch
):
    text = """
[llm]
default_profile = "personal"

[llm.profiles.personal]
provider = "codex-cli"
model = "gpt-test"
auth = "chatgpt"
timeout_seconds = 30

[llm.workflows.tots]
profile = "personal"
on_unavailable = "error"
"""
    config = load_config(write_config(tmp_path, text), environ={})
    calls = []
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, "Logged in using ChatGPT", ""
            )
        schema = Path(command[command.index("--output-schema") + 1])
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            f"Failed to read output schema file {schema}: "
            "No such file or directory (os error 2)\n",
        )

    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.subprocess.run", runner
    )
    result = config_diagnostics(config, environ={})

    assert len(calls) == 2
    assert calls[0][0][1:3] == ["login", "status"]
    assert Path(
        calls[1][0][calls[1][0].index("--output-schema") + 1]
    ).name == "missing-schema.json"
    assert result["profiles"][0]["ready"] is True
    assert result["profiles"][0]["reason"] is None
