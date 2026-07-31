import json
import shutil
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from matters.llm import (
    AuthenticationError,
    DependencyUnavailableError,
    GenerationError,
    GenerationTimeoutError,
    IncompleteResponseError,
    InvalidStructuredResponseError,
    RateLimitError,
    RefusalError,
    StructuredRequest,
    StructuredResult,
    create_generator,
    register_provider,
    resolve_generator,
)
from matters.llm.contract import validate_structured_data
from matters.llm.config import ProfileConfig
from matters.llm.providers.anthropic_api import AnthropicGenerator
from matters.llm.providers.codex_cli import (
    DISABLED_CAPABILITIES,
    CodexCLIGenerator,
)
from matters.llm.providers.openai_api import OpenAIGenerator


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "boolean"}},
    "required": ["answer"],
    "additionalProperties": False,
}

EXPECTED_DISABLED_CAPABILITIES = (
    "apps",
    "auth_elicitation",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode_host",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "shell_snapshot",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "unified_exec",
    "workspace_dependencies",
)


def request():
    return StructuredRequest("test-operation", "system", "user", SCHEMA, 100)


def formatted_exception(error):
    return "".join(traceback.format_exception(error))


def is_codex_preflight(command):
    schema = Path(command[command.index("--output-schema") + 1])
    return schema.name == "missing-schema.json"


def successful_codex_preflight(command):
    schema = Path(command[command.index("--output-schema") + 1])
    return subprocess.CompletedProcess(
        command,
        1,
        "",
        f"Failed to read output schema file {schema}: "
        "No such file or directory (os error 2)\n",
    )


def profile(provider, **overrides):
    values = {
        "name": "test",
        "provider": provider,
        "model": "model-test",
        "auth": "chatgpt" if provider == "codex-cli" else "api-key",
        "timeout_seconds": 10,
        "api_key_env": None if provider == "codex-cli" else "TEST_PROVIDER_KEY",
        "executable": "codex-test",
    }
    values.update(overrides)
    return ProfileConfig(**values)


def test_codex_check_rejects_non_chatgpt_auth(monkeypatch):
    monkeypatch.setattr("matters.llm.providers.codex_cli.shutil.which", lambda *_args, **_kwargs: "/bin/codex")

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, "Logged in using an API key", "")

    generator = CodexCLIGenerator(profile("codex-cli"), environ={"PATH": "/bin"}, runner=runner)
    readiness = generator.check()

    assert readiness.ready is False
    assert readiness.reason == "ChatGPT authentication required"


def test_codex_check_requires_exact_chatgpt_status_line(monkeypatch):
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, "Status: Logged in using ChatGPT for another profile", ""
        )

    readiness = CodexCLIGenerator(
        profile("codex-cli"), environ={"PATH": "/bin"}, runner=runner
    ).check()
    assert readiness.ready is False


def test_codex_check_preflights_exact_command_once_and_caches(monkeypatch):
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, "Logged in using ChatGPT", ""
            )
        return successful_codex_preflight(command)

    generator = CodexCLIGenerator(
        profile("codex-cli"), environ={"PATH": "/bin"}, runner=runner
    )

    first = generator.check()
    second = generator.check()

    assert first is second
    assert first.ready is True
    assert first.reason is None
    assert len(calls) == 2
    command, kwargs = calls[1]
    assert is_codex_preflight(command)
    assert "tools.view_image=false" not in command
    assert "--strict-config" in command
    assert kwargs["input"] == "Configuration preflight only."


def test_codex_check_rejects_unexpected_preflight_failure_without_raw_detail(
    monkeypatch,
):
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )

    def runner(command, **_kwargs):
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
            "No such file or directory (os error 2)\n"
            "secret invalid configuration detail\n",
        )

    readiness = CodexCLIGenerator(
        profile("codex-cli"), environ={}, runner=runner
    ).check()

    assert readiness.ready is False
    assert readiness.credential_available is True
    assert readiness.reason == "Codex isolation preflight failed"
    assert "secret" not in readiness.reason


def test_codex_generation_is_isolated_schema_bound_and_scrubbed(monkeypatch):
    monkeypatch.setattr("matters.llm.providers.codex_cli.shutil.which", lambda *_args, **_kwargs: "/bin/codex")
    calls = []
    temporary_paths = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if is_codex_preflight(command):
            return successful_codex_preflight(command)
        output = Path(command[command.index("--output-last-message") + 1])
        schema = Path(command[command.index("--output-schema") + 1])
        cwd = Path(command[command.index("--cd") + 1])
        temporary_paths.extend([output, schema, cwd])
        output.write_text('{"answer": true}')
        return subprocess.CompletedProcess(command, 0, "", "")

    generator = CodexCLIGenerator(
        profile("codex-cli"),
        environ={
            "PATH": "/bin",
            "HOME": "/safe/home",
            "OPENAI_API_KEY": "secret-key",
            "UNRELATED_TOKEN": "secret-token",
        },
        runner=runner,
    )
    result = generator.generate(request())
    command, kwargs = calls[-1]

    expected_command = [
        "/bin/codex",
        "exec",
        "--model",
        "model-test",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--sandbox",
        "read-only",
    ]
    for capability in EXPECTED_DISABLED_CAPABILITIES:
        expected_command.extend(("--disable", capability))
    expected_command.extend(
        [
            "--config",
            'web_search="disabled"',
            "--config",
            "tools.web_search=false",
            "--cd",
            command[command.index("--cd") + 1],
            "--output-schema",
            command[command.index("--output-schema") + 1],
            "--output-last-message",
            command[command.index("--output-last-message") + 1],
            "-",
        ]
    )

    assert result.data["answer"] is True
    assert DISABLED_CAPABILITIES == EXPECTED_DISABLED_CAPABILITIES
    assert command == expected_command
    assert kwargs["input"].startswith("system")
    assert "secret-key" not in repr(calls)
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "UNRELATED_TOKEN" not in kwargs["env"]
    assert kwargs["env"]["HOME"] == "/safe/home"
    assert all(not path.exists() for path in temporary_paths)


def test_codex_timeout_is_redacted(monkeypatch):
    monkeypatch.setattr("matters.llm.providers.codex_cli.shutil.which", lambda *_args, **_kwargs: "/bin/codex")
    temporary_paths = []

    def runner(command, **_kwargs):
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if is_codex_preflight(command):
            return successful_codex_preflight(command)
        temporary_paths.extend(
            [
                Path(command[command.index("--output-last-message") + 1]),
                Path(command[command.index("--output-schema") + 1]),
                Path(command[command.index("--cd") + 1]),
            ]
        )
        raise subprocess.TimeoutExpired(command, 1, output="secret output")

    generator = CodexCLIGenerator(profile("codex-cli"), environ={}, runner=runner)
    with pytest.raises(GenerationTimeoutError) as error:
        generator.generate(request())
    assert "secret output" not in str(error.value)
    assert "secret output" not in formatted_exception(error.value)
    assert all(not path.exists() for path in temporary_paths)


def test_codex_failure_is_redacted_and_cleans_temporary_files(monkeypatch):
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )
    temporary_paths = []

    def runner(command, **_kwargs):
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")
        if is_codex_preflight(command):
            return successful_codex_preflight(command)
        temporary_paths.extend(
            [
                Path(command[command.index("--output-last-message") + 1]),
                Path(command[command.index("--output-schema") + 1]),
                Path(command[command.index("--cd") + 1]),
            ]
        )
        return subprocess.CompletedProcess(
            command, 1, "secret model output", "secret provider failure"
        )

    generator = CodexCLIGenerator(profile("codex-cli"), environ={}, runner=runner)
    with pytest.raises(GenerationError) as error:
        generator.generate(request())

    assert "secret model output" not in str(error.value)
    assert "secret provider failure" not in str(error.value)
    formatted = formatted_exception(error.value)
    assert "secret model output" not in formatted
    assert "secret provider failure" not in formatted
    assert all(not path.exists() for path in temporary_paths)


def test_codex_invalid_output_traceback_is_redacted(monkeypatch):
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )
    sentinel = "RAW_CODEX_MODEL_OUTPUT_SENTINEL"

    def runner(command, **_kwargs):
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, "Logged in using ChatGPT", ""
            )
        if is_codex_preflight(command):
            return successful_codex_preflight(command)
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(sentinel)
        return subprocess.CompletedProcess(command, 0, "", "")

    generator = CodexCLIGenerator(
        profile("codex-cli"), environ={}, runner=runner
    )
    with pytest.raises(InvalidStructuredResponseError) as error:
        generator.generate(request())

    assert sentinel not in str(error.value)
    assert sentinel not in formatted_exception(error.value)


def test_codex_os_error_is_redacted(monkeypatch):
    monkeypatch.setattr(
        "matters.llm.providers.codex_cli.shutil.which",
        lambda *_args, **_kwargs: "/bin/codex",
    )
    sentinel = "RAW_CODEX_OS_ERROR_SENTINEL"

    def runner(command, **_kwargs):
        if command[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, "Logged in using ChatGPT", ""
            )
        if is_codex_preflight(command):
            return successful_codex_preflight(command)
        raise OSError(sentinel)

    generator = CodexCLIGenerator(
        profile("codex-cli"), environ={}, runner=runner
    )
    with pytest.raises(DependencyUnavailableError) as error:
        generator.generate(request())

    assert sentinel not in str(error.value)
    assert sentinel not in formatted_exception(error.value)


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex CLI not installed")
def test_installed_codex_accepts_isolation_preflight_without_generation():
    executable = shutil.which("codex")
    generator = CodexCLIGenerator(
        profile("codex-cli", executable=executable, timeout_seconds=30)
    )

    assert generator._run_preflight() is True


def test_structured_records_own_nested_mapping_values():
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    metadata = {"context": {"ids": ["matter-1"]}}
    request_record = StructuredRequest(
        "ownership", "system", "user", schema, metadata=metadata
    )
    data = {"items": ["first"]}
    usage = {"tokens": {"input": [1]}}
    result_record = StructuredResult(
        data, "test-provider", "test-model", usage=usage
    )

    schema["properties"]["items"]["items"]["type"] = "number"
    metadata["context"]["ids"].append("matter-2")
    data["items"].append("second")
    usage["tokens"]["input"].append(2)

    assert request_record.schema["properties"]["items"]["items"]["type"] == "string"
    assert request_record.metadata["context"]["ids"] == ["matter-1"]
    assert result_record.data["items"] == ["first"]
    assert result_record.usage["tokens"]["input"] == [1]


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_openai_responses_request_is_strict_and_tool_free():
    responses = FakeResponses(SimpleNamespace(status="completed", output_text='{"answer": true}', output=[], usage=None))
    client = SimpleNamespace(responses=responses)
    generator = OpenAIGenerator(profile("openai-api"), environ={"TEST_PROVIDER_KEY": "secret"}, client=client)

    result = generator.generate(request())
    call = responses.calls[0]

    assert result.data["answer"] is True
    assert call["text"]["format"]["strict"] is True
    assert call["text"]["format"]["schema"] == SCHEMA
    assert call["store"] is False
    assert "tools" not in call
    assert "secret" not in json.dumps(call)


def test_openai_refusal_incomplete_and_redacted_error():
    refusal = SimpleNamespace(
        status="completed",
        output_text="",
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])],
    )
    generator = OpenAIGenerator(profile("openai-api"), environ={"TEST_PROVIDER_KEY": "x"}, client=SimpleNamespace(responses=FakeResponses(refusal)))
    with pytest.raises(RefusalError):
        generator.generate(request())

    incomplete = SimpleNamespace(status="incomplete", output_text="", output=[])
    generator._client = SimpleNamespace(responses=FakeResponses(incomplete))
    with pytest.raises(IncompleteResponseError):
        generator.generate(request())

    generator._client = SimpleNamespace(responses=FakeResponses(error=RuntimeError("credential secret")))
    with pytest.raises(GenerationError) as error:
        generator.generate(request())
    assert "credential secret" not in str(error.value)
    assert "credential secret" not in formatted_exception(error.value)


def test_openai_invalid_output_traceback_is_redacted():
    sentinel = "RAW_OPENAI_MODEL_OUTPUT_SENTINEL"
    response = SimpleNamespace(
        status="completed", output_text=sentinel, output=[], usage=None
    )
    generator = OpenAIGenerator(
        profile("openai-api"),
        environ={"TEST_PROVIDER_KEY": "x"},
        client=SimpleNamespace(responses=FakeResponses(response)),
    )

    with pytest.raises(InvalidStructuredResponseError) as error:
        generator.generate(request())

    assert sentinel not in str(error.value)
    assert sentinel not in formatted_exception(error.value)


def test_sdk_rate_limit_is_normalized_without_raw_detail():
    class ProviderRateLimitError(Exception):
        pass

    responses = FakeResponses(error=ProviderRateLimitError("secret provider detail"))
    generator = OpenAIGenerator(
        profile("openai-api"),
        environ={"TEST_PROVIDER_KEY": "x"},
        client=SimpleNamespace(responses=responses),
    )
    with pytest.raises(RateLimitError) as error:
        generator.generate(request())
    assert "secret provider detail" not in str(error.value)
    assert "secret provider detail" not in formatted_exception(error.value)


def test_anthropic_messages_request_and_local_validation():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"answer": true}')],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=2, output_tokens=3),
        )

    generator = AnthropicGenerator(
        profile("anthropic-api"),
        environ={"TEST_PROVIDER_KEY": "secret"},
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
    )
    result = generator.generate(request())

    assert result.usage == {"input_tokens": 2, "output_tokens": 3}
    assert calls[0]["output_config"]["format"]["schema"] == SCHEMA
    assert "tools" not in calls[0]
    assert "secret" not in json.dumps(calls[0])

    generator._client.messages.create = lambda **_kwargs: SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"answer": "wrong"}')],
        stop_reason="end_turn",
        usage=None,
    )
    with pytest.raises(InvalidStructuredResponseError):
        generator.generate(request())


def test_anthropic_refusal_and_incomplete_are_normalized():
    response = SimpleNamespace(content=[], stop_reason="refusal", usage=None)
    generator = AnthropicGenerator(
        profile("anthropic-api"),
        environ={"TEST_PROVIDER_KEY": "x"},
        client=SimpleNamespace(messages=SimpleNamespace(create=lambda **_kwargs: response)),
    )
    with pytest.raises(RefusalError):
        generator.generate(request())

    response.stop_reason = "max_tokens"
    with pytest.raises(IncompleteResponseError):
        generator.generate(request())


def test_anthropic_failure_is_redacted():
    def fail(**_kwargs):
        raise RuntimeError("secret provider detail")

    generator = AnthropicGenerator(
        profile("anthropic-api"),
        environ={"TEST_PROVIDER_KEY": "x"},
        client=SimpleNamespace(messages=SimpleNamespace(create=fail)),
    )

    with pytest.raises(GenerationError) as error:
        generator.generate(request())
    assert "secret provider detail" not in str(error.value)
    assert "secret provider detail" not in formatted_exception(error.value)


def test_anthropic_invalid_output_traceback_is_redacted():
    sentinel = "RAW_ANTHROPIC_MODEL_OUTPUT_SENTINEL"
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=sentinel)],
        stop_reason="end_turn",
        usage=None,
    )
    generator = AnthropicGenerator(
        profile("anthropic-api"),
        environ={"TEST_PROVIDER_KEY": "x"},
        client=SimpleNamespace(
            messages=SimpleNamespace(create=lambda **_kwargs: response)
        ),
    )

    with pytest.raises(InvalidStructuredResponseError) as error:
        generator.generate(request())

    assert sentinel not in str(error.value)
    assert sentinel not in formatted_exception(error.value)


def test_api_readiness_requires_dependency_and_credential(monkeypatch):
    monkeypatch.setattr("matters.llm.providers.openai_api.importlib.util.find_spec", lambda _name: None)
    readiness = OpenAIGenerator(profile("openai-api"), environ={}).check()
    assert readiness.ready is False
    assert readiness.dependency_available is False
    assert readiness.credential_available is False


def test_authentication_error_does_not_name_or_value_the_credential():
    generator = OpenAIGenerator(profile("openai-api"), environ={}, client=SimpleNamespace())
    with pytest.raises(AuthenticationError) as error:
        generator.generate(request())
    assert "TEST_PROVIDER_KEY" not in str(error.value)


def test_invalid_schema_is_normalized_before_provider_use():
    invalid_schema = {
        "type": "object",
        "properties": {"answer": {"type": "secret-invalid-schema-type"}},
    }
    with pytest.raises(InvalidStructuredResponseError) as error:
        StructuredRequest(
            "schema-check", "system", "user", invalid_schema, 100
        )
    assert "secret-invalid-schema-type" not in str(error.value)
    assert "secret-invalid-schema-type" not in formatted_exception(error.value)


def test_validate_structured_data_normalizes_schema_error():
    invalid_schema = {
        "type": "object",
        "properties": {"answer": {"type": "raw-schema-detail"}},
    }
    with pytest.raises(InvalidStructuredResponseError) as error:
        validate_structured_data(
            {"answer": True}, invalid_schema, "test-provider", "schema-check"
        )
    assert "raw-schema-detail" not in str(error.value)
    assert "raw-schema-detail" not in formatted_exception(error.value)


def test_legacy_client_failure_traceback_is_redacted():
    sentinel = "RAW_LEGACY_PROVIDER_SENTINEL"

    def fail(**_kwargs):
        raise RuntimeError(sentinel)

    client = SimpleNamespace(messages=SimpleNamespace(create=fail))
    with pytest.warns(DeprecationWarning, match="deprecated"):
        selection = resolve_generator("tots", injected=client, environ={})

    with pytest.raises(GenerationError) as error:
        selection.generator.generate(request())

    assert sentinel not in str(error.value)
    assert sentinel not in formatted_exception(error.value)


def test_legacy_injection_model_precedence_by_workflow():
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_kwargs: None))
    with pytest.warns(DeprecationWarning, match="deprecated"):
        default = resolve_generator("extraction", injected=client, environ={})
    with pytest.warns(DeprecationWarning, match="deprecated"):
        environment = resolve_generator(
            "tots",
            injected=client,
            environ={"MATTERS_TOTS_MODEL": "environment-model"},
        )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        explicit = resolve_generator(
            "tots",
            injected=client,
            model_override="explicit-model",
            environ={"MATTERS_TOTS_MODEL": "environment-model"},
        )

    assert default.model == "claude-sonnet-4-6"
    assert environment.model == "environment-model"
    assert explicit.model == "explicit-model"


def test_programmatic_provider_registration_is_lazy():
    calls = []
    expected = object()

    def builder(selected_profile, *, environ=None):
        calls.append((selected_profile.name, environ))
        return expected

    register_provider("test-custom-provider", builder)
    custom_profile = profile("openai-api")
    custom_profile = ProfileConfig(
        custom_profile.name,
        "test-custom-provider",
        custom_profile.model,
        custom_profile.auth,
        custom_profile.timeout_seconds,
        custom_profile.api_key_env,
        custom_profile.executable,
    )

    assert calls == []
    assert create_generator(custom_profile, environ={"X": "Y"}) is expected
    assert calls == [("test", {"X": "Y"})]
