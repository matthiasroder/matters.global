"""Strict TOML configuration for model profiles and workflows."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from platformdirs import user_config_dir

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


PROVIDERS = {"codex-cli", "openai-api", "anthropic-api"}
_REGISTERED_PROVIDERS = set()
WORKFLOWS = {"extraction", "reconciliation", "tots"}
FALLBACKS = {
    "extraction": {"marker", "error"},
    "reconciliation": {"skip", "error"},
    "tots": {"error"},
}
DEFAULT_FALLBACKS = {"extraction": "marker", "reconciliation": "skip", "tots": "error"}
MODEL_ENV = {
    "extraction": "MATTERS_EXTRACT_MODEL",
    "reconciliation": "MATTERS_EXTRACT_MODEL",
    "tots": "MATTERS_TOTS_MODEL",
}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_FIELDS = {"api_key", "key", "token", "secret", "credential", "password"}


class ConfigError(ValueError):
    """A field-addressed configuration error that never contains secret values."""


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    provider: str
    model: str
    auth: str
    timeout_seconds: float = 120.0
    api_key_env: str | None = None
    executable: str = "codex"


@dataclass(frozen=True)
class WorkflowConfig:
    name: str
    profile: str | None = None
    on_unavailable: str = "error"


@dataclass(frozen=True)
class LLMConfig:
    path: Path
    exists: bool
    default_profile: str | None = None
    profiles: Mapping[str, ProfileConfig] = field(default_factory=dict)
    workflows: Mapping[str, WorkflowConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedProfile:
    profile: ProfileConfig
    workflow: WorkflowConfig


def register_provider_id(provider: str):
    """Allow a factory-registered provider id in subsequently loaded config."""
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider id must be a non-empty string")
    _REGISTERED_PROVIDERS.add(provider)


def resolve_config_path(path=None, *, environ=None) -> tuple[Path, bool]:
    environ = os.environ if environ is None else environ
    selected = path or environ.get("MATTERS_CONFIG")
    explicit = selected is not None
    target = (
        Path(selected).expanduser()
        if selected
        else Path(user_config_dir("matters")) / "config.toml"
    )
    if explicit and not target.is_file():
        raise ConfigError("config path: file does not exist")
    return target, target.is_file()


def load_config(path=None, *, environ=None) -> LLMConfig:
    target, exists = resolve_config_path(path, environ=environ)
    if not exists:
        return LLMConfig(path=target, exists=False)
    try:
        with target.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError:
        raise ConfigError("config: invalid TOML") from None
    except OSError:
        raise ConfigError("config: could not read file") from None
    _fields(document, {"llm"}, "config")
    llm = _table(document.get("llm", {}), "llm")
    _fields(llm, {"default_profile", "profiles", "workflows"}, "llm")

    profiles_raw = _table(llm.get("profiles", {}), "llm.profiles")
    profiles = {
        name: _profile(name, _table(raw, f"llm.profiles.{name}"))
        for name, raw in profiles_raw.items()
    }
    workflows_raw = _table(llm.get("workflows", {}), "llm.workflows")
    unknown_workflows = set(workflows_raw) - WORKFLOWS
    if unknown_workflows:
        name = sorted(unknown_workflows)[0]
        raise ConfigError(f"llm.workflows.{name}: unknown workflow")
    workflows = {
        name: _workflow(name, _table(raw, f"llm.workflows.{name}"))
        for name, raw in workflows_raw.items()
    }
    default_profile = llm.get("default_profile")
    if default_profile is not None and not isinstance(default_profile, str):
        raise ConfigError("llm.default_profile: expected string")
    if default_profile and default_profile not in profiles:
        raise ConfigError("llm.default_profile: unknown profile")
    for name, workflow in workflows.items():
        if workflow.profile and workflow.profile not in profiles:
            raise ConfigError(f"llm.workflows.{name}.profile: unknown profile")
    return LLMConfig(target, True, default_profile, profiles, workflows)


def resolve_workflow(
    config: LLMConfig,
    workflow: str,
    *,
    profile_override=None,
    model_override=None,
    environ=None,
) -> ResolvedProfile | None:
    if workflow not in WORKFLOWS:
        raise ConfigError(f"workflow: unknown workflow {workflow}")
    environ = os.environ if environ is None else environ
    workflow_config = config.workflows.get(
        workflow, WorkflowConfig(workflow, on_unavailable=DEFAULT_FALLBACKS[workflow])
    )
    profile_name = profile_override or workflow_config.profile or config.default_profile
    if not profile_name:
        return None
    if profile_name not in config.profiles:
        raise ConfigError(f"llm profile: unknown profile {profile_name}")
    profile = config.profiles[profile_name]
    effective_model = model_override or environ.get(MODEL_ENV[workflow]) or profile.model
    return ResolvedProfile(replace(profile, model=effective_model), workflow_config)


def config_diagnostics(config: LLMConfig, *, profile_name=None, environ=None):
    from .factory import create_generator

    names = [profile_name] if profile_name else sorted(config.profiles)
    diagnostics = []
    for name in names:
        if name not in config.profiles:
            raise ConfigError(f"llm profile: unknown profile {name}")
        profile = config.profiles[name]
        readiness = create_generator(profile, environ=environ).check()
        diagnostics.append({"profile": name, **readiness.as_dict()})
    workflows = {}
    for workflow in sorted(WORKFLOWS):
        resolved = resolve_workflow(config, workflow, environ=environ)
        workflows[workflow] = {
            "profile": resolved.profile.name if resolved else None,
            "on_unavailable": (
                resolved.workflow.on_unavailable
                if resolved
                else DEFAULT_FALLBACKS[workflow]
            ),
        }
    return {
        "config_path": str(config.path),
        "config_exists": config.exists,
        "default_profile": config.default_profile,
        "workflows": workflows,
        "profiles": diagnostics,
    }


def _profile(name, raw):
    path = f"llm.profiles.{name}"
    for field_name in raw:
        if field_name.lower() in SECRET_FIELDS:
            raise ConfigError(f"{path}.{field_name}: secret values are forbidden")
    allowed = {"provider", "model", "auth", "timeout_seconds", "api_key_env", "executable"}
    _fields(raw, allowed, path)
    provider = _required_string(raw, "provider", path)
    model = _required_string(raw, "model", path)
    is_custom = provider in _REGISTERED_PROVIDERS and provider not in PROVIDERS
    if provider not in PROVIDERS and not is_custom:
        raise ConfigError(f"{path}.provider: unknown provider")
    auth_default = (
        "chatgpt"
        if provider == "codex-cli"
        else ("custom" if is_custom else "api-key")
    )
    auth = raw.get("auth", auth_default)
    if is_custom:
        if not isinstance(auth, str) or not auth.strip():
            raise ConfigError(f"{path}.auth: non-empty string required")
    elif auth not in ({"chatgpt"} if provider == "codex-cli" else {"api-key"}):
        raise ConfigError(f"{path}.auth: unsupported auth mode")
    timeout = raw.get("timeout_seconds", 120)
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigError(f"{path}.timeout_seconds: expected a positive number")
    api_key_env = raw.get("api_key_env")
    if provider in {"openai-api", "anthropic-api"}:
        if not isinstance(api_key_env, str) or not ENV_NAME.fullmatch(api_key_env):
            raise ConfigError(f"{path}.api_key_env: valid environment name required")
    elif is_custom and api_key_env is not None:
        if not isinstance(api_key_env, str) or not ENV_NAME.fullmatch(api_key_env):
            raise ConfigError(f"{path}.api_key_env: expected a valid environment name")
    elif api_key_env is not None:
        raise ConfigError(f"{path}.api_key_env: not valid for codex-cli")
    executable = raw.get("executable", "codex")
    if not isinstance(executable, str) or not executable.strip():
        raise ConfigError(f"{path}.executable: expected non-empty string")
    return ProfileConfig(name, provider, model, auth, float(timeout), api_key_env, executable)


def _workflow(name, raw):
    path = f"llm.workflows.{name}"
    _fields(raw, {"profile", "on_unavailable"}, path)
    profile = raw.get("profile")
    if profile is not None and not isinstance(profile, str):
        raise ConfigError(f"{path}.profile: expected string")
    fallback = raw.get("on_unavailable", DEFAULT_FALLBACKS[name])
    if fallback not in FALLBACKS[name]:
        raise ConfigError(f"{path}.on_unavailable: unsupported fallback mode")
    return WorkflowConfig(name, profile, fallback)


def _fields(raw, allowed, path):
    unknown = set(raw) - allowed
    if unknown:
        name = sorted(unknown)[0]
        if name.lower() in SECRET_FIELDS:
            raise ConfigError(f"{path}.{name}: secret values are forbidden")
        raise ConfigError(f"{path}.{name}: unknown field")


def _table(value, path):
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected table")
    return value


def _required_string(raw, name, path):
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{name}: non-empty string required")
    return value
