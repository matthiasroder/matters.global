"""Provider registry, lazy construction, and workflow resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .config import (
    DEFAULT_FALLBACKS,
    MODEL_ENV,
    ProfileConfig,
    load_config,
    register_provider_id,
    resolve_workflow,
)
from .contract import StructuredGenerator


ProviderBuilder = Callable[..., StructuredGenerator]
_CUSTOM_PROVIDERS: dict[str, ProviderBuilder] = {}


@dataclass(frozen=True)
class GeneratorSelection:
    generator: StructuredGenerator
    profile: str
    provider: str
    model: str
    on_unavailable: str


def register_provider(provider: str, builder: ProviderBuilder, *, replace=False):
    if provider in _CUSTOM_PROVIDERS and not replace:
        raise ValueError(f"provider already registered: {provider}")
    register_provider_id(provider)
    _CUSTOM_PROVIDERS[provider] = builder


def create_generator(profile: ProfileConfig, *, environ=None) -> StructuredGenerator:
    if profile.provider in _CUSTOM_PROVIDERS:
        return _CUSTOM_PROVIDERS[profile.provider](profile, environ=environ)
    if profile.provider == "codex-cli":
        from .providers.codex_cli import CodexCLIGenerator

        return CodexCLIGenerator(profile, environ=environ)
    if profile.provider == "openai-api":
        from .providers.openai_api import OpenAIGenerator

        return OpenAIGenerator(profile, environ=environ)
    if profile.provider == "anthropic-api":
        from .providers.anthropic_api import AnthropicGenerator

        return AnthropicGenerator(profile, environ=environ)
    raise ValueError(f"unknown provider: {profile.provider}")


def resolve_generator(
    workflow,
    *,
    injected=None,
    config_path=None,
    profile_override=None,
    model_override=None,
    environ=None,
):
    if injected is not None:
        from .legacy import LEGACY_DEFAULT_MODEL, coerce_generator

        environ = os.environ if environ is None else environ
        legacy_model = (
            model_override
            or environ.get(MODEL_ENV[workflow])
            or LEGACY_DEFAULT_MODEL
        )
        generator = coerce_generator(injected, model=legacy_model)
        return GeneratorSelection(
            generator,
            "injected",
            generator.provider,
            generator.model,
            DEFAULT_FALLBACKS[workflow],
        )
    config = load_config(config_path, environ=environ)
    resolved = resolve_workflow(
        config,
        workflow,
        profile_override=profile_override,
        model_override=model_override,
        environ=environ,
    )
    if resolved is None:
        return None
    profile = resolved.profile
    return GeneratorSelection(
        create_generator(profile, environ=environ),
        profile.name,
        profile.provider,
        profile.model,
        resolved.workflow.on_unavailable,
    )
