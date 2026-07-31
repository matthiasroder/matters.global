"""Shared structured-generation records and sanitized failures."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import jsonschema


def _owned_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Own a deep copy while preventing replacement of top-level entries."""
    return MappingProxyType(copy.deepcopy(dict(value or {})))


@dataclass(frozen=True)
class StructuredRequest:
    """Top-level-frozen request owning copies of nested schema and metadata values."""

    operation: str
    system: str
    user: str
    schema: Mapping[str, Any]
    max_output_tokens: int = 4096
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.operation.strip():
            raise ValueError("structured request operation is required")
        if not isinstance(self.schema, Mapping):
            raise ValueError("structured request schema must have an object root")
        schema = _owned_mapping(self.schema)
        if schema.get("type") != "object":
            raise ValueError("structured request schema must have an object root")
        if self.max_output_tokens <= 0:
            raise ValueError("structured request max_output_tokens must be positive")
        validate_structured_schema(schema, "structured-generation", self.operation)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "metadata", _owned_mapping(self.metadata))


@dataclass(frozen=True)
class StructuredResult:
    """Top-level-frozen result owning copies of nested data and usage values."""

    data: Mapping[str, Any]
    provider: str
    model: str
    usage: Mapping[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "data", _owned_mapping(self.data))
        if self.usage is not None:
            object.__setattr__(self, "usage", _owned_mapping(self.usage))


@dataclass(frozen=True)
class Readiness:
    provider: str
    model: str
    ready: bool
    auth_method: str
    dependency_available: bool
    credential_available: bool
    reason: str | None = None

    def as_dict(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "ready": self.ready,
            "auth_method": self.auth_method,
            "dependency_available": self.dependency_available,
            "credential_available": self.credential_available,
            "reason": self.reason,
        }


@runtime_checkable
class StructuredGenerator(Protocol):
    provider: str
    model: str

    def generate(self, request: StructuredRequest) -> StructuredResult: ...

    def check(self) -> Readiness: ...


class GenerationError(RuntimeError):
    """Base class whose message is safe to show to users."""

    category = "generation"

    def __init__(self, provider: str, operation: str, detail: str | None = None):
        self.provider = provider
        self.operation = operation
        message = f"{provider} {operation} failed ({self.category})"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class ConfigurationError(GenerationError):
    category = "configuration"


class AuthenticationError(GenerationError):
    category = "authentication"


class DependencyUnavailableError(GenerationError):
    category = "dependency unavailable"


class GenerationTimeoutError(GenerationError):
    category = "timeout"


class RateLimitError(GenerationError):
    category = "rate limit"


class RefusalError(GenerationError):
    category = "refusal"


class IncompleteResponseError(GenerationError):
    category = "incomplete response"


class InvalidStructuredResponseError(GenerationError):
    category = "invalid structured response"


def validate_structured_schema(
    schema: Mapping[str, Any], provider: str, operation: str
):
    try:
        validator = jsonschema.validators.validator_for(dict(schema))
        validator.check_schema(dict(schema))
    except jsonschema.SchemaError:
        raise InvalidStructuredResponseError(provider, operation) from None


def validate_structured_data(
    data: Any, schema: Mapping[str, Any], provider: str, operation: str
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise InvalidStructuredResponseError(provider, operation)
    validate_structured_schema(schema, provider, operation)
    try:
        jsonschema.validate(data, dict(schema))
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        raise InvalidStructuredResponseError(provider, operation) from None
    return data


def normalize_provider_exception(provider: str, operation: str, error: Exception):
    """Map SDK failures by type without exposing the provider's raw message."""
    name = type(error).__name__.lower()
    if "authentication" in name or "permission" in name or "unauthorized" in name:
        return AuthenticationError(provider, operation)
    if "ratelimit" in name or "rate_limit" in name or "overloaded" in name:
        return RateLimitError(provider, operation)
    if "timeout" in name:
        return GenerationTimeoutError(provider, operation)
    return GenerationError(provider, operation)
