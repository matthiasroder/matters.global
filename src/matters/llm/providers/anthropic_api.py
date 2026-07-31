"""Anthropic Messages API structured-generation adapter."""

from __future__ import annotations

import importlib.util
import json
import os

from ..contract import (
    AuthenticationError,
    DependencyUnavailableError,
    IncompleteResponseError,
    InvalidStructuredResponseError,
    Readiness,
    RefusalError,
    StructuredRequest,
    StructuredResult,
    normalize_provider_exception,
    validate_structured_data,
)


class AnthropicGenerator:
    provider = "anthropic-api"

    def __init__(self, profile, *, environ=None, client=None):
        self.profile = profile
        self.model = profile.model
        self.environ = dict(os.environ if environ is None else environ)
        self._client = client

    def check(self):
        dependency = self._client is not None or importlib.util.find_spec("anthropic") is not None
        credential = bool(self.environ.get(self.profile.api_key_env or ""))
        return Readiness(
            self.provider,
            self.model,
            dependency and credential,
            "api-key-environment",
            dependency,
            credential,
            None
            if dependency and credential
            else ("Anthropic SDK unavailable" if not dependency else "credential unavailable"),
        )

    def generate(self, request: StructuredRequest):
        readiness = self.check()
        if not readiness.dependency_available:
            raise DependencyUnavailableError(self.provider, request.operation)
        if not readiness.credential_available:
            raise AuthenticationError(self.provider, request.operation)
        client = self._client or self._new_client(request.operation)
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=request.max_output_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config={
                    "format": {"type": "json_schema", "schema": dict(request.schema)}
                },
            )
        except Exception as error:
            raise normalize_provider_exception(
                self.provider, request.operation, error
            ) from None
        try:
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "refusal":
                raise RefusalError(self.provider, request.operation)
            if stop_reason == "max_tokens":
                raise IncompleteResponseError(self.provider, request.operation)
            text = next(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            data = json.loads(text)
            data = validate_structured_data(
                data, request.schema, self.provider, request.operation
            )
            usage = _usage(response)
        except (IncompleteResponseError, RefusalError, InvalidStructuredResponseError):
            raise
        except (AttributeError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
            raise InvalidStructuredResponseError(
                self.provider, request.operation
            ) from None
        return StructuredResult(data, self.provider, self.model, usage)

    def _new_client(self, operation):
        try:
            import anthropic

            return anthropic.Anthropic(
                api_key=self.environ[self.profile.api_key_env],
                timeout=self.profile.timeout_seconds,
            )
        except Exception as error:
            raise normalize_provider_exception(self.provider, operation, error) from None


def _usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return {key: int(value) for key, value in values.items() if value is not None} or None
