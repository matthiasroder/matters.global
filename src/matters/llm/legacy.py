"""Temporary compatibility for injected Anthropic-shaped clients."""

from __future__ import annotations

import json
import warnings

from .contract import (
    GenerationError,
    InvalidStructuredResponseError,
    Readiness,
    StructuredRequest,
    StructuredResult,
    validate_structured_data,
)


LEGACY_DEFAULT_MODEL = "claude-sonnet-4-6"


class LegacyMessagesGenerator:
    provider = "legacy-messages-client"

    def __init__(self, client, model=None):
        self.client = client
        self.model = model or LEGACY_DEFAULT_MODEL

    def check(self):
        return Readiness(self.provider, self.model, True, "injected", True, True)

    def generate(self, request: StructuredRequest):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=request.max_output_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config={
                    "format": {"type": "json_schema", "schema": dict(request.schema)}
                },
            )
        except Exception:
            raise GenerationError(self.provider, request.operation) from None
        try:
            text = next(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            )
            data = json.loads(text)
        except (AttributeError, StopIteration, TypeError, json.JSONDecodeError):
            raise InvalidStructuredResponseError(
                self.provider, request.operation
            ) from None
        data = validate_structured_data(
            data, request.schema, self.provider, request.operation
        )
        return StructuredResult(data, self.provider, self.model)


def coerce_generator(value, *, model=None):
    if hasattr(value, "generate") and hasattr(value, "check"):
        return value
    if hasattr(value, "messages") and hasattr(value.messages, "create"):
        warnings.warn(
            "Anthropic-shaped injected clients are deprecated; inject a "
            "StructuredGenerator instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return LegacyMessagesGenerator(value, model=model)
    raise TypeError("injected model object must implement StructuredGenerator")
