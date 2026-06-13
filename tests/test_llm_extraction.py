import json
from types import SimpleNamespace

import pytest

from matters import build_extraction_proposal, llm_extraction_proposal
from matters.llm_extraction import _llm_available


class FakeClient:
    """Stand-in for anthropic.Anthropic exposing messages.create(...)."""

    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        text = json.dumps(self._payload)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class ExplodingClient:
    """A client whose create() must never be called."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):  # pragma: no cover - asserted not called
        raise AssertionError("LLM client should not be called")


PAPER_PAYLOAD = {
    "candidates": [
        {
            "name": "Design-thinking training raises idea fluency",
            "kind": "finding",
            "description": "A four-week intervention increased fluency 23% vs controls.",
            "conditions": [
                {"label": "Effect replicates beyond the 120-student sample"},
                {"label": "Fluency gain holds at p < 0.05 against a control group"},
            ],
        },
        {
            "name": "Creativity gains do not transfer across domains",
            "kind": "risk",
            "description": "Gains did not carry to a domain-general insight task.",
            "conditions": [
                {"label": "Transfer is measured on an independent insight task"},
            ],
        },
    ],
    "dependency_candidates": [
        {
            "prerequisite": "Design-thinking training raises idea fluency",
            "dependent": "Creativity gains do not transfer across domains",
            "reason": "the transfer caveat depends on the fluency finding",
        },
        {
            "prerequisite": "build_general_purpose_web_research_agent",
            "dependent": "Design-thinking training raises idea fluency",
            "reason": "the finding depends on source discovery",
        },
        {
            "prerequisite": "Some matter that was never extracted",
            "dependent": "Another phantom matter",
            "reason": "should be dropped — endpoints do not resolve",
        },
    ],
}


def test_llm_proposal_maps_structured_response():
    client = FakeClient(PAPER_PAYLOAD)

    proposal = llm_extraction_proposal(
        "Divergent thinking has long been used as a proxy for creative potential...",
        source_type="paper",
        existing_matters={"build_general_purpose_web_research_agent"},
        client=client,
        model="claude-sonnet-4-6",
    )

    assert proposal["engine"] == "llm"
    assert proposal["model"] == "claude-sonnet-4-6"
    assert proposal["source_type"] == "paper"
    assert proposal["requires_confirmation"] is True

    ids = [candidate["id"] for candidate in proposal["candidates"]]
    assert ids == [
        "design_thinking_training_raises_idea_fluency",
        "creativity_gains_do_not_transfer_across_domains",
    ]

    first = proposal["candidates"][0]
    assert first["source_type"] == "paper"
    assert first["conditions"][0] == {
        "label": "Effect replicates beyond the 120-student sample",
        "truth": False,
    }
    assert all(
        condition["truth"] is False
        for candidate in proposal["candidates"]
        for condition in candidate["conditions"]
    )


def test_llm_proposal_resolves_and_drops_dependencies():
    proposal = llm_extraction_proposal(
        "...",
        source_type="paper",
        existing_matters={"build_general_purpose_web_research_agent"},
        client=FakeClient(PAPER_PAYLOAD),
    )

    assert proposal["dependency_candidates"] == [
        {
            "prerequisite": "design_thinking_training_raises_idea_fluency",
            "dependent": "creativity_gains_do_not_transfer_across_domains",
            "reason": "the transfer caveat depends on the fluency finding",
        },
        {
            "prerequisite": "build_general_purpose_web_research_agent",
            "dependent": "design_thinking_training_raises_idea_fluency",
            "reason": "the finding depends on source discovery",
        },
    ]


def test_llm_proposal_sends_schema_and_model_to_client():
    client = FakeClient(PAPER_PAYLOAD)

    llm_extraction_proposal("text", client=client, model="claude-opus-4-8")

    call = client.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "candidates" in call["output_config"]["format"]["schema"]["properties"]


def test_llm_proposal_supplies_default_conditions_when_missing():
    payload = {
        "candidates": [
            {
                "name": "A bare matter",
                "kind": "goal",
                "description": "",
                "conditions": [],
            }
        ],
        "dependency_candidates": [],
    }

    proposal = llm_extraction_proposal("text", client=FakeClient(payload))

    candidate = proposal["candidates"][0]
    assert candidate["description"] == "Goal extracted from source."
    assert candidate["conditions"] == [
        {"label": "Resolved outcome is defined for: A bare matter", "truth": False},
        {"label": "Next concrete action is chosen for: A bare matter", "truth": False},
    ]


def test_llm_proposal_dedupes_repeated_names():
    payload = {
        "candidates": [
            {"name": "Same matter", "kind": "claim", "description": "x", "conditions": []},
            {"name": "Same matter", "kind": "claim", "description": "y", "conditions": []},
        ],
        "dependency_candidates": [],
    }

    proposal = llm_extraction_proposal("text", client=FakeClient(payload))

    assert [c["id"] for c in proposal["candidates"]] == ["same_matter", "same_matter_2"]


def test_build_falls_back_to_marker_on_llm_error():
    client = FakeClient(error=RuntimeError("api exploded"))

    proposal = build_extraction_proposal(
        "Goal: Build shared matter map\n",
        source_type="notes",
        client=client,
    )

    assert proposal["engine"] == "marker"
    assert "RuntimeError" in proposal["fallback_reason"]
    assert proposal["candidates"][0]["id"] == "build_shared_matter_map"
    assert proposal["requires_confirmation"] is True


def test_build_falls_back_on_unparseable_output():
    bad = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                content=[SimpleNamespace(type="text", text="not json")]
            )
        )
    )

    proposal = build_extraction_proposal("Goal: X\n", client=bad)

    assert proposal["engine"] == "marker"
    assert "JSONDecodeError" in proposal["fallback_reason"]


def test_build_uses_marker_when_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert _llm_available() is False

    proposal = build_extraction_proposal("Goal: Build shared matter map\n")

    assert proposal["engine"] == "marker"
    assert "fallback_reason" not in proposal
    assert proposal["candidates"][0]["id"] == "build_shared_matter_map"


def test_no_llm_flag_never_calls_client():
    proposal = build_extraction_proposal(
        "Goal: Build shared matter map\n",
        use_llm=False,
        client=ExplodingClient(),
    )

    assert proposal["engine"] == "marker"
    assert proposal["candidates"][0]["id"] == "build_shared_matter_map"


def test_llm_available_true_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert _llm_available() is True
