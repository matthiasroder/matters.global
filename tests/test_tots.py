import json
import traceback
from types import SimpleNamespace

import pytest

from matters import (
    Readiness,
    StructuredResult,
    TotsError,
    build_tots_context,
    build_tots_proposal,
    fit_bradley_terry_davidson,
    reconcile_ordered_judgments,
    schedule_swiss_pairs,
    select_diverse_finalists,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:Anthropic-shaped injected clients are deprecated.*:DeprecationWarning"
)


def graph_state():
    return (
        {"foundation", "blocked_prerequisite", "target", "downstream"},
        {
            "foundation": [{"label": "Background established", "truth": True}],
            "blocked_prerequisite": [
                {"label": "Blocking evidence collected", "truth": False}
            ],
            "target": [{"label": "Target question answered", "truth": False}],
            "downstream": [{"label": "Application validated", "truth": False}],
        },
        {
            ("foundation", "blocked_prerequisite"),
            ("blocked_prerequisite", "target"),
            ("target", "downstream"),
        },
    )


def candidate_payload(number, reference="condition:target:1"):
    return {
        "title": f"Candidate {number}",
        "hypothesis": f"Mechanism {number} explains the target",
        "mechanism": f"Intervention {number} changes the relevant mechanism",
        "assumptions": [f"Assumption {number}"],
        "predictions": [f"Prediction {number} differs from alternatives"],
        "test": {
            "description": f"Run discriminating test {number}",
            "supports_if": f"Outcome {number} is observed",
            "refutes_if": f"Outcome {number} is absent",
        },
        "supporting_evidence": [
            {"ref": reference, "claim": "The target condition is unresolved"}
        ],
        "contradicting_evidence": [],
    }


class FakeStageClient:
    def __init__(
        self,
        *,
        malformed_stage=None,
        error_stage=None,
        reference="condition:target:1",
        reject_expansion=False,
        malformed_child_entry=False,
    ):
        self.malformed_stage = malformed_stage
        self.error_stage = error_stage
        self.reference = reference
        self.reject_expansion = reject_expansion
        self.malformed_child_entry = malformed_child_entry
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        stage = kwargs["system"].split("[matters-tots:", 1)[1].split("]", 1)[0]
        if stage == self.error_stage:
            raise RuntimeError("secret provider detail")
        if stage == self.malformed_stage:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="not json")]
            )

        payload = json.loads(kwargs["messages"][0]["content"])
        if stage == "generation":
            response = {
                "candidates": [
                    candidate_payload(number, self.reference)
                    for number in range(1, payload["requested_candidates"] + 1)
                ]
            }
        elif stage == "reflection":
            response = {
                "reflections": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "graph_consistency": (
                            "fail"
                            if self.reject_expansion
                            and int(candidate["candidate_id"][1:]) >= 5
                            else "pass"
                        ),
                        "evidence_grounding": "pass",
                        "testability": "pass",
                        "issues": [],
                    }
                    for candidate in payload["candidates"]
                ]
            }
        elif stage == "proximity":
            ids = [candidate["candidate_id"] for candidate in payload["candidates"]]
            groups = [ids[:2]] + [[candidate_id] for candidate_id in ids[2:]]
            response = {
                "clusters": [
                    {
                        "candidate_ids": group,
                        "reason": "same direction" if len(group) > 1 else "distinct",
                    }
                    for group in groups
                    if group
                ]
            }
        elif stage == "pairwise":
            response = {
                "judgments": [self._judgment(comparison) for comparison in payload["comparisons"]]
            }
        elif stage == "expansion":
            parents = payload["parents"]
            operations = ["refine", "falsify", "simplify", "combine"]
            response = {
                "children": [
                    {
                        "parent_id": parents[index % len(parents)]["candidate_id"],
                        "operation": operations[index % len(operations)],
                        "candidate": candidate_payload(100 + index),
                    }
                    for index in range(payload["requested_children"])
                ]
            }
            if self.malformed_child_entry:
                response["children"].insert(0, "malformed")
        else:  # pragma: no cover - a new stage must be added deliberately
            raise AssertionError(stage)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(response))]
        )

    @staticmethod
    def _judgment(comparison):
        left_id = comparison["left"]["candidate_id"]
        right_id = comparison["right"]["candidate_id"]
        winner = "left" if left_id < right_id else "right"
        return {
            "comparison_id": comparison["comparison_id"],
            "winner": winner,
            "criterion_winners": {
                "contextual_grounding": winner,
                "inferential_insight": winner,
                "evidential_justification": winner,
                "analytical_utility": winner,
            },
            "reason": "The lower stable id wins in this deterministic fake.",
        }


class FakeStageGenerator:
    provider = "fake-provider"
    model = "fake-model"

    def __init__(self):
        self.client = FakeStageClient()

    def check(self):
        return Readiness(self.provider, self.model, True, "fake", True, True)

    def generate(self, request):
        response = self.client._create(
            model=self.model,
            max_tokens=request.max_output_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            output_config={"format": {"type": "json_schema", "schema": request.schema}},
        )
        return StructuredResult(
            json.loads(response.content[0].text), self.provider, self.model
        )


def test_context_contains_false_conditions_and_bounded_ancestry():
    matters, conditions, dependencies = graph_state()

    context = build_tots_context(
        "target",
        matters,
        conditions,
        dependencies,
        context_text="first line\nsecond line",
        context_cap=1,
    )

    assert context["target"]["false_conditions"] == [
        {"label": "Target question answered", "truth": False}
    ]
    assert context["direct_prerequisites"][0]["matter_id"] == "blocked_prerequisite"
    assert context["unresolved_prerequisite_ancestry"][0]["distance"] == 1
    assert context["supplemental_context"]["lines"][1] == {
        "line": 2,
        "text": "second line",
    }
    assert "truncated" in context["warnings"][0]


def test_tots_accepts_provider_neutral_generator():
    matters, conditions, dependencies = graph_state()
    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        breadth=2,
        depth=1,
        max_candidates=2,
        max_comparisons=2,
        generator=FakeStageGenerator(),
    )

    assert result["provider"] == "fake-provider"
    assert result["model"] == "fake-model"
    assert result["model_profile"] == "injected"


def test_legacy_tots_client_uses_workflow_model_environment(monkeypatch):
    matters, conditions, dependencies = graph_state()
    monkeypatch.setenv("MATTERS_TOTS_MODEL", "tots-environment-model")
    client = FakeStageClient()

    build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        breadth=2,
        depth=1,
        max_candidates=2,
        max_comparisons=2,
        client=client,
    )

    assert {call["model"] for call in client.calls} == {"tots-environment-model"}


def test_unknown_resolved_and_cyclic_targets_fail_before_model_use():
    matters, conditions, dependencies = graph_state()
    client = FakeStageClient()

    with pytest.raises(TotsError, match="unknown matter"):
        build_tots_proposal(
            "missing", matters, conditions, dependencies, client=client
        )
    with pytest.raises(TotsError, match="already resolved"):
        build_tots_proposal(
            "foundation", matters, conditions, dependencies, client=client
        )
    with pytest.raises(TotsError, match="cycle"):
        build_tots_proposal(
            "target",
            matters,
            conditions,
            dependencies | {("target", "foundation")},
            client=client,
        )
    assert client.calls == []


def test_schedule_is_connected_bounded_and_deterministic():
    ratings = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}

    first = schedule_swiss_pairs(["d", "c", "b", "a"], 4, seed_ratings=ratings)
    second = schedule_swiss_pairs(["a", "b", "c", "d"], 4, seed_ratings=ratings)

    assert first == second
    assert first[:3] == [("a", "b"), ("b", "c"), ("c", "d")]
    assert len(first) == 4


def test_reversed_order_requires_consistent_absolute_winner():
    first = {"left_id": "a", "right_id": "b", "winner": "left"}
    consistent = {"left_id": "b", "right_id": "a", "winner": "right"}
    biased = {"left_id": "b", "right_id": "a", "winner": "left"}

    assert reconcile_ordered_judgments(first, consistent) == "a"
    assert reconcile_ordered_judgments(first, biased) == "tie"


def test_btd_ranks_repeated_winner_and_handles_all_ties():
    comparisons = [
        {"candidate_a": "a", "candidate_b": "b", "outcome": "a"},
        {"candidate_a": "a", "candidate_b": "c", "outcome": "a"},
        {"candidate_a": "b", "candidate_b": "c", "outcome": "b"},
    ]

    result = fit_bradley_terry_davidson(["a", "b", "c"], comparisons)
    rank = {item["candidate_id"]: item["rank"] for item in result["ratings"]}
    assert rank == {"a": 1, "b": 2, "c": 3}

    tied = fit_bradley_terry_davidson(
        ["b", "a"],
        [{"candidate_a": "a", "candidate_b": "b", "outcome": "tie"}],
    )
    assert [item["candidate_id"] for item in tied["ratings"]] == ["a", "b"]
    assert {item["ability"] for item in tied["ratings"]} == {0.0}


def test_diverse_finalists_take_distinct_clusters_before_rank_fill():
    candidates = [
        {"candidate_id": "a", "cluster_id": "x", "tournament": {"rank": 1, "ability": 2}},
        {"candidate_id": "b", "cluster_id": "x", "tournament": {"rank": 2, "ability": 1}},
        {"candidate_id": "c", "cluster_id": "y", "tournament": {"rank": 3, "ability": 0}},
    ]

    finalists = select_diverse_finalists(candidates, limit=2)

    assert [item["candidate_id"] for item in finalists] == ["a", "c"]


def test_full_search_builds_bounded_immutable_tree_and_uses_complete_budget():
    matters, conditions, dependencies = graph_state()
    client = FakeStageClient()

    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        context_text="Observed result\nContradicting result",
        client=client,
        model="test-model",
    )

    assert result["ranking_semantics"] == "search_priority_not_truth"
    assert result["requires_confirmation"] is True
    assert result["state_modified"] is False
    assert result["budget"] == {
        "candidates_generated": 8,
        "candidate_limit": 8,
        "ordered_comparisons_used": 24,
        "ordered_comparison_limit": 24,
    }
    assert len(result["tree"]) == 8
    assert len(result["pairwise_comparisons"]) == 12
    assert len(result["finalists"]) == 3
    initial = {node["candidate_id"]: node for node in result["tree"][:4]}
    assert initial["c001"]["hypothesis"] == "Mechanism 1 explains the target"
    assert any(node["parent_id"] == "c001" for node in result["tree"][4:])
    assert initial["c001"]["expanded"] is True
    assert all(
        node["validation"]["status"] == "viable" for node in result["tree"]
    )
    assert {call["model"] for call in client.calls} == {"test-model"}
    assert result["provider"] == "legacy-messages-client"
    assert result["model_profile"] == "injected"


def test_external_hard_failure_excludes_candidate_from_tournament():
    matters, conditions, dependencies = graph_state()

    def evaluator(candidate, _context):
        if candidate["candidate_id"] == "c001":
            return {
                "type": "fail",
                "provenance": "deterministic-check",
                "reason": "contradicted by an external observation",
            }
        return {"type": "pass", "provenance": "deterministic-check"}

    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        breadth=4,
        depth=1,
        max_candidates=4,
        max_comparisons=6,
        client=FakeStageClient(),
        evaluator=evaluator,
    )

    rejected = next(node for node in result["tree"] if node["candidate_id"] == "c001")
    assert rejected["validation"]["status"] == "rejected"
    assert "c001" not in {item["candidate_id"] for item in result["finalists"]}
    assert all(
        "c001" not in (item["candidate_a"], item["candidate_b"])
        for item in result["pairwise_comparisons"]
    )


def test_explicit_external_preference_sorts_before_model_rank():
    matters, conditions, dependencies = graph_state()

    def evaluator(candidate, _context):
        return {
            "type": "human",
            "provenance": "reviewer-1",
            "preference": (
                "promote" if candidate["candidate_id"] == "c004" else "neutral"
            ),
        }

    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        breadth=4,
        depth=1,
        max_candidates=4,
        max_comparisons=6,
        client=FakeStageClient(),
        evaluator=evaluator,
    )

    assert result["selection_precedence"] == [
        "external_evaluator",
        "model_tournament",
    ]
    assert result["finalists"][0]["candidate_id"] == "c004"
    assert result["finalists"][0]["external_preference"] == 1


def test_external_preference_controls_which_parent_is_expanded():
    matters, conditions, dependencies = graph_state()

    def evaluator(candidate, _context):
        return {
            "type": "human",
            "provenance": "reviewer-1",
            "preference": (
                "promote" if candidate["candidate_id"] == "c004" else "neutral"
            ),
        }

    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        client=FakeStageClient(),
        evaluator=evaluator,
    )

    assert any(node["parent_id"] == "c004" for node in result["tree"][4:])
    promoted_parent = next(
        node for node in result["tree"] if node["candidate_id"] == "c004"
    )
    assert promoted_parent["expanded"] is True


def test_invalid_evidence_is_reported_and_removed_from_judge_inputs():
    matters, conditions, dependencies = graph_state()
    client = FakeStageClient(reference="matter:not_in_context")

    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        breadth=2,
        depth=1,
        max_candidates=2,
        max_comparisons=2,
        client=client,
    )

    assert "unsupported evidence references" in result["tree"][0]["validation"][
        "issues"
    ][0]
    pairwise_call = next(
        call for call in client.calls if "[matters-tots:pairwise]" in call["system"]
    )
    pairwise_payload = json.loads(pairwise_call["messages"][0]["content"])
    for comparison in pairwise_payload["comparisons"]:
        assert comparison["left"]["supporting_evidence"] == []
        assert comparison["right"]["supporting_evidence"] == []


def test_rejected_children_do_not_replace_viable_parents():
    matters, conditions, dependencies = graph_state()

    result = build_tots_proposal(
        "target",
        matters,
        conditions,
        dependencies,
        client=FakeStageClient(reject_expansion=True),
    )

    parents = {node["candidate_id"]: node for node in result["tree"][:4]}
    assert parents["c001"]["expanded"] is False
    assert parents["c003"]["expanded"] is False
    assert {item["candidate_id"] for item in result["finalists"]}.issubset(parents)
    assert all(
        node["validation"]["status"] == "rejected" for node in result["tree"][4:]
    )


def test_schema_invalid_expansion_is_rejected_at_model_boundary():
    matters, conditions, dependencies = graph_state()

    with pytest.raises(TotsError, match="expansion model response was not valid JSON"):
        build_tots_proposal(
            "target",
            matters,
            conditions,
            dependencies,
            client=FakeStageClient(malformed_child_entry=True),
        )


def test_model_boundary_reports_stage_without_leaking_provider_detail():
    matters, conditions, dependencies = graph_state()

    with pytest.raises(TotsError, match="generation model call failed") as error:
        build_tots_proposal(
            "target",
            matters,
            conditions,
            dependencies,
            client=FakeStageClient(error_stage="generation"),
        )
    assert "secret provider detail" not in str(error.value)
    assert "secret provider detail" not in "".join(
        traceback.format_exception(error.value)
    )

    with pytest.raises(TotsError, match="not valid JSON") as malformed_error:
        build_tots_proposal(
            "target",
            matters,
            conditions,
            dependencies,
            client=FakeStageClient(malformed_stage="generation"),
        )
    assert "not json" not in str(malformed_error.value)
    assert "not json" not in "".join(
        traceback.format_exception(malformed_error.value)
    )


def test_configuration_rejects_unconnected_comparison_budget():
    matters, conditions, dependencies = graph_state()

    with pytest.raises(TotsError, match="too small"):
        build_tots_proposal(
            "target",
            matters,
            conditions,
            dependencies,
            breadth=4,
            max_comparisons=4,
            client=FakeStageClient(),
        )


def test_missing_model_profile_fails_explicitly(monkeypatch, tmp_path):
    matters, conditions, dependencies = graph_state()
    monkeypatch.setattr("matters.llm.config.user_config_dir", lambda _name: str(tmp_path))

    with pytest.raises(TotsError, match="no model profile configured"):
        build_tots_proposal("target", matters, conditions, dependencies)


def test_api_key_does_not_silently_select_tots_provider(monkeypatch, tmp_path):
    matters, conditions, dependencies = graph_state()
    monkeypatch.setattr("matters.llm.config.user_config_dir", lambda _name: str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with pytest.raises(TotsError, match="no model profile configured") as error:
        build_tots_proposal("target", matters, conditions, dependencies)
    assert "test-key" not in str(error.value)
