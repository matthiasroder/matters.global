"""LLM-based candidate matter extraction.

This is the semantic counterpart to the deterministic marker extractor in
``extraction.py``. It reads prose (paper abstracts, sections, notes) and asks a
model to surface the source's actual claims, contributions, and findings as
matters, with evidence-grounded conditions and semantic dependency candidates.

The marker extractor stays the offline fallback: ``build_extraction_proposal``
uses the LLM when an API key is available and degrades to the marker engine on
any failure, so callers always get a valid proposal in the existing contract.
"""

import importlib.util
import json
import os

from .extraction import dedupe_candidates, extraction_proposal, slugify


DEFAULT_MODEL = "claude-sonnet-4-6"

MATTER_KINDS = (
    "claim",
    "finding",
    "contribution",
    "method",
    "goal",
    "problem",
    "risk",
    "decision",
    "question",
    "concern",
)

SYSTEM_PROMPT = """\
You read source text and extract the "matters" it bears — the things worth
tracking — to build a matters graph.

A matter is anything worth tracking: a concern, goal, decision, responsibility,
risk, question, claim, or finding. In research literature, a matter is anything
the work establishes about its subject, or leaves open about it.

For every matter, your most important judgment is its RESOLUTION STATUS:

- resolved — the source treats this as settled, as far as the source itself
  establishes it: a demonstrated finding, a validated method, a delivered
  contribution, an achieved goal, an answered question. These are the
  established backdrop — what the field now takes as known.
- open — the source treats this as unresolved: an open question, an unmet goal,
  an unaddressed limitation or risk, a gap, an unresolved tension between
  results, or a stated aspiration. These are the live matters the work points
  toward but has not closed.

Judge status from MEANING, not wording. Do not rely on specific cue phrases (do
not key on "future work," "open question," "we propose," "limitation," or any
fixed signal). Infer status from how the authors frame each matter: something
asserted as demonstrated is resolved; something hedged, aimed at, questioned,
acknowledged as missing, or left in tension is open. You SHOULD infer open
matters that the framing implies even when the authors never label them as open
— but ground every matter in the text, and never invent one.

A good extraction contains BOTH the settled results and the open matters the
work raises. Do not return only the headline findings; surface the genuine open
questions, gaps, and aspirations too. Often the richest open matters are the
unmet criteria behind a resolved finding — the replication, mechanism,
generalization, or application it has not yet achieved.

For each matter produce:
- name: a concise, specific title (not a full sentence; no trailing period).
- kind: one of claim, finding, contribution, method, goal, problem, risk,
  decision, question, concern.
- description: 1-2 sentences grounding the matter in the source.
- status: "resolved" or "open".
- conditions: 2-4 observable, checkable criteria that define what it means for
  this matter to be resolved, each with truth = true (met by the source) or
  false (not met).
  * For a RESOLVED matter, list only criteria the source has actually
    established — ALL of them must be true. Do NOT attach stronger criteria the
    source did not meet (replication, external validation, generalization, a
    mechanism, application) as false conditions here.
  * When such a stronger, still-unmet criterion is worth tracking, capture it as
    its OWN separate OPEN matter (e.g., "Replicate the rigidity finding across
    independent samples") rather than as a false condition on the resolved
    matter.
  * For an OPEN matter, list the criteria that would close it — at least one
    must be false; any already-met sub-criteria may be true.
  Keep conditions concrete and evidence-grounded; avoid generic placeholders.
  Status and truths must agree: a resolved matter has all conditions true; an
  open matter has at least one condition false.

Also propose dependency_candidates between matters where one must be resolved or
established before another: {prerequisite, dependent, reason} — a method before
the findings that use it, an established result before the open question that
builds on it, evidence before the claim that rests on it. Endpoints may be a
matter you are extracting (use its name) or one of the provided existing matter
ids. Give a short human-readable reason. Only include genuine prerequisite
relationships, not topical similarity.

Extract every distinct matter the source supports — do not collapse the whole
text into one matter, and do not duplicate. Ground everything in the source.
"""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(MATTER_KINDS)},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["resolved", "open"]},
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "truth": {"type": "boolean"},
                            },
                            "required": ["label", "truth"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "kind", "description", "status", "conditions"],
                "additionalProperties": False,
            },
        },
        "dependency_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prerequisite": {"type": "string"},
                    "dependent": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["prerequisite", "dependent", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates", "dependency_candidates"],
    "additionalProperties": False,
}


def build_extraction_proposal(
    source_text,
    source_type="text",
    existing_matters=(),
    *,
    use_llm=True,
    client=None,
    model=None,
):
    """Return an extraction proposal, preferring the LLM engine.

    Falls back to the deterministic marker engine when the LLM is unavailable
    (no SDK, no API key) or fails (auth error, API error, unparseable output),
    so the caller always receives a valid proposal in the standard contract.
    """
    if use_llm and (client is not None or _llm_available()):
        try:
            return llm_extraction_proposal(
                source_text,
                source_type,
                existing_matters,
                client=client,
                model=model,
            )
        except Exception as error:  # noqa: BLE001 - any failure degrades gracefully
            proposal = _marker_proposal(source_text, source_type, existing_matters)
            proposal["fallback_reason"] = f"{type(error).__name__}: {error}"
            return proposal

    return _marker_proposal(source_text, source_type, existing_matters)


def llm_extraction_proposal(
    source_text,
    source_type="text",
    existing_matters=(),
    *,
    client=None,
    model=None,
):
    """Extract candidate matters with an LLM and return a standard proposal.

    ``client`` is any object exposing ``messages.create(...)`` like the Anthropic
    SDK client; injecting a fake keeps this fully testable offline. A real client
    is constructed lazily only when one is not provided.
    """
    client = client or _default_client()
    model = model or os.environ.get("MATTERS_EXTRACT_MODEL") or DEFAULT_MODEL

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": _user_content(source_text, source_type, existing_matters),
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}
        },
    )

    data = json.loads(_response_text(response))
    candidates = _candidates_from_llm(data, source_type)
    candidate_ids = [candidate["id"] for candidate in candidates]
    dependency_candidates = _dependencies_from_llm(
        data, candidate_ids, existing_matters
    )

    return {
        "source_type": source_type,
        "candidates": candidates,
        "dependency_candidates": dependency_candidates,
        "requires_confirmation": True,
        "engine": "llm",
        "model": model,
    }


def _marker_proposal(source_text, source_type, existing_matters):
    proposal = extraction_proposal(
        source_text, source_type=source_type, existing_matters=existing_matters
    )
    proposal["engine"] = "marker"
    return proposal


def _candidates_from_llm(data, source_type):
    candidates = []
    for item in data.get("candidates", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        kind = str(item.get("kind") or "").strip()
        description = str(item.get("description") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        status = status if status in ("resolved", "open") else "open"
        candidates.append(
            {
                "id": slugify(name),
                "name": name,
                "description": description or _default_description(kind),
                "kind": kind,
                "status": status,
                "source_type": source_type,
                "conditions": _conditions_from_llm(
                    item.get("conditions"), name, status
                ),
            }
        )
    return dedupe_candidates(candidates)


def _conditions_from_llm(raw_conditions, name, status="open"):
    conditions = []
    default_truth = status == "resolved"
    for entry in raw_conditions or []:
        if isinstance(entry, dict):
            label = str(entry.get("label") or "").strip()
            truth = entry.get("truth", default_truth)
        else:
            label = str(entry).strip()
            truth = default_truth
        if label:
            conditions.append({"label": label, "truth": bool(truth)})

    if conditions:
        return conditions

    return [
        {"label": f"Resolved outcome is defined for: {name}", "truth": False},
        {"label": f"Next concrete action is chosen for: {name}", "truth": False},
    ]


def _default_description(kind):
    if kind:
        return f"{kind.title()} extracted from source."
    return "Matter extracted from source."


def _dependencies_from_llm(data, candidate_ids, existing_matters):
    valid = set(candidate_ids) | set(existing_matters)
    proposals = []
    seen = set()
    for item in data.get("dependency_candidates", []):
        if not isinstance(item, dict):
            continue
        prerequisite = _resolve_endpoint(item.get("prerequisite"), valid)
        dependent = _resolve_endpoint(item.get("dependent"), valid)
        if not prerequisite or not dependent or prerequisite == dependent:
            continue
        key = (prerequisite, dependent)
        if key in seen:
            continue
        seen.add(key)
        reason = str(item.get("reason") or "").strip() or "semantic relation"
        proposals.append(
            {
                "prerequisite": prerequisite,
                "dependent": dependent,
                "reason": reason,
            }
        )
    return proposals


def _resolve_endpoint(value, valid):
    if not value:
        return None
    candidate = str(value).strip()
    if candidate in valid:
        return candidate
    slug = slugify(candidate)
    if slug in valid:
        return slug
    return None


def _user_content(source_text, source_type, existing_matters):
    existing = "\n".join(f"- {matter}" for matter in sorted(existing_matters))
    return (
        f"Source type: {source_type}\n\n"
        "Existing matter ids (propose dependencies against these where "
        f"warranted):\n{existing or '(none)'}\n\n"
        f"Source text:\n{source_text}"
    )


def _response_text(response):
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("LLM response contained no text block")


def _default_client():
    import anthropic

    return anthropic.Anthropic()


def _llm_available():
    if importlib.util.find_spec("anthropic") is None:
        return False
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
