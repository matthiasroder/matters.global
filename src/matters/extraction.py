"""Candidate matter extraction from source text."""

import re


MATTER_MARKERS = (
    "goal",
    "matter",
    "problem",
    "decision",
    "risk",
    "responsibility",
    "todo",
)


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "untitled_matter"


def extract_candidate_matters(source_text, source_type="text"):
    candidates = []
    for line in source_text.splitlines():
        candidate = extract_candidate_from_line(line, source_type)
        if candidate is not None:
            candidates.append(candidate)

    if candidates:
        return dedupe_candidates(candidates)

    fallback = source_text.strip().splitlines()
    if not fallback:
        return []

    title = fallback[0].strip()[:80]
    return [
        create_candidate(
            title,
            "Matter extracted from unstructured source text.",
            source_type,
        )
    ]


def extract_candidate_from_line(line, source_type):
    stripped = line.strip()
    if not stripped:
        return None

    checkbox_match = re.match(r"^[-*]\s+\[[ xX ]\]\s+(.+)$", stripped)
    if checkbox_match:
        title = checkbox_match.group(1).strip()
        return create_candidate(title, "Checkbox item extracted from source.", source_type)

    marker_match = re.match(
        rf"^(?:[-*]\s*)?({'|'.join(MATTER_MARKERS)})\s*:\s*(.+)$",
        stripped,
        flags=re.IGNORECASE,
    )
    if marker_match:
        marker = marker_match.group(1).lower()
        title = marker_match.group(2).strip()
        return create_candidate(title, f"{marker.title()} extracted from source.", source_type)

    return None


def create_candidate(title, description, source_type):
    return {
        "id": slugify(title),
        "name": title,
        "description": description,
        "source_type": source_type,
        "conditions": [
            {
                "label": f"Resolved outcome is defined for: {title}",
                "truth": False,
            },
            {
                "label": f"Next concrete action is chosen for: {title}",
                "truth": False,
            },
        ],
    }


def dedupe_candidates(candidates):
    out = []
    seen = set()
    for candidate in candidates:
        base_id = candidate["id"]
        candidate_id = base_id
        suffix = 2
        while candidate_id in seen:
            candidate_id = f"{base_id}_{suffix}"
            suffix += 1
        candidate = {**candidate, "id": candidate_id}
        seen.add(candidate_id)
        out.append(candidate)
    return out


def propose_dependency_candidates(candidates, existing_matters):
    existing = set(existing_matters)
    proposals = []

    for candidate in candidates:
        words = set(re.findall(r"[a-z0-9]+", candidate["name"].lower()))
        for matter in sorted(existing):
            matter_words = set(re.findall(r"[a-z0-9]+", matter.lower()))
            if words & matter_words:
                proposals.append(
                    {
                        "prerequisite": matter,
                        "dependent": candidate["id"],
                        "reason": "name overlap",
                    }
                )

    return proposals


def extraction_proposal(source_text, source_type="text", existing_matters=()):
    candidates = extract_candidate_matters(source_text, source_type)
    return {
        "source_type": source_type,
        "candidates": candidates,
        "dependency_candidates": propose_dependency_candidates(
            candidates, existing_matters
        ),
        "requires_confirmation": True,
    }
