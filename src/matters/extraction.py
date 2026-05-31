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

DEPENDENCY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "build",
    "can",
    "develop",
    "for",
    "from",
    "global",
    "in",
    "into",
    "make",
    "matters",
    "of",
    "on",
    "or",
    "publish",
    "resolve",
    "system",
    "the",
    "to",
    "use",
    "with",
}

CREATIVITY_RESEARCH_TERMS = {
    "application",
    "applications",
    "claim",
    "claims",
    "context",
    "creativity",
    "evidence",
    "finding",
    "findings",
    "intervention",
    "interventions",
    "research",
    "source",
    "sources",
}


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
        return create_candidate(
            title,
            "Checkbox item extracted from source.",
            source_type,
            marker="checkbox",
        )

    marker_match = match_marker_line(stripped)
    if marker_match:
        marker = marker_match["marker"].lower()
        title = marker_match["title"].strip()
        return create_candidate(
            title,
            f"{marker.title()} extracted from source.",
            source_type,
            marker=marker,
        )

    return None


def match_marker_line(stripped):
    direct_match = re.match(
        rf"^(?:[-*]\s*)?({'|'.join(MATTER_MARKERS)})\s*:\s*(.+)$",
        stripped,
        flags=re.IGNORECASE,
    )
    if direct_match:
        return {
            "marker": direct_match.group(1),
            "title": direct_match.group(2),
        }

    speaker_match = re.match(
        rf"^(?:[-*]\s*)?[^:]+:\s*({'|'.join(MATTER_MARKERS)})\s*:\s*(.+)$",
        stripped,
        flags=re.IGNORECASE,
    )
    if speaker_match:
        return {
            "marker": speaker_match.group(1),
            "title": speaker_match.group(2),
        }

    return None


def create_candidate(title, description, source_type, marker=None):
    return {
        "id": slugify(title),
        "name": title,
        "description": description,
        "source_type": source_type,
        "conditions": create_conditions(title, source_type, marker),
    }


def create_conditions(title, source_type, marker=None):
    if is_creativity_research_matter(title, source_type):
        labels = [
            f"Source context is captured for: {title}",
            f"Evidence quality and limits are reviewed for: {title}",
        ]

        if marker == "problem":
            labels.append(f"Research gap and affected context are observable for: {title}")
        elif marker == "risk":
            labels.append(f"Mitigation or transfer condition is defined for: {title}")
        elif marker == "decision":
            labels.append(
                f"Decision rationale and downstream graph impact are recorded for: {title}"
            )
        elif marker in {"goal", "matter", "responsibility"}:
            labels.append(
                f"Human reviewer accepts this creativity research matter before persistence: {title}"
            )
        else:
            labels.append(f"Next concrete action is chosen for: {title}")

        return [{"label": label, "truth": False} for label in labels]

    labels_by_marker = {
        "decision": [
            f"Decision criteria are documented for: {title}",
            f"Chosen option and next action are recorded for: {title}",
        ],
        "problem": [
            f"Problem state is observable for: {title}",
            f"First concrete step toward resolution is chosen for: {title}",
        ],
        "risk": [
            f"Risk impact and trigger are described for: {title}",
            f"Mitigation or owner is chosen for: {title}",
        ],
        "responsibility": [
            f"Owner and expected outcome are clear for: {title}",
            f"Next concrete action is chosen for: {title}",
        ],
    }
    labels = labels_by_marker.get(
        marker,
        [
            f"Resolved outcome is defined for: {title}",
            f"Next concrete action is chosen for: {title}",
        ],
    )
    return [{"label": label, "truth": False} for label in labels]


def is_creativity_research_matter(title, source_type):
    words = word_set(f"{source_type} {title}")
    return bool(words & CREATIVITY_RESEARCH_TERMS)


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
        words = meaningful_words(candidate["name"])
        for matter in sorted(existing):
            matter_words = meaningful_words(matter)
            if words & matter_words:
                proposals.append(
                    {
                        "prerequisite": matter,
                        "dependent": candidate["id"],
                        "reason": "name overlap",
                    }
                )

    return proposals


def word_set(value):
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def meaningful_words(value):
    return {word for word in word_set(value) if word not in DEPENDENCY_STOPWORDS}


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
