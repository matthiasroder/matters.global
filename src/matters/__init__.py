"""Reusable matters engine.

``resolved``, ``unresolved``, ``universe``, ``frontier``, ``horizon`` and
``descendants`` are re-exported here under the names they have always had,
but they are now :mod:`matters.graph_index`'s, not :mod:`matters.engine`'s.
Two consequences for anyone importing them, both deliberate at 0.1.0:

* they refuse a graph containing a cycle with
  :class:`~matters.graph_index.DependencyCycleError`, naming one concrete
  cycle, where the recursive versions sometimes raised
  ``ValueError("dependency cycle")`` and sometimes returned a wrong answer
  instead, depending on which condition happened to be false first;
* ``frontier``, ``horizon`` and ``descendants`` take the ``matters`` set as
  their second argument. They used to infer their nodes from the edge list,
  which cannot see an isolated matter and cannot tell a typo from a real id.

``prerequisites`` and ``dependents`` are unchanged and still come from
``engine``: they read the edge set and never traverse, so they answer even
on a state file that contains a cycle.
"""

from .engine import (
    as_condition_list,
    condition_label,
    create_condition,
    dependents,
    has_dependency_cycle,
    normalize_conditions,
    prerequisites,
    serialize_condition,
    truth,
)
from .extraction import (
    extract_candidate_matters,
    extraction_proposal,
    propose_dependency_candidates,
    slugify,
)
from .graph_index import (
    DependencyCycleError,
    descendants,
    frontier,
    horizon,
    resolved,
    universe,
    unresolved,
)
from .identity import (
    EmbeddingStore,
    FakeEmbedder,
    LocalEmbedder,
    classify_relationship,
    get_embedder,
    ingest_candidates,
    match_candidate,
    reconcile_candidates,
)
from .llm_extraction import (
    build_extraction_proposal,
    llm_extraction_proposal,
)
from .llm import (
    ConfigError,
    GenerationError,
    Readiness,
    StructuredGenerator,
    StructuredRequest,
    StructuredResult,
    create_generator,
    load_config,
    register_provider,
    resolve_generator,
)
from .reports import (
    false_condition_labels,
    format_unlock_report,
    propose_action,
    unlock_items,
    unlock_report,
)
from .sharing import PUBLIC, merge_public_state, public_state
from .storage import (
    DEFAULT_STATE_PATH,
    load_state,
    resolve_state_path,
    save_state,
)
from .tots import (
    TotsError,
    build_tots_context,
    build_tots_proposal,
    fit_bradley_terry_davidson,
    reconcile_ordered_judgments,
    schedule_swiss_pairs,
    select_diverse_finalists,
)
from .view import (
    build_view_payload,
    render_view_html,
    write_view,
)

__all__ = [
    "DEFAULT_STATE_PATH",
    "ConfigError",
    "DependencyCycleError",
    "EmbeddingStore",
    "FakeEmbedder",
    "GenerationError",
    "LocalEmbedder",
    "PUBLIC",
    "Readiness",
    "StructuredGenerator",
    "StructuredRequest",
    "StructuredResult",
    "TotsError",
    "as_condition_list",
    "build_extraction_proposal",
    "build_tots_context",
    "build_tots_proposal",
    "build_view_payload",
    "classify_relationship",
    "condition_label",
    "create_condition",
    "create_generator",
    "dependents",
    "descendants",
    "extract_candidate_matters",
    "extraction_proposal",
    "fit_bradley_terry_davidson",
    "get_embedder",
    "ingest_candidates",
    "llm_extraction_proposal",
    "match_candidate",
    "reconcile_candidates",
    "frontier",
    "false_condition_labels",
    "format_unlock_report",
    "has_dependency_cycle",
    "horizon",
    "load_config",
    "load_state",
    "merge_public_state",
    "normalize_conditions",
    "prerequisites",
    "propose_action",
    "propose_dependency_candidates",
    "public_state",
    "reconcile_ordered_judgments",
    "register_provider",
    "render_view_html",
    "resolve_generator",
    "resolve_state_path",
    "resolved",
    "save_state",
    "schedule_swiss_pairs",
    "select_diverse_finalists",
    "serialize_condition",
    "slugify",
    "truth",
    "unlock_items",
    "unlock_report",
    "universe",
    "unresolved",
    "write_view",
]
