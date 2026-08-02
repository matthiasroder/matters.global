"""Single source of truth for matters state mutation rules.

Both ``matters.cli`` and ``matters.web`` route every write through this
module. A copy of any rule here in either caller fails AC-13.

Import direction, enforced: this module imports ``.engine``,
``.graph_index``, ``.storage`` and ``.extraction`` only. It must never
import ``.cli`` or ``.web``.

Three index bases coexist in this codebase. Do not infer one from another:

===========  ==================================================================
Base         Where
===========  ==================================================================
1-based      CLI ``condition-ref``, ``show`` text output, the ``index``
             argument of :func:`normalize_condition` (fallback label only)
0-based      stored list position, web API ``index`` payload field,
             :func:`require_condition_index`, ``show --json`` ``index`` field
===========  ==================================================================

:func:`resolve_condition_ref` is the only 1-based to 0-based converter in
the codebase. An all-digit label is therefore unaddressable by label, which
is why ``show`` prints a number for every condition; and a reference
captured before a delete may point at a different condition afterwards,
because deleting renumbers everything after it.

POSIX only. ``fcntl`` is imported at module scope, exactly as ``web.py``
already does, so the package does not import on Windows. That is accepted
and explicit: a silently disabled advisory lock is worse than no lock.
"""

import copy
import fcntl
import json
import os
import re
import threading
from contextlib import contextmanager

from .engine import dependents as matter_dependents
from .engine import prerequisites as matter_prerequisites
from .engine import truth as condition_truth
from .extraction import slugify
from .graph_index import DependencyCycleError, GraphIndex
from .storage import load_state, resolve_state_path, save_state


MATTER_ID_PATTERN = re.compile(r"[a-z0-9_]+")
CONDITION_INDEX_PATTERN = re.compile(r"[0-9]+")
STATE_CYCLE_MESSAGE = "state dependency graph contains a cycle"

UNSET = object()


class RuleError(ValueError):
    """A rejected state mutation, carrying a semantic code.

    The code is deliberately not an ``HTTPStatus``: HTTP is ``web.py``'s
    concern. ``RuleError`` subclasses ``ValueError`` so ``parser.error``
    funnelling keeps working and nothing catching ``ValueError`` regresses.

    ``cycle`` is empty for every code except ``state_cycle``, where it
    carries the offending cycle as structured data so a caller does not have
    to parse it back out of the message. It is a keyword with a default
    because every existing raise site is positional ``(message, code)``.
    """

    def __init__(self, message, code="invalid", cycle=()):
        super().__init__(message)
        self.code = code
        self.cycle = tuple(cycle)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


class StateMutationLocks:
    def __init__(self):
        self._locks = {}
        self._guard = threading.Lock()

    @contextmanager
    def lock(self, state_path):
        lock_key = str(resolve_state_path(state_path))
        with self._guard:
            lock = self._locks.setdefault(lock_key, threading.RLock())
        with lock:
            yield


state_mutation_locks = StateMutationLocks()


def state_lock_path(state_path):
    """Return the sidecar lock path for ``state_path``. Pure, no I/O.

    The state file itself cannot be the lock target: ``save_state`` writes
    via ``mkstemp`` + ``os.replace``, which swaps the inode. A writer holding
    ``flock`` on the old inode releases nothing when the path starts pointing
    at a new one, so mutual exclusion would silently not exist. The sidecar's
    inode is stable because nothing ever replaces or unlinks it.
    """

    path = resolve_state_path(state_path)
    return path.parent / f".{path.name}.lock"


@contextmanager
def state_lock(state_path):
    """Hold the process-local lock and then the advisory file lock.

    Order matters and is not negotiable: process-local ``RLock`` first, then
    ``flock``. ``flock`` is owned by the open file description, so two
    threads in one process holding two descriptors do exclude each other.
    Taking the file lock outside the ``RLock`` makes same-process writers
    collide instead of serialising.

    The file lock is not reentrant across two descriptors, therefore exactly
    one :func:`state_transaction` per command, never nested. Compose by
    mutating one draft, never by calling another rules op inside a
    transaction.

    Non-blocking, single attempt, no retry, no timeout. Cleanup is automatic:
    the kernel drops the lock when the descriptor closes or the process dies,
    including on SIGKILL. The lock file stays empty and is never unlinked
    (unlink-while-locked is itself a race).
    """

    lock_path = state_lock_path(state_path)
    with state_mutation_locks.lock(state_path):
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as error:
            raise RuleError(
                f"cannot lock state directory: {lock_path.parent}: {error.strerror}"
            ) from error
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuleError(
                    f"state file is locked by another matters process: "
                    f"{lock_path} (is 'matters web' running?)",
                    "locked",
                ) from error
            yield lock_path
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------


class StateDraft:
    """The mutable state a :func:`state_transaction` hands to its caller.

    Callers mutate ``matters``, ``conditions`` and ``dependencies`` in place.
    ``index`` is the :class:`GraphIndex` built from the state as loaded, or
    ``None`` when the transaction was opened with ``require_acyclic=False``
    -- in which case the caller must not read it, because no index of a
    cyclic graph exists to hand over. See :func:`state_transaction`.
    """

    def __init__(self, path, matters, conditions, dependencies):
        self.path = path
        self.matters = matters
        self.conditions = conditions
        self.dependencies = dependencies
        self.index = None

    def as_tuple(self):
        return self.matters, self.conditions, self.dependencies


def load_state_or_rule_error(path, load=None):
    """Load state, translating every failure mode into a ``RuleError``."""

    loader = load or load_state
    try:
        return loader(path)
    except json.JSONDecodeError as error:
        raise RuleError(f"state file is not valid JSON: {path}: {error.msg}") from error
    except OSError as error:
        raise RuleError(
            f"state file is not readable: {path}: {error.strerror}"
        ) from error
    except (KeyError, TypeError, ValueError) as error:
        # Valid JSON, invalid graph. The catch set matches the one
        # ``web.validate_switch_state_path`` already uses against the same
        # ``load_state`` call, and for the same reason: a structurally wrong
        # file reaches the engine, which raises whatever its own traversal
        # raises. A scalar ``conditions[<matter>]`` value surfaces as
        # ``TypeError`` ('int' object is not iterable), a missing top-level
        # key as ``KeyError``; neither is a caller bug, both are bad input.
        raise RuleError(
            f"state file is not a valid matters graph: {path}: {error}"
        ) from error


@contextmanager
def state_transaction(
    state_path, *, require_exists=False, require_acyclic=True, load=None
):
    """Own one whole read-modify-write against ``state_path``.

    Saves on clean exit **only if the draft differs from the loaded
    snapshot**. That is load-bearing, not an optimisation: ``save_state``
    rewrites the file in canonical form, so an unconditional save would
    change the bytes of a file that a no-op command touched.

    Reads take no lock at all. ``os.replace`` makes every write atomic, so a
    reader always sees a complete file; locking reads would add contention
    for nothing and would create a lock file for read-only commands.

    ``require_acyclic`` decides two things at once, and they are the same
    decision: whether a state file that already contains a cycle is refused
    before the caller is yielded to, and whether ``draft.index`` exists.

    ================  ======================================================
    Value             Contract
    ================  ======================================================
    ``True``          Default. A pre-existing cycle raises ``state_cycle``
                      naming the cycle, before the caller sees the draft and
                      before anything is written. ``draft.index`` is the
                      index of the state **as loaded** -- valid to read, and
                      stale the moment the caller mutates the draft.
    ``False``         Nothing is refused for shape, and ``draft.index`` is
                      ``None``, never a partial or stale index, so a caller
                      that opts out must not read it. Only a mutation that
                      cannot create a cycle may opt out: ``create`` (AC-21)
                      and the two edge-removal verbs, which are how a cyclic
                      file gets repaired.
    ================  ======================================================
    """

    path = resolve_state_path(state_path)
    if require_exists and not path.exists():
        raise RuleError(f"state file does not exist: {path}", "not_found")

    with state_lock(path):
        if require_exists and not path.exists():
            raise RuleError(f"state file does not exist: {path}", "not_found")

        matters, conditions, dependencies = load_state_or_rule_error(path, load)
        draft = StateDraft(path, matters, conditions, dependencies)
        if require_acyclic:
            draft.index = require_acyclic_index(matters, conditions, dependencies)
        snapshot = copy.deepcopy((matters, conditions, dependencies))

        yield draft

        if draft.as_tuple() == snapshot:
            return
        try:
            save_state(
                draft.matters, draft.conditions, draft.dependencies, path=path
            )
        except ValueError as error:
            raise RuleError(str(error)) from error


# ---------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------


def require_matter_id_syntax(matter_id):
    if not MATTER_ID_PATTERN.fullmatch(matter_id):
        raise RuleError(
            "matter id must contain lowercase letters, numbers, and underscores only"
        )
    return matter_id


def normalized_matter_id(payload):
    raw_id = str(payload.get("id") or "").strip()
    title = str(payload.get("title") or payload.get("name") or "").strip()
    matter_id = raw_id or slugify(title)
    if not matter_id:
        raise RuleError("matter id or title is required")
    return require_matter_id_syntax(matter_id)


def normalize_condition(condition, index):
    """Normalise one condition payload. Never raises.

    ``index`` is **1-based** and is used only to build the fallback label.
    """

    if isinstance(condition, str):
        label = condition.strip()
        truth_value = False
    else:
        label = str(condition.get("label") or "").strip()
        truth_value = condition_truth(condition.get("truth", False))
    if not label:
        label = f"Unlabeled condition {index}"
    return {"label": label, "truth": truth_value}


def require_condition_index(payload, conditions):
    """Validate the **0-based** ``index`` field of a web API payload."""

    try:
        index = int(payload["index"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuleError("valid condition index is required") from error
    if index < 0 or index >= len(conditions):
        raise RuleError("condition index is out of range", "not_found")
    return index


def dependency_endpoints(payload, matters):
    source = str(payload.get("source") or payload.get("prerequisite") or "").strip()
    target = str(payload.get("target") or payload.get("dependent") or "").strip()
    if not source or not target:
        raise RuleError("dependency source and target are required")
    if source not in matters:
        raise RuleError(f"unknown dependency source: {source}", "not_found")
    if target not in matters:
        raise RuleError(f"unknown dependency target: {target}", "not_found")
    return source, target


def require_matter(matter_id, matters):
    if not matter_id:
        raise RuleError("matter id is required")
    if matter_id not in matters:
        raise RuleError(f"unknown matter: {matter_id}", "not_found")


# ---------------------------------------------------------------------------
# Cycle rules -- one implementation (GraphIndex), no fourth copy
# ---------------------------------------------------------------------------


def format_cycle(cycle):
    """Render one cycle as ``a -> b -> c -> a``, closing the loop.

    The only renderer of a cycle in the codebase. ASCII ``->`` on purpose:
    this string reaches a terminal under an arbitrary locale, where a unicode
    arrow can come out as mojibake or raise on encode, and it matches the
    arrow ``create`` already prints for a dependency.

    A self-loop renders as ``a -> a``, because :attr:`DependencyCycleError`
    stores each id once and this is where the closing edge is made visible.
    """

    if not cycle:
        return ""
    cycle = tuple(cycle)
    return " -> ".join(cycle + (cycle[0],))


def state_cycle_message(cycle):
    """Extend the pinned cycle sentence with the cycle itself.

    :data:`STATE_CYCLE_MESSAGE` stays intact and leading. It is a wire
    contract -- the web API answers it verbatim in the body of a 422 for a
    graph read -- so naming the cycle may only ever append to it.
    """

    named = format_cycle(cycle)
    return f"{STATE_CYCLE_MESSAGE}: {named}" if named else STATE_CYCLE_MESSAGE


def build_index(matters, conditions, dependencies):
    """Build a graph index, refusing a cyclic graph with the pinned message.

    The message is the bare sentence, and the cycle rides along as structured
    data on the error. Rendering it into the text is
    :func:`require_acyclic_index`'s job, on the refusal-to-write path where
    a person needs to know which edge to remove; a graph *read* keeps
    answering the stable sentence it has always answered.
    """

    try:
        return GraphIndex(matters, conditions, dependencies)
    except DependencyCycleError as error:
        raise RuleError(
            STATE_CYCLE_MESSAGE, "state_cycle", cycle=error.cycle
        ) from error


def require_acyclic_index(matters, conditions, dependencies):
    """Build the index a write needs, naming the cycle when it refuses.

    Goes through :func:`build_index` by name, so a caller that replaces that
    rule replaces it here too (AC-13), and re-raises anything it did not
    recognise untouched.
    """

    try:
        return build_index(matters, conditions, dependencies)
    except RuleError as error:
        if error.code != "state_cycle" or not error.cycle:
            raise
        raise RuleError(
            state_cycle_message(error.cycle), "state_cycle", cycle=error.cycle
        ) from error


def has_cycle(matters, conditions, dependencies):
    try:
        GraphIndex(matters, conditions, dependencies)
    except DependencyCycleError:
        return True
    return False


def would_create_cycle(matters, conditions, dependencies, edge):
    return has_cycle(matters, conditions, set(dependencies) | {edge})


def unblocked_matters(before, after):
    """Matters that were blocked before a mutation and are not blocked after.

    Not just direct dependents: the visible consequence of a mutation is the
    whole set that stopped being blocked. Matters that no longer exist after
    the mutation are excluded rather than raising.
    """

    return sorted(
        matter
        for matter in before.blocked
        if matter in after.universe or after.resolved.get(matter, False)
    )


def format_matter_list(matter_ids):
    return ", ".join(matter_ids) if matter_ids else "none"


def format_dependency_count(count):
    """Render a dependency count with a noun that agrees with it.

    Lives here rather than in a caller because both surfaces need it: the
    refusal message below is built in this module, and the CLI prints the
    same phrase on success.
    """

    if not count:
        return "no dependencies"
    return f"{count} dependency" if count == 1 else f"{count} dependencies"


# ---------------------------------------------------------------------------
# Condition editing primitives
# ---------------------------------------------------------------------------


def append_condition(current, condition):
    """Return ``current`` plus one normalised condition.

    The fallback-label index is ``len(current) + 1`` -- 1-based (AC-14).
    """

    return current + [normalize_condition(condition, len(current) + 1)]


def apply_condition_patch(current, index, *, label=UNSET, truth=UNSET):
    """Return the condition at 0-based ``index`` with the given fields set.

    Unset fields are preserved, which is what keeps ``edit-condition`` from
    clearing a truth value and ``mark`` from clearing a label.
    """

    updated = dict(current[index])
    if label is not UNSET:
        updated["label"] = str(label).strip()
    if truth is not UNSET:
        updated["truth"] = condition_truth(truth)
    return normalize_condition(updated, index + 1)


def delete_condition_at(current, index):
    """Delete the condition at 0-based ``index`` in place, returning it.

    Renumbering of later conditions is inherent to list deletion (E-11).
    """

    removed = current[index]
    del current[index]
    return removed


def resolve_condition_ref(matter_id, current, ref):
    """Resolve a CLI ``condition-ref`` to a **0-based** list position.

    This is the single owner of condition addressing and the only 1-based to
    0-based conversion in the codebase.

    An argument that is entirely ASCII digits is an index, so ``"0"`` is
    always out of range and never means the first condition. Anything else
    (including ``"-1"``, ``"1.0"`` and ``"1 "``) is a label, matched exactly
    and case-sensitively; two conditions sharing a label raise rather than
    silently resolving to the first.
    """

    ref_text = str(ref)
    if CONDITION_INDEX_PATTERN.fullmatch(ref_text):
        number = int(ref_text)
        if number < 1 or number > len(current):
            raise RuleError(
                f"matter {matter_id} has no condition {number} "
                f"(it has {len(current)})",
                "not_found",
            )
        return number - 1

    needle = ref_text.strip()
    positions = [
        position
        for position, condition in enumerate(current)
        if condition.get("label") == needle
    ]
    if not positions:
        raise RuleError(
            f"matter {matter_id} has no condition matching {ref_text!r}; "
            f"run `matters show {matter_id}` to see condition numbers",
            "not_found",
        )
    if len(positions) > 1:
        listed = ", ".join(str(position + 1) for position in positions)
        raise RuleError(
            f"matter {matter_id} has {len(positions)} conditions labelled "
            f"{ref_text!r} (positions {listed}); address it by number"
        )
    return positions[0]


# ---------------------------------------------------------------------------
# Create-expression parsing (moved from cli.py, byte-identical in behavior)
# ---------------------------------------------------------------------------


def create_matters_from_expression(expression, matters, conditions, dependencies):
    parsed_matters = parse_create_expression(expression)
    ids = [matter["id"] for matter in parsed_matters]
    duplicate_ids = sorted({matter_id for matter_id in ids if ids.count(matter_id) > 1})
    if duplicate_ids:
        raise ValueError("duplicate matter ids in expression: " + ", ".join(duplicate_ids))

    existing_ids = sorted(set(ids) & matters)
    if existing_ids:
        raise ValueError("matter already exists: " + ", ".join(existing_ids))

    for parsed_matter in parsed_matters:
        matter_id = parsed_matter["id"]
        matters.add(matter_id)
        conditions[matter_id] = [
            {"label": parsed_matter["condition"], "truth": False}
        ]

    for prerequisite, dependent in zip(parsed_matters[1:], parsed_matters):
        dependencies.add((prerequisite["id"], dependent["id"]))

    return parsed_matters


def parse_create_expression(expression):
    segments = [segment.strip() for segment in expression.split(">")]
    segments = [segment for segment in segments if segment]
    if not segments:
        raise ValueError("matter expression is empty")

    return [parse_create_segment(segment) for segment in segments]


def parse_create_segment(segment):
    name = segment
    condition = None

    if segment.endswith(")"):
        start = segment.rfind("(")
        if start > 0:
            name = segment[:start].strip()
            condition = segment[start + 1 : -1].strip()

    if not name:
        raise ValueError("matter name cannot be empty")

    if not condition:
        condition = f"Resolved: {name}"

    return {"id": slugify(name), "name": name, "condition": condition}


# ---------------------------------------------------------------------------
# Web-shaped operations
#
# These four take the web API payload shapes and return None; the web
# wrappers call graph_payload themselves.
# ---------------------------------------------------------------------------


def create_matter(state_path, payload, *, load=None):
    with state_transaction(state_path, load=load) as draft:
        matter_id = normalized_matter_id(payload)
        if matter_id in draft.matters:
            raise RuleError(f"matter already exists: {matter_id}", "conflict")

        condition_payloads = payload.get("conditions") or [
            {"label": f"Resolved: {matter_id.replace('_', ' ')}", "truth": False}
        ]
        normalized_conditions = [
            normalize_condition(condition, index)
            for index, condition in enumerate(condition_payloads, start=1)
        ]

        draft.matters.add(matter_id)
        draft.conditions[matter_id] = normalized_conditions


def update_conditions(state_path, matter_id, payload, *, load=None):
    with state_transaction(state_path, load=load) as draft:
        if matter_id not in draft.matters:
            raise RuleError(f"unknown matter: {matter_id}", "not_found")

        action = payload.get("action")
        current = list(draft.conditions.get(matter_id, []))

        if "conditions" in payload:
            current = [
                normalize_condition(condition, index)
                for index, condition in enumerate(payload["conditions"], start=1)
            ]
        elif action == "toggle":
            index = require_condition_index(payload, current)
            current[index]["truth"] = not condition_truth(current[index])
        elif action == "delete":
            index = require_condition_index(payload, current)
            delete_condition_at(current, index)
        else:
            index = payload.get("index")
            if index is None:
                current = append_condition(current, payload)
            else:
                index = require_condition_index(payload, current)
                patch = {}
                if "label" in payload:
                    patch["label"] = payload["label"]
                if "truth" in payload:
                    patch["truth"] = payload["truth"]
                current[index] = apply_condition_patch(current, index, **patch)

        draft.conditions[matter_id] = current


def add_dependency(state_path, payload, *, load=None):
    with state_transaction(state_path, load=load) as draft:
        source, target = dependency_endpoints(payload, draft.matters)
        if would_create_cycle(
            draft.matters, draft.conditions, draft.dependencies, (source, target)
        ):
            raise RuleError("dependency would create a cycle")
        draft.dependencies.add((source, target))


def remove_dependency(state_path, payload, *, load=None):
    """Remove one edge, and do so even on an already cyclic file.

    ``require_acyclic=False`` for the same reason as :func:`unlink`, which is
    this operation's CLI twin: removing an edge cannot create a cycle, and
    refusing here would leave the web with no way at all to repair a state
    file that acquired one. Touches no derived value, so ``draft.index``
    being ``None`` costs it nothing.
    """

    with state_transaction(state_path, require_acyclic=False, load=load) as draft:
        source, target = dependency_endpoints(payload, draft.matters)
        draft.dependencies.discard((source, target))


# ---------------------------------------------------------------------------
# CLI-shaped operations
#
# These return a result dict so cli.py can print its one-line confirmation
# without recomputing anything.
# ---------------------------------------------------------------------------


def set_condition_truth(state_path, matter_id, ref, truth_value):
    """Set one condition's truth. ``truth_value`` must be a real ``bool``.

    Deliberately *not* funnelled through :func:`engine.truth`, which
    truthy-coerces: ``truth("false")`` is ``True``, so a caller handing over
    an unconverted CLI word would silently write the opposite of what the
    user asked for. Callers must convert first; a non-bool is a caller bug
    and is rejected before any file is opened or locked.

    ``engine.truth`` itself is unchanged: the web path legitimately receives
    JSON booleans through :func:`normalize_condition` and
    :func:`apply_condition_patch`, and its coercion must keep working there.
    """

    if not isinstance(truth_value, bool):
        raise RuleError(
            "condition truth must be true or false, not "
            f"{type(truth_value).__name__}: {truth_value!r}"
        )

    with state_transaction(state_path, require_exists=True) as draft:
        require_matter(matter_id, draft.matters)
        current = list(draft.conditions.get(matter_id, []))
        position = resolve_condition_ref(matter_id, current, ref)
        desired = truth_value
        changed = condition_truth(current[position]) != desired
        current[position] = apply_condition_patch(current, position, truth=desired)
        draft.conditions[matter_id] = current
        return {
            "matter": matter_id,
            "position": position + 1,
            "label": current[position]["label"],
            "truth": desired,
            "changed": changed,
        }


def add_condition(state_path, matter_id, label):
    with state_transaction(state_path, require_exists=True) as draft:
        require_matter(matter_id, draft.matters)
        current = append_condition(
            list(draft.conditions.get(matter_id, [])), label
        )
        draft.conditions[matter_id] = current
        return {
            "matter": matter_id,
            "position": len(current),
            "label": current[-1]["label"],
            "truth": current[-1]["truth"],
            "changed": True,
        }


def edit_condition_label(state_path, matter_id, ref, label):
    with state_transaction(state_path, require_exists=True) as draft:
        require_matter(matter_id, draft.matters)
        current = list(draft.conditions.get(matter_id, []))
        position = resolve_condition_ref(matter_id, current, ref)
        previous_label = current[position]["label"]
        current[position] = apply_condition_patch(current, position, label=label)
        draft.conditions[matter_id] = current
        return {
            "matter": matter_id,
            "position": position + 1,
            "label": current[position]["label"],
            "previous_label": previous_label,
            "truth": current[position]["truth"],
            "changed": current[position]["label"] != previous_label,
        }


def delete_condition(state_path, matter_id, ref, *, confirmed=False):
    with state_transaction(state_path, require_exists=True) as draft:
        require_matter(matter_id, draft.matters)
        current = list(draft.conditions.get(matter_id, []))
        position = resolve_condition_ref(matter_id, current, ref)
        removed = delete_condition_at(current, position)
        draft.conditions[matter_id] = current

        emptied = not current
        after = build_index(draft.matters, draft.conditions, draft.dependencies)
        unblocked = unblocked_matters(draft.index, after)

        if emptied and not confirmed:
            raise RuleError(
                f"deleting the last condition of {matter_id} will make it "
                f"resolved, unblocking: {format_matter_list(unblocked)} "
                f"— rerun with --yes",
                "conflict",
            )

        return {
            "matter": matter_id,
            "position": position + 1,
            "label": removed["label"],
            "truth": removed["truth"],
            "emptied": emptied,
            "unblocked": unblocked,
            "changed": True,
        }


def link(state_path, dependent, prerequisite):
    """Add the edge for "``dependent`` needs ``prerequisite``".

    Stored as ``(prerequisite, dependent)`` -- source first -- matching
    engine semantics and the order ``create`` already writes.
    """

    with state_transaction(state_path, require_exists=True) as draft:
        source, target = dependency_endpoints(
            {"source": prerequisite, "target": dependent}, draft.matters
        )
        changed = (source, target) not in draft.dependencies
        if changed:
            if would_create_cycle(
                draft.matters, draft.conditions, draft.dependencies, (source, target)
            ):
                raise RuleError(
                    f"{target} cannot require {source}: "
                    f"the dependency would create a cycle"
                )
            draft.dependencies.add((source, target))
        return {
            "dependent": target,
            "prerequisite": source,
            "changed": changed,
        }


def unlink(state_path, dependent, prerequisite):
    """Remove the edge for "``dependent`` needs ``prerequisite``".

    The one write verb that runs against a state file which already contains
    a cycle. Removing an edge cannot create one, so the guard has nothing to
    protect here, and it is the only repair a person has: with it refused,
    a file that acquired a cycle -- by hand, by an older tool, by an agent
    writing JSON directly -- could be fixed only by editing JSON by hand.
    Every other write verb still refuses.

    A repair is not guaranteed to be finished by one call: a graph can hold
    several cycles, and the next write refusal names the next one.

    ``draft.index`` is ``None`` on this path. This function reads no derived
    value, and must not start to.
    """

    with state_transaction(
        state_path, require_exists=True, require_acyclic=False
    ) as draft:
        source, target = dependency_endpoints(
            {"source": prerequisite, "target": dependent}, draft.matters
        )
        changed = (source, target) in draft.dependencies
        draft.dependencies.discard((source, target))
        return {
            "dependent": target,
            "prerequisite": source,
            "changed": changed,
        }


def delete_matter(state_path, matter_id, *, cascade=False, confirmed=False):
    with state_transaction(state_path, require_exists=True) as draft:
        require_matter(matter_id, draft.matters)
        blocking_dependents = sorted(draft.index.dependents[matter_id])
        if blocking_dependents and not cascade:
            raise RuleError(
                f"matter {matter_id} is required by: "
                f"{format_matter_list(blocking_dependents)} — rerun with "
                f"--cascade to delete it and those dependencies, or unlink "
                f"them first",
                "conflict",
            )

        removed_edges = sorted(
            edge for edge in draft.dependencies if matter_id in edge
        )
        draft.matters.discard(matter_id)
        draft.conditions.pop(matter_id, None)
        for edge in removed_edges:
            draft.dependencies.discard(edge)

        after = build_index(draft.matters, draft.conditions, draft.dependencies)
        unblocked = unblocked_matters(draft.index, after)

        if not confirmed:
            raise RuleError(
                f"deleting matter {matter_id} removes "
                f"{format_dependency_count(len(removed_edges))}; "
                f"unblocked: {format_matter_list(unblocked)} "
                f"— rerun with --yes",
                "conflict",
            )

        return {
            "matter": matter_id,
            "removed_edges": [list(edge) for edge in removed_edges],
            "unblocked": unblocked,
            "changed": True,
        }


# ---------------------------------------------------------------------------
# Read helpers -- no lock, no index, no writes
# ---------------------------------------------------------------------------


def describe_matter(state_path, matter_id):
    """Stored facts about one matter. No derived resolved/actionable status.

    ``index`` is **0-based**, matching the stored list position and the web
    API payload field. The CLI ``condition-ref`` and the ``show`` text output
    are 1-based; that split is deliberate (machine versus human) and must not
    be papered over.
    """

    path = resolve_state_path(state_path)
    matters, conditions, dependencies = load_state_or_rule_error(path)
    require_matter(matter_id, matters)
    return {
        "id": matter_id,
        "conditions": [
            {
                "index": position,
                "label": condition["label"],
                "truth": condition_truth(condition),
            }
            for position, condition in enumerate(conditions.get(matter_id, []))
        ],
        "prerequisites": sorted(matter_prerequisites(matter_id, dependencies)),
        "dependents": sorted(matter_dependents(matter_id, dependencies)),
    }


def list_matters(state_path):
    path = resolve_state_path(state_path)
    matters, _conditions, _dependencies = load_state_or_rule_error(path)
    return sorted(matters)
