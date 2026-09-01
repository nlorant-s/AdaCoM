# -*- coding: utf-8 -*-
"""Context-lock and lineage utilities for AdaCoM with a thinking-enabled agent.

Why this exists
---------------
When the frozen agent runs with extended thinking, the Anthropic Messages API
requires the thinking block of the *in-flight* tool-use cycle (the most recent
assistant message that issued a tool_use) to be passed back verbatim, with its
signature. If the context manager rewrites or deletes that message the next
agent call 400s. Deleting the paired tool_result is equally fatal: the repair
pass in ``perform_modifications`` would then delete the orphaned tool_use
message — the one holding the thinking block.

Rules enforced here
-------------------
* ``LOCKED``       – the latest assistant message: no op may target it.
* ``REWRITE_ONLY`` – the tool_result message(s) paired with it: ops may rewrite
  ``new_content`` (the repo keeps it a paired tool_result) but may not delete
  it (empty ``new_content``) or merge it into a span with other messages.
* Everything older is unrestricted (older thinking blocks are not required by
  the API and may be stripped/rewritten).

Two enforcement points, both provided:
1. Prompt side – ``format_msgs_with_locks`` annotates the serialized context so
   the manager can see what it may not touch.
2. Apply side  – ``filter_modifications`` drops violating ops before
   ``perform_modifications`` and returns a violation list you can turn into a
   format penalty during RL.

Lineage
-------
The repo addresses messages by *positional index*, re-numbered every round, so
there is no stable identity to reconstruct a message's edit history. We stash
a stable uid in ``Msg.metadata["cm_uid"]`` and record, per manager step, which
uids each op consumed and which uid it produced. That is the serial-reproduction
chain for the drift analysis.

This module is duck-typed on ``Msg`` (needs ``.role``, ``.content``,
``.metadata``) so it can be unit-tested without agentscope's dependencies.
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

LOCKED = "locked"
REWRITE_ONLY = "rewrite_only"

# Appended to the manager prompt (after the Strategy Guidelines section).
LOCK_PROMPT_ADDENDUM = """
### Locked Messages
Some messages in the Agent Context carry a `"lock"` field:
- `"lock": "locked"` — you MUST NOT include this message's id in any modification. It is the agent's in-flight reasoning and tool call and cannot be altered.
- `"lock": "rewrite_only"` — you MAY rewrite this message's content (non-empty `new_content`, targeting exactly this one id) but MUST NOT delete it or merge it with other messages.
Modifications that violate a lock are discarded.
"""


# ----------------------------------------------------------------------------
# Block helpers (mirror memorymanager.is_tool_use / is_tool_result)
# ----------------------------------------------------------------------------
def _blocks(msg: Any) -> List[dict]:
    c = getattr(msg, "content", None)
    if isinstance(c, list):
        return [b for b in c if isinstance(b, dict)]
    return []


def _has_type(msg: Any, t: str) -> bool:
    return any(b.get("type") == t for b in _blocks(msg))


def _tool_use_ids(msg: Any) -> Set[str]:
    return {b.get("id") for b in _blocks(msg) if b.get("type") == "tool_use" and b.get("id")}


def _tool_result_ids(msg: Any) -> Set[str]:
    return {b.get("id") for b in _blocks(msg) if b.get("type") == "tool_result" and b.get("id")}


# ----------------------------------------------------------------------------
# Lock computation
# ----------------------------------------------------------------------------
def compute_locks(chat_history: Sequence[Any], thinking_enabled: bool = True) -> Dict[int, str]:
    """Return {index: LOCKED | REWRITE_ONLY} for the current history.

    If thinking is disabled the API imposes no constraint and we return {}.
    (You may still want to lock for a cleaner MDP; pass thinking_enabled=True
    to force it.)
    """
    if not thinking_enabled or not chat_history:
        return {}

    # Latest assistant message.
    last_a = None
    for i in range(len(chat_history) - 1, -1, -1):
        if getattr(chat_history[i], "role", None) == "assistant":
            last_a = i
            break
    if last_a is None:
        return {}

    # Only an in-flight tool cycle needs protection: the latest assistant
    # message must contain a tool_use (a final text answer ends the cycle, and
    # in this harness `finish` terminates the rollout anyway). We still lock
    # plain-text latest messages when they carry a thinking block, to be safe.
    if not (_has_type(chat_history[last_a], "tool_use") or _has_type(chat_history[last_a], "thinking")):
        return {}

    locks: Dict[int, str] = {last_a: LOCKED}
    call_ids = _tool_use_ids(chat_history[last_a])
    for j in range(last_a + 1, len(chat_history)):
        if getattr(chat_history[j], "role", None) == "user" and (_tool_result_ids(chat_history[j]) & call_ids):
            locks[j] = REWRITE_ONLY
    return locks


# ----------------------------------------------------------------------------
# Apply-side enforcement
# ----------------------------------------------------------------------------
@dataclass
class LockViolation:
    op_index: int
    ids: List[int]
    lock: str
    reason: str


def _normalize_ids(ids: Any) -> List[int]:
    if not isinstance(ids, list):
        ids = [ids]
    out = []
    for x in ids:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def filter_modifications(
    modifications: List[dict],
    chat_history: Sequence[Any],
    thinking_enabled: bool = True,
) -> Tuple[List[dict], List[LockViolation]]:
    """Drop ops that violate locks. Returns (kept_ops, violations).

    Does not mutate the input list.
    """
    locks = compute_locks(chat_history, thinking_enabled)
    if not locks:
        return list(modifications), []

    kept: List[dict] = []
    violations: List[LockViolation] = []
    for k, op in enumerate(modifications):
        ids = _normalize_ids(op.get("ids", []))
        new_content = op.get("new_content", "")
        is_delete = not (isinstance(new_content, str) and new_content.strip() != "") and not (
            isinstance(new_content, list) and len(new_content) > 0
        )
        violated: Optional[LockViolation] = None
        for i in ids:
            lock = locks.get(i)
            if lock == LOCKED:
                violated = LockViolation(k, ids, lock, "targets locked latest assistant message")
                break
            if lock == REWRITE_ONLY:
                if is_delete:
                    violated = LockViolation(k, ids, lock, "deletes rewrite-only tool_result")
                    break
                if len(ids) != 1:
                    violated = LockViolation(k, ids, lock, "merges rewrite-only tool_result with other messages")
                    break
        if violated is None:
            kept.append(op)
        else:
            violations.append(violated)
    return kept, violations


# ----------------------------------------------------------------------------
# Prompt-side annotation
# ----------------------------------------------------------------------------
def format_msgs_with_locks(
    chat_history: Sequence[Any],
    thinking_enabled: bool = True,
    strip_content_ids: bool = True,
    strip_thinking: bool = True,
) -> str:
    """Serialize history like ``utils.format_msgs`` but add a ``lock`` field
    where applicable and (by default) drop thinking blocks from the manager's
    view — they are large and not the manager's business.
    """
    locks = compute_locks(chat_history, thinking_enabled)
    results = []
    for idx, msg in enumerate(chat_history):
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if content is None:
            content = ""
        if isinstance(content, str):
            unit: Dict[str, Any] = {"role": role, "content": content, "id": idx}
        elif isinstance(content, list):
            unit = {"role": role, "content": [], "id": idx}
            for c in content:
                if isinstance(c, dict):
                    if strip_thinking and c.get("type") in ("thinking", "redacted_thinking"):
                        continue
                    if strip_content_ids and c.get("type") in ("tool_use", "tool_result") and "id" in c:
                        c = {k: v for k, v in c.items() if k != "id"}
                unit["content"].append(c)
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
        if idx in locks:
            unit["lock"] = locks[idx]
        results.append(unit)
    return json.dumps(results, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------------
def ensure_uids(chat_history: Sequence[Any]) -> None:
    """Give every message a stable ``metadata['cm_uid']`` (idempotent)."""
    for m in chat_history:
        md = getattr(m, "metadata", None)
        if md is None:
            try:
                m.metadata = {}
                md = m.metadata
            except AttributeError:
                continue
        if "cm_uid" not in md:
            md["cm_uid"] = uuid.uuid4().hex[:12]


@dataclass
class LineageRecord:
    step: int
    op_index: int
    consumed_uids: List[str]
    consumed_roles: List[str]
    produced_uid: Optional[str]      # None => deletion
    produced_role: Optional[str]
    justification: str = ""
    new_content_chars: int = 0
    consumed_chars: int = 0


def _msg_chars(msg: Any) -> int:
    c = getattr(msg, "content", "")
    try:
        return len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False))
    except Exception:
        return 0


def plan_lineage(
    modifications: List[dict],
    chat_history_before: Sequence[Any],
    step: int,
) -> Tuple[List[LineageRecord], Dict[int, str]]:
    """Compute lineage records for the ops about to be applied.

    Returns (records, produced_uid_by_first_index). The second value lets you
    stamp the produced uid onto whatever message ends up at ``ids[0]`` after
    ``perform_modifications`` runs (the repo replaces in place at ids[0] and
    deletes ids[1:]).
    """
    ensure_uids(chat_history_before)
    records: List[LineageRecord] = []
    produced: Dict[int, str] = {}
    n = len(chat_history_before)
    for k, op in enumerate(modifications):
        ids = [i for i in _normalize_ids(op.get("ids", [])) if 0 <= i < n]
        if not ids:
            continue
        new_content = op.get("new_content", "")
        is_delete = not (isinstance(new_content, str) and new_content.strip() != "") and not (
            isinstance(new_content, list) and len(new_content) > 0
        )
        consumed = [chat_history_before[i] for i in ids]
        produced_uid = None if is_delete else uuid.uuid4().hex[:12]
        if produced_uid:
            produced[ids[0]] = produced_uid
        records.append(
            LineageRecord(
                step=step,
                op_index=k,
                consumed_uids=[m.metadata.get("cm_uid") for m in consumed],
                consumed_roles=[getattr(m, "role", None) for m in consumed],
                produced_uid=produced_uid,
                produced_role=None if is_delete else op.get("role"),
                justification=str(op.get("justification", ""))[:500],
                new_content_chars=0 if is_delete else (len(new_content) if isinstance(new_content, str) else len(json.dumps(new_content))),
                consumed_chars=sum(_msg_chars(m) for m in consumed),
            )
        )
    return records, produced


def stamp_produced_uids(chat_history_after: Sequence[Any], chat_history_before: Sequence[Any], produced: Dict[int, str]) -> None:
    """After perform_modifications, the message that was at before-index ids[0]
    survives (rewritten) at a shifted position. Match by object identity where
    possible; fall back to metadata absence.
    """
    before_objs = {id(m): i for i, m in enumerate(chat_history_before)}
    for m in chat_history_after:
        md = getattr(m, "metadata", None)
        if md is None:
            try:
                m.metadata = {}
                md = m.metadata
            except AttributeError:
                continue
        # Freshly constructed replacement Msg objects have no cm_uid yet.
        if "cm_uid" not in md:
            # Find which before-index it replaced: the repo rebuilds Msg at ids[0],
            # so pick the first unassigned produced uid in order.
            for bi in sorted(produced):
                uid = produced[bi]
                if uid and not any(getattr(x, "metadata", {}).get("cm_uid") == uid for x in chat_history_after):
                    md["cm_uid"] = uid
                    break
            else:
                md["cm_uid"] = uuid.uuid4().hex[:12]


def snapshot(chat_history: Sequence[Any]) -> List[dict]:
    """JSON-safe deep snapshot of a history (for pre/post logging)."""
    out = []
    for m in chat_history:
        out.append(
            {
                "cm_uid": (getattr(m, "metadata", None) or {}).get("cm_uid"),
                "role": getattr(m, "role", None),
                "content": copy.deepcopy(getattr(m, "content", None)),
            }
        )
    return out


def records_to_json(records: List[LineageRecord]) -> List[dict]:
    return [asdict(r) for r in records]
