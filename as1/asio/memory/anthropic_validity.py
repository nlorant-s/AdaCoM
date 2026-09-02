# -*- coding: utf-8 -*-
"""Round-trip validation of a managed context against Anthropic message rules.

The manager rewrites ``_chat_history`` between agent steps. The repair pass in
``perform_modifications`` fixes the common damage (orphan tool_use / tool_result
pairs), but it is not a proof: this module is the check that runs *after* the
edits and says whether the history the agent is about to be sent is one the
Messages API will accept.

Rules checked (see docs/anthropic_api_notes.md for the sourced statements):

* ``FIRST_NOT_USER``      – the first message must be a user message.
* ``SYSTEM_IN_MESSAGES``  – ``system`` is a top-level request parameter; there
  is no system role inside ``messages``.
* ``ORPHAN_TOOL_USE``     – a ``tool_use`` with no matching ``tool_result``.
* ``ORPHAN_TOOL_RESULT``  – a ``tool_result`` with no matching ``tool_use``.
* ``TOOL_RESULT_NOT_ADJACENT`` – the ``tool_result`` exists but is not in the
  message immediately after the one holding its ``tool_use``.
* ``EMPTY_MESSAGE``       – a message with no content at all.
* ``THINKING_MISSING`` / ``THINKING_MODIFIED`` / ``THINKING_NOT_FIRST`` – the
  in-flight cycle's thinking block must round-trip verbatim, signature
  included, and (in manual thinking mode) start the assistant message.

The thinking checks compare against a snapshot taken before the manager ran
(``capture_thinking_snapshot``), which is what makes "modified" detectable at
all: a rewritten signature is still a well-formed block.

Duck-typed on ``Msg`` (needs ``.role`` and ``.content``) so it can be unit
tested without agentscope. Blocks are the agentscope shapes — ``tool_use``
carries ``id``, ``tool_result`` carries ``id``/``output`` — and the Anthropic
wire spellings (``tool_use_id``, ``content``) are accepted too.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

ERROR = "error"
WARNING = "warning"

FIRST_NOT_USER = "FIRST_NOT_USER"
SYSTEM_IN_MESSAGES = "SYSTEM_IN_MESSAGES"
ORPHAN_TOOL_USE = "ORPHAN_TOOL_USE"
ORPHAN_TOOL_RESULT = "ORPHAN_TOOL_RESULT"
TOOL_RESULT_NOT_ADJACENT = "TOOL_RESULT_NOT_ADJACENT"
EMPTY_MESSAGE = "EMPTY_MESSAGE"
THINKING_MISSING = "THINKING_MISSING"
THINKING_MODIFIED = "THINKING_MODIFIED"
THINKING_NOT_FIRST = "THINKING_NOT_FIRST"

THINKING_TYPES = ("thinking", "redacted_thinking")


@dataclass
class Violation:
    code: str
    index: Optional[int]
    detail: str
    severity: str = ERROR

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Block helpers
# ---------------------------------------------------------------------------
def _blocks(msg: Any) -> List[dict]:
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _role(msg: Any) -> Optional[str]:
    return getattr(msg, "role", None)


def _is_empty(msg: Any) -> bool:
    content = getattr(msg, "content", None)
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        return len(content) == 0
    return False


def _call_id(block: dict) -> Optional[str]:
    # agentscope uses "id" on both block types; the wire format uses
    # "tool_use_id" on tool_result.
    return block.get("id") or block.get("tool_use_id")


def _thinking_blocks(msg: Any) -> List[dict]:
    return [b for b in _blocks(msg) if b.get("type") in THINKING_TYPES]


def _latest_assistant_index(chat_history: Sequence[Any]) -> Optional[int]:
    for i in range(len(chat_history) - 1, -1, -1):
        if _role(chat_history[i]) == "assistant":
            return i
    return None


# ---------------------------------------------------------------------------
# Thinking snapshot
# ---------------------------------------------------------------------------
def capture_thinking_snapshot(chat_history: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """Deep-copy the latest assistant message's thinking blocks.

    Returns ``None`` when there is nothing to protect (no assistant message, or
    it carries no thinking block). Take this *before* applying modifications;
    pass it to :func:`validate_anthropic_history` afterwards.
    """
    idx = _latest_assistant_index(chat_history)
    if idx is None:
        return None
    blocks = _thinking_blocks(chat_history[idx])
    if not blocks:
        return None
    return {
        "index": idx,
        "blocks": copy.deepcopy(blocks),
        "leads_message": _blocks(chat_history[idx])[0].get("type") in THINKING_TYPES,
    }


def _same_thinking(a: dict, b: dict) -> bool:
    """Blocks round-trip verbatim, so compare the fields the API reads."""
    if a.get("type") != b.get("type"):
        return False
    if a.get("signature") != b.get("signature"):
        return False
    return a.get("thinking", a.get("data")) == b.get("thinking", b.get("data"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_anthropic_history(
    chat_history: Sequence[Any],
    thinking_snapshot: Optional[Dict[str, Any]] = None,
    thinking_enabled: bool = True,
) -> List[Violation]:
    """Return every rule the history breaks; empty list means API-valid."""
    violations: List[Violation] = []
    n = len(chat_history)
    if n == 0:
        return violations

    if _role(chat_history[0]) != "user":
        violations.append(Violation(
            FIRST_NOT_USER, 0,
            f"first message has role {_role(chat_history[0])!r}, expected 'user'",
        ))

    for i, msg in enumerate(chat_history):
        if _role(msg) == "system":
            violations.append(Violation(
                SYSTEM_IN_MESSAGES, i,
                "system is a top-level request parameter, not a message role",
            ))
        if _is_empty(msg):
            violations.append(Violation(EMPTY_MESSAGE, i, "message has no content"))

    # tool_use <-> tool_result pairing, first-come-first-served by call id.
    uses: Dict[str, List[int]] = {}
    results: Dict[str, List[int]] = {}
    for i, msg in enumerate(chat_history):
        for b in _blocks(msg):
            cid = _call_id(b)
            if not cid:
                continue
            if b.get("type") == "tool_use":
                uses.setdefault(cid, []).append(i)
            elif b.get("type") == "tool_result":
                results.setdefault(cid, []).append(i)

    for cid, indices in uses.items():
        result_indices = list(results.get(cid, []))
        for use_idx in indices:
            following = [r for r in result_indices if r > use_idx]
            if not following:
                violations.append(Violation(
                    ORPHAN_TOOL_USE, use_idx,
                    f"tool_use {cid!r} has no following tool_result",
                ))
                continue
            result_idx = following[0]
            result_indices.remove(result_idx)
            if result_idx != use_idx + 1:
                violations.append(Violation(
                    TOOL_RESULT_NOT_ADJACENT, use_idx,
                    f"tool_result for {cid!r} is at index {result_idx}, "
                    f"not immediately after the tool_use at {use_idx}",
                ))

    for cid, indices in results.items():
        use_indices = list(uses.get(cid, []))
        for result_idx in indices:
            preceding = [u for u in use_indices if u < result_idx]
            if not preceding:
                violations.append(Violation(
                    ORPHAN_TOOL_RESULT, result_idx,
                    f"tool_result {cid!r} has no preceding tool_use",
                ))
                continue
            use_indices.remove(preceding[-1])

    if thinking_enabled and thinking_snapshot:
        violations.extend(_check_thinking(chat_history, thinking_snapshot))

    return violations


def _check_thinking(chat_history: Sequence[Any],
                    snapshot: Dict[str, Any]) -> List[Violation]:
    expected = snapshot.get("blocks") or []
    idx = _latest_assistant_index(chat_history)
    if idx is None:
        return [Violation(THINKING_MISSING, None,
                          "no assistant message left to carry the in-flight thinking block")]

    actual = _thinking_blocks(chat_history[idx])
    if not actual:
        return [Violation(
            THINKING_MISSING, idx,
            "latest assistant message lost its thinking block; the API requires "
            "the in-flight cycle's block to be passed back verbatim",
        )]

    out: List[Violation] = []
    if len(actual) != len(expected) or not all(
        _same_thinking(a, b) for a, b in zip(actual, expected)
    ):
        out.append(Violation(
            THINKING_MODIFIED, idx,
            f"thinking block(s) changed: {len(expected)} before, {len(actual)} after "
            "(text or signature differs)",
        ))

    blocks = _blocks(chat_history[idx])
    if snapshot.get("leads_message") and blocks and blocks[0].get("type") not in THINKING_TYPES:
        out.append(Violation(
            THINKING_NOT_FIRST, idx,
            "the assistant message no longer begins with its thinking block "
            "(required in manual thinking mode)",
        ))
    return out


def violations_to_json(violations: Sequence[Violation]) -> List[dict]:
    return [v.to_dict() for v in violations]


def format_violations(violations: Sequence[Violation]) -> str:
    return "; ".join(f"{v.code}@{v.index}: {v.detail}" for v in violations)


def has_errors(violations: Sequence[Violation]) -> bool:
    return any(v.severity == ERROR for v in violations)


class AnthropicValidityError(Exception):
    """Raised when a managed context would be rejected by the Messages API."""

    def __init__(self, violations: Sequence[Violation]):
        self.violations = list(violations)
        super().__init__(format_violations(violations))
