"""Round-trip validation of the managed context (TASK 3).

Synthetic histories, including the lock edge cases from test_context_lock.py:
the manager rewriting the locked latest assistant message, and deleting the
rewrite-only tool_result that pairs with it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from asio.memory.anthropic_validity import (  # noqa: E402
    EMPTY_MESSAGE, FIRST_NOT_USER, ORPHAN_TOOL_RESULT, ORPHAN_TOOL_USE,
    SYSTEM_IN_MESSAGES, THINKING_MISSING, THINKING_MODIFIED, THINKING_NOT_FIRST,
    TOOL_RESULT_NOT_ADJACENT, AnthropicValidityError, capture_thinking_snapshot,
    format_violations, has_errors, validate_anthropic_history, violations_to_json,
)


class Msg:
    def __init__(self, role, content, metadata=None):
        self.role, self.content, self.metadata = role, content, metadata


def q():
    return Msg("user", "Find the person who ...")


def note():
    return Msg("user", [{"type": "text", "text": "Working note: constraints A,B"}])


def a(cid, thinking=True, sig="sig"):
    blocks = ([{"type": "thinking", "thinking": "let me search", "signature": sig}]
              if thinking else [])
    return Msg("assistant", blocks + [
        {"type": "text", "text": "Searching."},
        {"type": "tool_use", "id": cid, "name": "search", "input": {"q": "x"}},
    ])


def o(cid):
    return Msg("user", [{"type": "tool_result", "id": cid, "output": "doc 1 ... doc 2 ..."}])


def history():
    return [q(), note(), a("c1"), o("c1"), a("c2"), o("c2")]


def codes(violations):
    return [v.code for v in violations]


def test_valid_history_has_no_violations():
    h = history()
    assert validate_anthropic_history(h, capture_thinking_snapshot(h)) == []


def test_first_message_must_be_user_and_no_system_role():
    h = [Msg("assistant", "hi"), Msg("system", "you are helpful"), q()]
    v = validate_anthropic_history(h)
    assert FIRST_NOT_USER in codes(v) and SYSTEM_IN_MESSAGES in codes(v)


def test_empty_message_is_flagged():
    h = history() + [Msg("user", [])]
    assert EMPTY_MESSAGE in codes(validate_anthropic_history(h))


def test_orphan_tool_result_when_tool_use_rewritten_to_text():
    # The manager rewrote a1 (which held the tool_use) but kept o1.
    h = history()
    h[2] = Msg("assistant", [{"type": "text", "text": "Round 1: I searched."}])
    v = validate_anthropic_history(h, capture_thinking_snapshot(h))
    assert ORPHAN_TOOL_RESULT in codes(v)


def test_orphan_tool_use_when_result_deleted():
    # Lock edge case: deleting the rewrite-only tool_result orphans the
    # tool_use in the locked assistant message.
    h = history()
    del h[5]
    v = validate_anthropic_history(h, capture_thinking_snapshot(h))
    assert ORPHAN_TOOL_USE in codes(v)
    assert has_errors(v)


def test_tool_result_not_adjacent():
    h = history()
    h.insert(5, note())  # a note wedged between a2 and its tool_result
    v = validate_anthropic_history(h, capture_thinking_snapshot(h))
    assert codes(v) == [TOOL_RESULT_NOT_ADJACENT]


def test_thinking_block_lost_is_detected():
    # Lock edge case: an op targeting the locked latest assistant message.
    h = history()
    snap = capture_thinking_snapshot(h)
    h[4] = Msg("assistant", [{"type": "text", "text": "rewritten a2"},
                             {"type": "tool_use", "id": "c2", "name": "search", "input": {}}])
    assert THINKING_MISSING in codes(validate_anthropic_history(h, snap))


def test_thinking_block_modified_is_detected():
    h = history()
    snap = capture_thinking_snapshot(h)
    h[4] = a("c2", sig="tampered-signature")
    v = validate_anthropic_history(h, snap)
    assert codes(v) == [THINKING_MODIFIED]
    # Same check catches an edited thinking text with an intact signature.
    h[4] = Msg("assistant", [{"type": "thinking", "thinking": "summarised", "signature": "sig"},
                             {"type": "tool_use", "id": "c2", "name": "search", "input": {}}])
    assert THINKING_MODIFIED in codes(validate_anthropic_history(h, snap))


def test_thinking_block_must_lead_the_message():
    h = history()
    snap = capture_thinking_snapshot(h)
    blocks = list(h[4].content)
    h[4] = Msg("assistant", blocks[1:] + [blocks[0]])  # thinking moved to the end
    assert THINKING_NOT_FIRST in codes(validate_anthropic_history(h, snap))


def test_snapshot_is_none_without_thinking_and_checks_are_skipped():
    h = [q(), a("c1", thinking=False), o("c1")]
    assert capture_thinking_snapshot(h) is None
    assert validate_anthropic_history(h, None) == []


def test_reporting_helpers():
    h = history()
    del h[5]
    v = validate_anthropic_history(h, capture_thinking_snapshot(h))
    assert violations_to_json(v)[0]["code"] == ORPHAN_TOOL_USE
    assert "ORPHAN_TOOL_USE" in format_violations(v)
    err = AnthropicValidityError(v)
    assert err.violations == v and "ORPHAN_TOOL_USE" in str(err)


# --- wiring: perform_modifications runs the lock, then the validator --------

def _manager(**config):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stubs
    stubs.install()
    from asio.memory.memorymanager import MemoryManager
    cfg = {"chat_tokenizer": stubs.StubTokenizer(), "enable_vdb": False,
           "agent_thinking_enabled": True, "anthropic_validity_mode": "raise"}
    cfg.update(config)
    return MemoryManager(name="manager", model=stubs.StubModel(), formatter=None, config=cfg)


def test_perform_modifications_keeps_context_valid():
    import asyncio

    m = _manager()
    m._chat_history = history()
    mods = [
        {"ids": [2, 3], "role": "user", "justification": "old round",
         "new_content": "Round 1: searched x, got docs"},   # legal
        {"ids": [4], "role": "assistant", "justification": "shorten",
         "new_content": "rewritten a2"},                     # targets LOCKED
        {"ids": [5], "role": "user", "justification": "drop", "new_content": ""},  # deletes REWRITE_ONLY
    ]
    asyncio.run(m.perform_modifications({"modifications": mods}))

    # The two illegal ops were dropped, so nothing invalid reached the API.
    assert [v.lock for v in m.lock_violations_last] == ["locked", "rewrite_only"]
    assert m.validity_violations_last == []
    assert m.last_results["validity_violations"] == []
    # The in-flight thinking block survived verbatim.
    thinking = [b for b in m._chat_history[-2].content if b.get("type") == "thinking"]
    assert thinking == [{"type": "thinking", "thinking": "let me search", "signature": "sig"}]


def test_perform_modifications_raises_when_context_is_invalid():
    import asyncio

    from asio.memory.memorymanager import ContextValidityError

    m = _manager()
    m._chat_history = history()
    # Deleting the task message and the note leaves an assistant message first,
    # which the API rejects. The lock does not cover this; the validator does.
    mods = [{"ids": [0, 1], "role": "user", "justification": "prune", "new_content": ""}]
    try:
        asyncio.run(m.perform_modifications({"modifications": mods}))
    except ContextValidityError as e:
        assert FIRST_NOT_USER in [v.code for v in e.violations]
        return
    raise AssertionError("expected ContextValidityError")


def test_validity_mode_log_does_not_raise():
    import asyncio

    m = _manager(anthropic_validity_mode="log")
    m._chat_history = history()
    mods = [{"ids": [0, 1], "role": "user", "justification": "prune", "new_content": ""}]
    asyncio.run(m.perform_modifications({"modifications": mods}))
    assert FIRST_NOT_USER in [v.code for v in m.validity_violations_last]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
