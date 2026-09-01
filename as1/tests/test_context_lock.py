import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asio.memory.context_lock import (
    compute_locks, filter_modifications, format_msgs_with_locks,
    plan_lineage, ensure_uids, LOCKED, REWRITE_ONLY,
)


class Msg:  # duck-typed stand-in for agentscope.message.Msg
    def __init__(self, role, content, metadata=None):
        self.role, self.content, self.metadata = role, content, metadata


def q():  return Msg("user", "Find the person who ...")
def note(): return Msg("user", [{"type": "text", "text": "Working note: constraints A,B"}])
def a(cid, thinking=True):
    blocks = ([{"type": "thinking", "thinking": "let me search", "signature": "sig"}] if thinking else []) \
        + [{"type": "text", "text": "Searching."}, {"type": "tool_use", "id": cid, "name": "search", "input": {"q": "x"}}]
    return Msg("assistant", blocks)
def o(cid): return Msg("user", [{"type": "tool_result", "id": cid, "output": "doc 1 ... doc 2 ..."}])


def history():
    # q, note, a1, o1, a2, o2  (a2/o2 = in-flight cycle)
    return [q(), note(), a("c1"), o("c1"), a("c2"), o("c2")]


def test_locks_latest_cycle_only():
    locks = compute_locks(history(), thinking_enabled=True)
    assert locks == {4: LOCKED, 5: REWRITE_ONLY}


def test_no_locks_when_thinking_disabled():
    assert compute_locks(history(), thinking_enabled=False) == {}


def test_no_lock_when_latest_assistant_is_final_text():
    h = history() + [Msg("assistant", [{"type": "text", "text": "The answer is X"}])]
    assert compute_locks(h) == {}


def test_filter_drops_locked_and_delete_of_rewrite_only():
    h = history()
    mods = [
        {"ids": [2, 3], "role": "user", "justification": "old", "new_content": "Round1: searched x, got docs"},  # ok
        {"ids": [4], "role": "assistant", "justification": "x", "new_content": "rewritten a2"},              # LOCKED
        {"ids": [5], "role": "user", "justification": "x", "new_content": ""},                                # delete rewrite-only
        {"ids": [5], "role": "user", "justification": "x", "new_content": "doc 1 summary"},                  # ok rewrite
        {"ids": [1, 5], "role": "user", "justification": "x", "new_content": "merged"},                      # merge with rewrite-only
    ]
    kept, viol = filter_modifications(mods, h)
    assert [m["ids"] for m in kept] == [[2, 3], [5]]
    assert [v.op_index for v in viol] == [1, 2, 4]
    assert viol[0].lock == LOCKED and viol[1].lock == REWRITE_ONLY


def test_format_annotates_locks_and_strips_thinking():
    s = format_msgs_with_locks(history())
    data = json.loads(s)
    assert data[4]["lock"] == "locked" and data[5]["lock"] == "rewrite_only"
    assert "lock" not in data[2]
    assert all(b["type"] != "thinking" for u in data if isinstance(u["content"], list) for b in u["content"])
    assert "id" not in data[4]["content"][-1]  # content ids stripped


def test_lineage_records_consumed_and_produced():
    h = history()
    mods = [
        {"ids": [2, 3], "role": "user", "justification": "merge", "new_content": "Round1 summary"},
        {"ids": [1], "role": "user", "justification": "drop", "new_content": ""},
    ]
    recs, produced = plan_lineage(mods, h, step=7)
    assert all(m.metadata["cm_uid"] for m in h)
    assert recs[0].consumed_uids == [h[2].metadata["cm_uid"], h[3].metadata["cm_uid"]]
    assert recs[0].produced_uid is not None and produced[2] == recs[0].produced_uid
    assert recs[1].produced_uid is None and recs[1].consumed_roles == ["user"]
    assert recs[0].step == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
