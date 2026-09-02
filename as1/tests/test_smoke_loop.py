"""End-to-end smoke test of the managed loop, all models mocked (TASK 8).

Three agent steps against a real MemoryManager:

  step 1  the manager rewrites the rewrite-only tool_result   -> legal
  step 2  the manager edits the locked latest assistant msg   -> dropped
  step 3  the manager merges an old tool pair (legal) and
          deletes the rewrite-only tool_result                -> partly dropped

Asserts the history the agent would be sent stays API-valid at every step (the
Task 3 validator), that the in-flight thinking block round-trips verbatim, that
violations are recorded and priced, and that the lineage JSONL has one line per
manager round with pre/post snapshots.

No network, no GPU, no real model: fake agent model, fake tool, fake manager.
"""
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.agent.bcp_worker import BCPWorker  # noqa: E402
from asio.memory.anthropic_validity import (  # noqa: E402
    capture_thinking_snapshot, format_violations, validate_anthropic_history,
)
from asio.memory.context_lock import LOCKED, REWRITE_ONLY, compute_locks  # noqa: E402
from asio.memory.memorymanager import MemoryManager  # noqa: E402
from asio.utils.cost_guard import CostTracker, ModelPrice  # noqa: E402

Msg = sys.modules["agentscope"].message.Msg


# --- fakes ------------------------------------------------------------------
class FakeAgentModel:
    """Emits an assistant turn with a signed thinking block and a tool_use."""

    model_name = "claude-haiku-4-5"

    def __init__(self):
        self.step = 0

    def respond(self):
        self.step += 1
        cid = f"call_{self.step}"
        return stubs.StubResponse(content=[
            {"type": "thinking",
             "thinking": f"step {self.step}: I should search for the next clue",
             "signature": f"sig-{self.step}"},
            {"type": "text", "text": f"Searching (step {self.step})."},
            {"type": "tool_use", "id": cid, "name": "search",
             "input": {"query": f"clue {self.step}"}},
        ], usage={"input_tokens": 1000 * self.step, "output_tokens": 200})


def fake_tool(call_id, step):
    """Anthropic tool results are user-role messages."""
    return Msg("user", [{"type": "tool_result", "id": call_id, "name": "search",
                         "output": f"doc_{step}a: long snippet ... doc_{step}b: more text ..."}],
               "user")


class FakeManagerModel:
    """Returns the edit plan for the current round, computed from the history."""

    def __init__(self, memory_box):
        self.memory_box = memory_box
        self.round = 0
        self.plans = []

    async def __call__(self, messages, **kwargs):
        self.round += 1
        history = self.memory_box[0]._chat_history
        locks = compute_locks(history, thinking_enabled=True)
        locked = [i for i, v in locks.items() if v == LOCKED][0]
        rewrite_only = [i for i, v in locks.items() if v == REWRITE_ONLY][0]

        if self.round == 1:
            ops = [{"ids": [rewrite_only], "role": "user",
                    "justification": "compress the tool output",
                    "new_content": "doc_1a, doc_1b: both mention the 1997 merger."}]
        elif self.round == 2:
            ops = [{"ids": [locked], "role": "assistant",
                    "justification": "the reasoning is redundant",
                    "new_content": "I searched and found the merger date."}]
        else:
            ops = [
                {"ids": [1, 2], "role": "user",
                 "justification": "fold round 1 into a note",
                 "new_content": "Round 1: searched 'clue 1'; found the 1997 merger."},
                {"ids": [rewrite_only], "role": "user",
                 "justification": "no longer needed", "new_content": ""},
            ]
        self.plans.append(ops)
        return stubs.StubResponse(text=json.dumps({"modifications": ops}))


class FakeWorker:
    """The slice of BCPWorker the loop's bookkeeping touches."""

    record_lock_violations = BCPWorker._record_lock_violation_penalty
    record_cost = BCPWorker._record_agent_cost

    def __init__(self, memory, cost_tracker):
        self.memory = memory
        self.cost_tracker = cost_tracker
        self.calculate_reward = True
        self.intermediate_rewards = []
        self.current_iteration = 0
        self.compression_step = 0
        self.experiment_logger = stubs.StubLogger()
        self.abort_rollout = False
        self.abort_reason = None


# --- the loop ---------------------------------------------------------------
async def _run_loop(lineage_path, steps=3):
    memory_box = [None]
    manager_model = FakeManagerModel(memory_box)
    memory = MemoryManager(
        name="MemoryManager",
        model=manager_model,
        formatter=stubs.StubFormatter(),
        config={
            "chat_tokenizer": stubs.StubTokenizer(),
            "enable_vdb": False,
            "agent_thinking_enabled": True,
            "lock_violation_penalty": -0.5,
            "lineage_log_path": lineage_path,
            "anthropic_validity_mode": "raise",
            "background_info_key": "anthropic_tool_use",
            "max_model_len": 32768,
        },
    )
    memory_box[0] = memory

    agent = FakeAgentModel()
    tracker = CostTracker(
        pricing={"claude-haiku-4-5": ModelPrice(input=1.0, output=5.0)},
        model_name=agent.model_name, max_usd=100.0,
    )
    worker = FakeWorker(memory, tracker)

    await memory.add(Msg("user", [{"type": "text", "text": "Question: who signed it?"}], "user"))

    violations_by_step, validity_by_step = [], []
    for step in range(1, steps + 1):
        worker.current_iteration = step - 1

        response = agent.respond()
        worker.record_cost(response)
        assistant = Msg("assistant", response.content, "assistant")
        await memory.add(assistant)

        call_id = response.content[-1]["id"]
        await memory.add(fake_tool(call_id, step))     # closes the round -> manager runs

        violations_by_step.append(list(memory.lock_violations_last))
        worker.record_lock_violations()
        worker.compression_step += 1

        # What the agent would be sent next must be valid on its own terms.
        history = await memory.get_memory()
        validity_by_step.append(validate_anthropic_history(
            history, thinking_snapshot=capture_thinking_snapshot(history)))

    return memory, worker, manager_model, violations_by_step, validity_by_step


def test_three_step_loop_keeps_the_context_api_valid():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lineage.jsonl")
        memory, worker, manager, violations, validity = asyncio.run(_run_loop(path))

        assert manager.round == 3, "the manager must run once per closed round"
        for step, v in enumerate(validity, 1):
            assert v == [], f"step {step}: {format_violations(v)}"


def test_lock_violations_are_recorded_and_priced():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lineage.jsonl")
        memory, worker, manager, violations, _ = asyncio.run(_run_loop(path))

        assert violations[0] == []                                  # legal rewrite
        assert [v.lock for v in violations[1]] == [LOCKED]           # locked edit
        assert [v.lock for v in violations[2]] == [REWRITE_ONLY]     # delete of the pair
        # The legal merge in step 3 still went through.
        assert manager.plans[2][0]["ids"] == [1, 2]

        penalties = [r for r in worker.intermediate_rewards if r["reason"] == "lock_violation"]
        assert [p["step"] for p in penalties] == [1, 2]              # steps 2 and 3, 0-indexed
        assert all(p["reward"] == -0.5 for p in penalties)


def test_in_flight_thinking_block_survives_verbatim():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lineage.jsonl")
        memory, _, _, _, _ = asyncio.run(_run_loop(path))

        latest = [m for m in memory._chat_history if m.role == "assistant"][-1]
        thinking = [b for b in latest.content if b.get("type") == "thinking"]
        assert thinking == [{"type": "thinking",
                             "thinking": "step 3: I should search for the next clue",
                             "signature": "sig-3"}]


def test_lineage_jsonl_has_one_line_per_round_with_snapshots():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lineage.jsonl")
        memory, _, _, _, _ = asyncio.run(_run_loop(path))

        lines = [json.loads(l) for l in open(path)]
        assert len(lines) == 3
        for i, line in enumerate(lines, 1):
            assert line["round"] == i
            assert line["pre"] and line["post"]
            assert all("cm_uid" in m and "role" in m and "content" in m for m in line["pre"])
            assert line["validity_violations"] == []
        # Round 1 rewrote one message: same length, one changed uid.
        assert len(lines[0]["post"]) == len(lines[0]["pre"])
        # Round 3 merged two messages away and had its delete dropped.
        assert len(lines[2]["post"]) == len(lines[2]["pre"]) - 1
        assert lines[1]["lock_violations"] and lines[2]["lock_violations"]
        # Lineage links the merged message to the two it consumed.
        merged = [r for r in lines[2]["lineage"] if len(r["consumed_uids"]) == 2]
        assert merged and merged[0]["produced_uid"]
        assert all(u for u in merged[0]["consumed_uids"])


def test_agent_cost_is_tracked_across_the_loop():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lineage.jsonl")
        _, worker, _, _, _ = asyncio.run(_run_loop(path))

        t = worker.cost_tracker
        assert t.totals.calls == 3
        assert t.totals.input_tokens == 1000 + 2000 + 3000
        assert t.totals.output_tokens == 600
        assert abs(t.cost_usd - (6000 * 1.0 + 600 * 5.0) / 1_000_000) < 1e-12
        assert worker.abort_rollout is False


def test_user_role_tool_result_closes_a_round():
    """Regression: with the direct Anthropic API tool results are user-role
    messages (the Messages API has no system role in `messages`), and those
    must still close the round that invokes the manager."""
    async def run():
        box = [None]
        manager = FakeManagerModel(box)
        memory = MemoryManager(
            name="m", model=manager, formatter=stubs.StubFormatter(),
            config={"chat_tokenizer": stubs.StubTokenizer(), "enable_vdb": False,
                    "agent_thinking_enabled": True, "max_model_len": 32768},
        )
        box[0] = memory
        agent = FakeAgentModel()
        await memory.add(Msg("user", [{"type": "text", "text": "Q"}], "user"))
        response = agent.respond()
        await memory.add(Msg("assistant", response.content, "assistant"))
        assert memory.round_number == 0 and manager.round == 0   # no result yet
        await memory.add(fake_tool(response.content[-1]["id"], 1))
        return memory, manager

    memory, manager = asyncio.run(run())
    assert memory.round_number == 1
    assert manager.round == 1
    # A plain user message (a manager-authored note) must not close a round.
    asyncio.run(memory.add(Msg("user", [{"type": "text", "text": "note"}], "user")))
    assert memory.round_number == 1 and manager.round == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
