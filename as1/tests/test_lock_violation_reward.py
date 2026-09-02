"""Lock violations become a format penalty during RL (TASK 4).

Default is a silent drop (penalty None), which is what SFT / warm-up wants;
setting memory_config.lock_violation_penalty turns it into a learning signal.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.agent.bcp_worker import BCPWorker  # noqa: E402
from asio.agent.memory_reward_utils import build_lock_violation_reward  # noqa: E402


class FakeMemory:
    def __init__(self, violations=None, penalty=None):
        self.lock_violations_last = list(violations or [])
        self.lock_violation_penalty = penalty
        self.last_results = None


class FakeWorker:
    """Just the attributes _record_lock_violation_penalty touches."""

    def __init__(self, memory, calculate_reward=True):
        self.memory = memory
        self.calculate_reward = calculate_reward
        self.intermediate_rewards = []
        self.current_iteration = 3
        self.compression_step = 7
        self.experiment_logger = stubs.StubLogger()

    record = BCPWorker._record_lock_violation_penalty


VIOLATIONS = [
    {"op_index": 1, "ids": [4], "lock": "locked",
     "reason": "targets locked latest assistant message"},
    {"op_index": 2, "ids": [5], "lock": "rewrite_only",
     "reason": "deletes rewrite-only tool_result"},
]


def test_builder_returns_none_without_penalty_or_violations():
    assert build_lock_violation_reward(VIOLATIONS, None, 0, 0) is None
    assert build_lock_violation_reward([], -0.5, 0, 0) is None


def test_builder_shape_matches_the_other_intermediate_rewards():
    entry = build_lock_violation_reward(VIOLATIONS, -0.5, iteration=3, step=7)
    assert entry["reason"] == "lock_violation"
    assert entry["reward"] == -0.5          # flat magnitude, not scaled by count
    assert entry["count"] == 2
    assert entry["iteration"] == 3 and entry["step"] == 7
    assert entry["violations"] == VIOLATIONS
    assert "locked" in entry["text_content"]
    # Same keys the degeneration / no_change entries carry.
    assert {"text_content", "reward", "reason", "iteration", "step"} <= set(entry)


def test_penalty_recorded_on_the_current_manager_step():
    w = FakeWorker(FakeMemory(VIOLATIONS, penalty=-0.5))
    w.record()
    assert len(w.intermediate_rewards) == 1
    assert w.intermediate_rewards[0]["reward"] == -0.5
    assert w.intermediate_rewards[0]["step"] == 7
    # Consumed: a second pass in the same round must not double-charge.
    w.record()
    assert len(w.intermediate_rewards) == 1


def test_silent_drop_by_default():
    w = FakeWorker(FakeMemory(VIOLATIONS, penalty=None))
    w.record()
    assert w.intermediate_rewards == []
    assert w.memory.lock_violations_last == []  # still consumed


def test_no_violations_is_a_noop():
    w = FakeWorker(FakeMemory([], penalty=-0.5))
    w.record()
    assert w.intermediate_rewards == []


def test_reward_calculation_disabled():
    w = FakeWorker(FakeMemory(VIOLATIONS, penalty=-0.5), calculate_reward=False)
    w.record()
    assert w.intermediate_rewards == []


def test_penalty_reaches_the_worker_from_memory_config():
    """memory_config.lock_violation_penalty -> MemoryManager -> worker."""
    from asio.memory.memorymanager import MemoryManager

    m = MemoryManager(
        name="manager", model=stubs.StubModel(), formatter=None,
        config={"chat_tokenizer": stubs.StubTokenizer(), "enable_vdb": False,
                "agent_thinking_enabled": True, "lock_violation_penalty": -0.5},
    )
    m.lock_violations_last = list(VIOLATIONS)
    w = FakeWorker(m)
    w.record()
    assert w.intermediate_rewards[0]["reward"] == -0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
