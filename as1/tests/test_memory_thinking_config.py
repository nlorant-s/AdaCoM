"""MemoryManager picks up the thinking flag / lineage path from memory_config.

Covers TASK 1: the workflows thread `agent_thinking_enabled` and
`lineage_log_path` into the manager's config. The workflow modules themselves
pull in Trinity + Ray, so the threading logic lives in two pure helpers
(`apply_thinking_memory_config`, `resolve_lineage_log_path`) which are tested
here directly, plus a MemoryManager built with a stub model and tokenizer.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.memory.context_lock import (  # noqa: E402
    apply_thinking_memory_config,
    resolve_lineage_log_path,
)
from asio.memory.memorymanager import MemoryManager  # noqa: E402


def _manager(**config):
    cfg = {
        "chat_tokenizer": stubs.StubTokenizer(),  # no HF download in tests
        "enable_vdb": False,
    }
    cfg.update(config)
    return MemoryManager(
        name="manager", model=stubs.StubModel(), formatter=None, config=cfg,
    )


def test_apply_thinking_memory_config_sets_flag():
    cfg = {}
    apply_thinking_memory_config(cfg, True)
    assert cfg["agent_thinking_enabled"] is True
    apply_thinking_memory_config(cfg, False)
    assert cfg["agent_thinking_enabled"] is False
    # Non-bool config values (yaml "true"/None) are coerced.
    assert apply_thinking_memory_config({}, None)["agent_thinking_enabled"] is False


def test_lineage_path_explicit_wins():
    with tempfile.TemporaryDirectory() as d:
        explicit = os.path.join(d, "nested", "mine.jsonl")
        path = resolve_lineage_log_path(
            {"lineage_log_path": explicit, "lineage_log_dir": os.path.join(d, "other")},
            run_dir=d, task_id=7, run_id=1,
        )
        assert path == explicit
        assert os.path.isdir(os.path.dirname(explicit))  # parent created


def test_lineage_path_from_dir_is_per_rollout():
    with tempfile.TemporaryDirectory() as d:
        a = resolve_lineage_log_path({"lineage_log_dir": d}, task_id="t1", run_id=0)
        b = resolve_lineage_log_path({"lineage_log_dir": d}, task_id="t1", run_id=1)
        assert a != b and a.endswith(os.path.join("task_t1", "run_0.jsonl"))


def test_lineage_path_falls_back_to_run_dir_and_can_be_disabled():
    with tempfile.TemporaryDirectory() as d:
        assert resolve_lineage_log_path({}, run_dir=d, task_id=1) == os.path.join(d, "lineage.jsonl")
        assert resolve_lineage_log_path({}, run_dir=None, task_id=1) is None
        assert resolve_lineage_log_path({"lineage_logging": False}, run_dir=d, task_id=1) is None


def test_manager_reads_thinking_config():
    with tempfile.TemporaryDirectory() as d:
        cfg = apply_thinking_memory_config({}, True)
        cfg["lineage_log_path"] = resolve_lineage_log_path(
            {"lineage_log_dir": d}, task_id="t9", run_id=2
        )
        cfg["lock_violation_penalty"] = -0.5
        m = _manager(**cfg)
        assert m.agent_thinking_enabled is True
        assert m.lineage_log_path.endswith(os.path.join("task_t9", "run_2.jsonl"))
        assert m.lock_violation_penalty == -0.5
        assert m.lock_violations_last == []


def test_manager_defaults_thinking_off():
    m = _manager()
    assert m.agent_thinking_enabled is False
    assert m.lineage_log_path is None
    assert m.lock_violation_penalty is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
