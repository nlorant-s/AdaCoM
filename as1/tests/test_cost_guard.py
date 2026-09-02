"""Per-run token / dollar budget for the frozen agent (TASK 6).

Prices are never in code: every test loads them from a file, the same way the
worker does.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.utils.cost_guard import (  # noqa: E402
    CostBudgetExceeded, CostTracker, batch_spend, build_cost_tracker,
    extract_usage, load_pricing, reset_batch_spend, resolve_price,
)

PRICES = {
    "models": {
        "claude-haiku-4-5": {"input": 1.0, "output": 5.0,
                             "cache_write_5m": 1.25, "cache_read": 0.10},
        "claude-sonnet-5": {"input": 2.0, "output": 10.0,
                            "cache_write_5m": 2.50, "cache_read": 0.20},
    }
}


class Usage:  # agentscope ChatUsage stand-in
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.time = 0.1


def _pricing_file(tmpdir, data=None):
    path = os.path.join(tmpdir, "pricing.json")
    with open(path, "w") as f:
        json.dump(data or PRICES, f)
    return path


def test_pricing_is_loaded_from_a_file():
    with tempfile.TemporaryDirectory() as d:
        pricing = load_pricing(_pricing_file(d))
        assert pricing["claude-haiku-4-5"].input == 1.0
        assert pricing["claude-sonnet-5"].output == 10.0
        try:
            load_pricing(os.path.join(d, "nope.json"))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def test_shipped_pricing_file_is_readable_and_covers_both_agents():
    """examples/dev/anthropic_pricing.yaml, parsed without PyYAML if needed."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "examples", "dev", "anthropic_pricing.yaml")
    text = open(path).read()
    for model in ("claude-haiku-4-5", "claude-sonnet-5"):
        assert model in text, model
    assert "platform.claude.com" in text and "Checked:" in text


def test_model_ids_match_by_prefix():
    with tempfile.TemporaryDirectory() as d:
        pricing = load_pricing(_pricing_file(d))
        assert resolve_price(pricing, "claude-haiku-4-5-20251001").input == 1.0
        assert resolve_price(pricing, "gpt-4o") is None


def test_usage_extraction_handles_object_dict_and_openai_shapes():
    assert extract_usage(Usage(100, 20))["input_tokens"] == 100
    raw = {"input_tokens": 5, "output_tokens": 6,
           "cache_creation_input_tokens": 7, "cache_read_input_tokens": 8}
    assert extract_usage(raw) == {"input_tokens": 5, "output_tokens": 6,
                                  "cache_write_tokens": 7, "cache_read_tokens": 8}
    assert extract_usage({"prompt_tokens": 3, "completion_tokens": 4})["output_tokens"] == 4
    assert extract_usage(None) == {}


def test_cost_arithmetic():
    with tempfile.TemporaryDirectory() as d:
        t = CostTracker(load_pricing(_pricing_file(d)), model_name="claude-haiku-4-5")
        # 1M input @ $1 + 1M output @ $5
        t.record_usage(Usage(1_000_000, 1_000_000))
        assert abs(t.cost_usd - 6.0) < 1e-9
        # cache write @1.25, cache read @0.10
        t.record_usage({"input_tokens": 0, "output_tokens": 0,
                        "cache_creation_input_tokens": 1_000_000,
                        "cache_read_input_tokens": 1_000_000})
        assert abs(t.cost_usd - (6.0 + 1.35)) < 1e-9
        assert t.totals.calls == 2


def test_dollar_cap_trips():
    with tempfile.TemporaryDirectory() as d:
        t = CostTracker(load_pricing(_pricing_file(d)),
                        model_name="claude-sonnet-5", max_usd=0.05)
        t.record_usage(Usage(10_000, 1_000))     # $0.02 + $0.01
        assert not t.over_budget()
        t.record_usage(Usage(10_000, 1_000))
        assert t.over_budget()
        try:
            t.check_budget()
        except CostBudgetExceeded as e:
            assert e.snapshot["cost_usd"] > 0.05
            return
        raise AssertionError("expected CostBudgetExceeded")


def test_token_cap_trips_independently_of_price():
    with tempfile.TemporaryDirectory() as d:
        t = CostTracker(load_pricing(_pricing_file(d)),
                        model_name="unpriced-model", max_tokens=1_000)
        t.record_usage(Usage(900, 200))
        assert t.over_budget() and t.cost_usd == 0.0
        assert "unpriced-model" in t.unpriced_models


def test_finalize_logs_the_rollout_and_accumulates_the_batch():
    reset_batch_spend()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "cost", "spend.jsonl")
        for _ in range(2):
            t = CostTracker(load_pricing(_pricing_file(d)),
                            model_name="claude-haiku-4-5", log_path=log,
                            batch_id="batch-7", experiment_logger=stubs.StubLogger())
            t.record_usage(Usage(1_000_000, 0))   # $1 each
            snap = t.finalize()
        assert snap["batch_rollouts"] == 2
        assert abs(snap["batch_cumulative_usd"] - 2.0) < 1e-9
        assert abs(batch_spend("batch-7")["cost_usd"] - 2.0) < 1e-9
        lines = [json.loads(l) for l in open(log)]
        assert len(lines) == 2 and lines[0]["model"] == "claude-haiku-4-5"
    reset_batch_spend()


def test_builder_requires_a_pricing_path_and_honours_enabled():
    assert build_cost_tracker(None) is None
    assert build_cost_tracker({"enabled": False, "pricing_path": "x"}) is None
    try:
        build_cost_tracker({"max_usd": 1.0})
    except ValueError as e:
        assert "pricing_path" in str(e)
    else:
        raise AssertionError("expected ValueError without pricing_path")
    with tempfile.TemporaryDirectory() as d:
        t = build_cost_tracker({"pricing_path": _pricing_file(d), "max_usd": 2.0},
                               model_name="claude-sonnet-5")
        assert t.max_usd == 2.0 and t.model_name == "claude-sonnet-5"


def test_worker_aborts_the_rollout_when_the_cap_is_hit():
    from asio.agent.bcp_worker import BCPWorker

    class FakeResponse:
        def __init__(self, usage):
            self.usage = usage

    class FakeWorker:
        experiment_logger = stubs.StubLogger()
        abort_rollout = False
        abort_reason = None
        record = BCPWorker._record_agent_cost

    with tempfile.TemporaryDirectory() as d:
        w = FakeWorker()
        w.cost_tracker = CostTracker(load_pricing(_pricing_file(d)),
                                     model_name="claude-haiku-4-5", max_usd=0.5)
        w.record(FakeResponse(Usage(100_000, 0)))       # $0.10
        assert w.abort_rollout is False
        w.record(FakeResponse(Usage(1_000_000, 0)))     # $1.10 total
        assert w.abort_rollout is True
        assert w.abort_reason == "cost_budget_exceeded"
        assert any("COST_GUARD" in m for m in w.experiment_logger.messages("warning"))

    # No tracker configured: never aborts.
    w2 = FakeWorker()
    w2.cost_tracker = None
    w2.record(FakeResponse(Usage(10 ** 9, 10 ** 9)))
    assert w2.abort_rollout is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
