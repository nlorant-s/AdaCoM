"""The Unity dev config says what the harness needs it to say (TASK 7).

Parses with PyYAML when available; falls back to text assertions so the test
still runs on a bare checkout.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
CONFIG = os.path.join(REPO, "examples", "dev", "haiku_untrained_1gpu.yaml")
SBATCH = os.path.join(REPO, "examples", "dev", "haiku_untrained_1gpu.sbatch")

sys.path.insert(0, os.path.join(REPO, "as1"))


def _load():
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(open(CONFIG))


def _workflow_args(cfg):
    return cfg["buffer"]["explorer_input"]["eval_tasksets"][0]["workflow_args"]


def test_config_exists_and_parses():
    assert os.path.exists(CONFIG)
    cfg = _load()
    if cfg is None:
        assert "agent_model_name: claude-haiku-4-5" in open(CONFIG).read()
        return
    assert cfg["mode"] == "bench"                      # untrained manager
    assert cfg["cluster"] == {"node_num": 1, "gpu_per_node": 1}
    assert cfg["explorer"]["rollout_model"]["engine_num"] == 1
    assert cfg["buffer"]["batch_size"] == 5            # 5 dev tasks


def test_agent_is_haiku_with_thinking_on():
    cfg = _load()
    if cfg is None:
        return
    wa = _workflow_args(cfg)
    assert wa["agent_model_name"] == "claude-haiku-4-5"
    assert wa["agent_enable_thinking"] is True
    assert wa["agent_base_url"] is None                # direct Anthropic, not DashScope
    assert wa["max_iterations"] == 15


def test_thinking_budget_is_valid_for_haiku():
    cfg = _load()
    if cfg is None:
        return
    wa = _workflow_args(cfg)
    from asio.utils.retry import (MIN_THINKING_BUDGET_TOKENS,
                                  classify_anthropic_thinking_mode,
                                  get_anthropic_thinking_kwargs)

    tc = wa["agent_thinking_config"]
    assert classify_anthropic_thinking_mode(wa["agent_model_name"]) == "budget"
    assert tc["budget_tokens"] >= MIN_THINKING_BUDGET_TOKENS
    # Must leave room for the response, or the API 400s.
    assert tc["budget_tokens"] < wa["agent_max_tokens"]
    kwargs = get_anthropic_thinking_kwargs(wa["agent_model_name"], True, tc)
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": tc["budget_tokens"]}


def test_lineage_and_cost_guard_are_on():
    cfg = _load()
    if cfg is None:
        return
    wa = _workflow_args(cfg)
    mem = wa["memory_config"]
    assert mem["lineage_log_dir"]
    assert mem.get("lineage_logging", True) is not False
    assert mem["anthropic_validity_mode"] == "raise"
    assert mem["background_info_key"] == "anthropic_tool_use"
    assert mem["lock_violation_penalty"] is None       # warm-up: silent drop

    cost = wa["agent_cost_config"]
    assert cost["max_usd"] and cost["max_tokens"] and cost["log_path"]
    pricing = os.path.join(REPO, cost["pricing_path"])
    assert os.path.exists(pricing), pricing


def test_pricing_covers_the_agent_this_config_uses():
    cfg = _load()
    if cfg is None:
        return
    from asio.utils.cost_guard import load_pricing, resolve_price

    wa = _workflow_args(cfg)
    pricing = load_pricing(os.path.join(REPO, wa["agent_cost_config"]["pricing_path"]))
    price = resolve_price(pricing, wa["agent_model_name"])
    assert price is not None and price.input > 0 and price.output > 0


def test_sbatch_matches_the_spec_dev_allocation():
    text = open(SBATCH).read()
    for line in ("-p gpu-preempt", "-t 02:00:00", "--gpus=1",
                 "--constraint=l40s", "--mem=64G", "-c 8"):
        assert line in text, line
    # Secrets from the environment only, never committed.
    assert "ANTHROPIC_API_KEY:?" in text
    assert "sk-ant" not in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
