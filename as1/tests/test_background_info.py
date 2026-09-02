"""Background_info is selectable per run (TASK 5).

The manager prompt itself is untouched; only what fills its {{Background}}
slot changes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.agent.background_info import (  # noqa: E402
    ANTHROPIC_TOOL_USE_BACKGROUND_INFO, BACKGROUND_INFO_VARIANTS,
    BCP_BACKGROUND_INFO, MCP_BACKGROUND_INFO, resolve_background_info,
)


def test_defaults_are_the_workers_original_texts():
    from asio.agent.bcp_worker import Background_info as bcp_bg
    from asio.agent.mcp_worker import Background_info as mcp_bg

    assert bcp_bg == BCP_BACKGROUND_INFO
    assert mcp_bg == MCP_BACKGROUND_INFO
    assert resolve_background_info({}, "bcp") == BCP_BACKGROUND_INFO
    assert resolve_background_info({"use_bg_info": True}, "mcp") == MCP_BACKGROUND_INFO


def test_key_selects_a_variant():
    cfg = {"use_bg_info": True, "background_info_key": "anthropic_tool_use"}
    assert resolve_background_info(cfg, "bcp") == ANTHROPIC_TOOL_USE_BACKGROUND_INFO
    assert set(BACKGROUND_INFO_VARIANTS) == {"bcp", "mcp", "anthropic_tool_use"}


def test_explicit_text_wins_and_unknown_key_raises():
    assert resolve_background_info({"background_info": "custom"}, "bcp") == "custom"
    # An empty string is not a choice; fall through to the key.
    assert resolve_background_info({"background_info": "  "}, "bcp") == BCP_BACKGROUND_INFO
    try:
        resolve_background_info({"background_info_key": "nope"}, "bcp")
    except ValueError as e:
        assert "anthropic_tool_use" in str(e)
        return
    raise AssertionError("expected ValueError for an unknown key")


def test_anthropic_variant_describes_protocol_and_lock():
    text = ANTHROPIC_TOOL_USE_BACKGROUND_INFO
    for expected in ("tool_use", "tool_result", "thinking", "locked", "rewrite_only"):
        assert expected in text, expected
    # Task-agnostic: no BrowseComp tool names.
    assert "get_document" not in text and "search:" not in text


def test_worker_resolves_the_variant_at_memory_creation():
    from asio.agent.bcp_worker import BCPWorker

    captured = {}

    class FakeSelf:
        experiment_logger = None

        def __init__(self, cfg):
            self.cfg = cfg

    cfg = {"use_bg_info": True, "background_info_key": "anthropic_tool_use",
           "chat_tokenizer": stubs.StubTokenizer(), "enable_vdb": False,
           "_api_model": stubs.StubModel(), "_api_formatter": None}
    memory = BCPWorker._create_custom_memory(FakeSelf(cfg), "MemoryManager", cfg)
    captured["bg"] = memory.config["background_info"]
    assert captured["bg"] == ANTHROPIC_TOOL_USE_BACKGROUND_INFO
    # And it lands in the manager prompt's {{Background}} slot.
    assert "{{Background}}" in memory.manager_step_2_prompt


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
