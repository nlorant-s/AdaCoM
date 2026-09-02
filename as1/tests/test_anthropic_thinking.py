"""Direct-Anthropic provider detection and thinking parameters (TASK 2).

Parameter shapes verified 2026-09 against
https://platform.claude.com/docs/en/build-with-claude/extended-thinking,
.../thinking and .../effort:
  * claude-haiku-4-5 (dev agent)  -> thinking={"type":"enabled","budget_tokens":N}
  * claude-sonnet-5 (experiments) -> thinking={"type":"adaptive"} (+ output_config.effort);
    "enabled" returns 400 there.
  * With thinking on, temperature/top_k are rejected; Claude 4.7+ rejects any
    non-default temperature/top_p/top_k regardless of thinking.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.utils.retry import (  # noqa: E402
    ANTHROPIC_EFFORT_LEVELS,
    DEFAULT_THINKING_BUDGET_TOKENS,
    anthropic_rejects_sampling_params,
    classify_anthropic_thinking_mode,
    detect_model_provider,
    get_anthropic_sampling_kwargs,
    get_thinking_kwargs,
    parse_claude_version,
)


class AnthropicChatModel:  # direct Messages API client
    def __init__(self, model_name):
        self.model_name = model_name


class DashScopeClaudeChatModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.provider = "r"


class OpenAIChatModel:
    def __init__(self, model_name, base_url=""):
        self.model_name = model_name
        self.base_url = base_url


class TrinityChatModel:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"


def test_detect_provider_recognises_anthropic_chat_model():
    assert detect_model_provider(AnthropicChatModel("claude-haiku-4-5")) == "anthropic"
    assert detect_model_provider(AnthropicChatModel("claude-sonnet-5")) == "anthropic"
    # DashScope Claude keeps OpenAI request formatting.
    assert detect_model_provider(DashScopeClaudeChatModel("claude-sonnet-4-20250514")) == "openai"
    assert detect_model_provider(TrinityChatModel()) == "openai"


def test_parse_claude_version():
    assert parse_claude_version("claude-haiku-4-5") == (4, 5)
    assert parse_claude_version("claude-sonnet-5") == (5, 0)
    assert parse_claude_version("claude-opus-4-1-20250805") == (4, 1)
    assert parse_claude_version("claude-3-7-sonnet-20250219") == (3, 7)
    assert parse_claude_version("claude-sonnet-4-6") == (4, 6)
    assert parse_claude_version("claude-mythos") is None


def test_thinking_mode_classification():
    assert classify_anthropic_thinking_mode("claude-haiku-4-5") == "budget"
    assert classify_anthropic_thinking_mode("claude-opus-4-5-20251101") == "budget"
    assert classify_anthropic_thinking_mode("claude-3-7-sonnet-20250219") == "budget"
    assert classify_anthropic_thinking_mode("claude-sonnet-5") == "adaptive"
    assert classify_anthropic_thinking_mode("claude-opus-4-7") == "adaptive"
    assert classify_anthropic_thinking_mode("claude-sonnet-4-6") == "adaptive"
    assert classify_anthropic_thinking_mode("claude-fable-5-1") == "adaptive"
    assert classify_anthropic_thinking_mode("claude-3-5-sonnet-20241022") == "unsupported"
    # Unknown ids assume "newer than we know": adaptive, never a 400 "enabled".
    assert classify_anthropic_thinking_mode("claude-something-new") == "adaptive"


def test_haiku_45_gets_budget_thinking():
    kw = get_thinking_kwargs(AnthropicChatModel("claude-haiku-4-5"), True)
    assert kw == {"thinking": {"type": "enabled",
                               "budget_tokens": DEFAULT_THINKING_BUDGET_TOKENS}}
    kw = get_thinking_kwargs(AnthropicChatModel("claude-haiku-4-5"), True,
                             {"budget_tokens": 2048})
    assert kw["thinking"]["budget_tokens"] == 2048
    assert "output_config" not in kw


def test_sonnet_5_gets_adaptive_thinking():
    m = AnthropicChatModel("claude-sonnet-5")
    assert get_thinking_kwargs(m, True) == {"thinking": {"type": "adaptive"}}
    kw = get_thinking_kwargs(m, True, {"effort": "medium"})
    assert kw == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    # A budget must never be sent to an adaptive-only model.
    assert "budget_tokens" not in str(get_thinking_kwargs(m, True, {"budget_tokens": 8000}))


def test_thinking_off_and_mode_override():
    m = AnthropicChatModel("claude-haiku-4-5")
    assert get_thinking_kwargs(m, False) == {}
    assert get_thinking_kwargs(m, True, {"mode": "off"}) == {}
    assert get_thinking_kwargs(m, True, {"mode": "adaptive"}) == {"thinking": {"type": "adaptive"}}
    assert get_thinking_kwargs(AnthropicChatModel("claude-sonnet-5"), True,
                               {"mode": "budget", "budget_tokens": 1024}
                               )["thinking"]["type"] == "enabled"


def test_invalid_budget_and_effort_are_rejected():
    m = AnthropicChatModel("claude-haiku-4-5")
    for cfg, model in (({"budget_tokens": 512}, m),
                       ({"effort": "turbo"}, AnthropicChatModel("claude-sonnet-5"))):
        try:
            get_thinking_kwargs(model, True, cfg)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {cfg}")
    assert set(ANTHROPIC_EFFORT_LEVELS) == {"low", "medium", "high", "xhigh", "max"}


def test_other_providers_unchanged():
    # DashScope Claude still uses extra_body, and only for whitelisted models.
    kw = get_thinking_kwargs(DashScopeClaudeChatModel("claude-sonnet-4-20250514"), True)
    assert kw == {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}}}
    assert get_thinking_kwargs(DashScopeClaudeChatModel("claude-opus-4-20250514"), True) == {}
    assert get_thinking_kwargs(OpenAIChatModel("qwen3-max"), True) == {
        "extra_body": {"enable_thinking": True}}
    # An OpenAI-compatible relay serving claude-* must not get top-level thinking.
    assert get_thinking_kwargs(OpenAIChatModel("claude-sonnet-5", "https://relay/v1"), True) == {}


def test_sampling_kwargs_respect_thinking_and_model():
    # Thinking on: no sampling params at all (temperature/top_k incompatible).
    assert get_anthropic_sampling_kwargs("claude-haiku-4-5", 0, True) == {}
    # Thinking off on an older model: temperature only, never with top_p.
    kw = get_anthropic_sampling_kwargs("claude-haiku-4-5", 0, False)
    assert kw == {"temperature": 0} and "top_p" not in kw
    # 4.7+ rejects non-default sampling whether or not thinking is on.
    assert get_anthropic_sampling_kwargs("claude-sonnet-5", 0, False) == {}
    assert get_anthropic_sampling_kwargs("claude-opus-4-7", 0.7, False) == {}
    assert anthropic_rejects_sampling_params("claude-sonnet-5") is True
    assert anthropic_rejects_sampling_params("claude-haiku-4-5") is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
