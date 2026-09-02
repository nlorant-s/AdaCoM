"""Direct-Anthropic provider detection and thinking parameters (TASK 2).

This fork sends exactly one thinking shape, adaptive:
``thinking={"type": "adaptive"}`` with depth via ``output_config.effort``.
Manual extended thinking (``budget_tokens``) is deliberately unsupported --
Claude 4.7+ rejects it with a 400, and supporting both would mean the dev agent
exercised a request shape the experiments never send. Models that predate
adaptive thinking must run with thinking off.

Shapes verified 2026-09 against
https://platform.claude.com/docs/en/build-with-claude/{extended-thinking,thinking,effort}.
With thinking on, temperature/top_k are rejected; Claude 4.7+ rejects any
non-default temperature/top_p/top_k regardless of thinking.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stubs  # noqa: E402

stubs.install()

from asio.utils.retry import (  # noqa: E402
    ANTHROPIC_EFFORT_LEVELS,
    ThinkingUnsupportedError,
    anthropic_rejects_sampling_params,
    check_thinking_supported,
    detect_model_provider,
    get_anthropic_sampling_kwargs,
    get_thinking_kwargs,
    parse_claude_version,
    supports_adaptive_thinking,
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


def test_adaptive_support_by_model():
    assert supports_adaptive_thinking("claude-sonnet-5") is True
    assert supports_adaptive_thinking("claude-opus-4-7") is True
    assert supports_adaptive_thinking("claude-sonnet-4-6") is True
    assert supports_adaptive_thinking("claude-fable-5-1") is True
    # Pre-4.6 models would need the deprecated budget shape.
    assert supports_adaptive_thinking("claude-haiku-4-5") is False
    assert supports_adaptive_thinking("claude-opus-4-5-20251101") is False
    assert supports_adaptive_thinking("claude-3-7-sonnet-20250219") is False
    # Unknown ids assume "newer than we know": adaptive, never a 400 "enabled".
    assert supports_adaptive_thinking("claude-something-new") is True


def test_sonnet_5_gets_adaptive_thinking():
    m = AnthropicChatModel("claude-sonnet-5")
    assert get_thinking_kwargs(m, True) == {"thinking": {"type": "adaptive"}}
    kw = get_thinking_kwargs(m, True, {"effort": "medium"})
    assert kw == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    # No budget is ever sent, whatever the config says.
    assert "budget_tokens" not in str(get_thinking_kwargs(m, True, {"budget_tokens": 8000}))


def test_thinking_on_a_pre_adaptive_model_fails_loudly():
    """Haiku 4.5 can still be the agent -- with thinking off."""
    m = AnthropicChatModel("claude-haiku-4-5")
    assert get_thinking_kwargs(m, False) == {}          # fine
    try:
        get_thinking_kwargs(m, True)
    except ThinkingUnsupportedError as e:
        assert "claude-sonnet-5" in str(e) and "agent_enable_thinking: false" in str(e)
    else:
        raise AssertionError("expected ThinkingUnsupportedError")
    # The same guard runs at config-parse time, before anything is spent.
    check_thinking_supported("claude-sonnet-5")
    try:
        check_thinking_supported("claude-haiku-4-5")
    except ThinkingUnsupportedError:
        return
    raise AssertionError("expected ThinkingUnsupportedError")


def test_thinking_can_be_disabled_from_the_config():
    m = AnthropicChatModel("claude-sonnet-5")
    assert get_thinking_kwargs(m, False) == {}
    assert get_thinking_kwargs(m, True, {"enabled": False}) == {}


def test_invalid_effort_is_rejected():
    try:
        get_thinking_kwargs(AnthropicChatModel("claude-sonnet-5"), True, {"effort": "turbo"})
    except ValueError as e:
        assert "effort" in str(e)
    else:
        raise AssertionError("expected ValueError for an invalid effort")
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
    # Thinking off on an older model (the harness-check agent): temperature
    # only, never alongside top_p.
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
