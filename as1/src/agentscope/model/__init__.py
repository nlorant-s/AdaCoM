# -*- coding: utf-8 -*-
"""The model module."""

from ._model_base import ChatModelBase
from ._model_response import ChatResponse
from ._dashscope_model import DashScopeChatModel
from ._dashscope_claude_model import DashScopeClaudeChatModel
from ._openai_model import OpenAIChatModel
from ._anthropic_model import AnthropicChatModel
from ._ollama_model import OllamaChatModel
from ._gemini_model import GeminiChatModel
from ._trinity_model import TrinityChatModel
from ._llm_hooks import (
    enable_auto_llm_logging,
    disable_auto_llm_logging,
    create_auto_logging_hook,
    auto_llm_logging_hook,
)

__all__ = [
    "ChatModelBase",
    "ChatResponse",
    "DashScopeChatModel",
    "DashScopeClaudeChatModel",
    "OpenAIChatModel",
    "AnthropicChatModel",
    "OllamaChatModel",
    "GeminiChatModel",
    "TrinityChatModel",
    "enable_auto_llm_logging",
    "disable_auto_llm_logging",
    "create_auto_logging_hook",
    "auto_llm_logging_hook",
]
