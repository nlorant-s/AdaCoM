# -*- coding: utf-8 -*-
"""Lightweight stand-ins for as1's heavy runtime dependencies.

Unit tests must run CPU-only, without vLLM/Ray/HF (see CLAUDE.md), but
``asio.memory.memorymanager`` imports agentscope, transformers, tiktoken and
json_repair at module load. ``install()`` puts minimal stubs in ``sys.modules``
**only for packages that are actually missing**, so on a full install the same
tests exercise the real objects.

Nothing here is used by production code.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import uuid


def _missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


class StubTokenizer:
    """Whitespace tokenizer with the encode/decode surface the manager uses."""

    def __init__(self, chars_per_token: int = 4):
        self.chars_per_token = chars_per_token

    def encode(self, text, **_kwargs):
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, default=str)
        n = max(1, len(text) // self.chars_per_token) if text else 0
        return list(range(n))

    def decode(self, tokens, **_kwargs):
        return " ".join(str(t) for t in tokens)

    def __call__(self, text, **_kwargs):
        return {"input_ids": self.encode(text)}


def _install_agentscope() -> None:
    pkg = _module("agentscope")
    pkg.__path__ = []

    message = _module("agentscope.message")

    class Msg:
        def __init__(self, name, content, role, metadata=None, timestamp=None,
                     invocation_id=None):
            assert isinstance(content, (list, str))
            assert role in ("user", "assistant", "system")
            self.name = name
            self.content = content
            self.role = role
            self.metadata = metadata
            self.id = uuid.uuid4().hex[:12]
            self.timestamp = timestamp
            self.invocation_id = invocation_id

        def to_dict(self):
            return {"id": self.id, "name": self.name, "role": self.role,
                    "content": self.content, "metadata": self.metadata}

        def get_text_content(self):
            if isinstance(self.content, str):
                return self.content
            return "".join(
                b.get("text", "") for b in self.content
                if isinstance(b, dict) and b.get("type") == "text"
            )

        def get_content_blocks(self, block_type=None):
            if isinstance(self.content, str):
                blocks = [{"type": "text", "text": self.content}]
            else:
                blocks = [b for b in self.content if isinstance(b, dict)]
            if block_type is None:
                return blocks
            return [b for b in blocks if b.get("type") == block_type]

        def __repr__(self):
            return f"Msg(role={self.role!r}, content={self.content!r})"

    def _block(**kwargs):
        return dict(kwargs)

    message.Msg = Msg
    for _name in ("TextBlock", "ToolUseBlock", "ToolResultBlock", "ThinkingBlock",
                  "ImageBlock", "AudioBlock", "VideoBlock", "ContentBlock"):
        setattr(message, _name, _block)
    pkg.message = message

    memory = _module("agentscope.memory")

    class MemoryBase:
        pass

    memory.MemoryBase = MemoryBase
    pkg.memory = memory

    model = _module("agentscope.model")

    class ChatModelBase:
        pass

    class TrinityChatModel(ChatModelBase):
        pass

    class AnthropicChatModel(ChatModelBase):
        pass

    class OpenAIChatModel(ChatModelBase):
        pass

    class DashScopeClaudeChatModel(ChatModelBase):
        pass

    model.ChatModelBase = ChatModelBase
    model.TrinityChatModel = TrinityChatModel
    model.AnthropicChatModel = AnthropicChatModel
    model.OpenAIChatModel = OpenAIChatModel
    model.DashScopeClaudeChatModel = DashScopeClaudeChatModel
    model.enable_auto_llm_logging = lambda *a, **k: None
    model.disable_auto_llm_logging = lambda *a, **k: None
    pkg.model = model

    formatter = _module("agentscope.formatter")

    class FormatterBase:
        pass

    formatter.FormatterBase = FormatterBase
    formatter.OpenAIChatFormatter = type("OpenAIChatFormatter", (FormatterBase,), {})
    formatter.AnthropicChatFormatter = type("AnthropicChatFormatter", (FormatterBase,), {})
    pkg.formatter = formatter

    agent = _module("agentscope.agent")

    class AgentBase:
        def __init__(self, *args, **kwargs):
            pass

    class ReActAgent(AgentBase):
        pass

    agent.AgentBase = AgentBase
    agent.ReActAgent = ReActAgent
    pkg.agent = agent

    tool = _module("agentscope.tool")

    class ToolResponse:
        def __init__(self, content=None, metadata=None, **kwargs):
            self.content = content
            self.metadata = metadata

    class Toolkit:
        def __init__(self, *args, **kwargs):
            self.tools = {}

        def register_tool_function(self, func, **kwargs):
            self.tools[getattr(func, "__name__", str(func))] = func

        def get_json_schemas(self):
            return []

    tool.ToolResponse = ToolResponse
    tool.Toolkit = Toolkit
    pkg.tool = tool

    types_mod = _module("agentscope.types")
    types_mod.JSONSerializableObject = object
    pkg.types = types_mod


def _install_transformers() -> None:
    mod = _module("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return StubTokenizer()

    mod.AutoTokenizer = AutoTokenizer


def _install_tiktoken() -> None:
    mod = _module("tiktoken")
    mod.get_encoding = lambda *_a, **_k: StubTokenizer()
    mod.encoding_for_model = lambda *_a, **_k: StubTokenizer()


def _install_json_repair() -> None:
    mod = _module("json_repair")

    def repair_json(text, return_objects=False, **_kwargs):
        try:
            obj = json.loads(text)
        except Exception:
            obj = {}
        return obj if return_objects else json.dumps(obj)

    mod.repair_json = repair_json
    mod.loads = lambda text, **_k: repair_json(text, return_objects=True)


_INSTALLERS = {
    "agentscope": _install_agentscope,
    "transformers": _install_transformers,
    "tiktoken": _install_tiktoken,
    "json_repair": _install_json_repair,
}


def install() -> list:
    """Stub every missing dependency; return the names that were stubbed."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    stubbed = []
    for name, installer in _INSTALLERS.items():
        if _missing(name) and name not in sys.modules:
            installer()
            stubbed.append(name)
    return stubbed


class StubModel:
    """Minimal ChatModelBase stand-in: returns canned responses, records calls."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        return None


class StubFormatter:
    """FormatterBase stand-in: passes messages through as plain dicts."""

    async def format(self, msgs=None, **_kwargs):
        out = []
        for m in msgs or []:
            out.append({"role": getattr(m, "role", "user"),
                        "content": getattr(m, "content", "")})
        return out


class StubResponse:
    """ChatResponse stand-in with content blocks and usage."""

    def __init__(self, text="", content=None, usage=None):
        self.content = content if content is not None else [{"type": "text", "text": text}]
        self.usage = usage


class StubLogger:
    """ExperimentLogger stand-in that keeps everything in memory."""

    def __init__(self):
        self.records = []

    def _log(self, level, msg):
        self.records.append((level, str(msg)))

    def log_debug(self, msg):
        self._log("debug", msg)

    def log_info(self, msg):
        self._log("info", msg)

    def log_warning(self, msg):
        self._log("warning", msg)

    def log_error(self, msg):
        self._log("error", msg)

    def messages(self, level=None):
        return [m for lv, m in self.records if level is None or lv == level]
