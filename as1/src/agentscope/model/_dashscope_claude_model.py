# -*- coding: utf-8 -*-
"""DashScope Claude model class for calling Claude models via DashScope's OpenAI-compatible API."""

from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Literal,
    Type,
    List,
)
from collections import OrderedDict

from pydantic import BaseModel

from ._model_base import ChatModelBase
from ._model_response import ChatResponse
from ._model_usage import ChatUsage
from .._logging import logger
from .._utils._common import (
    _json_loads_with_repair,
    _create_tool_from_base_model,
)
from ..message import TextBlock, ToolUseBlock, ThinkingBlock
from ..tracing import trace_llm
from ..types import JSONSerializableObject


class DashScopeClaudeChatModel(ChatModelBase):
    """DashScope Claude model class.

    This model class is specifically designed for calling Claude models through
    DashScope's OpenAI-compatible API endpoint with native Anthropic protocol support.

    Key features:
    - Uses OpenAI-compatible endpoint (/v1/chat/completions)
    - Adds dashscope_extend_params for native protocol support
    - Parses Anthropic-format responses

    Example usage:
        model = DashScopeClaudeChatModel(
            model_name="claude-sonnet-4-5-20250929",
            api_key="your-dashscope-api-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        stream: bool = False,
        max_tokens: int = 4096,
        provider: str = "r",
        generate_kwargs: dict[str, JSONSerializableObject] | None = None,
    ) -> None:
        """Initialize the DashScope Claude chat model.

        Uses raw HTTP POST (httpx) to DashScope's /chat/completions endpoint
        so the Claude-native response shape ({content:[...], usage:{...}})
        is parsed directly, bypassing the OpenAI SDK which expects OpenAI
        ChatCompletion schema and hangs/errors on Claude-native payloads.

        Args:
            model_name: The Claude model name (e.g., "aws.claude-3-7-sonnet-20250219")
            api_key: DashScope API key
            base_url: DashScope OpenAI-compatible API base URL
            stream: streaming not supported in this client; kept for API compat
            max_tokens: Maximum tokens for response
            provider: DashScope provider code (default "r" for Claude relay)
            generate_kwargs: Extra keyword arguments for generation
        """
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx is required for DashScopeClaudeChatModel. `pip install httpx`."
            ) from e

        super().__init__(model_name, stream=False)

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.provider = provider
        self.generate_kwargs = generate_kwargs or {}

    @trace_llm
    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Get the response from DashScope Claude API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Tool schemas in OpenAI format
            tool_choice: Tool choice control
            structured_model: Pydantic model for structured output
            **kwargs: Additional generation arguments

        Returns:
            ChatResponse or async generator of ChatResponse for streaming
        """
        # Save original args/kwargs for hooks
        original_args = (messages,)
        original_kwargs = {
            "tools": tools,
            "tool_choice": tool_choice,
            "structured_model": structured_model,
            **kwargs,
        }

        # Process messages for DashScope Claude native protocol:
        # 1. Extract system messages into top-level `system` string (Anthropic native format)
        # 2. Remove 'name' field (Anthropic doesn't accept 'name' in messages)
        # 3. Convert tool messages to user messages with tool_result content (Anthropic doesn't accept role="tool")
        # 4. Convert assistant tool_calls to Anthropic tool_use format
        system_texts: list[str] = []
        processed_messages = []
        for msg in messages:
            role = msg.get("role")

            if role == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    text = "\n".join(parts)
                else:
                    text = str(content)
                if text:
                    system_texts.append(text)

            elif role == "tool":
                # Convert tool result to Anthropic format: role="user" with tool_result content block
                tool_call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                processed_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content if isinstance(content, str) else str(content)
                        }
                    ]
                })

            elif role == "assistant":
                # Convert assistant message with tool_calls to Anthropic format
                content = msg.get("content")
                tool_calls = msg.get("tool_calls", [])

                # Build content blocks
                content_blocks = []

                # Add text content if present
                if content:
                    if isinstance(content, str):
                        content_blocks.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                content_blocks.append(block)
                            elif isinstance(block, str):
                                content_blocks.append({"type": "text", "text": block})

                # Convert OpenAI tool_calls to Anthropic tool_use blocks
                for tc in tool_calls:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        arguments = func.get("arguments", "{}")
                        if isinstance(arguments, str):
                            import json as json_module
                            try:
                                arguments = json_module.loads(arguments)
                            except:
                                arguments = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": arguments
                        })

                processed_messages.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else None
                })

            else:
                # User or other roles - copy with cleaned content
                clean_msg = {"role": role}
                content = msg.get("content")
                if content is not None:
                    clean_msg["content"] = content
                processed_messages.append(clean_msg)

        # Merge any caller-provided extra_body (e.g. {"thinking": {...}}) with our
        # dashscope routing params instead of letting **kwargs clobber them.
        caller_extra_body = kwargs.pop("extra_body", None) or {}
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_tokens": self.max_tokens,
            "stream": False,
            "dashscope_extend_params": {
                "provider": self.provider,
                "using_native_protocol": True,
            },
            **self.generate_kwargs,
            **kwargs,
        }
        if system_texts:
            body["system"] = "\n\n".join(system_texts)

        if tools:
            body["tools"] = self._format_tools_for_anthropic(tools)

        if tool_choice:
            self._validate_tool_choice(tool_choice, tools)
            body["tool_choice"] = self._format_tool_choice(tool_choice)

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden."
                )
            format_tool = _create_tool_from_base_model(structured_model)
            body["tools"] = self._format_tools_for_anthropic([format_tool])
            body["tool_choice"] = {"type": "tool", "name": format_tool["function"]["name"]}

        # Any caller extra_body keys (e.g. "thinking") go at top level for
        # dashscope's Claude native protocol.
        for k, v in caller_extra_body.items():
            body.setdefault(k, v)

        start_datetime = datetime.now()

        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=300) as http_client:
            resp = await http_client.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"DashScope Claude HTTP {resp.status_code}: {resp.text[:500]}"
                )
            raw_response = resp.json()

        parsed_response = await self._parse_response(
            start_datetime, raw_response, structured_model,
        )

        final_response = self._call_post_llm_hooks(
            parsed_response,
            *original_args,
            **original_kwargs,
        )
        return final_response

    async def _parse_response(
        self,
        start_datetime: datetime,
        response: Any,
        structured_model: Type[BaseModel] | None = None,
    ) -> ChatResponse:
        """Parse non-streaming response in Anthropic format.

        DashScope returns Anthropic-format response when using_native_protocol=true:
        {
            "content": [{"text": "...", "type": "text"}],
            "model": "...",
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {...}
        }
        """
        content_blocks: List[TextBlock | ToolUseBlock | ThinkingBlock] = []
        metadata = None

        # Response is the raw JSON dict from dashscope's Claude-native reply.
        raw_response = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(raw_response, dict):
            raw_response = {}

        # Handle Anthropic-format content
        content = raw_response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        content_blocks.append(
                            TextBlock(type="text", text=block.get("text", ""))
                        )
                    elif block_type == "thinking":
                        content_blocks.append(
                            ThinkingBlock(type="thinking", thinking=block.get("thinking", ""))
                        )
                    elif block_type == "tool_use":
                        content_blocks.append(
                            ToolUseBlock(
                                type="tool_use",
                                id=block.get("id", ""),
                                name=block.get("name", ""),
                                input=block.get("input", {}),
                            )
                        )
                        if structured_model:
                            metadata = block.get("input", {})

        # Parse usage
        usage = None
        raw_usage = raw_response.get("usage", {})
        if raw_usage:
            usage = ChatUsage(
                input_tokens=raw_usage.get("prompt_tokens", 0) or raw_usage.get("input_tokens", 0),
                output_tokens=raw_usage.get("completion_tokens", 0) or raw_usage.get("output_tokens", 0),
                time=(datetime.now() - start_datetime).total_seconds(),
            )

        return ChatResponse(
            content=content_blocks,
            usage=usage,
            metadata=metadata,
        )

    async def _parse_stream_response(
        self,
        start_datetime: datetime,
        response: Any,
        structured_model: Type[BaseModel] | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Parse streaming response in Anthropic format."""
        text_buffer = ""
        thinking_buffer = ""
        tool_calls = OrderedDict()
        usage = None
        metadata = None

        async for chunk in response:
            # Handle different chunk formats
            raw_chunk = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk

            # For streaming, content might come in different events
            content = raw_chunk.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            text_buffer = block.get("text", "")
                        elif block_type == "thinking":
                            thinking_buffer = block.get("thinking", "")
                        elif block_type == "tool_use":
                            tool_id = block.get("id", "")
                            tool_calls[tool_id] = {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": block.get("name", ""),
                                "input": block.get("input", {}),
                            }

            # Handle delta content for OpenAI-style streaming
            choices = raw_chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    text_buffer += delta["content"]

            # Update usage if available
            raw_usage = raw_chunk.get("usage", {})
            if raw_usage:
                usage = ChatUsage(
                    input_tokens=raw_usage.get("prompt_tokens", 0) or raw_usage.get("input_tokens", 0),
                    output_tokens=raw_usage.get("completion_tokens", 0) or raw_usage.get("output_tokens", 0),
                    time=(datetime.now() - start_datetime).total_seconds(),
                )

            # Build content blocks
            content_blocks: List[TextBlock | ToolUseBlock | ThinkingBlock] = []
            if thinking_buffer:
                content_blocks.append(ThinkingBlock(type="thinking", thinking=thinking_buffer))
            if text_buffer:
                content_blocks.append(TextBlock(type="text", text=text_buffer))
            for tool_call in tool_calls.values():
                content_blocks.append(
                    ToolUseBlock(
                        type=tool_call["type"],
                        id=tool_call["id"],
                        name=tool_call["name"],
                        input=tool_call["input"],
                    )
                )
                if structured_model:
                    metadata = tool_call["input"]

            if content_blocks:
                yield ChatResponse(
                    content=content_blocks,
                    usage=usage,
                    metadata=metadata,
                )

    def _clean_schema_properties(self, schema: dict) -> dict:
        """Remove unsupported fields from JSON schema for Anthropic API.

        Anthropic's API is strict about which JSON Schema fields are allowed.
        Fields like 'default' are not permitted.
        """
        if not isinstance(schema, dict):
            return schema

        # Fields allowed in Anthropic tool schemas
        allowed_property_fields = {"type", "description", "enum", "items", "properties", "required", "anyOf", "oneOf", "allOf"}

        cleaned = {}
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                # Clean each property
                cleaned_props = {}
                for prop_name, prop_value in value.items():
                    if isinstance(prop_value, dict):
                        cleaned_prop = {k: v for k, v in prop_value.items() if k in allowed_property_fields}
                        # Recursively clean nested structures
                        if "items" in cleaned_prop and isinstance(cleaned_prop["items"], dict):
                            cleaned_prop["items"] = self._clean_schema_properties(cleaned_prop["items"])
                        if "properties" in cleaned_prop:
                            cleaned_prop = self._clean_schema_properties(cleaned_prop)
                        cleaned_props[prop_name] = cleaned_prop
                    else:
                        cleaned_props[prop_name] = prop_value
                cleaned["properties"] = cleaned_props
            elif key == "items" and isinstance(value, dict):
                cleaned["items"] = self._clean_schema_properties(value)
            else:
                cleaned[key] = value

        return cleaned

    def _format_tools_for_anthropic(
        self,
        schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format.

        OpenAI format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Anthropic format:
        {"name": ..., "description": ..., "input_schema": ...}
        """
        formatted = []
        for schema in schemas:
            if "function" in schema:
                func = schema["function"]
                input_schema = func.get("parameters", {})
                # Clean the schema to remove unsupported fields
                input_schema = self._clean_schema_properties(input_schema)
                formatted.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": input_schema,
                })
            else:
                # Already in Anthropic format, but still clean the schema
                if "input_schema" in schema:
                    schema = schema.copy()
                    schema["input_schema"] = self._clean_schema_properties(schema["input_schema"])
                formatted.append(schema)
        return formatted

    def _format_tool_choice(
        self,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None,
    ) -> dict | None:
        """Format tool_choice for Anthropic API."""
        if tool_choice is None:
            return None

        mapping = {
            "auto": {"type": "auto"},
            "none": {"type": "none"},
            "any": {"type": "any"},
            "required": {"type": "any"},
        }
        if tool_choice in mapping:
            return mapping[tool_choice]

        # Specific tool name
        return {"type": "tool", "name": tool_choice}
