import asyncio
import time
import traceback
from collections.abc import AsyncIterable
from typing import Any, Callable, Optional, Literal
from ..logger import ExperimentLogger
from .response_shape import normalize_response_content
import random


def convert_tools_openai_to_anthropic(tools: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style tool schemas to Anthropic-style tool schemas.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Anthropic format:
    [{"name": "...", "description": "...", "input_schema": {...}}]

    Args:
        tools: List of tools in OpenAI format

    Returns:
        List of tools in Anthropic format
    """
    if not tools:
        return tools

    converted = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            # OpenAI format - convert to Anthropic format
            func = tool["function"]
            converted.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
        elif "name" in tool and "input_schema" in tool:
            # Already in Anthropic format
            converted.append(tool)
        else:
            # Unknown format, pass through
            converted.append(tool)

    return converted


def get_thinking_kwargs(model, enable_thinking: bool = True) -> dict:
    """Get thinking/reasoning kwargs based on model type.

    Returns extra kwargs to enable thinking for models that support it.
    For models that don't support thinking, returns empty dict.

    Supported models:
    - qwen3-max: extra_body={"enable_thinking": True}
    - doubao-*: extra_body={"enable_thinking": True}
    - DashScopeClaudeChatModel: extra_body={"thinking": {"type": "enabled", "budget_tokens": 4096}}
    - GeminiChatModel: thinking_config set at init time (not here)
    """
    model_class_name = type(model).__name__.lower()
    model_name = (getattr(model, "model_name", "") or "").lower()

    # DashScope Claude: thinking via extra_body. Dashscope proxy hangs silently
    # when `thinking` is present for models that don't natively support it (and
    # even type=disabled triggers hangs for opus). Whitelist only the claude
    # models known to accept thinking; for everything else omit the key.
    if "dashscopeclaude" in model_class_name:
        if not enable_thinking:
            return {}
        thinking_capable = ("claude-sonnet-4" in model_name or "claude-3-7-sonnet" in model_name)
        if not thinking_capable:
            return {}
        return {"extra_body": {
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }}

    # OpenAI-compatible models with enable_thinking support
    thinking_models = ["qwen3-max", "doubao"]
    if any(t in model_name for t in thinking_models):
        return {"extra_body": {"enable_thinking": enable_thinking}}

    return {}


def detect_model_provider(model) -> Literal["openai", "anthropic", "unknown"]:
    """
    Detect the model provider from the model object.

    This function determines the API format used for requests, which affects
    message formatting (e.g., whether role="system" is allowed).

    Args:
        model: The model object (AgentScope model)

    Returns:
        Provider name: "openai", "anthropic", or "unknown"
    """
    # Check model class name
    model_class_name = type(model).__name__.lower()

    # DashScopeClaudeChatModel uses OpenAI-compatible endpoint for requests
    # so it should be treated as "openai" for request formatting purposes
    if "dashscopeclaude" in model_class_name:
        return "openai"

    if "anthropic" in model_class_name or "claude" in model_class_name or "bedrock" in model_class_name:
        return "anthropic"

    # Check model_name attribute if available
    model_name = getattr(model, "model_name", "") or ""
    model_name_lower = model_name.lower()

    # Skip claude/anthropic detection if it's a DashScope model
    # (DashScope models use OpenAI-compatible request format)
    if hasattr(model, "provider") or "dashscope" in model_class_name:
        return "openai"

    if "claude" in model_name_lower or "anthropic" in model_name_lower:
        return "anthropic"

    # Check for Bedrock by looking at client or base_url
    if hasattr(model, "client") and model.client:
        client_class = type(model.client).__name__.lower()
        if "bedrock" in client_class:
            return "anthropic"

    if hasattr(model, "base_url"):
        base_url = str(getattr(model, "base_url", "") or "")
        if "bedrock" in base_url.lower() or "anthropic" in base_url.lower():
            return "anthropic"

    return "openai"


async def retry_model_call(
    model_func: Callable,
    *args,
    max_retries: int = 5,
    sleep_seconds: int = 10,
    experiment_logger: Optional[ExperimentLogger] = None,
    **kwargs
) -> Any:
    """
    Retry function for model calls that may fail.
    
    Args:
        model_func: The model function to call (e.g., self.model)
        *args: Arguments to pass to the model function
        max_retries: Maximum number of retry attempts (default: 3)
        sleep_seconds: Number of seconds to sleep between retries (default: 10)
        experiment_logger: Logger instance for tracking retry attempts
        **kwargs: Keyword arguments to pass to the model function
        
    Returns:
        The result from the successful model call
        
    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if experiment_logger:
                experiment_logger.log_debug(f"Model call attempt {attempt + 1}/{max_retries + 1}")
            
            # Call the model function
            result = await model_func(*args, **kwargs)

            # If the model returned an async generator (stream mode), consume
            # it and keep the last (cumulative) ChatResponse.
            # isinstance avoids triggering __getattr__ on Msg-like dict objects
            # where missing-key lookup raises KeyError (hasattr only catches AttributeError).
            if isinstance(result, AsyncIterable):
                last = None
                async for chunk in result:
                    last = chunk
                result = last

            # Apply all provider-specific response shape normalization in one
            # place (see asio.utils.response_shape) so every worker sees
            # OpenAI-canonical content and never has to post-process on its
            # own. Covers kimi id collisions, minimax XML tool calls, gpt-oss
            # text tool calls, channel markers, thinking blocks.
            if result is not None and hasattr(result, "content"):
                result.content = normalize_response_content(
                    result.content, logger=experiment_logger
                )

            if experiment_logger:
                experiment_logger.log_debug(f"Model call succeeded on attempt {attempt + 1}")

            return result
            
        except Exception as e:
            last_exception = e
            tb = traceback.format_exc()
            
            # Check if this is a non-retryable error.
            # Note: some OpenAI-compatible relays return a generic 403
            # `bad_response_status_code` for transient upstream failures.
            # Treat only clearly permanent 403s (e.g. moderation / auth) as
            # non-retryable; otherwise allow the normal retry loop to handle
            # them.
            error_str = str(e)
            error_str_lower = error_str.lower()

            non_retryable_errors = [
                "data_inspection_failed",  # DashScope content moderation
                "Error code: 400",         # Bad request errors
                "'max_tokens' or 'max_completion_tokens' is too large",
            ]

            non_retryable_403_markers = [
                "data_inspection_failed",
                "inappropriate content",
                "invalid_api_key",
                "insufficient_permissions",
                "permission denied",
            ]

            is_non_retryable = any(error_marker in error_str for error_marker in non_retryable_errors)
            if not is_non_retryable and "Error code: 403" in error_str:
                is_non_retryable = any(marker in error_str_lower for marker in non_retryable_403_markers)
            
            if is_non_retryable:
                if experiment_logger:
                    experiment_logger.log_error(
                        f"Non-retryable error encountered: {e}\n"
                        f"Failing immediately without retry.\n"
                        f"Traceback: {tb}"
                    )
                # Raise immediately for non-retryable errors
                raise
            
            if experiment_logger:
                experiment_logger.log_warning(
                    f"Model call failed on attempt {attempt + 1}/{max_retries + 1}. error: {e}\n"
                    # f"Traceback: {tb}\n"
                    # f"Args: {args}\n"
                    # f"Kwargs: {kwargs}\n"
                )
            
            # Don't sleep after the last attempt
            if attempt < max_retries:
                tmp_sleep_seconds = random.uniform(0, sleep_seconds)
                if experiment_logger:
                    experiment_logger.log_debug(f"Sleeping for {tmp_sleep_seconds:.2f} seconds before retry...")
                await asyncio.sleep(tmp_sleep_seconds)
    
    # All retries failed, log final failure and raise the last exception
    if experiment_logger:
        experiment_logger.log_error(
            f"Model call failed after {max_retries + 1} attempts. "
            f"Model function: {model_func}\n"
            f"Args: {args}\n"
            f"Kwargs: {kwargs}\n"
            f"Final error: {last_exception}\n"
            f"Traceback: {traceback.format_exc()}"
        )
    
    raise last_exception
