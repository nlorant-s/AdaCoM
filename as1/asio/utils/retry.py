import asyncio
import re
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


# ---------------------------------------------------------------------------
# Direct Anthropic (Messages API) thinking configuration
# ---------------------------------------------------------------------------
# Verified against https://platform.claude.com/docs/en/build-with-claude/
# {extended-thinking,thinking,effort} (docs.claude.com redirects there):
#
#  * Manual extended thinking is `thinking={"type": "enabled",
#    "budget_tokens": N}` with N >= 1024 and N < max_tokens. It is the *only*
#    thinking mode on Claude 4.5 and earlier — including claude-haiku-4-5,
#    our dev agent.
#  * Adaptive thinking is `thinking={"type": "adaptive"}`, with depth steered
#    by `output_config={"effort": ...}` instead of a token budget. Claude 4.7
#    and later — including claude-sonnet-5, the experiment agent — reject
#    `type: "enabled"` with a 400.
#  * Claude 4.6 accepts both; the docs recommend adaptive there.
#  * Sampling: on Claude 4.7+ any non-default temperature/top_p/top_k is a 400
#    on every request. On older models the restriction applies only while
#    thinking is on (temperature and top_k incompatible, top_p allowed in
#    [0.95, 1]). So a thinking-enabled Claude agent sends no sampling params
#    at all — see get_anthropic_sampling_kwargs.
MIN_THINKING_BUDGET_TOKENS = 1024
DEFAULT_THINKING_BUDGET_TOKENS = 4096
ANTHROPIC_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# thinking mode by version: >= this uses adaptive, >= 3.7 uses budget.
_ADAPTIVE_FROM = (4, 6)
_BUDGET_FROM = (3, 7)
# Non-default sampling params are rejected outright from this version on.
_NO_SAMPLING_FROM = (4, 7)
# Model families released after the 4.x line; they are all adaptive-only.
_POST_4X_FAMILIES = ("fable", "mythos")


def parse_claude_version(model_name: str):
    """Best-effort (major, minor) for a Claude model id, or None.

    Handles both orderings used by Anthropic ids: ``claude-sonnet-4-6``,
    ``claude-sonnet-5``, ``claude-haiku-4-5``, ``claude-3-7-sonnet-20250219``,
    ``claude-opus-4-1-20250805``. Trailing date stamps are ignored.
    """
    name = (model_name or "").lower()
    numbers = []
    for part in re.split(r"[^0-9]+", name):
        if part:
            numbers.append(int(part))
    numbers = [n for n in numbers if n < 100]  # drop 20250219-style stamps
    if not numbers:
        return None
    major = numbers[0]
    minor = numbers[1] if len(numbers) > 1 else 0
    return (major, minor)


def classify_anthropic_thinking_mode(model_name: str) -> str:
    """Return "adaptive", "budget" or "unsupported" for a direct-Anthropic id."""
    name = (model_name or "").lower()
    if any(f in name for f in _POST_4X_FAMILIES):
        return "adaptive"
    version = parse_claude_version(name)
    if version is None:
        # Unknown id: assume it is newer than what we know about. "enabled"
        # 400s on new models, "adaptive" is the forward-compatible guess.
        return "adaptive"
    if version >= _ADAPTIVE_FROM:
        return "adaptive"
    if version >= _BUDGET_FROM:
        return "budget"
    return "unsupported"


def anthropic_rejects_sampling_params(model_name: str) -> bool:
    """True if the model 400s on any non-default temperature/top_p/top_k."""
    name = (model_name or "").lower()
    if any(f in name for f in _POST_4X_FAMILIES):
        return True
    version = parse_claude_version(name)
    if version is None:
        return True
    return version >= _NO_SAMPLING_FROM


def _is_direct_anthropic(model) -> bool:
    """True for the direct Anthropic Messages API client.

    Deliberately class-based, not name-based: an OpenAI-compatible relay
    serving a claude-* model is detected as provider "anthropic" for message
    formatting, but its request kwargs are OpenAI-shaped, so a top-level
    `thinking` would be wrong there. Excludes the DashScope proxy, which takes
    thinking through extra_body.
    """
    model_class_name = type(model).__name__.lower()
    if "dashscope" in model_class_name:
        return False
    return "anthropic" in model_class_name or "bedrock" in model_class_name


def get_anthropic_thinking_kwargs(model_name: str, enable_thinking: bool = True,
                                  thinking_config: Optional[dict] = None) -> dict:
    """Request kwargs enabling thinking on the direct Anthropic client.

    ``thinking_config`` (from the run config) may carry:
      - ``mode``: "auto" (default), "adaptive", "budget" or "off"
      - ``budget_tokens``: budget-mode thinking budget (default 4096, min 1024)
      - ``effort``: adaptive-mode effort level; omitted unless set, since the
        API default ("high") is what omitting the parameter means.
    """
    cfg = dict(thinking_config or {})
    if not enable_thinking or cfg.get("mode") == "off":
        return {}

    mode = cfg.get("mode", "auto")
    if mode in (None, "auto"):
        mode = classify_anthropic_thinking_mode(model_name)
    if mode == "unsupported":
        return {}

    if mode == "adaptive":
        kwargs = {"thinking": {"type": "adaptive"}}
        effort = cfg.get("effort")
        if effort:
            effort = str(effort).lower()
            if effort not in ANTHROPIC_EFFORT_LEVELS:
                raise ValueError(
                    f"Invalid effort {effort!r}; expected one of {ANTHROPIC_EFFORT_LEVELS}"
                )
            kwargs["output_config"] = {"effort": effort}
        return kwargs

    if mode == "budget":
        budget = int(cfg.get("budget_tokens", DEFAULT_THINKING_BUDGET_TOKENS))
        if budget < MIN_THINKING_BUDGET_TOKENS:
            raise ValueError(
                f"budget_tokens={budget} is below the API minimum "
                f"{MIN_THINKING_BUDGET_TOKENS}"
            )
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    raise ValueError(f"Unknown thinking mode {mode!r}")


def get_anthropic_sampling_kwargs(model_name: str, temperature: float = 0,
                                  enable_thinking: bool = True) -> dict:
    """Sampling kwargs that a Claude model will actually accept.

    Empty whenever thinking is on (temperature and top_k are incompatible with
    thinking) or the model rejects non-default sampling outright (4.7+).
    Consequence for us: a thinking-enabled agent cannot be pinned to
    temperature 0, so rollout variance is higher than the paper's setup.
    """
    if enable_thinking or anthropic_rejects_sampling_params(model_name):
        return {}
    # top_p is never sent alongside temperature: Claude rejects both together.
    return {"temperature": temperature}


def get_thinking_kwargs(model, enable_thinking: bool = True,
                        thinking_config: Optional[dict] = None) -> dict:
    """Get thinking/reasoning kwargs based on model type.

    Returns extra kwargs to enable thinking for models that support it.
    For models that don't support thinking, returns empty dict.

    Supported models:
    - qwen3-max: extra_body={"enable_thinking": True}
    - doubao-*: extra_body={"enable_thinking": True}
    - DashScopeClaudeChatModel: extra_body={"thinking": {"type": "enabled", "budget_tokens": 4096}}
    - AnthropicChatModel (direct Messages API): top-level `thinking`, either
      {"type": "enabled", "budget_tokens": N} or {"type": "adaptive"} plus an
      optional output_config.effort — see get_anthropic_thinking_kwargs.
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

    # Direct Anthropic client: kwargs go straight into messages.create().
    if _is_direct_anthropic(model):
        return get_anthropic_thinking_kwargs(model_name, enable_thinking, thinking_config)

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

    # agentscope's AnthropicChatModel talks to the Messages API directly, so
    # request kwargs (thinking, output_config) are top-level, not extra_body.
    if "anthropicchatmodel" in model_class_name:
        return "anthropic"

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
