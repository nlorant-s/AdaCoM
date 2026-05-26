# -*- coding: utf-8 -*-
"""LLM hooks for automatic logging and monitoring."""

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


def auto_llm_logging_hook(
    model_instance: Any,
    response: Any,
    call_info: Dict[str, Any],
    logger_instance: Optional[Any] = None,
) -> None:
    """
    Automatic LLM logging hook that logs every LLM call.
    
    This hook is designed to be registered as a post_llm_call hook
    and will automatically log all LLM invocations using the provided
    logger instance or a default logging mechanism.
    
    Args:
        model_instance: The model instance that made the call
        response: The response from the LLM
        call_info: Dictionary containing original call arguments
        logger_instance: Optional logger instance to use for logging
    """
    if logger_instance is not None:
        # Use provided logger (e.g., ExperimentLogger)
            args = call_info.get('args', ())
            kwargs = call_info.get('kwargs', {})
            
            # Extract messages from args
            messages = args[0] if args else kwargs.get('messages', [])
            
            # Log the LLM invocation
            logger_instance.log_llm_invocation(
                model_class=model_instance.__class__.__name__,
                model_name=model_instance.model_name,
                messages=messages,
                response=response,
                response_type=str(type(response)),  # Add debug info
                **{k: v for k, v in kwargs.items() if k != 'messages'}
            )


def _fallback_log_llm_call(
    model_instance: Any,
    response: Any,
    call_info: Dict[str, Any],
    log_dir: str = "./llm_logs"
) -> None:
    """
    Fallback logging mechanism when no logger is provided.
    
    Args:
        model_instance: The model instance
        response: The LLM response
        call_info: Call information
        log_dir: Directory to save logs
    """
    try:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"llm_call_{model_instance.__class__.__name__}_{timestamp}.json"
        filepath = log_path / filename
        
        # Prepare log data
        args = call_info.get('args', ())
        kwargs = call_info.get('kwargs', {})
        messages = args[0] if args else kwargs.get('messages', [])
        
        log_data = {
            "timestamp": timestamp,
            "model_class": model_instance.__class__.__name__,
            "model_name": model_instance.model_name,
            "messages": messages,
            "arguments": {k: v for k, v in kwargs.items() if k != 'messages'},
            "response": _serialize_response(response)
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"LLM call logged to: {filepath}")
        
    except Exception as e:
        print(f"Warning: Fallback LLM logging failed: {e}")


def _serialize_response(response: Any) -> Any:
    """
    Serialize response object to JSON-compatible format.
    
    Args:
        response: Response object from LLM
        
    Returns:
        JSON-serializable representation
    """
    try:
        if response is None:
            return None
        elif isinstance(response, (str, int, float, bool)):
            return response
        elif isinstance(response, dict):
            # This handles ChatResponse (which inherits from dict) and regular dicts
            result = {}
            for k, v in response.items():
                try:
                    result[k] = _serialize_response(v)
                except Exception:
                    result[k] = str(v)
            return result
        elif isinstance(response, (list, tuple)):
            return [_serialize_response(item) for item in response]
        elif hasattr(response, '__dict__'):
            # Convert object to dictionary, including all attributes
            result = {}
            for k, v in response.__dict__.items():
                if not k.startswith('_'):
                    try:
                        result[k] = _serialize_response(v)
                    except Exception:
                        # If serialization fails for an attribute, convert to string
                        result[k] = str(v)
            # Also try to get common attributes directly
            for attr in ['content', 'message', 'text', 'choices', 'usage', 'metadata']:
                if hasattr(response, attr) and attr not in result:
                    try:
                        result[attr] = _serialize_response(getattr(response, attr))
                    except Exception:
                        result[attr] = str(getattr(response, attr))
            return result
        else:
            return str(response)
    except Exception as e:
        # If all else fails, return a string representation with debug info
        return {
            "serialization_error": str(e),
            "response_type": str(type(response)),
            "response_str": str(response)
        }


def create_auto_logging_hook(logger_instance: Optional[Any] = None):
    """
    Create an auto-logging hook with a specific logger instance.
    
    Args:
        logger_instance: Optional logger instance (e.g., ExperimentLogger)
        
    Returns:
        A configured auto-logging hook function
    """
    def hook(model_instance: Any, response: Any, call_info: Dict[str, Any]) -> None:
        auto_llm_logging_hook(model_instance, response, call_info, logger_instance)
    
    return hook


def enable_auto_llm_logging(
    models: list = None,
    logger_instance: Optional[Any] = None,
    hook_name: str = "auto_llm_logging"
) -> None:
    """
    Enable automatic LLM logging for specified model classes.

    Args:
        models: List of model classes to enable logging for.
               If None, enables for OpenAIChatModel, DashScopeChatModel, and DashScopeClaudeChatModel.
        logger_instance: Optional logger instance to use
        hook_name: Name for the hook registration
    """
    from ._openai_model import OpenAIChatModel
    from ._dashscope_model import DashScopeChatModel
    from ._dashscope_claude_model import DashScopeClaudeChatModel
    from ._gemini_model import GeminiChatModel
    from ._anthropic_model import AnthropicChatModel
    from ._ollama_model import OllamaChatModel

    if models is None:
        models = [
            OpenAIChatModel, DashScopeChatModel, DashScopeClaudeChatModel,
            GeminiChatModel, AnthropicChatModel, OllamaChatModel,
        ]

    hook = create_auto_logging_hook(logger_instance)

    for model_class in models:
        model_class.register_class_hook(
            "post_llm_call",
            hook_name,
            hook
        )


def disable_auto_llm_logging(
    models: list = None,
    hook_name: str = "auto_llm_logging"
) -> None:
    """
    Disable automatic LLM logging for specified model classes.

    Args:
        models: List of model classes to disable logging for.
               If None, disables for all known model classes.
        hook_name: Name of the hook to remove
    """
    from ._openai_model import OpenAIChatModel
    from ._dashscope_model import DashScopeChatModel
    from ._dashscope_claude_model import DashScopeClaudeChatModel
    from ._gemini_model import GeminiChatModel
    from ._anthropic_model import AnthropicChatModel
    from ._ollama_model import OllamaChatModel

    if models is None:
        models = [
            OpenAIChatModel, DashScopeChatModel, DashScopeClaudeChatModel,
            GeminiChatModel, AnthropicChatModel, OllamaChatModel,
        ]
    
    for model_class in models:
        try:
            model_class.remove_class_hook("post_llm_call", hook_name)
        except ValueError:
            # Hook doesn't exist, ignore
            pass
    
    # print(f"Auto LLM logging disabled for {[m.__name__ for m in models]}")