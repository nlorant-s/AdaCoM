# -*- coding: utf-8 -*-
"""The chat model base class."""

from abc import abstractmethod
from typing import AsyncGenerator, Any, Callable
from collections import OrderedDict

from ._model_response import ChatResponse

TOOL_CHOICE_MODES = ["auto", "none", "any", "required"]


class ChatModelBase:
    """Base class for chat models."""

    model_name: str
    """The model name"""

    stream: bool
    """Is the model output streaming or not"""

    _class_pre_llm_call_hooks: dict[str, Callable] = OrderedDict()
    """Class-level hooks called before LLM invocation"""

    _class_post_llm_call_hooks: dict[str, Callable] = OrderedDict()
    """Class-level hooks called after LLM invocation"""

    def __init__(
        self,
        model_name: str,
        stream: bool,
    ) -> None:
        """Initialize the chat model base class.

        Args:
            model_name (`str`):
                The name of the model
            stream (`bool`):
                Whether the model output is streaming or not
        """
        self.model_name = model_name
        self.stream = stream
        
        # Initialize instance-level hooks
        self._instance_pre_llm_call_hooks = OrderedDict()
        self._instance_post_llm_call_hooks = OrderedDict()

    @abstractmethod
    async def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        pass

    def _validate_tool_choice(
        self,
        tool_choice: str,
        tools: list[dict] | None,
    ) -> None:
        """
        Validate tool_choice parameter.

        Args:
            tool_choice (`str`):
                Tool choice mode or function name
            tools (`list[dict] | None`):
                Available tools list
        Raises:
            TypeError: If tool_choice is not string
            ValueError: If tool_choice is invalid
        """
        if not isinstance(tool_choice, str):
            raise TypeError(
                f"tool_choice must be str, got {type(tool_choice)}",
            )
        if tool_choice in TOOL_CHOICE_MODES:
            return

        available_functions = [tool["function"]["name"] for tool in tools]

        if tool_choice not in available_functions:
            all_options = TOOL_CHOICE_MODES + available_functions
            raise ValueError(
                f"Invalid tool_choice '{tool_choice}'. "
                f"Available options: {', '.join(sorted(all_options))}",
            )

    def _call_pre_llm_hooks(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call pre-LLM hooks and return modified kwargs.
        
        Args:
            *args: Positional arguments for the LLM call
            **kwargs: Keyword arguments for the LLM call
            
        Returns:
            Modified kwargs after processing through hooks
        """
        modified_kwargs = dict(kwargs)
        modified_kwargs['args'] = args
        
        # Call class-level hooks first
        for hook in self._class_pre_llm_call_hooks.values():
            result = hook(self, modified_kwargs)
            if result is not None:
                modified_kwargs.update(result)
        
        # Call instance-level hooks
        for hook in self._instance_pre_llm_call_hooks.values():
            result = hook(self, modified_kwargs)
            if result is not None:
                modified_kwargs.update(result)
                
        return modified_kwargs

    def _call_post_llm_hooks(
        self, 
        response: Any, 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """Call post-LLM hooks and return potentially modified response.
        
        Args:
            response: The LLM response
            *args: Original positional arguments
            **kwargs: Original keyword arguments
            
        Returns:
            Potentially modified response
        """
        modified_response = response
        call_info = {'args': args, 'kwargs': kwargs}
        
        # Call class-level hooks first
        for hook in self._class_post_llm_call_hooks.values():
            result = hook(self, modified_response, call_info)
            if result is not None:
                modified_response = result
        
        # Call instance-level hooks
        for hook in self._instance_post_llm_call_hooks.values():
            result = hook(self, modified_response, call_info)
            if result is not None:
                modified_response = result
                
        return modified_response

    @classmethod
    def register_class_hook(
        cls,
        hook_type: str,
        hook_name: str,
        hook: Callable,
    ) -> None:
        """Register a class-level hook.
        
        Args:
            hook_type: Either 'pre_llm_call' or 'post_llm_call'
            hook_name: Unique name for the hook
            hook: The hook function
        """
        if hook_type not in ['pre_llm_call', 'post_llm_call']:
            raise ValueError(f"Invalid hook type: {hook_type}")
            
        hooks = getattr(cls, f"_class_{hook_type}_hooks")
        hooks[hook_name] = hook

    @classmethod
    def remove_class_hook(cls, hook_type: str, hook_name: str) -> None:
        """Remove a class-level hook.
        
        Args:
            hook_type: Either 'pre_llm_call' or 'post_llm_call'
            hook_name: Name of the hook to remove
        """
        if hook_type not in ['pre_llm_call', 'post_llm_call']:
            raise ValueError(f"Invalid hook type: {hook_type}")
            
        hooks = getattr(cls, f"_class_{hook_type}_hooks")
        if hook_name in hooks:
            del hooks[hook_name]
        else:
            raise ValueError(f"Hook '{hook_name}' not found")

    def register_instance_hook(
        self,
        hook_type: str,
        hook_name: str,
        hook: Callable,
    ) -> None:
        """Register an instance-level hook.
        
        Args:
            hook_type: Either 'pre_llm_call' or 'post_llm_call'
            hook_name: Unique name for the hook
            hook: The hook function
        """
        if hook_type not in ['pre_llm_call', 'post_llm_call']:
            raise ValueError(f"Invalid hook type: {hook_type}")
            
        hooks = getattr(self, f"_instance_{hook_type}_hooks")
        hooks[hook_name] = hook

    def remove_instance_hook(self, hook_type: str, hook_name: str) -> None:
        """Remove an instance-level hook.
        
        Args:
            hook_type: Either 'pre_llm_call' or 'post_llm_call'
            hook_name: Name of the hook to remove
        """
        if hook_type not in ['pre_llm_call', 'post_llm_call']:
            raise ValueError(f"Invalid hook type: {hook_type}")
            
        hooks = getattr(self, f"_instance_{hook_type}_hooks")
        if hook_name in hooks:
            del hooks[hook_name]
        else:
            raise ValueError(f"Hook '{hook_name}' not found")
