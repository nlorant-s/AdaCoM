from abc import ABC, abstractmethod
from typing import Any, List
from agentscope.message import Msg, ToolUseBlock, TextBlock, ContentBlock

class BaseMemory(ABC):
    """Base class for memory."""
    def __init__(self):
        pass






    @abstractmethod
    def retrieve(self, uid: str, query: str, **kwargs) -> Any|bool:
        """retrieve memory"""
        pass

    @abstractmethod
    def add_memory(self, uid: str, content: List[Msg], **kwargs) -> Any:
        """Save content to memory."""
        pass

    @abstractmethod
    def process_content(self, uid: str, content: List[Msg]|Msg):
        """extract info in content for memory."""
        pass

    @abstractmethod
    def delete(self, uid: str, key: Any) -> None:
        """Delete part of memory by some criteria."""
        pass

    @abstractmethod
    def clear_memory(self, uid: str,) -> None:
        """Clear all memory."""
        pass


    @abstractmethod
    def show_all_memory(self, uid: str) -> Any:
        """Show all memory."""

