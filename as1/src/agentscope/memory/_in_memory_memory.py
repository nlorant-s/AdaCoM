# -*- coding: utf-8 -*-
"""The dialogue memory class"""

from typing import Union, Iterable, Any
import json
import os

from ._memory_base import MemoryBase
from ..message import Msg


class InMemoryMemory(MemoryBase):
    """The in-memory memory class for storing messages."""

    def __init__(
        self,
        debug_dir: str = None,
    ) -> None:
        """Initialize the in-memory memory object."""
        super().__init__()
        self.content: list[Msg] = []
        self.debug_dir = debug_dir

    def state_dict(self) -> dict:
        """Convert the current memory into JSON data format."""
        return {
            "content": [_.to_dict() for _ in self.content],
        }

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
    ) -> None:
        """Load the memory from JSON data.

        Args:
            state_dict (`dict`):
                The state dictionary to load, which should have a "content"
                field.
            strict (`bool`, defaults to `True`):
                If `True`, raises an error if any key in the module is not
                found in the state_dict. If `False`, skips missing keys.
        """
        self.content = []
        for data in state_dict["content"]:
            data.pop("type", None)
            self.content.append(Msg.from_dict(data))

    async def size(self) -> int:
        """The size of the memory."""
        return len(self.content)

    async def retrieve(self, *args: Any, **kwargs: Any) -> None:
        """Retrieve items from the memory."""
        raise NotImplementedError(
            "The retrieve method is not implemented in "
            f"{self.__class__.__name__} class.",
        )

    async def delete(self, index: Union[Iterable, int]) -> None:
        """Delete the specified item by index(es).

        Args:
            index (`Union[Iterable, int]`):
                The index to delete.
        """
        if isinstance(index, int):
            index = [index]

        invalid_index = [_ for _ in index if 0 > _ or _ >= len(self.content)]

        if invalid_index:
            raise IndexError(
                f"The index {invalid_index} does not exist.",
            )

        self.content = [
            _ for idx, _ in enumerate(self.content) if idx not in index
        ]

    async def add(
        self,
        memories: Union[list[Msg], Msg, None],
        allow_duplicates: bool = False,
    ) -> None:
        """Add message into the memory.

        Args:
            memories (`Union[list[Msg], Msg, None]`):
                The message to add.
            allow_duplicates (`bool`, defaults to `False`):
                If allow adding duplicate messages (with the same id) into
                the memory.
        """
        if memories is None:
            return

        if isinstance(memories, Msg):
            memories = [memories]

        if not isinstance(memories, list):
            raise TypeError(
                f"The memories should be a list of Msg or a single Msg, "
                f"but got {type(memories)}.",
            )

        for msg in memories:
            if not isinstance(msg, Msg):
                raise TypeError(
                    f"The memories should be a list of Msg or a single Msg, "
                    f"but got {type(msg)}.",
                )

        if not allow_duplicates:
            existing_ids = [_.id for _ in self.content]
            memories = [_ for _ in memories if _.id not in existing_ids]
        self.content.extend(memories)
        self._save_debug_history(memories)

    async def get_memory(self) -> list[Msg]:
        """Get the memory content."""
        return self.content

    async def clear(self) -> None:
        """Clear the memory content."""
        self.content = []

    def _output_format(self, item: Msg) -> dict:
        """Convert a message to output format similar to memorymanager.py."""
        if hasattr(item, 'to_dict'):
            return item.to_dict()
        else:
            return {
                "role": getattr(item, 'role', 'user'),
                "content": getattr(item, 'content', str(item)),
                "name": getattr(item, 'name', '')
            }

    def _save_debug_history(self, msgs: list[Msg], hint: str = None) -> None:
        """Save the chat history to memory.json file for debugging purposes."""
        if not self.debug_dir:
            return
        
        memory_file = os.path.join(self.debug_dir, "memory.json")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        with open(memory_file, "a", encoding="utf-8") as f:
            if hint:
                f.write(hint + '\n')
            if msgs:
                all_items = []
                for item in msgs:
                    all_items.append(self._output_format(item))
                f.write(json.dumps(all_items, indent=2, ensure_ascii=False))
            f.write("\n===========END===========\n")
