"""Reader of the buffer."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BufferReader(ABC):
    """Interface of the buffer reader."""

    @abstractmethod
    def read(
        self, batch_size: Optional[int] = None, min_version: Optional[int] = None
    ) -> List:
        """Read from buffer. If ``min_version`` is set, drop experiences with older model_version."""

    @abstractmethod
    async def read_async(
        self, batch_size: Optional[int] = None, min_version: Optional[int] = None
    ) -> List:
        """Read from buffer asynchronously. If ``min_version`` is set, drop experiences with older model_version."""

    def __len__(self) -> int:
        """Get the number of samples in buffer."""
        raise NotImplementedError

    def state_dict(self) -> Dict:
        """Return the state of the reader as a dict.
        Returns:
            A dict containing the reader state. At minimum, it should contain
            the `current_index` field.
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict) -> None:
        raise NotImplementedError
