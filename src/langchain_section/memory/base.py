
from abc import ABC, abstractmethod
from langchain_core.chat_history import BaseChatMessageHistory

class BaseMemoryBackend(ABC):

    @abstractmethod
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieve the conversation history for a given session ID."""

    @abstractmethod
    def clear_history(self, session_id: str) -> None:
        """Clear the conversation history for a given session ID."""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all session IDs that have stored conversation histories."""