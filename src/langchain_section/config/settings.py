import os
from dotenv import load_dotenv

load_dotenv()

class LangChainSettings:
    """Class to hold LangChain settings."""

    CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    DEFAULT_TEMPERATURE: float = 0.7
    LOW_TEMPERATURE: float = 0.3
    MAX_RETRIES: int = 3

    SQLITE_DB_PATH: str = "data/chat_history.db"
    CHROMA_PATH: str = "./data/langchain_chroma"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5

    COST_INPUT_PER_MILLION: float = 0.15
    COST_OUTPUT_PER_MILLION: float = 0.60

    @classmethod
    def validate(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
settings = LangChainSettings()

