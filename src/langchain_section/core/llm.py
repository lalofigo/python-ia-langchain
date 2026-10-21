from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from src.langchain_section.config.settings import settings

@lru_cache(maxsize=1)
def get_llm(temperature: float = None) -> ChatOpenAI:
    """Get a cached instance of the ChatOpenAI model."""
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        temperature=temperature or settings.DEFAULT_TEMPERATURE,
        max_retries=settings.MAX_RETRIES
    )

@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Get a cached instance of the OpenAIEmbeddings model."""
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)