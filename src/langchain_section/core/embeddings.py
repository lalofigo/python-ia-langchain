
import os
from langchain_chroma import Chroma
from src.langchain_section.config.settings import settings
from src.langchain_section.core.llm import get_embeddings

def get_or_create_vectorstore(
        collection_name: str,
        persist_path: str = None,
) -> Chroma:
    """Get or create a Chroma vectorstore for embeddings."""
    persist_path = persist_path or settings.CHROMA_PATH
    os.makedirs(persist_path, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_path,
        embedding_function=get_embeddings(),
    )