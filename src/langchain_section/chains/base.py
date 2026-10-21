
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from src.langchain_section.config.settings import settings
from src.langchain_section.core.llm import get_llm

def build_assistant_chain(system_prompt: str = None) -> Runnable:
    """Builds a chain for the assistant with optional system prompt."""
    # Create the system message
    default_system = """Eres un asistente técnico experto en Python e IA. 
    Tienes acceso al historial competo de la conversación y a la información de contexto del usuario.
    Usalo para dar respuestas precisas y útiles. Responde en español."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or default_system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | get_llm() | StrOutputParser()