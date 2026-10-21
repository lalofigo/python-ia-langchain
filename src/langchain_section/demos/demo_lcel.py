from src.langchain_section.core.llm import get_llm
#from src.langchain_section.config.settings import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

llm = get_llm(0.2)

def demo_simple_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un experto en {tema}. Responde en una forma precisa, maximo una linea."),
            ("human", "{pregunta}")
        ]
    )

    parser = StrOutputParser()

    #Cadena simple
    chain = prompt | llm | parser

    response = chain.invoke({
        "tema": "amor", 
        "pregunta": "Escribe una frase o pensamiento corto para conquistar a un hombre gay."
    })

    print("Response from simple chain:")
    print(response)
    print("--"*30)

def demo_steps_inspection() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente tecnico"),
        ("human", "{pregunta}")
    ])

    parser = StrOutputParser()

    input_message = {"pregunta": "¿Qué es un decorador en phyton?"}

    #Paso 1
    messages = prompt.invoke(input_message)
    print("paso 1 - Prompt Output")
    print(f"Tipo: {type(messages).__name__}")
    print(f"Mensajes: {messages.messages}")

    #Paso 2
    ai_message = llm.invoke(messages)
    print("paso 2 - LLM Output")
    print(f"Tipo: {type(ai_message).__name__}")
    print(f"AI message: {ai_message}")
    print(f"Contenido: {ai_message.content[:50]}")

    #Paso 3:Parser extraer texto
    text=parser.invoke(ai_message)
    print("paso 3 - parser Output")
    print(f"Tipo: {type(text).__name__}")
    print(f"Texto: {text[:50]}")

def demo_batch() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Clasifica a la cantante en buena, mala, o pesima."),
        ("human", "{pregunta}")
    ])

    chain = prompt | llm | StrOutputParser()
    inputs = [
        {"pregunta": "Clasifica a la cantante La Tigresa del Oriente."},
        {"pregunta": "Clasifica a la cantante Madonna."},
        {"pregunta": "Clasifica a la cantante Britney Spears."},
        {"pregunta": "Clasifica a la cantante Lindsay Lohan."}
    ]

    results = chain.batch(inputs)

    print("Batch results:")
    for i, result in enumerate(results):
        print(f"Input: {inputs[i]['pregunta']}")
        print(f"Output: {result}")
        print("--"*30)

def demo_streaming() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explicame de manera sencilla conceptos tecnicos."),
        ("human", "En un parrafo sencillo explica que es: {concepto}")
    ])

    parser = StrOutputParser()

    chain = prompt | llm | parser

    print("Streaming response:")
    print("IA: ", end="", flush=True)

    for chunk in chain.stream({"concepto": "tokens en IA"}):
        print(chunk, end="", flush=True)

    print()

def demo_passthrough() -> None:
    def search_context(query: str) -> str:
        # Simula una búsqueda en una base de datos o motor de búsqueda
        contexts = {
            "Python": "Python es un lenguaje de programación interpretado, de alto nivel y propósito general.",
            "Decorador": "Un decorador en Python es una función que modifica el comportamiento de otra función.",
            "IA": "La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas."
        }

        for keyword, context in contexts.items():
            if keyword.lower() in query:
                return context
        return "No se encontró contexto relevante."
    retriever = RunnableLambda(search_context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responde usando esye contexto: {contexto}"),
        ("human", "Dame información sobre: {query}")
    ])

    chain = (
        { "contexto": retriever, "query": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"Que es un Decorador"})
    print("Response from passthrough chain:")
    print(response)
    print()

if __name__ == "__main__":
    print("--"*30)
    print("Running demo_simple_chain()...")
    print("--"*30)
    #demo_simple_chain()
    #demo_steps_inspection()
    #demo_batch()
    #demo_streaming()
    demo_passthrough()
    