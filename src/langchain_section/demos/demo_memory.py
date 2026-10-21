
from langchain_core.runnables import RunnableWithMessageHistory
from src.langchain_section.memory.base import BaseMemoryBackend
from src.langchain_section.chains.base import build_assistant_chain
from src.langchain_section.memory.sqlite_memory import SQLiteMemoryBackend


def build_chatbot(backend: BaseMemoryBackend) -> RunnableWithMessageHistory:
    """Builds a chatbot with memory backend."""
    chain = build_assistant_chain()
    return RunnableWithMessageHistory(
        input_messages_key="input", 
        history_messages_key="history",
        runnable=chain,
        get_session_history=backend.get_history
    )

def run_chat_session(
        chatbot: RunnableWithMessageHistory,
        backend: BaseMemoryBackend,
        session_id: str
) -> None:
    print(f"Starting chat session with ID: {session_id}")

    messages = backend.get_history(session_id).messages

    if messages:
        print("Previous conversation history ===>")
        for message in messages:
            print(f"{message.type}: {message.content}")
    else:
        print("No previous conversation history.")
    
    print("Command: 'historial' to view history, 'limpiar' to clear history, 'salir' to exit.")

    while True:
        try:
            user_input = input("🧑🏽‍💻 You: ")
            if user_input.lower() == "salir":
                print("🚶🏽 Exiting chat session.")
                break
            elif user_input.lower() == "historial":
                history = backend.get_history(session_id).messages
                print("📚 Conversation history:")
                for message in history:
                    print(f"{message.type}: {message.content[:70]}")
                continue
            elif user_input.lower() == "limpiar":
                backend.clear_history(session_id)
                print("🆑 Conversation history cleared.")
                continue
            elif user_input.lower() == "sesiones":
                sessions = backend.list_sessions()
                print("🗂️ Active sessions:")
                for s in sessions:
                    print(f"- {s}")
                continue

            response = chatbot.invoke(
                {"input": user_input}, 
                config={"configurable": {"session_id": session_id}}
            )
            print(f"🤖 Bot: {response}")
        except KeyboardInterrupt:
            print("\n🚶🏽 Exiting chat session.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    print("=== 💎Chatbot with Memory Backend Demo💎 ===")

    print()
    print("Select memory backend:")
    print("1. SQLite")
    backend_choice = input("Enter choice (1): ").strip()

    if backend_choice == "1":
        backend = SQLiteMemoryBackend("./data/chat_history.db")
        print(f"💿 Using SQLite backend with DB path: {backend.db_path}")
    else:  
        print("❌ Invalid choice. Exiting.")
        exit(1)

    chatbot = build_chatbot(backend)
    run_chat_session(chatbot, backend, session_id="default_session")