# main_agent.py  –  Agent RAG “agentique” avec mémoire par session
# ---------------------------------------------------------------------
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

from tools import retrieve_it , explain                   # tool décoré @tool
from prompt_agent import prompt_agent             # prompt décrivant le tool
from conversation_summary_memory import ConversationSummaryBufferMessageHistory


load_dotenv(Path(__file__).resolve().parent / ".env")


llm_agent       = ChatOpenAI(model="gpt-4o", temperature=0.3 , streaming=True)
summarizer_llm  = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.2)

# ---------------------------------------------------------------------
# 2) Tools et chaîne agent
# ---------------------------------------------------------------------
tools        = [retrieve_it ,explain]
agent_chain  = create_tool_calling_agent(llm_agent, tools, prompt_agent)


agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)


chat_map = {}

def get_chat_history(session_id: str, k: int = 4):
    """Crée/retourne la mémoire résumée d'une session."""
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(
            llm=summarizer_llm,
            k=k
        )
    return chat_map[session_id]


agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(id="session_id", annotation=str, default="default"),
        ConfigurableFieldSpec(id="k",           annotation=int,  default=4),
    ],
)


def rag_response_agentic(query: str,
                         session_id: str = "default",
                         k: int = 4) -> str:
    result = agent_with_memory.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id, "k": k}
        }
    )
    return result["output"]

# ---------------------------------------------------------------------
# 7) Petit test CLIENT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("### Agentic RAG – Mode CLIENT , taper quitter pour quitter le mode ###")
    session = "cli_user"
    while True:
        q = input("\n💬 > ")
        if q.lower() in {"quit", "exit",'quitter'}:
            break
        print("🤖", rag_response_agentic(q, session_id=session))
