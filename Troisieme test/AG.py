
import os
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from supabase.client import Client, create_client
from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from conversation_summary_memory import ConversationSummaryBufferMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

# =============== 0. PARAMÈTRES GLOBAUX ===============
LANGUAGE_MODEL  = "llama3.2:1b-instruct-fp16"


# initiate supabase database
supabase_url = os.environ.get("https://pqjohzljlyaysybtfupj.supabase.co")
supabase_key = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBxam9oemxqbHlheXN5YnRmdXBqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTEwMTQ2NDYsImV4cCI6MjA2NjU5MDY0Nn0.POc8MRRb8K1WOBeRzmcsHR4z-fTFoqW2qG-uxOn_KUo")

supabase: Client = create_client("https://pqjohzljlyaysybtfupj.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBxam9oemxqbHlheXN5YnRmdXBqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTEwMTQ2NDYsImV4cCI6MjA2NjU5MDY0Nn0.POc8MRRb8K1WOBeRzmcsHR4z-fTFoqW2qG-uxOn_KUo")

# initiate embeddings model
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=EMBEDDING_MODEL,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# initiate large language model (temperature = 0)
llm = ChatOllama(model=LANGUAGE_MODEL, temperature=0.3)

def get_chat_history(session_id: str, llm: ChatOllama, k: int):
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(llm=llm, k=k)
    return chat_map[session_id]

# fetch the prompt from the prompt hub
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're an IT support assistant. When answering a user's question,  you should always start by using the tools provided to retrieve relevant information ;(e.g., from a documentation retriever, database, or API). "
        "The results of any tool call will be provided in the 'scratchpad' below."
        "If the scratchpad contains an answer, respond to the user clearly and helpfully using only that information. "
        "Do not use any external knowledge or assumptions."
        "If the scratchpad is empty or no relevant information was found, do not guess. "
        "Instead, politely ask the user for more details or clarification that might help narrow down the issue."
        "Your goal is to solve IT problems related to office tools like Outlook, Teams, Windows, or email setups in a professional and accessible way."
        "Always keep your answers concise, polite, and practical."
        "Never run a second tool once you have a scratchpad result. Only one tool call is allowed per interaction."

    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# create the tools
@tool(response_format="content_and_artifact")
def retrieve_solution_faiss(query: str, top_k: int = 3):
    # Générer l'embedding de la requête
    vector_store.similarity_search(query, k=2)
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

    results = []
    for idx, score in zip(idxs[0], dists[0]):
        if score < 0.75:
            continue  # Ignorer les correspondances trop faibles
        blk = entries[idx]
        results.append(
            {"problem": blk["problem"], "solution": blk["solution"], "score": float(score)}
        )

    return results






def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# invoke the agent
response = agent_executor.invoke({"input": "why is agentic rag better than naive rag?"})

# put the result on the screen
print(response["output"])