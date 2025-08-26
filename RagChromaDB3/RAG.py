# RAG.py  – prompt complet, few‑shots, memory, rag_chain
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from conversation_summary_memory import ConversationSummaryBufferMessageHistory
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

LANGUAGE_MODEL  = os.getenv("LANGUAGE_MODEL",  "llama3.2:1b-instruct-fp16")
llm = ChatOllama(model=LANGUAGE_MODEL, temperature=0)

prompt_text = """
Tu es l’assistant support Akuiteo.
Ta mission :
— Fournir une réponse claire et fiable à l’utilisateur, en français.
— T’appuyer UNIQUEMENT sur les informations contenues dans {CONTEXT}.
— Ne jamais inventer de donnée; si le CONTEXT est vide ou hors‑sujet, réponds franchement : «Je n’ai rien trouvé dans le Livre d’or sur ce sujet. »

ATTENTION :
— Ne commence jamais par résumer la question ou le prompt; va directement à la réponse.
— N’écris jamais de code; donne uniquement des explications sous forme de phrase courte, suivie, si utile, d’une liste d’étapes ou de points clés (bullet points, pas de H1/H2).
— Aucun JSON, aucune balise Markdown, aucun code.

Ton style :
– Chaleureux et professionnel.
– Sans jargon technique inutile; privilégie des verbes d’action («Ouvrez…», «Cliquez…»).
– Si des montants, chemins de menu ou champs précis sont présents dans le CONTEXT, cite-les tels quels.

Le CONTEXT (extraits du Livre d’or Akuiteo) suit; ignore toute autre source d’information.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt_text),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{query}")
])

chat_map = {}
def get_chat_history(session_id: str):
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(
            llm=llm,
            k=4
        )
    return chat_map[session_id]

rag_chain = RunnableWithMessageHistory(
    prompt_template | llm,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
)