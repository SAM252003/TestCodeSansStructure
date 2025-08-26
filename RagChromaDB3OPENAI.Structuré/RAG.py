# RAG.py  – prompt complet, few‑shots, memory, rag_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from conversation_summary_memory import ConversationSummaryBufferMessageHistory
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL", "gpt-3.5-turbo-0125")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model=LANGUAGE_MODEL,
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

prompt_text = """
Tu es l’assistant support Akuiteo.
Ta mission  
—Fournir une réponse claire et fiable à l’utilisateur, en français.  
-T’appuyer UNIQUEMENT sur les informations contenues dans {CONTEXT}.  
—Ne jamais inventer de données; si le CONTEXT est vide ou hors‑sujet, dis-le franchement: «Je n’ai rien trouvé dans le Livred’or sur ce sujet.»

ATTENTION : N’écris jamais de code; donne uniquement des explications sous forme
   de phrase courte suivie, si utile, d’une liste d’étapes.

Réponds sous la forme:  
• une phrase‑réponse courte (résumé)  
• puis, si utile, une liste d’étapes ou de points clés (bullet points, pas de Markdown H1/H2).  
Aucun JSON, aucune balise Markdown, aucun code.

Ton style  
–Ton chaleureux, professionnel.  
–Si des montants, chemins de menu, ou champs précis sont présents dans le CONTEXT, cite‑les exactement.

Suit le CONTEXT (extraits du Livre d’or Akuiteo) ; ignore toute autre source d’information.
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