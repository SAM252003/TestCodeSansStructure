from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.main.rag.conversation_summary_memory import ConversationSummaryBufferMessageHistory
from app.main.components.llm.llm_component import LLMComponent

# Initialise le LLM singleton existant
llm = LLMComponent().get_llm()

# Prompt systemique central (peut être externalisé si besoin)
prompt_text = """
Tu es l’assistant support Akuiteo.
Ta mission :
— Fournir une réponse claire et fiable à l’utilisateur, en français.
— T’appuyer UNIQUEMENT sur les informations contenues dans {CONTEXT}.
— Ne jamais inventer de données; si le CONTEXT est vide ou hors-sujet, dis-le franchement : «Je n’ai rien trouvé dans le Livre d’or sur ce sujet.»

Réponds sous la forme :
• une phrase-réponse courte (résumé)
• puis, si utile, une liste d’étapes ou de points clés.

Ton style :
– Chaleureux et professionnel.
– Pas de jargon inutile; privilégie des verbes d’action («Ouvrez…», «Cliquez…»).
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt_text),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{query}")
])

# Gestion de l'historique par session
chat_map: dict[str, ConversationSummaryBufferMessageHistory] = {}

def get_chat_history(session_id: str):
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(
            llm=llm,
            k=4
        )
    return chat_map[session_id]

# Chaîne RAG avec mémoire par session
rag_chain = RunnableWithMessageHistory(
    prompt_template | llm,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
)
