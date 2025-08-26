
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
import os
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from conversation_summary_memory import ConversationSummaryBufferMessageHistory

""""
os.environ["LANGCHAIN_API_KEY"]     = "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"
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]     = "Projet"
"""

LANGUAGE_MODEL  = "llama3.2:1b-instruct-fp16"
llm = ChatOllama(model=LANGUAGE_MODEL, temperature=0.3)

prompt = """Tu es un assistant expert en support IT pour les entreprises.

Tu reçois :
- Une **question utilisateur** : {query}
- Un **contexte extrait automatiquement** de la base documentaire de l’entreprise (historique de tickets, guides internes, logs, etc.) voici le context {CONTEXT}
- Soit poli , si l'utilisateur te dis bonjour tu lui repondra  : 'Bonjour comment puis je t'aider aujourd'hui ?' et si il te dit enrevoir :  'Enrevoir , j'espère avoir été utile !'
Ta mission est de résoudre la question **de façon précise et fiable**.

Voici comment procéder :
1. Raisonne étape par étape dans ta tête pour comprendre le problème et planifier une solution (ne montre pas ce raisonnement à l’utilisateur).
2. Utilise uniquement les informations du contexte si possible. Ne devine pas si tu n’es pas sûr.
3. À la fin, donne une **réponse concise et professionnelle**, directement exploitable par l’utilisateur final.
4. 2. Si le contexte est pertinent, utilise-le uniquement. Sinon, appuie-toi sur ta propre expertise pour répondre. Il ne faut **jamais dire que tu ne sais pas** : réfléchis au mieux avec les infos disponibles.


⚠️ **Important : tu ne dois JAMAIS afficher ton raisonnement. Seule la réponse finale doit être visible ET aucune phrase d'introduction**


    N'oublie pas de t'aider du context : {CONTEXT}
    """

example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
examples = [
    {
        "input": "Outlook ne reçoit plus mes mails. Comment corriger ça ?",
        "output": (
            "## Réception de mails impossible dans Outlook\n\n"
            "*Examiner les règles de boîte de réception.\n"
            "*Vérifier les quotas de boîte pleine.\n"
            "*Contrôler pare-feu, antivirus, et configuration Exchange\n"
            "*Redémarrer l’ordinateur sans application tierce.\n"
            "*Analyser les logs et les codes d erreur.\n"
            "*Comparer avec un autre compte utilisateur."
            "*Escalader vers le support si besoin, avec logs.\n"
        )
    },
    {
        "input": "Mon Outlook ne peut pas envoyer de mails, que dois-je faire ?",
        "output": (
            "## Diagnostic – Envoi d'e-mails impossible dans Outlook\n\n"
            "*Vérifier la configuration du compte (adresse, mot de passe)."
            "*Contrôler les paramètres SMTP et la connectivité réseau.\n\n"
            "*Redémarrer l'ordinateur et tester sans logiciel tiers actif.\n"
            "* Analyser les journaux d erreur (logs Outlook, Event Viewer)\".\n"
            "*Consulter la documentation Microsoft pour les cas similaires.\n"
            "*Tester avec un autre compte ou poste..\n"
        )
    }
]
few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    MessagesPlaceholder(variable_name="history"),
    few_shot_prompt,
    ("user", "{query}"),
])

pipeline = prompt_template | llm
chat_map = {}

def get_chat_history(session_id: str, llm: ChatOllama, k: int):
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(llm=llm, k=k)
    return chat_map[session_id]

rag_chain = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="session_id", annotation=str, default="id_default"),
        ConfigurableFieldSpec(id="llm",         annotation=ChatOllama, default=llm),
        ConfigurableFieldSpec(id="k",           annotation=int,        default=4),
    ],
)

