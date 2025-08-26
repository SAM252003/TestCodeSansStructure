# RAG.py  – prompt complet, few‑shots, memory, rag_chain
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from conversation_summary_memory import ConversationSummaryBufferMessageHistory
import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
)

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.2,
    top_p=0.9
)

llm = HuggingFacePipeline(pipeline=hf_pipe)
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
–Pas de jargon technique inutile; privilégie des verbes d’action («Ouvrez…», «Cliquez…»).  
–Si des montants, chemins de menu, ou champs précis sont présents dans le CONTEXT, cite‑les exactement.

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