# conversation_summary_memory.py

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI  # ou ton import ChatOpenAI

class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory):

    def __init__(self, llm: ChatOpenAI, k: int):
        # Ne pas appeler super().__init__ : on gère soi‑même
        self.llm = llm
        self.k   = k
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        existing_summary = None
        old_messages     = None

        # Si un résumé existe déjà en position 0, on le vire
        if self.messages and isinstance(self.messages[0], SystemMessage):
            existing_summary = self.messages.pop(0)

        # Ajoute les nouveaux messages
        self.messages.extend(messages)

        # On découpe si > k
        if len(self.messages) > self.k:
            old_messages  = self.messages[: self.k]
            self.messages = self.messages[-self.k :]

        if not old_messages:
            return

        # Prépare le prompt pour résumer
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Étant donné le résumé existant et les anciens messages, "
                "génère un résumé concis (court paragraphe) avec le maximum d'informations."
            ),
            HumanMessagePromptTemplate.from_template(
                "Résumé existant :\n{existing_summary}\n\n"
                "Messages à résumer :\n{old_messages}"
            ),
        ])

        # Appelle le LLM
        result = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                old_messages=old_messages
            )
        )
        new_summary = result.content if hasattr(result, "content") else str(result)

        # Insère le nouveau résumé en tête
        self.messages = [SystemMessage(content=new_summary)] + self.messages

    def clear(self) -> None:
        """Vide totalement l’historique."""
        self.messages = []

