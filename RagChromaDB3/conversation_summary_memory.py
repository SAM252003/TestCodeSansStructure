from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field


class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOllama = Field(default_factory=ChatOllama)
    k: int = Field(default=4)

    def __init__(self, llm: ChatOllama, k: int):
        super().__init__(llm=llm, k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:

        existing_summary = None
        old_messages = None

        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):

            existing_summary: str | None = self.messages.pop(0)

        self.messages.extend(messages)

        if len(self.messages) > self.k:

            old_messages = self.messages[:self.k]
            self.messages = self.messages[-self.k:]
        if old_messages is None:


            return
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                  "Étant donné le résumé de la conversation existante et les nouveaux messages, "
                  "générez un nouveau résumé de la conversation. Veillez à conserver "
                  "autant d'informations pertinentes que possible MAIS faites en sorte que le résumé "
                  "soit concis et ne dépasse pas la longueur d'un court paragraphe."
            ),
            HumanMessagePromptTemplate.from_template(
                "Conversation existante:\n{existing_summary}\n\n"
                "New messages:\n{old_messages}"
            )
        ])

        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                old_messages=old_messages
            )
        )

        self.messages = [SystemMessage(content=new_summary.content)] + self.messages

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []