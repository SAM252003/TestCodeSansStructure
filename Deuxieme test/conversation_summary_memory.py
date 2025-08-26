from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field


class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOllama = Field(default_factory=ChatOllama)
    k: int = Field(default_factory=int)

    def __init__(self, llm: ChatOllama, k: int):
        super().__init__(llm=llm, k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:

        existing_summary = None
        old_messages = None

        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):

            existing_summary: str | None = self.messages.pop(0)
        # add the new messages to the history
        self.messages.extend(messages)
        # check if we have too many messages
        if len(self.messages) > self.k:
            # pull out the oldest messages...
            old_messages = self.messages[:self.k]
            # ...and keep only the most recent messages
            self.messages = self.messages[-self.k:]
        if old_messages is None:

            # if we have no old_messages, we have nothing to update in summary
            return
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible BUT keep the summary "
                "concise and no more than a short paragraph in length."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{old_messages}"
            )
        ])
        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                old_messages=old_messages
            )
        )
        # prepend the new summary to the history
        self.messages = [SystemMessage(content=new_summary.content)] + self.messages

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []