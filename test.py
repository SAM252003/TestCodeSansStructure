import os
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.chains import LLMChain



# Modèle
model_name = "llama3.2:1b-instruct-fp16"
llm = ChatOllama(temperature=0.2, model=model_name)

# Données
context = """Aurelio AI is an AI company developing tooling for AI engineers...
(the rest of your context)
"""
query = "what does Aurelio AI do?"

# Prompt system/user
system_template = SystemMessagePromptTemplate.from_template("""
Answer the user's query based on the context below.                 
If you cannot answer the question using the
provided information answer with "I don't know".

Always answer in markdown format. When doing so please
provide headers, short summaries, follow with bullet
points, then conclude.

Context: {context}
""")

user_template = HumanMessagePromptTemplate.from_template("{query}")

# Chat prompt template complet
prompt_template = ChatPromptTemplate.from_messages([
    system_template,
    user_template
])

# Format des messages
messages = prompt_template.format_messages(context=context, query=query)

# Appel du modèle
response = llm.invoke(messages)
print(response.content)
