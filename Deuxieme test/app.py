import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main_agent import agent_with_memory      # le runnable streamable

st.set_page_config(page_title="Agentic RAG IT", page_icon="🛠️")
st.title("🛠️ Agentic RAG IT Support")

# Historique d’affichage
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

question = st.chat_input("Posez votre question…")

if question:
    # 1. Affiche la question
    st.session_state.messages.append(HumanMessage(question))
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Placeholder pour la réponse en flux
    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""

        # 3. Stream
        stream = agent_with_memory.stream(
            {"input": question},
            config={"configurable": {"session_id": "streamlit_user", "k": 4}}
        )

        for chunk in stream:
            # Cas 1 : chunk est déjà du texte
            if isinstance(chunk, str):
                delta = chunk
            # Cas 2 : chunk est un dict {"output": "..."}
            else:
                delta = chunk.get("output", "") or chunk.get("delta", "")
            partial += delta
            placeholder.markdown(partial + "▌")   # curseur

        placeholder.markdown(partial)             # version finale

        # 4. Stocke la réponse dans l’historique
        st.session_state.messages.append(AIMessage(partial))
