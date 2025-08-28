from dotenv import load_dotenv
import os
import json
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ─────────────────────────────
# Setup
# ─────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq Chatbot with Memory", page_icon="💬")
st.title("💬 Groq Chatbot with Memory")

# ─────────────────────────────
# Sidebar controls
# ─────────────────────────────
with st.sidebar:
    st.subheader("Controls")
    model_name = st.selectbox(
        "Groq Model",
        ["deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.1-8b-instant"],
        index=2
    )

    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 300, 150)

    system_prompt = st.text_area(
        "System prompt (rules)",
        value="You are a helpful, concise teaching assistant. Use short, clear explanations."
    )
    st.caption("Tip: Lower temperature for factual tasks; raise for brainstorming.")

    if st.button("🧹 Clear chat"):
        # reset only our single source of truth: the message history
        st.session_state.pop("history", None)
        st.rerun()

# ─────────────────────────────
# API key guard
# ─────────────────────────────
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Add it to your .env or deployment secrets.")
    st.stop()

# ─────────────────────────────
# Initialize single history
# ─────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

# ─────────────────────────────
# LLM + prompt + chain
# ─────────────────────────────
# ChatGroq reads GROQ_API_KEY from env
llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
)

# Role-structured prompt: System → History → Human
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Wrap with message history
chat_with_history = RunnableWithMessageHistory(
    chain,
    # Given a session_id, return the corresponding history object
    # Here we use a single-session app, so we return the one stored in session_state
    lambda session_id: st.session_state.history,
    input_messages_key="input",
    history_messages_key="history",
)

# ─────────────────────────────
# Render existing conversation
# ─────────────────────────────
# Only show human/ai messages (skip system if present)
for msg in st.session_state.history.messages:
    role = getattr(msg, "type", None) or getattr(msg, "role", "")
    content = msg.content
    if role == "human":
        st.chat_message("user").write(content)
    elif role in ("ai", "assistant"):
        st.chat_message("assistant").write(content)

# ─────────────────────────────
# Handle user turn
# ─────────────────────────────
user_input = st.chat_input("Type your message...")

if user_input:
    # Display the user message in the UI
    st.chat_message("user").write(user_input)

    # Invoke the chain with history tracking
    # session_id can be any stable string; using "default" for single-user app
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            response_text = chat_with_history.invoke(
                {"input": user_input, "system_prompt": system_prompt},
                config={"configurable": {"session_id": "default"}},
            )
        except Exception as e:
            st.error(f"Model error: {e}")
            response_text = ""

        # Typing effect (optional)
        typed = ""
        for ch in response_text:
            typed += ch
            placeholder.markdown(typed)

# ─────────────────────────────
# Download chat history (JSON)
# ─────────────────────────────
if st.session_state.history.messages:
    # Convert LangChain messages to simple {role, text}
    export = []
    for m in st.session_state.history.messages:
        role = getattr(m, "type", None) or getattr(m, "role", "")
        if role == "human":
            export.append({"role": "user", "text": m.content})
        elif role in ("ai", "assistant"):
            export.append({"role": "assistant", "text": m.content})
        # Skip system in export (can add if you want)

    st.download_button(
        "⬇️ Download chat JSON",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )
