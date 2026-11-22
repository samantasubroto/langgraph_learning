import os
import uuid
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def build_chatbot():
    """Build the graph once. The cached MemorySaver keeps history across reruns."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    def chat_node(state: ChatState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=MemorySaver())


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def thread_title(thread_id: str) -> str:
    """Use the first user message as the chat's label, or a placeholder if empty."""
    history = chatbot.get_state({"configurable": {"thread_id": thread_id}}).values.get(
        "messages", []
    )
    for message in history:
        if isinstance(message, HumanMessage):
            text = message.content.strip()
            return text[:30] + ("..." if len(text) > 30 else "")
    return "New chat"


def start_new_chat():
    new_id = generate_thread_id()
    st.session_state.chat_threads.append(new_id)
    st.session_state.thread_id = new_id


if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
    st.stop()

chatbot = build_chatbot()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
    st.session_state.chat_threads = [st.session_state.thread_id]

config = {"configurable": {"thread_id": st.session_state.thread_id}}

with st.sidebar:
    st.header("Chat")
    if st.button("New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.caption("Past chats")
    # Most recent thread first.
    for thread_id in reversed(st.session_state.chat_threads):
        is_current = thread_id == st.session_state.thread_id
        label = thread_title(thread_id)
        if st.button(label, key=f"thread-{thread_id}", use_container_width=True,
                     type="primary" if is_current else "secondary"):
            st.session_state.thread_id = thread_id
            st.rerun()

    st.divider()
    st.caption(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")

st.title("Chatbot")

# Replay the saved conversation for this thread.
history = chatbot.get_state(config).values.get("messages", [])
for message in history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Type your message"):
    with st.chat_message("user"):
        st.markdown(prompt)

    def token_stream():
        """Yield the reply text chunk by chunk as the model writes it."""
        for chunk, _metadata in chatbot.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    with st.chat_message("assistant"):
        try:
            st.write_stream(token_stream())
        except Exception as exc:
            st.error(f"Request failed: {exc}")