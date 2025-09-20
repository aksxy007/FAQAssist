import streamlit as st
import uuid
import json
import traceback
from pathlib import Path
from ai.react_agent.agent import get_react_agent

agent = get_react_agent()

st.write("FAQSAP")
st.caption("Demo app for FAQ assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your FAQ assistant. How can I help you today?"}
    ]

if "thread_id" not in st.session_state:
    if (thread_file := Path("thread_id.txt")).exists():
        st.session_state.thread_id = thread_file.read_text().strip()
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        thread_file.write_text(st.session_state.thread_id)

def get_assistant_response(user_input):
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )
        content = response["messages"][-1].content
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"answer": content}  # fallback
    except Exception as e:
        print(f"Error: {e}-{traceback.format_exc()}")
        return {"answer": "Sorry, something went wrong."}

# Chat input
if prompt := st.chat_input("Ask your question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        response = get_assistant_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Rendering loop
for message in st.session_state.messages:
    if message["role"] == "assistant":
        response = message["content"]
        if isinstance(response, dict):
            with st.chat_message("assistant"):
                st.markdown(response.get("answer", "Sorry, I don't know the answer to that."))
                with st.expander("Reasoning", expanded=False):
                    st.markdown(response.get("reasoning", "No reasoning provided"))
        else:
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        with st.chat_message("user"):
            st.markdown(message["content"])
