# ai/react_agent/agent.py
import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from ai.react_agent.tools import retrieve_from_vector_store
from ai.react_agent.prompts import get_system_prompt
from langchain_groq.chat_models import ChatGroq

llm = ChatGroq(model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"), reasoning_format="parsed")

tools = [retrieve_from_vector_store]
memory = InMemorySaver()

system_prompt = get_system_prompt()

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

def get_react_agent():
    return agent
