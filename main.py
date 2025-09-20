import uuid
import json
from pathlib import Path
import traceback
import time

from ai.react_agent.agent import get_react_agent

agent = get_react_agent()

THREAD_ID_FILE = Path("thread_id.txt")
if THREAD_ID_FILE.exists():
    thread_id = THREAD_ID_FILE.read_text().strip()
else:
    thread_id = str(uuid.uuid4())
    THREAD_ID_FILE.write_text(thread_id)

config = {"configurable": {"thread_id": thread_id}}
print(f"Persistent thread_id: {thread_id}")
print("Ask your questions (type 'exit' to quit):")

# test_queries = [
# "What was NVIDIA's total revenue in fiscal year 2024?",
# "What percentage of Google's 2023 revenue came from advertising?",
# "How much did Microsoft's cloud revenue grow from 2022 to 2023?",
# "Which of the three companies had the highest gross margin in 2023?", 
# "Which company had the highest operating margin in 2024?",
# "Compare the R&D spending as a percentage of revenue across all three companies in 2023""How did each company's operating margin change from 2022 to 2024?",
# "What are the main AI risks mentioned by each company and how do they differ?"
# ]


# Run agent loop
while True:
    user_input = input("\n> ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_input}],
        },config=config)
        # The agent returns structured JSON content
        # print(response["messages"][-1]["content"])
        content = response["messages"][-1].content
        # print(content)
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            output_json = json.loads(content)
        except json.JSONDecodeError:
            output_json = {"raw_output": content}  # fallback

        print("\nAgent Response:")
        print(json.dumps(output_json, indent=4))

    except Exception as e:
        print(f"Error: {e}-{traceback.format_exc()}")



# for query in test_queries:
#     print(f"User Query:\n> {query}")
#     try:
#         response = agent.invoke({
#             "messages": [{"role": "user", "content": query}],
#         },config=config)
#         # The agent returns structured JSON content
#         # print(response["messages"][-1]["content"])
#         content = response["messages"][-1].content
#         # print(content)
#         content = content.replace("```json", "").replace("```", "").strip()
#         try:
#             output_json = json.loads(content)
#         except json.JSONDecodeError:
#             output_json = {"raw_output": content}  # fallback

#         print("\nAgent Response:")
#         print(json.dumps(output_json, indent=4))

#     except Exception as e:
#         print(f"Error: {e}-{traceback.format_exc()}")
    
#     time.sleep(10)