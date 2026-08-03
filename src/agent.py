import os
import warnings
warnings.filterwarnings("ignore") # Suppress warnings for a cleaner output

from langgraph.prebuilt import create_react_agent # The *create_react_agent* function is "deprecated" but works fine for our use case
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from src.rag import retrieval_tool
from src.tools import (
    prediction_tool, dataset_stats_tool, csv_lookup_tool
)

# AGENT INITIALIZATION #
TOOLS = [retrieval_tool, prediction_tool, dataset_stats_tool, csv_lookup_tool]

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please, set an OpenAI API-key before running.")

print("Using OpenAI GPT-4o-mini as the agent's language model.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    checkpointer=memory,
)

# CHAT FUNCTION #
def chat(message: str, session_id: str = "default") -> str:
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    return result["messages"][-1].content

# INTERACTIVE TESTING #
if __name__ == "__main__":
    print("\n=================================")
    print("  Task 3 - Interactive Agent Test")
    print("  Type 'exit' to exit")
    print("=================================\n")

    session = "test_session"

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "exit", "q"):
            break
        if not user_input:
            continue
        response = chat(user_input, session_id=session)
        print(f"Agent: {response}\n")