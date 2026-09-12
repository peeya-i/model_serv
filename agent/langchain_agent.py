import os
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. Configuration - Connect to your local model server
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "your-internal-secure-gateway-token-xyz")
MODEL_NAME = os.environ.get("MODEL_NAME", "Llama-3.2-3B-Instruct")

llm = ChatOpenAI(
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=512,
)

# 2. Define Agent Tools
@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Safely calculates a basic mathematical expression, e.g. '125 * 8'."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


# Bind tools to the model
tools = [get_current_time, calculate]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


def run_agent(query: str) -> str:
    """Executes a complete reasoning and tool execution cycle with LangChain."""
    messages = [
        SystemMessage(
            content="You are a helpful assistant with access to local tools. Always call tools when needed."
        ),
        HumanMessage(content=query),
    ]

    print(f"\n[User Query]: {query}")

    # 1. Query model with tool schemas
    ai_msg = llm_with_tools.invoke(messages)

    # 2. If the model decided to call tools, execute them
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"[Tool Call]: {tool_name} with arguments: {tool_args}")

            if tool_name in tools_by_name:
                tool_output = tools_by_name[tool_name].invoke(tool_args)
            else:
                tool_output = f"Error: Tool {tool_name} not found"

            print(f"[Tool Output]: {tool_output}")

            # Append assistant message with single tool call to match Llama-3.2 chat template
            messages.append(ai_msg)
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        # 3. Generate final response with tool observations
        final_response = llm.invoke(messages)
        print(f"[Agent Response]: {final_response.content}")
        return final_response.content

    print(f"[Agent Response]: {ai_msg.content}")
    return ai_msg.content


if __name__ == "__main__":
    print("=== Testing LangChain Local Server Agent ===")
    run_agent("What time is it right now?")
    run_agent("Calculate 125 * 8")
