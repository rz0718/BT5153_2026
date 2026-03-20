#!/usr/bin/env python3
"""
LangChain tool-calling demo with local Ollama model `qwen2.5:3b`.

Five-step flow:
1) Define tools
2) Pass tools + user message to the model
3) Model decides whether to call tools
4) Execute tool(s) in app code
5) Send tool result(s) back to model for final answer

Run:
    python tool_call/tool_calling_demo.py
"""

from __future__ import annotations

import ast
from typing import Dict

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import OpenAI

# STEP 1: Define the tools (LangChain @tool decorator + typed args/docstrings)
@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather  in a location."""
    mock = {"London": 15, "Tokyo": 22, "Paris": 18, "Singapore": 28}
    temp = mock.get(location.split(",")[0].strip(), 20)
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32
        return f"{location}: {temp:.1f}F, sunny"
    return f"{location}: {temp:.1f}C, sunny"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only numbers and + - * / ( ) allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [get_current_weather, calculate]
TOOLS_BY_NAME: Dict[str, object] = {t.name: t for t in TOOLS}


def run_tool(name: str, args: dict) -> str:
    tool_obj = TOOLS_BY_NAME.get(name)
    if tool_obj is None:
        return f"Unknown tool: {name}"
    return str(tool_obj.invoke(args))


def run_demo(question: str, max_rounds: int = 5) -> None:
    # Local Ollama model (ensure `ollama serve` is running and model is pulled).
    #model = ChatOllama(model="qwen2.5:3b", temperature=0)
    # OpenAI model 
    model = OpenAI(model='gpt-4o-min')
    # STEP 2: Pass tools + user query to the LLM
    model_with_tools = model.bind_tools(TOOLS)
    messages = [HumanMessage(content=question)]

    for _ in range(max_rounds):
        response = model_with_tools.invoke(messages)
        print("Model:", response.content)
        print(response)
        # STEP 3: If no tool calls, we have a final answer
        if not response.tool_calls:
            print("Answer:", response.content or "(no content)")
            return

        messages.append(response)

        # STEP 4: Execute each tool call in application code
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args", {})
            print(f"Tool requested -> {name}({args})")
            tool_result = run_tool(name, args)

            # STEP 5: Send tool result back to the LLM via ToolMessage
            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )

    print("(Model kept requesting tools; increase max_rounds for full flow.)")


if __name__ == "__main__":
    #run_demo("What's the weather like in Singapore?")
    print("--------------------------------")
    #run_demo("What is 4 * 7 / 3?")
    print()
    #run_demo("Hello, how are you?")
    print()
    run_demo("Check the weather in Singapore and London and get the sum of their tempeature.")