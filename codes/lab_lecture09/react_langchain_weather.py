#!/usr/bin/env python3
"""
ReAct Agent with Weather Tools using LangChain and Ollama qwen2.5:3b

This script combines:
- Tools and model (qwen2.5:3b) from tool_calling_langchain.py
- TRUE ReAct implementation with explicit Thought/Action/Observation format

The goal is to test whether ReAct can solve multi-step tool call problems
like "Check the weather in Singapore and London and get the sum of their temperature."

Note: This uses a manual ReAct implementation since newer LangChain versions
have removed create_react_agent in favor of create_agent.
"""

from __future__ import annotations

from typing import Dict
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent


# STEP 1: Define the tools (same as tool_calling_langchain.py)
@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather in a location.
    
    Args:
        location: The location to get weather for (e.g., "London", "Tokyo")
        unit: Temperature unit, either "celsius" or "fahrenheit" (default: "celsius")
    
    Returns:
        Weather information for the location
    """
    mock = {"London": 15, "Tokyo": 22, "Paris": 18, "Singapore": 28}
    temp = mock.get(location.split(",")[0].strip(), 20)
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32
        return f"{location}: {temp:.1f}F, sunny"
    return f"{location}: {temp:.1f}C, sunny"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "4 + 7")
    
    Returns:
        The result of the calculation
    """
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only numbers and + - * / ( ) allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [get_current_weather, calculate]


def create_react_system_prompt():
    """Create the ReAct system prompt that enforces explicit reasoning format
    
    This system prompt instructs the model to use the ReAct format:
    - Thought: reasoning step
    - Action: tool to call (with proper function call)
    - Observation: (model waits for tool result)
    - Repeat until final answer
    """
    return """You are a helpful assistant that uses tools to answer questions.

When answering questions, you must follow this exact format:

1. Think about what you need to do
2. Call the appropriate tool(s) using function calling
3. Wait for the observation (tool result)
4. Repeat steps 1-3 as needed
5. Provide a final answer once you have all the information

For multi-step problems:
- Break down the problem into individual steps
- Call tools one at a time or in parallel when needed
- Use the results from previous tool calls to inform next steps
- Extract specific information (like numbers) from tool results before using them in calculations

Remember:
- Be explicit about your reasoning
- Show your work step by step
- When you have the final answer, state it clearly"""


def query(question: str, verbose: bool = True):
    """
    Execute a query using ReAct-style agent with Ollama qwen2.5:3b
    
    This implements ReAct reasoning by:
    1. Using a system prompt that encourages step-by-step reasoning
    2. Allowing the agent to call tools iteratively
    3. Providing observations back to the agent after each tool call
    
    Args:
        question: The question to answer
        verbose: Whether to print detailed execution logs
    
    Returns:
        The final answer from the agent
    """
    # Use local Ollama model (same as tool_calling_langchain.py)
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    
    # Create agent with ReAct-style system prompt
    system_prompt = create_react_system_prompt()
    agent_executor = create_agent(llm, TOOLS, system_prompt=system_prompt)
    
    try:
        if verbose:
            print(f"\nQuestion: {question}\n")
            print("=" * 70)
        
        result_messages = []
        for event in agent_executor.stream(
            {"messages": [("user", question)]},
            stream_mode="values"
        ):
            if verbose:
                message = event["messages"][-1]
                if hasattr(message, 'content') and message.content:
                    msg_type = message.type.upper()
                    content = message.content
                    if msg_type == "AI":
                        print(f"\n[AGENT REASONING]:\n{content}")
                    elif msg_type == "TOOL":
                        print(f"\n[OBSERVATION]:\n{content}")
                    elif msg_type == "HUMAN":
                        print(f"\n[QUESTION]: {content}")
                elif hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"\n[ACTIONS]:")
                    for tool_call in message.tool_calls:
                        print(f"  - {tool_call['name']}({tool_call['args']})")
            result_messages = event["messages"]
        
        if verbose:
            print("\n" + "=" * 70)
        
        # Get the final answer from the last AI message
        for msg in reversed(result_messages):
            if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
                return msg.content
        
        return "No answer generated"
    except Exception as e:
        return f"Error executing query: {str(e)}"


if __name__ == "__main__":
    print("=" * 70)
    print("ReAct Agent with Weather Tools (qwen2.5:3b)")
    print("=" * 70)
    
    
    print('==================Test 4: Multi-Step Tool Call====================')
    print("Query: Check the weather in Singapore and London and get temperatures in Singapore + temperatures in London.")
    answer = query("Check the weather in Singapore and London and get the sum of their temperature.")
    print(f"\nFinal Answer: {answer}\n")
