#!/usr/bin/env python3
"""
Tool Calling Demo — 5-step process: define tools → pass to LLM → LLM requests tool →
execute tool → send result back → final answer. Run: python tool_calling_demo.py
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# STEP 1: Define the tools (name, description, parameters as JSON schema)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather in a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City/country e.g. London, UK",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def get_current_weather(location: str, unit: str = "celsius") -> str:
    mock = {"London": 15, "Tokyo": 22, "Paris": 18}
    temp = mock.get(location.split(",")[0].strip(), 20)
    if unit == "fahrenheit":
        temp = temp * 9 // 5 + 32
    return f"{location}: {temp}°{'F' if unit == 'fahrenheit' else 'C'}, sunny"


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only numbers and + - * / ( ) allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOL_FNS = {"get_current_weather": get_current_weather, "calculate": calculate}


def run_tool(name: str, args: dict) -> str:
    fn = TOOL_FNS.get(name)
    return str(fn(**args)) if fn else f"Unknown tool: {name}"


def run_demo(question: str, max_rounds: int = 5):
    messages = [{"role": "user", "content": question}]

    for _ in range(max_rounds):
        # STEP 2: Pass tools & query to the LLM
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        print(msg)
        # STEP 3: LLM decides — if no tool call, we have the final answer
        if not getattr(msg, "tool_calls", None):
            print("Answer:", msg.content or "(no content)")
            return

        # STEP 3 (continued): LLM requested tool call(s) — add assistant message with tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": t.id,
                        "type": "function",
                        "function": {
                            "name": t.function.name,
                            "arguments": t.function.arguments,
                        },
                    }
                    for t in msg.tool_calls
                ],
            }
        )
        # STEP 4: Execute the tool (your application code)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = run_tool(name, args)
            # STEP 5: Send results back to the LLM (append as tool-role messages)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # After tool results are sent, one more call yields the final answer
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    final_msg = final.choices[0].message
    if getattr(final_msg, "tool_calls", None):
        print("(LLM requested another tool; increase max_rounds for full flow)")
    else:
        print("Answer:", final_msg.content or "(no content)")


if __name__ == "__main__":
    run_demo("What's the weather like in London?")
    print()
    run_demo("What is 4 * 7 / 3?")
