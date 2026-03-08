"""
ReAct Agent Implementation using LangChain
Based on the ReAct prompting paradigm: https://arxiv.org/abs/2210.03629

This script demonstrates how to create a ReAct (Reasoning and Acting) agent using LangChain,
similar to the custom implementation in react_basic.py but leveraging LangChain's built-in
agent framework.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

load_dotenv()


def get_langchain_llm():
    """Initialize and return a LangChain ChatOpenAI instance"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY before running the ReAct demo query.")
    return ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


@tool
def calculate(expression: str) -> float:
    """Runs a calculation and returns the number. Uses Python so be sure to use floating point syntax if necessary.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "4 * 7 / 3")
    
    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def course_info(query: str) -> str:
    """Search BT5153 course information for topics like schedule, assignments, grading, contact info, etc.
    
    Args:
        query: The topic to search for (e.g., "assignments", "schedule", "grading")
    
    Returns:
        Relevant course information
    """
    course_data = {
        "schedule": "Classes are on Fridays at COM1-0204. Key dates: Assignment I due 02/13, Assignment II due 02/27, Assignment III due 03/13, Kaggle starts 03/20, Final presentations 05/01",
        "assignments": "Individual Assignments are 50% total: Assignment 1 (10%), Assignment 2 (10%), Assignment 3 (10%), Kaggle Competition (20%)",
        "grading": "Assessment breakdown: Attendance Check (10%), Individual Assignments (50%), Group Project (40%)",
        "contact": "Lecturer: Rui Zhao (diszr@nus.edu.sg), TAs: Dingyu Shi (dingyushi@u.nus.edu), Yang Ding (yding@u.nus.edu)",
        "project": "Group Project (40%): Project proposal (5%), Project presentation (20%), Project final report (15%). Teams of 4-5 members required.",
        "topics": "Course covers advanced ML techniques, NLP, LLMs, transformers, RAG, agent design patterns, model evaluation and deployment",
        "venue": "Class venue is COM1-0204, Fridays",
        "prerequisites": "Basic Python programming knowledge and basic math knowledge required"
    }
    
    query_lower = query.lower()
    
    for key, info in course_data.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return info
    
    return "BT5153 Applied Machine Learning for Business Analytics - NUS MSBA Spring 2026. For specific information, try searching for: schedule, assignments, grading, contact, project, topics, venue, or prerequisites."


def create_react_prompt():
    """Create the ReAct prompt template for the agent"""
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    
    return PromptTemplate.from_template(template)


def query(question: str, max_iterations: int = 5, verbose: bool = True):
    """
    Execute a query using the LangChain ReAct agent
    
    Args:
        question: The question to answer
        max_iterations: Maximum number of reasoning/action loops
        verbose: Whether to print detailed execution logs
    
    Returns:
        The final answer from the agent
    """
    llm = get_langchain_llm()
    
    tools = [calculate, course_info]
    
    prompt = create_react_prompt()
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True
    )
    
    try:
        result = agent_executor.invoke({"input": question})
        return result.get("output", "No answer generated")
    except Exception as e:
        return f"Error executing query: {str(e)}"


if __name__ == "__main__":
    print("=" * 50)
    print("LangChain ReAct Agent Demo")
    print("=" * 50)
    
    print('\n==================Test 1====================')
    print("Query: Hello, what is your name")
    answer = query("Hello, what is your name")
    print(f"\nFinal Answer: {answer}\n")
    
    print('==================Test 2====================')
    print("Query: What is 5 + 6, then times by 30")
    answer = query("What is 5 + 6, then times by 30")
    print(f"\nFinal Answer: {answer}\n")
    
    print('==================Test 3====================')
    print("Query: What are the assignment weights in BT5153?")
    answer = query("What are the assignment weights in BT5153?")
    print(f"\nFinal Answer: {answer}\n")
    
    print('==================Test 4====================')
    print("Query: If attendance is worth 10/10 points, what's the minimum I need on individual assignments to pass BT5153 with 50% overall?")
    answer = query("If attendance is worth 10/10 points, what's the minimum I need on individual assignments to pass BT5153 with 50% overall?")
    print(f"\nFinal Answer: {answer}\n")
    
    print('==================Test 5 (Multi-turn)====================')
    print("Query: If I score 85% on Assignment 1 and 90% on Assignment 2 in BT5153, what would be my total points from these two assignments?")
    answer = query("If I score 85% on Assignment 1 and 90% on Assignment 2 in BT5153, what would be my total points from these two assignments?")
    print(f"\nFinal Answer: {answer}\n")
