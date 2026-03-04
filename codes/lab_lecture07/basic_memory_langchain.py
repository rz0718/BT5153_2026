"""
Basic chat with memory — LangChain version.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm
history = []


def chat(user_input: str):
    result = chain.invoke({"input": user_input, "history": history})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=result.content))
    return result.content


print(chat("My name is RZ. I am at BT5153 class"))
print(chat("What is my name? and what class am I in?"))
