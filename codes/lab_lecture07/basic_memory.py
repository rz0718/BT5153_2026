from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# This list stores the conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."},
]


def get_chat_response(user_input):
    # Add the new user message to the history
    conversation_history.append({"role": "user", "content": user_input})

    # Send the entire history to the API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=conversation_history
    )
    ai_response_content = response.choices[0].message.content
    # Add the AI's response to the history for the next turn
    conversation_history.append({"role": "assistant", "content": ai_response_content})
    return ai_response_content


# Example usage:
print(get_chat_response("My name is RZ. I am at BT5153 class"))
print(get_chat_response("What is my name? and what class am I in?"))
