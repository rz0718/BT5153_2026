import openai
import re
import httpx
from dotenv import load_dotenv
import os
load_dotenv()

model_used = "gpt-3.5-turbo-0125" #"gpt-3.5-turbo-0125"   #gpt-4o-2024-08-06
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY before running the ReAct demo query.")
    return openai.OpenAI(api_key=api_key)
 
class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        client = get_openai_client()
        completion = client.chat.completions.create(model=model_used, messages=self.messages, temperature=0)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content

prompt = """
You run in a loop of Thought, Action, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

course_info:
e.g. course_info: assignments
Search BT5153 course information for topics like schedule, assignments, grading, contact info, etc.

Example session:

Question: What are the assignment weights in BT5153?
Thought: I need to look up assignment information for the BT5153 course
Action: course_info: assignments

You will be called again with this:

Observation: Individual Assignments are 50% total: Assignment 1 (10%), Assignment 2 (10%), Assignment 3 (10%), Kaggle Competition (20%)

You then output:

Answer: The individual assignments in BT5153 are worth 50% total, broken down as: Assignment 1 (10%), Assignment 2 (10%), Assignment 3 (10%), and Kaggle Competition (20%).
""".strip()


action_re = re.compile('^Action: (\w+): (.*)$') # match the format: "Action: Thought: something"

def query(question, max_turns=5):
    i = 0
    bot = ChatBot(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return




def course_info(query):
    """Simple demo function to retrieve BT5153 course information"""
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
    
    # Simple keyword matching for demo purposes
    for key, info in course_data.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return info
    
    # Default fallback
    return "BT5153 Applied Machine Learning for Business Analytics - NUS MSBA Spring 2026. For specific information, try searching for: schedule, assignments, grading, contact, project, topics, venue, or prerequisites."

def calculate(what):
    return eval(what)

known_actions = {
    "calculate": calculate,
    "course_info": course_info
}

if __name__ == "__main__":
    # Example queries for the actions - this will require multiple tool calls
    print('==================Test 1====================')
    query("Hello, what is your name")
    print('==================Test 2====================')
    query("What is 5 + 6, than times by 30")
    print('==================Test 3====================')
    query("What are the assignment weights in BT5153?")
    print('==================Test 4====================')
    query("If I get 60% on attendence, get 50% on projects and all points on assignment, what's my final score?")





"""
Multi-turn example session:

Question: If I score 85% on Assignment 1 and 90% on Assignment 2 in BT5153, what would be my total points from these two assignments?
Thought: I need to first find out the weights for Assignment 1 and Assignment 2 in BT5153
Action: course_info: assignments

You will be called again with this:

Observation: Individual Assignments are 50% total: Assignment 1 (10%), Assignment 2 (10%), Assignment 3 (10%), Kaggle Competition (20%)

Thought: Now I know Assignment 1 is worth 10% and Assignment 2 is worth 10%. I need to calculate the weighted scores: (85% * 10%) + (90% * 10%)
Action: calculate: (85 * 0.10) + (90 * 0.10)

You will be called again with this:

Observation: 17.5

You then output:

Answer: Based on the BT5153 grading scheme, if you score 85% on Assignment 1 (worth 10%) and 90% on Assignment 2 (worth 10%), you would earn 17.5 points out of the total 50 points available for individual assignments.

"""


# Note: If a question doesn't require any tools (calculate or course_info), simply provide a direct answer without using the Thought-Action-Observation pattern.