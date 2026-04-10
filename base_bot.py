import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model=os.getenv('LLM_MODEL_ID'),
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)

def chatbot(state:State):
    return {"messages":[llm.invoke(state['messages'])]}

graph_builder.add_node(chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()

def stream_graph_update(user_input:str):
    for event in graph.stream({'messages':[{'role':'user', 'content':user_input}]}):
        for value in event.values():
            print('Assistant:', value['messages'][-1].content)

while True:
    user_input = input('User: ')
    if user_input.lower() in ['quit', 'q', 'exit']:
        break
    stream_graph_update(user_input)