import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv


load_dotenv()

# ==== 状态定义 ====
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ==== 工具定义 ====
search_tool = TavilySearch(
    api_key=os.getenv('TAVILY_API_KEY'),
    max_results=1,
    topic="general"
)
tools = ToolNode([search_tool])

# ==== 大模型定义 ====
llm = ChatOpenAI(
    model=os.getenv('LLM_MODEL_ID'),
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)
llm = llm.bind_tools([search_tool])
def chatbot(state:State):
    response = llm.invoke(state['messages'])
    return {'messages':[response]}

# ==== 记忆管理 ====
memory = MemorySaver()

def create_graph():
    workflow = StateGraph(State)

    workflow.add_node('chatbot', chatbot)
    workflow.add_node('tools', tools)

    workflow.add_edge(START, 'chatbot')
    workflow.add_conditional_edges('chatbot', tools_condition)
    workflow.add_edge('tools', 'chatbot')
    app = workflow.compile(checkpointer=memory)
    
    return app


app = create_graph()

def stream_graph_update(user_input:str):
    input = {'messages':[{'role':'user', 'content':user_input}]}
    config = {'configurable':{'thread_id':'1'}}
    for output in app.stream(input, config=config):
            for _, node_output in output.items():
                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]
                    if isinstance(latest_message, AIMessage):
                        print('Assistant:', latest_message.content)

while True:
    user_input = input('User: ')
    if user_input.lower() in ['quit', 'q', 'exit']:
        print('GoodBye!')
        break
    stream_graph_update(user_input)