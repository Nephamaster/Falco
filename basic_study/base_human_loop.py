import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
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

@tool
def human_assistance(query:str)->str:
    """Request assistance from human"""
    human_response = interrupt(query)
    if isinstance(human_response, dict):
        return human_response.get("data", "")
    return str(human_response)
tools = [search_tool, human_assistance]
tool_node = ToolNode(tools)

# ==== 大模型定义 ====
llm = ChatOpenAI(
    model=os.getenv('LLM_MODEL_ID'),
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)
llm = llm.bind_tools(tools)
def chatbot(state:State):
    response = llm.invoke(state['messages'])
    assert len(response.tool_calls) <= 1
    return {'messages':[response]}

# ==== 记忆管理 ====
memory = MemorySaver()

def create_graph():
    workflow = StateGraph(State)

    workflow.add_node('chatbot', chatbot)
    workflow.add_node('tools', tool_node)

    workflow.add_edge(START, 'chatbot')
    workflow.add_conditional_edges('chatbot', tools_condition)
    workflow.add_edge('tools', 'chatbot')
    app = workflow.compile(checkpointer=memory)
    
    return app


app = create_graph()


def get_user_input(interrupt_info):
    print("=== HUMAN IN THE LOOP ===")
    print("收到中断请求：")
    print(json.dumps(interrupt_info, ensure_ascii=False, indent=2))
    user_text = input("请输入人工回复: ").strip()
    return {"data": user_text}


def run_stream(input_data, config):
    """
    统一处理一次 stream。
    返回:
    - None: 正常结束，无中断
    - Command(resume=...): 需要恢复
    """
    for output in app.stream(
        input_data,
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        # v2 下 output 是字典，包含 type / data
        if output["type"] == "messages":
            msg, metadata = output["data"]
            if isinstance(msg, AIMessageChunk) and msg.content:
                print('Assistant:', msg.content, end="", flush=True)

        elif output["type"] == "updates":
            data = output["data"]
            if "__interrupt__" in data:
                interrupt_obj = data["__interrupt__"][0]
                interrupt_info = interrupt_obj.value
                human_response = get_user_input(interrupt_info)
                return Command(resume=human_response)
    return None


def stream_graph_update(user_input:str):
    input = {'messages':[{'role':'user', 'content':user_input}]}
    config = {'configurable':{'thread_id':'1'}}
    resume_cmd = run_stream(input,config)
    while resume_cmd is not None:
        print("\n=== RESUMING GRAPH ===")
        resume_cmd = run_stream(resume_cmd,config)
    print("\n")


while True:
    user_input = input('User: ')
    # user_input = "我需要一些用来构建智能体的专业指导。你能帮我请求协助吗？"
    if user_input.lower() in ['quit', 'q', 'exit']:
        print('GoodBye!')
        break
    stream_graph_update(user_input)