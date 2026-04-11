import ast
import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import InjectedToolCallId, tool
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
    review_title: str
    release_date: str

# ==== 工具定义 ====
search_tool = TavilySearch(
    api_key=os.getenv('TAVILY_API_KEY'),
    max_results=1,
    topic="general"
)

@tool
def human_assistance(
    review_title: str, release_date: str, tool_call_id: Annotated[str, InjectedToolCallId]
)->str:
    """Request assistance from human.
    
    Note that because we are generating a ToolMessage for a state update, we
    generally require the ID of the corresponding tool call. We can use
    LangChain's InjectedToolCallId to signal that this argument should not
    be revealed to the model in the tool's schema.
    """
    human_response = interrupt(
        {
            "question": "这对吗？",
            "review_title": review_title,
            "release_date": release_date,
        }
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = review_title
        verified_birthday = release_date
        response = "正确"
    else:
        verified_name = human_response.get("name", review_title)
        verified_birthday = human_response.get("birthday", release_date)
        response = f"纠正: {human_response}"
    state_update = {
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        "review_title": verified_name,
        "release_date": verified_birthday,
    }
    return Command(update=state_update)
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
    user_text = input("请输入人工回复(dict格式): ").strip()
    try:
        user_data = ast.literal_eval(user_text)
        if not isinstance(user_data, dict):
            raise ValueError("输入不是 dict")
        return user_data
    except Exception as e:
        print(f"输入格式错误: {e}")
        print("将默认视为未修正")
        return {"correct": "yes"}


def run_stream(input_data, config):
    """
    统一处理一次 stream。
    返回:
    - None: 正常结束，无中断
    - Command(resume=...): 需要恢复
    """
    started_printing = False
    for output in app.stream(
        input_data,
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        # v2 下 output 是字典，包含 type / data
        if output["type"] == "messages":
            msg, metadata = output["data"]
            if not isinstance(msg, AIMessageChunk):
                continue
            text = msg.content
            if not text:
                continue
            if metadata.get("langgraph_node") != "chatbot":
                continue
            if not started_printing:
                print("Assistant: ", end="", flush=True)
                started_printing = True
            print(text, end="", flush=True)

        elif output["type"] == "updates":
            data = output["data"]
            if "__interrupt__" in data:
                interrupt_obj = data["__interrupt__"][0]
                interrupt_info = interrupt_obj.value
                human_response = get_user_input(interrupt_info)
                return Command(resume=human_response)
    if started_printing:
        print()
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
    # user_input = (
    #     "你能查看LangGraph是何时发布的吗？"
    #     "当你找到答案时, 用 `human_assistance` 工具进行审查。"
    # )
    # user_input = "我需要一些用来构建智能体的专业指导。你能帮我请求协助吗？"
    if user_input.lower() in ['quit', 'q', 'exit']:
        print('GoodBye!')
        break
    stream_graph_update(user_input)