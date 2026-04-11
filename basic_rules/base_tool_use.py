import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools:list)-> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs:dict):
        if messages := inputs.get('messages', []):
            message = messages[-1]
        else:
            raise ValueError('No message found in `input`')
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call['name']].invoke(tool_call['args'])
            outputs.append(
                ToolMessage(
                    json.dumps(tool_result),
                    name=tool_call['name'],
                    tool_call_id=tool_call['id']
                )
            )
        return {'messages': outputs}

search_tool = TavilySearch(
    api_key=os.getenv('TAVILY_API_KEY'),
    max_results=1,
    topic="general"
)
# tools = BasicToolNode([search_tool])
tools = ToolNode([search_tool])

llm = ChatOpenAI(
    model=os.getenv('LLM_MODEL_ID'),
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY')
)
llm = llm.bind_tools([search_tool])
def chatbot(state:State):
    response = llm.invoke(state['messages'])
    # print('==== AI Messages ====')
    # print('content:', response.content)
    # print('tool_calls:', getattr(response, 'tool_calls', []))
    return {'messages':[response]}

def route_tools(state:State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(f'No message found in `state` to tool edge: {state}')
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return 'tools'
    return END

workflow = StateGraph(State)

workflow.add_node('chatbot', chatbot)
workflow.add_node('tools', tools)

workflow.add_edge(START, 'chatbot')
# workflow.add_conditional_edges('chatbot', route_tools, {'tools':'tools', END:END})
workflow.add_conditional_edges('chatbot', tools_condition)
workflow.add_edge('tools', 'chatbot')

app = workflow.compile()

def stream_graph_update(user_input:str):
    for output in app.stream({'messages':[{'role':'user', 'content':user_input}]}):
            for node_name, node_output in output.items():
                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]
                    if isinstance(latest_message, AIMessage):
                        print(latest_message.content)

while True:
    user_input = input('User: ')
    if user_input.lower() in ['quit', 'q', 'exit']:
        break
    stream_graph_update(user_input)