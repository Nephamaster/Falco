import os
from dotenv import load_dotenv
from typing_extensions import TypedDict, Literal
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing import Annotated, List
import operator


load_dotenv()


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )

# Graph state
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# LLM
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "qwen3.5-27b"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

# Nodes
def llm_call_generator(state: State):
    """LLM generates a joke"""
    if state.get("feedback"):
        msg = llm.invoke(
            f"""Write a joke below about {state['topic']} but taking into account the feedback: {state['feedback']}."""
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""
    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)
# Add the nodes
optimizer_builder.add_node('generator', llm_call_generator)
optimizer_builder.add_node('evaluator', llm_call_evaluator)
# Add the edges
optimizer_builder.add_edge(START, 'generator')
optimizer_builder.add_edge('generator', 'evaluator')
optimizer_builder.add_conditional_edges(
    'evaluator',
    route_joke,
    {'Accepted': END, 'Rejected + Feedback':'generator'}
)

# Compile the workflow
optimizer = optimizer_builder.compile()

# Invoke
state = optimizer.invoke({"topic": "Donald Trump"})
print(state["joke"])