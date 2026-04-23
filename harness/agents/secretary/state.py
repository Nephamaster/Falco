from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class FalcoState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: str
    context_block: str
    soul_block: str
    skills_block: str
    mcp_block: str
    workspace_block: str
    working_directory: str
