from typing import TypedDict
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    intent: str

    business_context: str

    sql_result: dict

    retrieved_chunks: list

    final_response: str