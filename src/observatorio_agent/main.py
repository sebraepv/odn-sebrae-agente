from __future__ import annotations

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from graph.state import AgentState

from graph.nodes.router import router
from graph.nodes.data_analyst import sql_node
from graph.nodes.rag_retrivier import rag_node
from graph.nodes.response import response_node

from langchain_azure_ai.agents.hosting import ResponsesHostServer

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


# ── Chat model ───────────────────────────────────────────────────────
def _build_chat_model() -> ChatOpenAI:
    deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME","gpt-5.4-mini")
    endpoint = os.getenv("OPEN_AI_ENDPOINT")
    api_key = os.getenv("FOUNDRY_API_KEY")

    return ChatOpenAI(
        model=deployment,
        base_url=endpoint,
        api_key=api_key,
        use_responses_api=True,
        output_version="responses/v1",
    )


# ── Graph ────────────────────────────────────────────────────────────
# Reuse the workflow state definition declared in graph/state.py and the
# nodes already implemented for this project.
State = AgentState


def _build_graph(model: ChatOpenAI):

    builder = StateGraph(State)

    async def router_node(state):
        return await router(state, model=model)

    async def sql_node_wrapper(state):
        return await sql_node(state, model=model)

    async def rag_node_wrapper(state):
        return await rag_node(state, model=model)

    async def response_node_wrapper(state):
        return await response_node(state, model=model)


    builder.add_node("router", router_node)
    builder.add_node("sql_node", sql_node_wrapper)
    builder.add_node("rag_node", rag_node_wrapper)
    builder.add_node("response_node", response_node_wrapper)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: state.get("intent", "qualitative"),
        {
            "quantitative": "sql_node",
            "qualitative": "rag_node",
        },
    )
    builder.add_edge("sql_node", "response_node")
    builder.add_edge("rag_node", "response_node")
    builder.add_edge("response_node", END)

    return builder.compile()


# ── Entrypoint ───────────────────────────────────────────────────────
def main() -> None:
    model = _build_chat_model()
    graph = _build_graph(model=model)

    port = int(os.environ.get("PORT", "8088"))
    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
     main()
