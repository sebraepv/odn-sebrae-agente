from langchain_core.messages import AIMessage


async def response_node(state, model):

    return {
        "messages": [
            AIMessage(
                content=state["final_response"]
            )
        ]
    }