from langchain_core.messages import HumanMessage


async def router(state, model):
    """Route the conversation while accepting the configured model.

    ``model`` is injected by the graph so this node has the same interface as
    other model-backed nodes, even though the current routing rules are local.
    """

    question = next(
        message.content.lower()
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )

    quantitative_terms = [
        "quantos",
        "quantidade",
        "total",
        "número",
        "valor",
        "quantas",
        "top"
    ]

    if any(
        word in question
        for word in quantitative_terms
    ):
        return {
            "intent": "quantitative"
        }

    return {
        "intent": "qualitative"
    }