import asyncio

from langchain_core.messages import HumanMessage

from main import _build_chat_model, _build_graph


async def run(pergunta: str):
    graph = _build_graph(model=_build_chat_model())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=pergunta)]}
    )

    print(f"\nPergunta: {pergunta}")
    print(f"Intent:   {result.get('intent')}")
    print(f"Resposta: {result['messages'][-1].content}")


async def main():
    await run("Quantas empresas ativas existem em Florianópolis?")
    await run("Quantos MEIs ativos existem em Florianópolis?")
    await run("Qual a quantidade de empresas da regional Sul?")
    await run("Como evoluiu a abertura de empresas no 2º trimestre de 2026?")


if __name__ == "__main__":
    asyncio.run(main())