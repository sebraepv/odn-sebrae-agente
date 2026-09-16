"""Nó quantitativo (Data Analyst) — fontes estáticas tidy.

Fluxo do nó:
    1. Carrega as regras de negócio aplicáveis do Vault do Obsidian.
    2. Extrai município / regional / porte da pergunta.
    3. Executa a agregação determinística em pandas.
    4. Monta a resposta com métrica, escopo, fonte e data de referência.

O LLM não gera nem executa código: apenas os parâmetros são extraídos.
"""

from __future__ import annotations

from typing import Any

from services.static_loader import (
    QueryResult,
    breakdown_por_porte,
    query_empresas,
    ranking_municipios,
)
from services.vault_loader import build_business_context

PORTE_LABEL = {
    "MEI": "MEIs ativos",
    "ME": "Microempresas ativas",
    "EPP": "Empresas de pequeno porte ativas",
}

LABEL_PADRAO = "Empresas ativas"

RANKING_TERMS = (
    "ranking",
    "top ",
    "maiores",
    "principais municipios",
    "principais municípios",
    "quais municipios",
    "quais municípios",
)

BREAKDOWN_TERMS = (
    "por porte",
    "distribuicao",
    "distribuição",
    "composicao",
    "composição",
    "abertura por porte",
)


def _fmt(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _label(result: QueryResult) -> str:
    return PORTE_LABEL.get(result.porte or "", LABEL_PADRAO)


def _wants(query: str, terms: tuple[str, ...]) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in terms)


def _bloco_ranking(result: QueryResult, query: str) -> list[str]:
    regional = (
        result.escopo_label.replace("Regional ", "")
        if result.escopo == "regional"
        else None
    )

    itens = ranking_municipios(
        limite=10,
        porte=result.porte,
        regional=regional,
    )

    if not itens:
        return []

    linhas = ["", "**Municípios com maiores volumes**", ""]
    linhas += [
        f"{i}. {item['municipio']} ({item['regional']}): "
        f"{_fmt(int(item['total']))}"
        for i, item in enumerate(itens, start=1)
    ]
    return linhas


def _bloco_breakdown(result: QueryResult) -> list[str]:
    itens = breakdown_por_porte(
        escopo_label=result.escopo_label,
        escopo=result.escopo,
    )

    if not itens or result.porte:
        return []

    total = sum(int(i["total"]) for i in itens) or 1

    linhas = ["", "**Distribuição por porte**", ""]
    linhas += [
        f"- {item['porte']}: {_fmt(int(item['total']))} "
        f"({int(item['total']) / total:.1%})"
        for item in itens
    ]
    return linhas


def _montar_resposta(result: QueryResult, query: str) -> str:
    linhas = [
        f"**{_label(result)} — {result.escopo_label}**",
        "",
        f"- Valor apurado: {_fmt(result.valor or 0)}",
        f"- Recorte territorial: {result.escopo_label}",
        f"- Data de referência: {result.data_referencia or 'não informada'}",
    ]

    if result.escopo != "municipio":
        municipios = len({r["municipio"] for r in result.detalhe})
        linhas.append(f"- Municípios considerados: {municipios}")

    if result.fonte:
        linhas.append(f"- Fonte: {result.fonte}")

    if _wants(query, BREAKDOWN_TERMS):
        linhas += _bloco_breakdown(result)

    if _wants(query, RANKING_TERMS):
        linhas += _bloco_ranking(result, query)

    linhas += [
        "",
        "_Resultado calculado sobre fonte estática do Observatório, "
        "conforme regras de negócio vigentes._",
    ]

    return "\n".join(linhas)


async def sql_node(state, model=None) -> dict[str, Any]:
    user_query = state["messages"][-1].content

    business_context = build_business_context(user_query, limit=3)

    result = query_empresas(user_query)

    if not result.encontrado:
        return {
            "business_context": business_context,
            "query_result": {},
            "final_response": result.mensagem,
        }

    return {
        "business_context": business_context,
        "query_result": {
            "indicador": result.indicador,
            "escopo": result.escopo,
            "escopo_label": result.escopo_label,
            "porte": result.porte,
            "valor": result.valor,
            "linhas": result.linhas,
            "data_referencia": result.data_referencia,
            "fonte": result.fonte,
        },
        "final_response": _montar_resposta(result, user_query),
    }
