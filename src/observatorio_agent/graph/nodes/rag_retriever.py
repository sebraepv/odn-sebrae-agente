"""Nó qualitativo (Knowledge Retrieval) — busca vetorial no Azure SQL.

Consome a mesma configuração da ingestão: chunks enriquecidos com
metadados por LLM e vetorizados em 1536 dimensões.

A consulta é fixa e parametrizada — o LLM apenas extrai filtros da
pergunta e redige a síntese final. Não há geração de SQL.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

import pyodbc
from openai import OpenAI

DIMENSOES = 1536
TOP_K = 6
DISTANCIA_MAXIMA = 0.50

_ANO_RE = re.compile(r"\b(20\d{2})\b")

REGIONAIS = (
    "Meio Oeste", "Oeste", "Extremo Oeste", "Centro Norte",
    "Grande Florianópolis", "Serra", "Vale do Itajaí",
    "Foz do Itajaí", "Norte", "Sul",
)

FALLBACK = (
    "Desculpe, não encontrei essa informação nos estudos do Observatório "
    "de Negócios. Recomendo consultar o Portal oficial: "
    "https://www.sebrae-sc.com.br/observatorio"
)


@dataclass(frozen=True)
class Trecho:
    chunk_id: int
    documento_titulo: str
    heading_path: str
    texto: str
    ano: int | None
    regional: str | None
    fonte: str | None
    palavras_chave: str | None
    indicadores: str | None
    distancia: float

    @property
    def referencia(self) -> str:
        partes = [self.documento_titulo]

        if self.ano:
            partes.append(str(self.ano))

        # O H1 do documento costuma repetir o título; só cita a seção
        # quando ela acrescenta informação.
        if self.heading_path:
            secao = self.heading_path
            prefixo = f"{self.documento_titulo} > "
            if secao.startswith(prefixo):
                secao = secao[len(prefixo):]
            if secao and secao != self.documento_titulo:
                partes.append(secao)

        return " — ".join(partes)


# ── Extração de filtros ──────────────────────────────────────────────


def _normalizar(texto: Any) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", str(texto))
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def extrair_filtros(pergunta: str) -> dict[str, Any]:
    normalizada = _normalizar(pergunta)
    filtros: dict[str, Any] = {}

    anos = [int(a) for a in _ANO_RE.findall(pergunta)]
    if anos:
        filtros["ano"] = max(anos)

    # Ordem decrescente de tamanho: evita que "Extremo Oeste" case como
    # "Oeste" ou "Centro Norte" case como "Norte".
    for regional in sorted(REGIONAIS, key=len, reverse=True):
        if _normalizar(regional) in normalizada:
            filtros["regional"] = regional
            break

    return filtros


# ── Infraestrutura ───────────────────────────────────────────────────


def _conectar() -> pyodbc.Connection:
    conn_str = os.getenv("SQL_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("SQL_CONNECTION_STRING não definida.")
    return pyodbc.connect(conn_str, autocommit=True)


def _embedar(pergunta: str) -> list[float]:
    cliente = OpenAI(
        api_key=os.environ["FOUNDRY_API_KEY"],
        base_url=os.environ["OPEN_AI_ENDPOINT"],
    )

    modelo = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    resposta = cliente.embeddings.create(model=modelo, input=[pergunta])

    return resposta.data[0].embedding


# ── Busca ────────────────────────────────────────────────────────────


def buscar(
    pergunta: str,
    top_k: int = TOP_K,
    distancia_maxima: float = DISTANCIA_MAXIMA,
    filtros: dict[str, Any] | None = None,
) -> list[Trecho]:
    """Executa a procedure de busca com os filtros extraídos."""
    filtros = filtros if filtros is not None else extrair_filtros(pergunta)

    vetor = _embedar(pergunta)
    vetor_json = "[" + ",".join(f"{v:.8f}" for v in vetor) + "]"

    conn = _conectar()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "{CALL dbo.sp_buscar_chunks (?,?,?,?,?,?)}",
            vetor_json,
            top_k,
            distancia_maxima,
            filtros.get("ano"),
            filtros.get("regional"),
            filtros.get("tema"),
        )
        linhas = cursor.fetchall()
    finally:
        conn.close()

    return [
        Trecho(
            chunk_id=linha[0],
            documento_titulo=linha[1],
            heading_path=linha[2] or "",
            texto=linha[3],
            ano=linha[4],
            regional=linha[5],
            fonte=linha[6],
            palavras_chave=linha[7],
            indicadores=linha[8],
            distancia=float(linha[9]),
        )
        for linha in linhas
    ]


# ── Síntese ──────────────────────────────────────────────────────────

_PROMPT = """Você é o agente do Observatório de Negócios do Sebrae/SC.

Responda à pergunta usando EXCLUSIVAMENTE os trechos fornecidos.
Não estime, não extrapole e não use conhecimento externo.
Se os trechos não sustentarem a resposta, diga que não encontrou.

Cite o estudo que fundamenta cada afirmação.
Seja objetivo, use markdown e evite texto desnecessário.

TRECHOS:
{contexto}

PERGUNTA:
{pergunta}
"""


def montar_contexto(trechos: Sequence[Trecho]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {t.referencia}\n{t.texto}"
        for i, t in enumerate(trechos, start=1)
    )


def _texto_da_resposta(resposta: Any) -> str:
    """Extrai o texto da resposta do modelo, seja ela uma string ou um objeto
    de mensagem do LangChain.
    """
    if isinstance(resposta, str):
        return resposta

    try:
        return resposta.content
    except AttributeError:
        raise ValueError(
            "Resposta inesperada: não é string nem objeto com atributo "
            "'content'."
        )

async def rag_node(state, model=None) -> dict[str, Any]:
    pergunta = state["messages"][-1].content
    filtros = extrair_filtros(pergunta)

    try:
        trechos = buscar(pergunta, filtros=filtros)
    except Exception as exc:  # noqa: BLE001
        return {
            "retrieved_chunks": [],
            "final_response": (
                "Não foi possível consultar a base de estudos no momento. "
                f"Detalhe técnico: {exc}"
            ),
        }

    if not trechos:
        return {"retrieved_chunks": [], "final_response": FALLBACK}

    recuperados = [
        {
            "chunk_id": t.chunk_id,
            "documento": t.documento_titulo,
            "secao": t.heading_path,
            "ano": t.ano,
            "distancia": round(t.distancia, 4),
        }
        for t in trechos
    ]

    if model is None:
        corpo = "\n\n".join(
            f"**{t.referencia}**\n\n{t.texto}" for t in trechos
        )
        return {"retrieved_chunks": recuperados, "final_response": corpo}

    resposta = await model.ainvoke(
        _PROMPT.format(
            contexto=montar_contexto(trechos),
            pergunta=pergunta,
        )
    )

    fontes = "\n".join(f"- {t.referencia}" for t in trechos)

    return {
        "retrieved_chunks": recuperados,
        "final_response": (
            f"{_texto_da_resposta(resposta)}\n\n**Estudos consultados**\n\n{fontes}"
        ),
    }
