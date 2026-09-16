"""Estratégias de embedding conforme Mishra et al. (2026).

Referência:
    Mishra, P. P.; Yeole, K. P.; Keshavamurthy, R.; Surana, M. B.;
    Sarayloo, F. "A Systematic Framework for Enterprise Knowledge
    Retrieval: Leveraging LLM-Generated Metadata to Enhance RAG
    Systems". IEEE CAI 2026. arXiv:2512.05411.

Três estratégias:
    1. content_only    — baseline, apenas o conteúdo do chunk.
    2. tfidf_weighted  — combinação ponderada 90:10 entre o vetor de
                         conteúdo e o vetor de metadados, com os termos
                         de metadados priorizados por TF-IDF e o
                         resultado renormalizado (L2).
    3. prefix_fusion   — metadados concatenados como prefixo textual,
                         gerando um único embedding.

Resultados do paper: recursive + tfidf_weighted alcançou 82.5% de
precisão; naive + prefix_fusion obteve o melhor NDCG (0.813).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

PESO_CONTEUDO = 0.90
PESO_METADADOS = 0.10

MAX_TERMOS_TFIDF = 25

_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")

_STOPWORDS = {
    "para", "com", "por", "dos", "das", "nos", "nas", "que", "uma", "uns",
    "the", "and", "for", "are", "was", "this", "that", "from", "with",
    "sobre", "como", "mais", "seus", "suas", "pelo", "pela", "entre",
    "ser", "foi", "sao", "tem", "pode", "deve", "essa", "esse", "isso",
}


# ── Estrutura de metadados ───────────────────────────────────────────


@dataclass
class MetadadosChunk:
    """Metadados gerados por LLM nas três dimensões do paper."""

    # Estrutural
    tipo_conteudo: str | None = None
    palavras_chave: list[str] = field(default_factory=list)
    resumo: str | None = None

    # Técnica
    entidades: list[str] = field(default_factory=list)
    indicadores: list[str] = field(default_factory=list)
    fontes_dados: list[str] = field(default_factory=list)

    # Contextual
    intencao: str | None = None
    perguntas_atendidas: list[str] = field(default_factory=list)

    def como_texto(self) -> str:
        """Serializa os metadados como texto para embedding."""
        partes: list[str] = []

        if self.tipo_conteudo:
            partes.append(f"Tipo: {self.tipo_conteudo}")
        if self.palavras_chave:
            partes.append(f"Palavras-chave: {', '.join(self.palavras_chave)}")
        if self.indicadores:
            partes.append(f"Indicadores: {', '.join(self.indicadores)}")
        if self.entidades:
            partes.append(f"Entidades: {', '.join(self.entidades)}")
        if self.fontes_dados:
            partes.append(f"Fontes: {', '.join(self.fontes_dados)}")
        if self.intencao:
            partes.append(f"Intenção: {self.intencao}")
        if self.perguntas_atendidas:
            perguntas = "; ".join(self.perguntas_atendidas)
            partes.append(f"Responde a: {perguntas}")
        if self.resumo:
            partes.append(f"Resumo: {self.resumo}")

        return "\n".join(partes)

    def termos(self) -> list[str]:
        """Termos candidatos à ponderação TF-IDF."""
        bruto: list[str] = []
        bruto += self.palavras_chave
        bruto += self.indicadores
        bruto += self.entidades
        bruto += self.fontes_dados

        if self.tipo_conteudo:
            bruto.append(self.tipo_conteudo)
        if self.intencao:
            bruto.append(self.intencao)

        return [t for t in (str(x).strip() for x in bruto) if t]

    def esta_vazio(self) -> bool:
        return not self.como_texto().strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tipo_conteudo": self.tipo_conteudo,
            "palavras_chave": self.palavras_chave,
            "resumo": self.resumo,
            "entidades": self.entidades,
            "indicadores": self.indicadores,
            "fontes_dados": self.fontes_dados,
            "intencao": self.intencao,
            "perguntas_atendidas": self.perguntas_atendidas,
        }

    @classmethod
    def from_dict(cls, dados: dict[str, Any]) -> "MetadadosChunk":
        def lista(valor: Any) -> list[str]:
            if isinstance(valor, (list, tuple, set)):
                return [str(v).strip() for v in valor if str(v).strip()]
            if valor:
                return [str(valor).strip()]
            return []

        return cls(
            tipo_conteudo=dados.get("tipo_conteudo") or None,
            palavras_chave=lista(dados.get("palavras_chave")),
            resumo=dados.get("resumo") or None,
            entidades=lista(dados.get("entidades")),
            indicadores=lista(dados.get("indicadores")),
            fontes_dados=lista(dados.get("fontes_dados")),
            intencao=dados.get("intencao") or None,
            perguntas_atendidas=lista(dados.get("perguntas_atendidas")),
        )


# ── Utilidades vetoriais ─────────────────────────────────────────────


def normalizar_l2(vetor: Sequence[float]) -> list[float]:
    norma = math.sqrt(sum(v * v for v in vetor))
    if norma == 0.0:
        return list(vetor)
    return [v / norma for v in vetor]


def combinar_ponderado(
    vetor_conteudo: Sequence[float],
    vetor_metadados: Sequence[float],
    peso_conteudo: float = PESO_CONTEUDO,
    peso_metadados: float = PESO_METADADOS,
) -> list[float]:
    """Combinação linear seguida de renormalização L2.

    A renormalização é essencial: sem ela, a magnitude resultante varia
    conforme o alinhamento entre os vetores, distorcendo a comparação
    por distância de cosseno entre chunks.
    """
    if len(vetor_conteudo) != len(vetor_metadados):
        raise ValueError(
            f"Dimensões incompatíveis: {len(vetor_conteudo)} "
            f"vs {len(vetor_metadados)}."
        )

    combinado = [
        peso_conteudo * c + peso_metadados * m
        for c, m in zip(vetor_conteudo, vetor_metadados)
    ]

    return normalizar_l2(combinado)


# ── TF-IDF sobre termos de metadados ─────────────────────────────────


def _tokenizar(texto: str) -> list[str]:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", str(texto).lower())
        if unicodedata.category(c) != "Mn"
    )
    return [t for t in _TOKEN_RE.findall(sem_acento) if t not in _STOPWORDS]


class PonderadorTfIdf:
    """Calcula IDF sobre o corpus de metadados.

    Termos que aparecem em quase todos os chunks (ex.: "sebrae",
    "observatorio") recebem peso baixo; termos discriminativos
    (ex.: "saldo_caged", "turismo") recebem peso alto.
    """

    def __init__(self) -> None:
        self._df: dict[str, int] = {}
        self._n_documentos = 0
        self._ajustado = False

    def ajustar(self, corpus_metadados: Iterable[MetadadosChunk]) -> "PonderadorTfIdf":
        self._df.clear()
        self._n_documentos = 0

        for metadados in corpus_metadados:
            self._n_documentos += 1
            vistos = {
                token
                for termo in metadados.termos()
                for token in _tokenizar(termo)
            }
            for token in vistos:
                self._df[token] = self._df.get(token, 0) + 1

        self._ajustado = True
        return self

    def idf(self, token: str) -> float:
        if self._n_documentos == 0:
            return 1.0
        df = self._df.get(token, 0)
        return math.log((1 + self._n_documentos) / (1 + df)) + 1.0

    def termos_relevantes(
        self,
        metadados: MetadadosChunk,
        limite: int = MAX_TERMOS_TFIDF,
    ) -> list[tuple[str, float]]:
        """Termos de metadados ordenados por peso TF-IDF."""
        if not self._ajustado:
            raise RuntimeError("Chame ajustar() antes de ponderar.")

        tf: dict[str, int] = {}
        original: dict[str, str] = {}

        for termo in metadados.termos():
            for token in _tokenizar(termo):
                tf[token] = tf.get(token, 0) + 1
                original.setdefault(token, termo)

        if not tf:
            return []

        total = sum(tf.values())

        pontuados = [
            (original[token], (contagem / total) * self.idf(token))
            for token, contagem in tf.items()
        ]

        pontuados.sort(key=lambda item: (-item[1], item[0]))

        vistos: set[str] = set()
        resultado: list[tuple[str, float]] = []

        for termo, peso in pontuados:
            chave = termo.lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            resultado.append((termo, peso))
            if len(resultado) >= limite:
                break

        return resultado

    def texto_ponderado(
        self,
        metadados: MetadadosChunk,
        limite: int = MAX_TERMOS_TFIDF,
    ) -> str:
        """Texto de metadados priorizado por TF-IDF, para embedding."""
        termos = self.termos_relevantes(metadados, limite)
        if not termos:
            return ""
        return ", ".join(termo for termo, _ in termos)


# ── As três estratégias ──────────────────────────────────────────────


def texto_content_only(chunk_texto: str, metadados: MetadadosChunk) -> str:
    """Baseline: ignora os metadados."""
    return chunk_texto


def texto_prefix_fusion(
    chunk_texto: str,
    metadados: MetadadosChunk,
    documento_titulo: str | None = None,
    heading_path: str | None = None,
) -> str:
    """Metadados como prefixo textual, seguidos do conteúdo.

    Melhor NDCG no paper quando combinado com chunking naive.
    """
    prefixo: list[str] = []

    if documento_titulo:
        prefixo.append(f"Documento: {documento_titulo}")
    if heading_path:
        prefixo.append(f"Seção: {heading_path}")

    corpo_metadados = metadados.como_texto()
    if corpo_metadados:
        prefixo.append(corpo_metadados)

    if not prefixo:
        return chunk_texto

    return "\n".join(prefixo) + "\n\n---\n\n" + chunk_texto


def embedding_tfidf_weighted(
    vetor_conteudo: Sequence[float],
    vetor_metadados: Sequence[float] | None,
    peso_conteudo: float = PESO_CONTEUDO,
    peso_metadados: float = PESO_METADADOS,
) -> list[float]:
    """Combinação 90:10 entre conteúdo e metadados.

    Quando não há metadados, retorna o vetor de conteúdo normalizado —
    degradando para o baseline em vez de falhar.
    """
    if vetor_metadados is None:
        return normalizar_l2(vetor_conteudo)

    return combinar_ponderado(
        vetor_conteudo,
        vetor_metadados,
        peso_conteudo,
        peso_metadados,
    )


ESTRATEGIAS_CHUNKING = {
    "naive": 1,
    "recursive": 2,
    "semantic": 3,
}

ESTRATEGIAS_EMBEDDING = {
    "content_only": 1,
    "tfidf_weighted": 2,
    "prefix_fusion": 3,
}
