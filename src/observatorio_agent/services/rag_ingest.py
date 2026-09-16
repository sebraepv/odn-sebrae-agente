"""Pipeline de ingestão RAG do Observatório de Negócios — Sebrae/SC.

Configuração única, baseada em Mishra et al. (2026), arXiv:2512.05411,
restrita à combinação de maior precisão reportada no paper (82,5%):

    1. Chunking recursivo por separadores hierárquicos
    2. Geração de metadados por LLM (estrutural, técnico, contextual)
    3. Enriquecimento TF-IDF weighted (90:10), renormalizado L2
    4. Embedding OpenAI text-embedding-3-small (1536 dimensões)
    5. Persistência transacional e idempotente no Azure SQL

Uso:
    python rag_ingest.py
    python rag_ingest.py --forcar
    python rag_ingest.py --sem-metadados     # pula a etapa de LLM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import truststore
truststore.inject_into_ssl()

import pyodbc
import yaml
from dotenv import load_dotenv
from openai import OpenAI

_RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(_RAIZ / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-ingest")

# ── Parâmetros ───────────────────────────────────────────────────────

DIMENSOES = 1536
TAMANHO_ALVO = 1200
SOBREPOSICAO = 200
PESO_CONTEUDO = 0.90
PESO_METADADOS = 0.10
MAX_TERMOS_TFIDF = 25
LOTE_EMBEDDING = 64
LOTE_INSERT = 200
MAX_TENTATIVAS = 4
PROMPT_VERSAO = "v1"

SEPARADORES = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_ANO_RE = re.compile(r"\b(19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_STOPWORDS = {
    "para", "com", "por", "dos", "das", "nos", "nas", "que", "uma",
    "sobre", "como", "mais", "seus", "suas", "pelo", "pela", "entre",
    "ser", "foi", "sao", "tem", "pode", "deve", "essa", "esse", "isso",
    "the", "and", "for", "are", "this", "that", "from", "with",
}


# ── Modelos ──────────────────────────────────────────────────────────


@dataclass
class Documento:
    chave: str
    titulo: str
    markdown: str
    body: str
    metadata: dict[str, Any]

    @property
    def conteudo_hash(self) -> str:
        return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()


@dataclass
class Metadados:
    """Metadados do LLM nas três dimensões do paper."""

    tipo_conteudo: str | None = None
    palavras_chave: list[str] = field(default_factory=list)
    resumo: str | None = None
    entidades: list[str] = field(default_factory=list)
    indicadores: list[str] = field(default_factory=list)
    fontes_dados: list[str] = field(default_factory=list)
    intencao: str | None = None
    perguntas_atendidas: list[str] = field(default_factory=list)

    def termos(self) -> list[str]:
        bruto = [
            *self.palavras_chave,
            *self.indicadores,
            *self.entidades,
            *self.fontes_dados,
        ]
        if self.tipo_conteudo:
            bruto.append(self.tipo_conteudo)
        if self.intencao:
            bruto.append(self.intencao)
        return [t for t in (str(x).strip() for x in bruto) if t]

    def vazio(self) -> bool:
        return not self.termos() and not self.resumo

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
    def from_dict(cls, dados: dict[str, Any]) -> "Metadados":
        def lista(valor: Any) -> list[str]:
            if isinstance(valor, (list, tuple, set)):
                return [str(v).strip() for v in valor if str(v).strip()]
            return [str(valor).strip()] if valor else []

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


@dataclass
class Chunk:
    documento_chave: str
    documento_titulo: str
    indice: int
    texto: str
    heading_path: str = ""
    metadados: Metadados = field(default_factory=Metadados)
    metadados_tfidf: str = ""

    @property
    def chunk_hash(self) -> str:
        base = f"{self.documento_chave}|{self.indice}|{self.texto}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()


# ── Leitura dos estudos ──────────────────────────────────────────────


def _normalizar(texto: Any) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", str(texto))
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def _como_texto(valor: Any) -> str | None:
    if valor is None:
        return None
    if isinstance(valor, (list, tuple, set)):
        itens = [str(v).strip() for v in valor if str(v).strip()]
        return ", ".join(itens) or None
    return str(valor).strip() or None


def ler_documento(path: Path, raiz: Path) -> Documento:
    markdown = path.read_text(encoding="utf-8")
    achado = _FRONTMATTER_RE.match(markdown)

    if achado:
        try:
            metadata = yaml.safe_load(achado.group("yaml")) or {}
        except yaml.YAMLError:
            metadata = {}
        body = achado.group("body")
    else:
        metadata, body = {}, markdown

    if not isinstance(metadata, dict):
        metadata = {}

    titulo = metadata.get("title") or metadata.get("titulo")

    if not titulo:
        for linha in body.splitlines():
            cabecalho = _HEADING_RE.match(linha.strip())
            if cabecalho and len(cabecalho.group(1)) == 1:
                titulo = cabecalho.group(2)
                break

    titulo = titulo or path.stem.replace("_", " ")

    ano = metadata.get("ano") or metadata.get("year")
    if ano is None:
        encontrado = _ANO_RE.search(path.stem) or _ANO_RE.search(str(titulo))
        ano = encontrado.group(0) if encontrado else None

    try:
        metadata["ano"] = int(ano) if ano is not None else None
    except (TypeError, ValueError):
        metadata["ano"] = None

    try:
        chave = str(path.relative_to(raiz)).replace("\\", "/")
    except ValueError:
        chave = path.name

    return Documento(
        chave=chave,
        titulo=str(titulo),
        markdown=markdown,
        body=body,
        metadata=metadata,
    )


# ── 1. Chunking recursivo ────────────────────────────────────────────


def chunk_recursive(
    texto: str,
    tamanho: int = TAMANHO_ALVO,
    sobreposicao: int = SOBREPOSICAO,
) -> list[str]:
    """Divisão hierárquica: do separador mais forte ao mais fraco."""
    limpo = texto.strip()
    if not limpo:
        return []

    def janela_fixa(fragmento: str) -> list[str]:
        partes, inicio, total = [], 0, len(fragmento)
        while inicio < total:
            fim = min(inicio + tamanho, total)
            pedaco = fragmento[inicio:fim].strip()
            if pedaco:
                partes.append(pedaco)
            if fim >= total:
                break
            inicio = max(fim - sobreposicao, inicio + 1)
        return partes

    def dividir(fragmento: str, nivel: int) -> list[str]:
        if len(fragmento) <= tamanho:
            return [fragmento] if fragmento.strip() else []

        if nivel >= len(SEPARADORES):
            return janela_fixa(fragmento)

        sep = SEPARADORES[nivel]
        blocos = [b for b in fragmento.split(sep) if b.strip()]

        if len(blocos) <= 1:
            return dividir(fragmento, nivel + 1)

        resultado: list[str] = []
        buffer: list[str] = []
        acumulado = 0

        for bloco in blocos:
            custo = len(bloco) + len(sep)

            if acumulado + custo > tamanho and buffer:
                agrupado = sep.join(buffer).strip()
                if len(agrupado) > tamanho:
                    resultado.extend(dividir(agrupado, nivel + 1))
                elif agrupado:
                    resultado.append(agrupado)

                cauda = buffer[-1]
                if len(cauda) <= sobreposicao:
                    buffer, acumulado = [cauda, bloco], len(cauda) + custo
                else:
                    buffer, acumulado = [bloco], custo
                continue

            buffer.append(bloco)
            acumulado += custo

        if buffer:
            agrupado = sep.join(buffer).strip()
            if len(agrupado) > tamanho:
                resultado.extend(dividir(agrupado, nivel + 1))
            elif agrupado:
                resultado.append(agrupado)

        return resultado

    return dividir(limpo, 0)


def mapear_headings(texto: str) -> list[tuple[int, str]]:
    mapa: list[tuple[int, str]] = []
    pilha: list[str] = []
    posicao = 0
    em_codigo = False

    for linha in texto.splitlines(keepends=True):
        despido = linha.strip()
        if despido.startswith("```"):
            em_codigo = not em_codigo

        cabecalho = None if em_codigo else _HEADING_RE.match(despido)

        if cabecalho:
            nivel = len(cabecalho.group(1))
            pilha = pilha[: nivel - 1]
            while len(pilha) < nivel - 1:
                pilha.append("")
            pilha.append(cabecalho.group(2).strip())
            mapa.append((posicao, " > ".join(p for p in pilha if p)))

        posicao += len(linha)

    return mapa


def gerar_chunks(doc: Documento) -> list[Chunk]:
    partes = chunk_recursive(doc.body)
    mapa = mapear_headings(doc.body)

    chunks: list[Chunk] = []
    cursor = 0

    for indice, parte in enumerate(partes):
        posicao = doc.body.find(parte[:80], cursor)
        if posicao == -1:
            posicao = cursor
        cursor = max(cursor, posicao)

        caminho = ""
        for inicio, path in mapa:
            if inicio <= posicao:
                caminho = path
            else:
                break

        chunks.append(
            Chunk(
                documento_chave=doc.chave,
                documento_titulo=doc.titulo,
                indice=indice,
                texto=parte,
                heading_path=caminho,
            )
        )

    return chunks


# ── 2. Metadados por LLM ─────────────────────────────────────────────

_SYSTEM = """Você extrai metadados estruturados de trechos de estudos do \
Observatório de Negócios do Sebrae/SC.

Extraia APENAS o que está explicitamente presente no trecho. Não infira, \
não complete com conhecimento externo e não invente termos.

Responda exclusivamente com um objeto JSON válido no formato:

{
  "tipo_conteudo": "<analise | tabela | metodologia | conclusao | \
contextualizacao | recomendacao>",
  "palavras_chave": ["<3 a 8 termos do próprio trecho>"],
  "resumo": "<uma frase objetiva, máximo 200 caracteres>",
  "entidades": ["<municípios, regionais, setores, órgãos citados>"],
  "indicadores": ["<indicadores ou métricas mencionados>"],
  "fontes_dados": ["<fontes de dados citadas>"],
  "intencao": "<o que este trecho informa, máximo 150 caracteres>",
  "perguntas_atendidas": ["<2 a 4 perguntas que o trecho responde>"]
}

Use listas vazias quando não houver o elemento. Escreva em português."""


def _extrair_json(conteudo: str) -> dict[str, Any]:
    texto = conteudo.strip()

    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto)

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    achado = _JSON_RE.search(texto)
    if achado:
        return json.loads(achado.group(0))

    raise ValueError("Resposta do LLM sem JSON válido.")


def gerar_metadados(
    cliente: OpenAI,
    modelo: str,
    chunk: Chunk,
) -> Metadados:
    mensagens = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Título do documento: {chunk.documento_titulo}\n"
                f"Seção: {chunk.heading_path or '(raiz)'}\n\n"
                f"TRECHO:\n{chunk.texto[:6000]}"
            ),
        },
    ]

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = cliente.chat.completions.create(
                model=modelo,
                messages=mensagens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return Metadados.from_dict(
                _extrair_json(resposta.choices[0].message.content or "")
            )
        except Exception as exc:  # noqa: BLE001
            if tentativa == MAX_TENTATIVAS:
                log.error("Metadados indisponíveis para chunk %d: %s",
                          chunk.indice, exc)
                return Metadados()
            time.sleep(2 ** tentativa)

    return Metadados()


# ── 3. Enriquecimento TF-IDF ─────────────────────────────────────────


def _tokenizar(texto: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(_normalizar(texto))
        if t not in _STOPWORDS
    ]


class PonderadorTfIdf:
    """IDF sobre o corpus de metadados.

    Termos presentes em quase todos os chunks (ex.: "sebrae") recebem
    peso baixo; termos discriminativos (ex.: "sazonalidade") recebem
    peso alto.
    """

    def __init__(self) -> None:
        self._df: dict[str, int] = {}
        self._n = 0

    def ajustar(self, corpus: Iterable[Metadados]) -> "PonderadorTfIdf":
        self._df.clear()
        self._n = 0

        for metadados in corpus:
            self._n += 1
            vistos = {
                token
                for termo in metadados.termos()
                for token in _tokenizar(termo)
            }
            for token in vistos:
                self._df[token] = self._df.get(token, 0) + 1

        return self

    def idf(self, token: str) -> float:
        if self._n == 0:
            return 1.0
        return math.log((1 + self._n) / (1 + self._df.get(token, 0))) + 1.0

    def texto_ponderado(
        self,
        metadados: Metadados,
        limite: int = MAX_TERMOS_TFIDF,
    ) -> str:
        tf: dict[str, int] = {}
        origem: dict[str, str] = {}

        for termo in metadados.termos():
            for token in _tokenizar(termo):
                tf[token] = tf.get(token, 0) + 1
                origem.setdefault(token, termo)

        if not tf:
            return ""

        total = sum(tf.values())

        pontuados = sorted(
            (
                (origem[token], (freq / total) * self.idf(token))
                for token, freq in tf.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )

        selecionados: list[str] = []
        vistos: set[str] = set()

        for termo, _ in pontuados:
            chave = termo.lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            selecionados.append(termo)
            if len(selecionados) >= limite:
                break

        return ", ".join(selecionados)


# ── 4. Embeddings ────────────────────────────────────────────────────


def normalizar_l2(vetor: Sequence[float]) -> list[float]:
    norma = math.sqrt(sum(v * v for v in vetor))
    return list(vetor) if norma == 0.0 else [v / norma for v in vetor]


def combinar(
    conteudo: Sequence[float],
    metadados: Sequence[float] | None,
) -> list[float]:
    """Combinação 90:10 com renormalização L2.

    A renormalização é essencial: sem ela a magnitude resultante varia
    conforme o alinhamento entre os vetores, distorcendo a comparação
    por distância de cosseno entre chunks.
    """
    if metadados is None:
        return normalizar_l2(conteudo)

    if len(conteudo) != len(metadados):
        raise ValueError(
            f"Dimensões incompatíveis: {len(conteudo)} vs {len(metadados)}."
        )

    return normalizar_l2([
        PESO_CONTEUDO * c + PESO_METADADOS * m
        for c, m in zip(conteudo, metadados)
    ])


def criar_clientes() -> tuple[OpenAI, str, str]:
    endpoint = os.getenv("OPEN_AI_ENDPOINT")
    api_key = os.getenv("FOUNDRY_API_KEY")

    if not endpoint or not api_key:
        raise RuntimeError(
            "OPEN_AI_ENDPOINT e FOUNDRY_API_KEY precisam estar definidos."
        )

    cliente = OpenAI(api_key=api_key, base_url=endpoint)

    return (
        cliente,
        os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini"),
    )


def embedar(
    cliente: OpenAI,
    modelo: str,
    textos: Sequence[str],
) -> list[list[float]]:
    vetores: list[list[float]] = []

    for inicio in range(0, len(textos), LOTE_EMBEDDING):
        lote = list(textos[inicio : inicio + LOTE_EMBEDDING])

        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                resposta = cliente.embeddings.create(model=modelo, input=lote)
                break
            except Exception:  # noqa: BLE001
                if tentativa == MAX_TENTATIVAS:
                    raise
                time.sleep(2 ** tentativa)

        for item in sorted(resposta.data, key=lambda d: d.index):
            if len(item.embedding) != DIMENSOES:
                raise RuntimeError(
                    f"Modelo '{modelo}' retornou {len(item.embedding)} "
                    f"dimensões; a coluna espera {DIMENSOES}."
                )
            vetores.append(item.embedding)

    return vetores


# ── 5. Persistência ──────────────────────────────────────────────────


def abrir_conexao() -> pyodbc.Connection:
    conn_str = os.getenv("SQL_CONNECTION_STRING")

    if not conn_str:
        raise RuntimeError("SQL_CONNECTION_STRING não definida.")

    if "pwd=" in conn_str.lower():
        log.warning(
            "Connection string com senha embutida. "
            "Prefira Authentication=ActiveDirectoryInteractive."
        )

    return pyodbc.connect(conn_str, autocommit=False)


def hash_documento_atual(conn: pyodbc.Connection, chave: str) -> str | None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT conteudo_hash FROM dbo.documentos WHERE documento_chave = ?",
            chave,
        )
        linha = cursor.fetchone()
        return linha[0] if linha else None
    finally:
        cursor.close()


def _vetor_json(vetor: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vetor) + "]"


_SQL_UPSERT_DOC = """
MERGE dbo.documentos AS destino
USING (SELECT ? AS documento_chave) AS origem
   ON destino.documento_chave = origem.documento_chave
WHEN MATCHED THEN UPDATE SET
    titulo = ?, conteudo_markdown = ?, conteudo_hash = ?,
    ano = ?, tema = ?, setor = ?, municipio = ?, regional = ?,
    fonte = ?, data_referencia = ?, atualizado_em = SYSUTCDATETIME()
WHEN NOT MATCHED THEN INSERT (
    documento_chave, titulo, conteudo_markdown, conteudo_hash,
    ano, tema, setor, municipio, regional, fonte, data_referencia
) VALUES (?,?,?,?,?,?,?,?,?,?,?);
"""

_SQL_INSERT_CHUNK = f"""
INSERT INTO dbo.chunks (
    documento_id, chunk_indice, chunk_texto, chunk_hash, n_caracteres,
    heading_path, tipo_conteudo, palavras_chave, resumo, entidades,
    indicadores, fontes_dados, intencao, perguntas_atendidas,
    metadados_json, metadados_tfidf, embedding, modelo_llm,
    modelo_embedding, peso_conteudo, prompt_versao
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
          CAST(CAST(? AS NVARCHAR(MAX)) AS VECTOR({DIMENSOES})),?,?,?,?);
"""


def gravar(
    conn: pyodbc.Connection,
    doc: Documento,
    chunks: Sequence[Chunk],
    vetores: Sequence[Sequence[float]],
    modelo_embedding: str,
    modelo_llm: str | None,
) -> None:
    """Regrava o documento inteiro numa única transação."""
    meta = doc.metadata

    valores_doc = (
        doc.titulo,
        doc.markdown,
        doc.conteudo_hash,
        meta.get("ano"),
        _como_texto(meta.get("tema")),
        _como_texto(meta.get("setor")),
        _como_texto(meta.get("municipio")),
        _como_texto(meta.get("regional")),
        _como_texto(meta.get("fonte")),
        _como_texto(meta.get("data_referencia")),
    )

    cursor = conn.cursor()
    try:
        cursor.execute(
            _SQL_UPSERT_DOC,
            doc.chave,
            *valores_doc,
            doc.chave,
            *valores_doc,
        )

        cursor.execute(
            "SELECT documento_id FROM dbo.documentos WHERE documento_chave = ?",
            doc.chave,
        )
        documento_id = cursor.fetchone()[0]

        cursor.execute(
            "DELETE FROM dbo.chunks WHERE documento_id = ?",
            documento_id,
        )

        linhas = []
        for chunk, vetor in zip(chunks, vetores):
            m = chunk.metadados
            linhas.append((
                documento_id,
                chunk.indice,
                chunk.texto,
                chunk.chunk_hash,
                len(chunk.texto),
                chunk.heading_path or None,
                m.tipo_conteudo,
                _como_texto(m.palavras_chave),
                m.resumo,
                _como_texto(m.entidades),
                _como_texto(m.indicadores),
                _como_texto(m.fontes_dados),
                m.intencao,
                _como_texto(m.perguntas_atendidas),
                json.dumps(m.to_dict(), ensure_ascii=False),
                chunk.metadados_tfidf or None,
                _vetor_json(vetor),
                modelo_llm,
                modelo_embedding,
                PESO_CONTEUDO,
                PROMPT_VERSAO,
            ))

        for inicio in range(0, len(linhas), LOTE_INSERT):
            cursor.executemany(
                _SQL_INSERT_CHUNK,
                linhas[inicio : inicio + LOTE_INSERT],
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


# ── Orquestração ─────────────────────────────────────────────────────


def ingerir(forcar: bool = False, com_metadados: bool = True) -> dict[str, int]:
    """Ingestão em duas passagens.

    O IDF precisa ser calculado sobre o corpus inteiro, não por documento.
    Ajustar o ponderador documento a documento faria termos genéricos do
    domínio (ex.: "Observatório", "Sebrae") receberem peso alto sempre que
    o estudo tivesse poucos chunks — exatamente o oposto do pretendido.
    Por isso a geração de metadados (passagem 1) precede a ponderação e a
    vetorização (passagem 2).
    """
    raiz = Path(os.getenv("ESTUDOS_PATH", _RAIZ / "estudos"))

    if not raiz.exists():
        raise RuntimeError(f"Diretório de estudos não encontrado: {raiz}")

    arquivos = sorted(raiz.rglob("*.md"))

    if not arquivos:
        raise RuntimeError(f"Nenhum estudo Markdown em {raiz}")

    cliente, modelo_emb, modelo_llm = criar_clientes()

    log.info("%d estudo(s) em %s", len(arquivos), raiz)
    log.info(
        "Embedding: %s (%d dim) | LLM: %s",
        modelo_emb,
        DIMENSOES,
        modelo_llm if com_metadados else "desativado",
    )

    resumo = {"documentos": 0, "inalterados": 0, "chunks": 0}
    conn = abrir_conexao()

    try:
        # ── Passagem 1: leitura, chunking e metadados ──────────────
        pendentes: list[tuple[Documento, list[Chunk]]] = []

        for arquivo in arquivos:
            doc = ler_documento(arquivo, raiz)

            atual = hash_documento_atual(conn, doc.chave)
            if not forcar and atual == doc.conteudo_hash:
                log.info("Inalterado: %s", doc.chave)
                resumo["inalterados"] += 1
                continue

            chunks = gerar_chunks(doc)

            if not chunks:
                log.warning("Sem conteúdo aproveitável: %s", doc.chave)
                continue

            log.info("Analisando %s (%d chunks)", doc.chave, len(chunks))

            if com_metadados:
                for posicao, chunk in enumerate(chunks, start=1):
                    chunk.metadados = gerar_metadados(
                        cliente, modelo_llm, chunk
                    )
                    if posicao % 10 == 0 or posicao == len(chunks):
                        log.info("  metadados %d/%d", posicao, len(chunks))

            pendentes.append((doc, chunks))

        if not pendentes:
            log.info("Nada a processar.")
            return resumo

        # ── Ponderação TF-IDF sobre o corpus completo ──────────────
        todos = [chunk for _, chunks in pendentes for chunk in chunks]

        ponderador = PonderadorTfIdf().ajustar(c.metadados for c in todos)

        for chunk in todos:
            chunk.metadados_tfidf = ponderador.texto_ponderado(chunk.metadados)

        log.info(
            "TF-IDF ajustado sobre %d chunks de %d documento(s).",
            len(todos),
            len(pendentes),
        )

        # ── Passagem 2: embeddings e persistência ──────────────────
        for doc, chunks in pendentes:
            log.info("Vetorizando %s (%d chunks)", doc.chave, len(chunks))

            vetores_conteudo = embedar(
                cliente, modelo_emb, [c.texto for c in chunks]
            )

            com_meta = [i for i, c in enumerate(chunks) if c.metadados_tfidf]

            vetores_meta: dict[int, list[float]] = {}
            if com_meta:
                brutos = embedar(
                    cliente,
                    modelo_emb,
                    [chunks[i].metadados_tfidf for i in com_meta],
                )
                vetores_meta = dict(zip(com_meta, brutos))

            finais = [
                combinar(vetores_conteudo[i], vetores_meta.get(i))
                for i in range(len(chunks))
            ]

            gravar(
                conn,
                doc,
                chunks,
                finais,
                modelo_emb,
                modelo_llm if com_metadados else None,
            )

            resumo["documentos"] += 1
            resumo["chunks"] += len(chunks)

    finally:
        conn.close()

    return resumo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingestão RAG do Observatório de Negócios."
    )
    parser.add_argument("--forcar", action="store_true",
                        help="Reprocessa documentos sem alteração.")
    parser.add_argument("--sem-metadados", action="store_true",
                        help="Pula a geração de metadados por LLM.")
    args = parser.parse_args()

    resultado = ingerir(
        forcar=args.forcar,
        com_metadados=not args.sem_metadados,
    )

    log.info(
        "Concluído: %d documento(s), %d inalterado(s), %d chunks.",
        resultado["documentos"],
        resultado["inalterados"],
        resultado["chunks"],
    )
