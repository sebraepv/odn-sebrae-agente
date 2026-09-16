"""Consulta determinística sobre fonte estática tidy (pandas).

Formato esperado (uma linha por município x porte x período):

    codigo_ibge | municipio | regional | porte | data_referencia | total

O LLM não gera código: apenas os parâmetros são extraídos da pergunta
e aplicados como filtros fixos sobre o DataFrame.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from services.etl_empresas import load_df_final

_DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"

DATA_PATH = Path(os.getenv("STATIC_DATA_PATH", _DEFAULT_DATA))

DEFAULT_DATASET = "empresas_por_municipio"

PORTE_ALIASES: dict[str, str] = {
    "mei": "MEI",
    "meis": "MEI",
    "microempreendedor": "MEI",
    "microempreendedores": "MEI",
    "me": "ME",
    "microempresa": "ME",
    "microempresas": "ME",
    "epp": "EPP",
    "pequeno porte": "EPP",
}

Escopo = Literal["municipio", "regional", "estado"]


# ── Resultado ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryResult:
    encontrado: bool
    valor: int | None = None
    indicador: str = ""
    escopo: Escopo = "estado"
    escopo_label: str = ""
    porte: str | None = None
    linhas: int = 0
    detalhe: list[dict[str, Any]] = field(default_factory=list)
    fonte: str = ""
    data_referencia: str = ""
    mensagem: str = ""


# ── Normalização ─────────────────────────────────────────────────────


def _normalize(text: Any) -> str:
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", stripped.lower()).strip()


# ── Carregamento ─────────────────────────────────────────────────────


def _signature(path: Path) -> float:
    try:
        return max(
            (p.stat().st_mtime for p in path.iterdir() if p.is_file()),
            default=0.0,
        )
    except OSError:
        return 0.0


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"codigo_ibge": str})
    return pd.read_json(path, dtype={"codigo_ibge": str})


def _from_df_final() -> pd.DataFrame:
    """Converte o recorte de porte de ``df_final`` para o esquema de consulta."""
    df_final = load_df_final()
    empresas = df_final.loc[df_final["tipo_categoria"] == "Porte"].copy()

    return empresas.rename(columns={
        "cod_ibge": "codigo_ibge",
        "Municipio": "municipio",
        "Regional": "regional",
        "categoria": "porte",
        "periodo": "data_referencia",
        "contagem por cnpj": "total",
    })[[
        "codigo_ibge", "municipio", "regional", "porte", "data_referencia", "total",
    ]]


@lru_cache(maxsize=1)
def _load_from_etl() -> pd.DataFrame:
    return _from_df_final()


@lru_cache(maxsize=4)
def _load(path: Path, dataset: str, signature: float) -> pd.DataFrame:
    for suffix in (".parquet", ".csv", ".json"):
        candidate = path / f"{dataset}{suffix}"
        if candidate.exists():
            df = _read_any(candidate)
            break
    else:
        return pd.DataFrame()

    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df = df.dropna(subset=["total"])
    df["total"] = df["total"].astype("int64")

    df["_municipio_norm"] = df["municipio"].map(_normalize)
    df["_regional_norm"] = df["regional"].map(_normalize)
    df["_porte_norm"] = df["porte"].map(_normalize)

    return df


def load_dataset(
    dataset: str = DEFAULT_DATASET,
    path: Path | None = None,
) -> pd.DataFrame:
    if path is None and dataset == DEFAULT_DATASET:
        df = _load_from_etl().copy()
    else:
        target = Path(path) if path else DATA_PATH
        if not target.exists():
            return pd.DataFrame()
        df = _load(target, dataset, _signature(target))

    if df.empty:
        return df

    df["codigo_ibge"] = df["codigo_ibge"].astype(str)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df = df.dropna(subset=["total"])
    df["total"] = df["total"].astype("int64")
    df["_municipio_norm"] = df["municipio"].map(_normalize)
    df["_regional_norm"] = df["regional"].map(_normalize)
    df["_porte_norm"] = df["porte"].map(_normalize)
    return df


# ── Extração de parâmetros ───────────────────────────────────────────


def detect_porte(query: str) -> str | None:
    normalized = _normalize(query)
    matches = [
        canonical
        for alias, canonical in PORTE_ALIASES.items()
        if re.search(rf"\b{re.escape(alias)}\b", normalized)
    ]
    return matches[0] if matches else None


def _detect_by_column(query: str, df: pd.DataFrame, column: str) -> str | None:
    normalized = _normalize(query)
    norm_column = f"_{column}_norm"

    pairs = (
        df[[column, norm_column]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    hits = [
        original
        for original, norm in pairs
        if norm and re.search(rf"\b{re.escape(norm)}\b", normalized)
    ]

    return max(hits, key=len) if hits else None


def detect_municipio(query: str, df: pd.DataFrame) -> str | None:
    return _detect_by_column(query, df, "municipio")


_LOCAL_RE = re.compile(
    r"\b(?:em|de|do|da|no|na|para)\s+"
    r"((?:[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç'-]+)"
    r"(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç'-]+)*)"
)

_REGIONAL_RE = re.compile(
    r"\bregion(?:al|ais)\s+(?:d[aeo]\s+)?"
    r"((?:[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç'-]+)"
    r"(?:\s+(?:d[aeo]s?\s+)?"
    r"[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\wÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç'-]+)*)"
)

_LOCAIS_GENERICOS = {
    "santa catarina", "sc", "estado", "brasil", "observatorio",
    "observatorio de negocios", "sebrae", "sebrae sc",
}


def detect_local_desconhecido(query: str, df: pd.DataFrame) -> str | None:
    """Detecta topônimo citado na pergunta que não existe na fonte.

    Evita o pior modo de falha: responder o total do estado quando o
    usuário perguntou por um município ausente da base.
    """
    conhecidos = set(df["_municipio_norm"]) | set(df["_regional_norm"])
    conhecidos |= _LOCAIS_GENERICOS

    candidatos = _REGIONAL_RE.findall(query) + _LOCAL_RE.findall(query)

    for bruto in candidatos:
        candidato = _normalize(bruto)
        if not candidato or candidato in conhecidos:
            continue
        if any(candidato in c or c in candidato for c in conhecidos):
            continue
        if candidato in {_normalize(p) for p in PORTE_ALIASES}:
            continue
        return bruto.strip()

    return None


def detect_regional(query: str, df: pd.DataFrame) -> str | None:
    return _detect_by_column(query, df, "regional")


# ── Consulta principal ───────────────────────────────────────────────


def query_empresas(
    user_query: str,
    dataset: str = DEFAULT_DATASET,
    path: Path | None = None,
) -> QueryResult:
    df = load_dataset(dataset, path)

    if df.empty:
        return QueryResult(
            encontrado=False,
            indicador=dataset,
            mensagem=(
                f"A fonte estática '{dataset}' não está disponível ou "
                f"está vazia."
            ),
        )

    porte = detect_porte(user_query)
    municipio = detect_municipio(user_query, df)
    regional = detect_regional(user_query, df)

    if municipio:
        escopo: Escopo = "municipio"
        escopo_label = municipio
        recorte = df[df["municipio"] == municipio]
    elif regional:
        escopo = "regional"
        escopo_label = f"Regional {regional}"
        recorte = df[df["regional"] == regional]
    else:
        desconhecido = detect_local_desconhecido(user_query, df)
        if desconhecido:
            return QueryResult(
                encontrado=False,
                indicador=dataset,
                porte=porte,
                mensagem=(
                    f"Não encontrei '{desconhecido}' na fonte estática do "
                    f"Observatório. Confirme a grafia do município ou da "
                    f"regional. Ausência de registro não deve ser "
                    f"interpretada como valor zero."
                ),
            )

        escopo = "estado"
        escopo_label = "Santa Catarina"
        recorte = df

    if porte:
        recorte = recorte[recorte["porte"] == porte]

    if recorte.empty:
        alvo = f"{escopo_label}"
        if porte:
            alvo += f" / porte {porte}"
        return QueryResult(
            encontrado=False,
            indicador=dataset,
            escopo=escopo,
            escopo_label=escopo_label,
            porte=porte,
            mensagem=(
                f"Não há registros na fonte para o recorte solicitado "
                f"({alvo}). Ausência de registro não deve ser interpretada "
                f"como valor zero."
            ),
        )

    valor = int(recorte["total"].sum())

    data_ref = ", ".join(
        sorted(recorte["data_referencia"].astype(str).unique())
    )

    fonte = ""
    if "fonte" in recorte.columns:
        fonte = ", ".join(sorted(recorte["fonte"].astype(str).unique()))

    detalhe = (
        recorte.drop(columns=[c for c in recorte.columns if c.startswith("_")])
        .to_dict(orient="records")
    )

    return QueryResult(
        encontrado=True,
        valor=valor,
        indicador=dataset,
        escopo=escopo,
        escopo_label=escopo_label,
        porte=porte,
        linhas=len(recorte),
        detalhe=detalhe,
        fonte=fonte,
        data_referencia=data_ref,
    )


# ── Agregações auxiliares ────────────────────────────────────────────


def breakdown_por_porte(
    escopo_label: str | None = None,
    escopo: Escopo = "estado",
    dataset: str = DEFAULT_DATASET,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    df = load_dataset(dataset, path)
    if df.empty:
        return []

    if escopo == "municipio" and escopo_label:
        df = df[df["municipio"] == escopo_label]
    elif escopo == "regional" and escopo_label:
        df = df[df["regional"] == escopo_label.replace("Regional ", "")]

    if df.empty:
        return []

    return (
        df.groupby("porte", as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)
        .to_dict(orient="records")
    )


def ranking_municipios(
    limite: int = 10,
    porte: str | None = None,
    regional: str | None = None,
    dataset: str = DEFAULT_DATASET,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    df = load_dataset(dataset, path)
    if df.empty:
        return []

    if porte:
        df = df[df["porte"] == porte]
    if regional:
        df = df[df["regional"] == regional]

    if df.empty:
        return []

    return (
        df.groupby(["municipio", "regional"], as_index=False)["total"]
        .sum()
        .nlargest(limite, "total")
        .to_dict(orient="records")
    )
