"""ETL de empresas de Santa Catarina.

Expõe ``load_df_final`` como fonte única dos dados estáticos. O carregamento
é adiado até a primeira consulta, evitando I/O durante a inicialização do agente.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import unicodedata

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data"

_COLUNAS_EMPRESAS = [
    "nr_cnpj", "nr_cnpj_raiz", "ds_tipo_estabelecimento", "dt_cadastro",
    "ds_situacao_cadastral", "ds_porte", "ds_natureza_juridica",
    "dt_situacao_cadastral", "sg_uf_ibge", "nm_uf_ibge", "cd_municipio",
    "nm_municipio", "cd_cnae_subclasse_num", "cd_cnae_subclasse",
    "cd_cnae_subclasse_completa", "ds_cnae_subclasse", "ds_setor_sebrae",
    "ds_sebrae_segmento", "ds_sebrae_subsegmento",
]


def _normalize_municipio(value: object) -> str:
    return unicodedata.normalize("NFKD", str(value)).encode("ASCII", "ignore").decode("utf-8")


def _normalize_municipio_code(value: object) -> object:
    """Converte códigos IBGE para uma chave textual consistente de junção."""
    if pd.isna(value):
        return pd.NA

    code = str(value).strip()
    return code[:-2] if code.endswith(".0") else code


@lru_cache(maxsize=1)
def load_df_final() -> pd.DataFrame:
    """Monta e retorna o DataFrame tidy consolidado usado pelo agente."""
    empresas_path = DATA_PATH / "empresasSC_setembro.csv"
    regionais_path = DATA_PATH / "divisaoMunRegionais.xlsx"

    empresas = pd.read_csv(
        empresas_path,
        sep=",",
        header=0,
        names=_COLUNAS_EMPRESAS,
        low_memory=False,
        converters={"cd_municipio": _normalize_municipio_code},
    )
    regionais = pd.read_excel(
        regionais_path,
        converters={"cod_ibge_municipio": _normalize_municipio_code},
    )
    merged = empresas.merge(
        regionais,
        left_on="cd_municipio",
        right_on="cod_ibge_municipio",
        how="left",
    ).rename(columns={"Gerência Regional Sebrae": "regional", "Território Sebrae": "territorio"})

    common = {
        "métrica": "quantidade_cnpjs",
        "unidade": "empresas",
        "periodo": "set-2026",
        "tipoIndicador": "integer",
        "definicao": "quantidade de empresas por categoria",
    }

    frames: list[pd.DataFrame] = []
    for source_column, category_type, indicator in (
        ("ds_porte", "Porte", "cnpjPorte"),
        ("ds_setor_sebrae", "Setor", "cnpjSetor"),
        ("ds_sebrae_segmento", "Segmento", "cnpjSegmento"),
    ):
        grouped = (
            merged.groupby(
                ["cd_municipio", "nm_municipio", "territorio", "regional", source_column],
                dropna=False,
            )["nr_cnpj"]
            .count()
            .reset_index(name="contagem por cnpj")
            .rename(columns={
                "cd_municipio": "cod_ibge",
                "nm_municipio": "Municipio",
                "territorio": "Territorio",
                "regional": "Regional",
                source_column: "categoria",
            })
        )
        grouped["tipo_categoria"] = category_type
        grouped["nmIndicador"] = indicator
        for column, value in common.items():
            grouped[column] = value
        frames.append(grouped)

    result = pd.concat(frames, ignore_index=True)
    result["categoria"] = result["categoria"].replace({
        "EMPRESA DE PEQUENO PORTE": "EPP",
        "MICRO EMPRESA": "ME",
        "MICROEMPREENDEDOR INDIVIDUAL": "MEI",
    })
    result["Municipio"] = result["Municipio"].map(_normalize_municipio)
    return result