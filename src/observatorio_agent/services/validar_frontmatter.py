"""Valida o frontmatter YAML dos estudos antes da ingestão.

Motivação: o parser do pipeline trata ``yaml.YAMLError`` devolvendo
metadados vazios. O estudo é indexado assim mesmo, mas sem ano, tema
nem regional — e passa a ser invisível para os filtros da busca, sem
que nenhum erro apareça no log.

Este script expõe essas falhas e, opcionalmente, corrige as mais comuns.

Uso:
    python validar_frontmatter.py
    python validar_frontmatter.py --origem ./estudos
    python validar_frontmatter.py --corrigir
    python validar_frontmatter.py --relatorio ./logs/frontmatter.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
import os

_RAIZ = Path(__file__).resolve().parents[1]

EXTENSOES = (".md", ".markdown")

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z", re.DOTALL
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_ANO_RE = re.compile(r"\b(19|20)\d{2}\b")

CAMPOS_ESSENCIAIS = ("title", "ano")
CAMPOS_RECOMENDADOS = ("tema", "regional", "fonte", "data_referencia")

REGIONAIS = (
    "Meio Oeste", "Oeste", "Extremo Oeste", "Centro Norte",
    "Grande Florianópolis", "Serra", "Vale do Itajaí",
    "Foz do Itajaí", "Norte", "Sul",
)

# Estudos de abrangência estadual não citam regional específica.
ABRANGENCIA_ESTADUAL = ("Santa Catarina", "SC", "Estadual")

# Valores que precisam de aspas por conterem caracteres especiais do YAML.
_PRECISA_ASPAS = re.compile(r"^[^\"'].*[:@`|>{}\[\]#&*!%].*$")


@dataclass
class Diagnostico:
    arquivo: str
    status: str
    erro: str | None = None
    campos_ausentes: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    corrigido: bool = False


# ── Detecção de problemas ────────────────────────────────────────────


def _diagnosticar_delimitador(bruto: str) -> str | None:
    """Explica por que a regex de frontmatter não casou."""
    if bruto.startswith("\ufeff"):
        return (
            "Arquivo começa com BOM (\\ufeff) antes do '---'. "
            "Salve como UTF-8 sem BOM."
        )

    if bruto.lstrip().startswith("---") and not bruto.startswith("---"):
        return (
            "Há linha em branco ou espaço antes do '---'. "
            "O bloco deve começar no primeiro caractere do arquivo."
        )

    primeira = bruto.splitlines()[0].strip() if bruto.splitlines() else ""

    if primeira.startswith("--") and not primeira.startswith("---"):
        return f"Delimitador incorreto: '{primeira}'. Use exatamente '---'."

    if bruto.startswith("---") and "\n---" not in bruto[3:]:
        return (
            "Bloco não foi fechado com '---'. "
            "Verifique se o fechamento usa '---' e não '...'."
        )

    if not bruto.startswith("---"):
        return "Arquivo não possui frontmatter."

    return "Frontmatter malformado (delimitadores não reconhecidos)."


def _explicar_erro_yaml(erro: yaml.YAMLError, bloco: str) -> str:
    """Traduz o erro do PyYAML para uma causa acionável."""
    # `problem` traz a causa específica; str(erro) traz só o contexto
    # genérico ("while scanning for the next token").
    problema = getattr(erro, "problem", None) or ""
    mensagem = f"{str(erro).splitlines()[0]} {problema}".strip()

    linha = None
    if hasattr(erro, "problem_mark") and erro.problem_mark:
        linha = erro.problem_mark.line + 1

    contexto = ""
    if linha:
        linhas = bloco.splitlines()
        if 0 < linha <= len(linhas):
            contexto = f" | linha {linha}: {linhas[linha - 1].strip()!r}"

    if "mapping values are not allowed" in mensagem:
        return (
            "Valor contém ':' sem aspas. Envolva em aspas duplas, "
            'ex.: title: "Panorama: abertura de empresas"' + contexto
        )

    if "'\\t'" in mensagem or "\t" in mensagem or "tab" in mensagem.lower():
        return (
            "Indentação com TAB. YAML aceita apenas espaços." + contexto
        )

    if "scanning a quoted scalar" in mensagem:
        return "Aspas abertas e não fechadas." + contexto

    if "special characters are not allowed" in mensagem:
        return "Caractere especial não escapado." + contexto

    if "could not find expected ':'" in mensagem:
        return (
            "Linha sem ':' ou valor iniciando com caractere reservado "
            "(@, `, %, *, &). Use aspas." + contexto
        )

    return mensagem + contexto


def _tentar_corrigir(bloco: str) -> str | None:
    """Aplica correções conservadoras ao bloco YAML.

    Só atua nos casos inequívocos: TAB no início da linha e valores
    escalares contendo ':' sem aspas.
    """
    linhas = bloco.splitlines()
    alterado = False
    saida: list[str] = []

    for linha in linhas:
        nova = linha

        if nova.startswith("\t") or "\t" in nova[: len(nova) - len(nova.lstrip())]:
            nova = nova.replace("\t", "  ")
            alterado = True

        chave_valor = re.match(r"^(\s*)([A-Za-z_][\w-]*):\s+(.+)$", nova)

        if chave_valor:
            recuo, chave, valor = chave_valor.groups()
            valor = valor.strip()

            ja_citado = valor.startswith(('"', "'"))
            eh_lista = valor.startswith("[")
            eh_comentario = valor.startswith("#")

            if (
                not ja_citado
                and not eh_lista
                and not eh_comentario
                and _PRECISA_ASPAS.match(valor)
            ):
                escapado = valor.replace('"', '\\"')
                nova = f'{recuo}{chave}: "{escapado}"'
                alterado = True

        saida.append(nova)

    return "\n".join(saida) if alterado else None


# ── Validação de um arquivo ──────────────────────────────────────────


def validar(caminho: Path, raiz: Path, corrigir: bool = False) -> Diagnostico:
    relativo = str(caminho.relative_to(raiz)).replace("\\", "/")

    bruto = caminho.read_bytes()

    try:
        texto = bruto.decode("utf-8")
    except UnicodeDecodeError:
        texto = bruto.decode("utf-8-sig", errors="replace")

    tinha_bom = texto.startswith("\ufeff")

    achado = _FRONTMATTER_RE.match(texto)

    # ── Delimitador não reconhecido ──────────────────────────────
    if not achado:
        motivo = _diagnosticar_delimitador(texto)

        # BOM e espaço em branco antes do '---' são removíveis com
        # segurança: não alteram o conteúdo do documento.
        if corrigir:
            limpo = texto.lstrip("\ufeff").lstrip()
            if limpo != texto and _FRONTMATTER_RE.match(limpo):
                caminho.write_text(limpo, encoding="utf-8")
                return validar(caminho, raiz, corrigir=False)._substituir(
                    corrigido=True
                )

        return Diagnostico(
            arquivo=relativo,
            status="sem_frontmatter",
            erro=motivo,
        )

    bloco = achado.group("yaml")
    corpo = achado.group("body")

    # ── YAML inválido ────────────────────────────────────────────
    try:
        metadata = yaml.safe_load(bloco)
    except yaml.YAMLError as erro:
        explicacao = _explicar_erro_yaml(erro, bloco)

        if corrigir:
            sugestao = _tentar_corrigir(bloco)

            if sugestao:
                try:
                    yaml.safe_load(sugestao)
                except yaml.YAMLError:
                    pass
                else:
                    novo = f"---\n{sugestao}\n---\n{corpo}"
                    caminho.write_text(novo, encoding="utf-8")
                    resultado = validar(caminho, raiz, corrigir=False)
                    return resultado._substituir(corrigido=True)

        return Diagnostico(
            arquivo=relativo,
            status="yaml_invalido",
            erro=explicacao,
        )

    if not isinstance(metadata, dict):
        return Diagnostico(
            arquivo=relativo,
            status="yaml_invalido",
            erro=(
                f"O bloco não é um mapa de chave/valor "
                f"(tipo: {type(metadata).__name__})."
            ),
        )

    # ── Campos e consistência ────────────────────────────────────
    ausentes: list[str] = []
    avisos: list[str] = []

    for campo in CAMPOS_ESSENCIAIS:
        valor = metadata.get(campo) or metadata.get(
            {"title": "titulo", "ano": "year"}.get(campo, campo)
        )
        if valor in (None, "", [], {}):
            ausentes.append(campo)

    for campo in CAMPOS_RECOMENDADOS:
        if metadata.get(campo) in (None, "", [], {}):
            avisos.append(f"Campo recomendado ausente: {campo}")

    ano = metadata.get("ano") or metadata.get("year")
    if ano is not None:
        try:
            ano_int = int(ano)
            if not 1990 <= ano_int <= 2100:
                avisos.append(f"Ano fora de faixa plausível: {ano}")
        except (TypeError, ValueError):
            avisos.append(f"Ano não numérico: {ano!r}")

    regional = metadata.get("regional")
    if isinstance(regional, str) and regional.strip():
        conhecidas = {r.lower() for j in (REGIONAIS, ABRANGENCIA_ESTADUAL) for r in j}
        if regional.strip().lower() not in conhecidas:
            avisos.append(
                f"Regional '{regional}' não está entre as 10 oficiais; "
                f"nem indica abrangência estadual; "
                f"o filtro por regional não vai encontrá-la."
            )

    if tinha_bom:
        avisos.append("Arquivo contém BOM; prefira UTF-8 sem BOM.")

    if not corpo.strip():
        avisos.append("Documento sem conteúdo após o frontmatter.")

    if not any(_HEADING_RE.match(l.strip()) for l in corpo.splitlines()):
        avisos.append(
            "Nenhum heading (#) no corpo: o chunking recursivo perde a "
            "hierarquia de seções."
        )

    status = "erro" if ausentes else ("aviso" if avisos else "ok")

    return Diagnostico(
        arquivo=relativo,
        status=status,
        campos_ausentes=ausentes,
        avisos=avisos,
    )


def _substituir(self: Diagnostico, **campos: Any) -> Diagnostico:
    dados = asdict(self)
    dados.update(campos)
    return Diagnostico(**dados)


Diagnostico._substituir = _substituir  # type: ignore[attr-defined]


# ── Execução ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida o frontmatter YAML dos estudos do Observatório."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path(os.getenv("ESTUDOS_PATH", _RAIZ / "estudos")),
        help="Diretório dos arquivos Markdown (default: ESTUDOS_PATH).",
    )
    parser.add_argument(
        "--corrigir",
        action="store_true",
        help="Corrige automaticamente BOM, TAB e valores com ':' sem aspas.",
    )
    parser.add_argument(
        "--relatorio",
        type=Path,
        default=None,
        help="Salva o diagnóstico completo em JSON.",
    )
    args = parser.parse_args()

    raiz = args.origem.resolve()

    if not raiz.exists():
        print(f"Diretório não encontrado: {raiz}")
        return 1

    arquivos = sorted(
        c for c in raiz.rglob("*")
        if c.is_file() and c.suffix.lower() in EXTENSOES
    )

    if not arquivos:
        print(f"Nenhum arquivo Markdown em {raiz}")
        return 1

    resultados = [validar(c, raiz, args.corrigir) for c in arquivos]

    ordem = {"sem_frontmatter": 0, "yaml_invalido": 1, "erro": 2, "aviso": 3, "ok": 4}
    resultados.sort(key=lambda d: (ordem.get(d.status, 9), d.arquivo))

    rotulos = {
        "sem_frontmatter": "SEM FRONTMATTER",
        "yaml_invalido": "YAML INVÁLIDO",
        "erro": "CAMPO OBRIGATÓRIO",
        "aviso": "AVISO",
        "ok": "OK",
    }

    print(f"\n{len(arquivos)} arquivo(s) em {raiz}\n")

    for d in resultados:
        if d.status == "ok" and not d.corrigido:
            continue

        marca = " [corrigido]" if d.corrigido else ""
        print(f"{rotulos[d.status]}{marca} — {d.arquivo}")

        if d.erro:
            print(f"    causa: {d.erro}")

        if d.campos_ausentes:
            print(f"    campos obrigatórios ausentes: {', '.join(d.campos_ausentes)}")

        for aviso in d.avisos:
            print(f"    aviso: {aviso}")

        print()

    totais: dict[str, int] = {}
    for d in resultados:
        totais[d.status] = totais.get(d.status, 0) + 1

    print("Resumo:")
    for status in ("ok", "aviso", "erro", "yaml_invalido", "sem_frontmatter"):
        if status in totais:
            print(f"  {rotulos[status]:20} {totais[status]}")

    corrigidos = sum(1 for d in resultados if d.corrigido)
    if corrigidos:
        print(f"  {'CORRIGIDOS':20} {corrigidos}")

    if args.relatorio:
        args.relatorio.parent.mkdir(parents=True, exist_ok=True)
        args.relatorio.write_text(
            json.dumps([asdict(d) for d in resultados], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nRelatório: {args.relatorio.resolve()}")

    bloqueantes = sum(
        totais.get(s, 0)
        for s in ("sem_frontmatter", "yaml_invalido", "erro")
    )

    return 1 if bloqueantes else 0


if __name__ == "__main__":
    sys.exit(main())
