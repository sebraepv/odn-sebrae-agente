"""Gera o frontmatter YAML dos estudos do Observatório.

Os arquivos produzidos pelo fluxo de conversão contêm apenas o corpo em
Markdown. Sem o bloco YAML, o pipeline grava ``tema``, ``setor``,
``municipio`` e ``regional`` como NULL — e esses campos alimentam os
filtros da busca vetorial.

Estratégia em duas camadas:

    1. Heurística determinística sobre nome do arquivo, título e corpo:
       resolve ano, regionais, municípios e temas por vocabulário
       controlado. Não custa nada e é reproduzível.

    2. LLM (opcional, --llm): preenche apenas o que a heurística não
       resolveu, lendo os primeiros trechos do documento.

O conteúdo do estudo nunca é alterado: o bloco é inserido acima do corpo.

Uso:
    python gerar_frontmatter.py --origem ../estudos --simular
    python gerar_frontmatter.py --origem ../estudos
    python gerar_frontmatter.py --origem ../estudos --llm
    python gerar_frontmatter.py --origem ../estudos --sobrescrever
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

_RAIZ = Path(__file__).resolve().parents[1]

log = logging.getLogger("frontmatter")

EXTENSOES = (".md", ".markdown")

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_ANO_RE = re.compile(r"\b(19|20)\d{2}\b")
_TRIMESTRE_RE = re.compile(
    r"\b(?:([1-4])\s*[ºo]?\s*(?:tri|trimestre)|t([1-4]))\b", re.IGNORECASE
)
_MES_ANO_RE = re.compile(
    r"\b(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[a-z]*"
    r"[\s/-]*(20\d{2})\b",
    re.IGNORECASE,
)

MESES = {
    "jan": "01", "fev": "02", "mar": "03", "abr": "04",
    "mai": "05", "jun": "06", "jul": "07", "ago": "08",
    "set": "09", "out": "10", "nov": "11", "dez": "12",
}

FONTE_PADRAO = "Observatório de Negócios - Sebrae/SC"

# As 10 regionais oficiais do Sebrae/SC.
REGIONAIS = (
    "Meio Oeste",
    "Oeste",
    "Extremo Oeste",
    "Centro Norte",
    "Grande Florianópolis",
    "Serra",
    "Vale do Itajaí",
    "Foz do Itajaí",
    "Norte",
    "Sul",
)

# TERRITORIOS = (
#     "Chapecó",
#     "Rio do Sul",
#     "Joaçaba",
#     "Florianópolis",
#     "São Miguel do Oeste",
#     "Blumenau",
#     "Joinville",
#     "Criciúma",
#     "Tubarão",
#     "Caçador",
#     "Itajaí",
#     "Jaraguá do Sul",
#     "Brusque",
#     "São Bento do Sul"
# )

# ESTADUAL = (
#     "Santa Catarina",
#     "Estadual"
# )

# Vocabulário controlado de temas do Observatório.
TEMAS = {
    "empresas": (
        "abertura de empresas", "empresas ativas", "baixa de empresas",
        "cnpj", "estabelecimento", "sobrevivencia de empresas",
    ),
    "empreendedorismo": (
        "empreendedor", "empreendedorismo", "mei",
        "microempreendedor", "perfil do empreendedor",
    ),
    "emprego": (
        "caged", "emprego formal", "saldo de empregos",
        "admissoes", "desligamentos", "massa salarial", "rais",
    ),
    "economia": (
        "pib", "inflacao", "ipca", "conjuntura", "atividade economica",
        "faturamento", "cenario economico",
    ),
    "consumo": (
        "potencial de consumo", "habitos de consumo", "varejo",
        "comportamento do consumidor", "ticket medio",
    ),
    "territorio": (
        "territorial", "regionalizacao", "municipios",
        "desenvolvimento regional",
    ),
    "turismo": (
        "turismo", "turista", "hotelaria", "ocupacao hoteleira",
        "sazonalidade", "eventos",
    ),
    "credito": (
        "credito", "financiamento", "juros", "endividamento",
        "capital de giro", "inadimplencia",
    ),
    "inovacao": (
        "inovacao", "startup", "tecnologia", "digitalizacao",
        "transformacao digital", "inteligencia artificial",
    ),
}

SETORES = {
    "Comércio": ("comercio", "varejo", "atacado", "lojista"),
    "Serviços": ("servicos", "prestador de servico"),
    "Indústria": ("industria", "industrial", "manufatura", "fabril"),
    "Agronegócio": ("agronegocio", "agropecuaria", "rural", "agricultura"),
    "Construção Civil": ("construcao civil", "construcao"),
    "Turismo": ("turismo", "hotelaria", "gastronomia"),
}

# Municípios citados com frequência nos estudos do Observatório.
MUNICIPIOS = (
    "Abdon Batista",  "Abelardo Luz",
    "Agrolândia", "Agronômica", "Água Doce", "Águas de Chapecó", "Águas Frias",
    "Águas Mornas", "Alfredo Wagner", "Alto Bela Vista", "Anchieta", "Angelina",
    "Anita Garibaldi", "Anitápolis", "Antônio Carlos", "Apiúna", "Arabutã", "Araquari",
    "Araranguá", "Armazém", "Arroio Trinta", "Arvoredo", "Ascurra", "Atalanta", "Aurora",
    "Balneário Arroio do Silva","Balneário Barra do Sul", "Balneário Camboriú",
    "Balneário Gaivota", "Balneário Piçarras", "Balneário Rincão", "Bandeirante",
    "Barra Bonita", "Barra Velha", "Bela Vista do Toldo", "Belmonte", "Benedito Novo", "Biguaçu",
    "Blumenau", "Bocaina do Sul", "Bom Jardim da Serra", "Bom Jesus", "Bom Jesus do Oeste", "Bom Retiro",
    "Bombinhas", "Botuverá", "Braço do Norte", "Braço do Trombudo", "Brunópolis", "Brusque",
    "Caçador", "Caibi", "Calmon", "Camboriú", "Campo Alegre", "Campo Belo do Sul",
    "Campo Erê", "Campos Novos", "Canelinha", "Canoinhas", "Capão Alto", "Capinzal",
    "Capivari de Baixo", "Catanduvas", "Caxambu do Sul", "Celso Ramos", "Cerro Negro", "Chapadão do Lageado",
    "Chapecó", "Cocal do Sul", "Concórdia", "Cordilheira Alta", "Coronel Freitas", "Coronel Martins",
    "Correia Pinto", "Corupá", "Criciúma", "Cunha Porã", "Cunhataí", "Curitibanos",
    "Descanso", "Dionísio Cerqueira", "Dona Emma", "Doutor Pedrinho", "Entre Rios", "Ermo",
    "Erval Velho", "Faxinal dos Guedes", "Flor do Sertão", "Florianópolis", "Formosa do Sul", "Forquilhinha", "Fraiburgo",
    "Frei Rogério", "Galvão", "Garopaba", "Garuva", "Gaspar", "Governador Celso Ramos",
    "Grão Pará", "Gravatal", "Guabiruba", "Guaraciaba", "Guaramirim", "Guarujá do Sul", "Guatambú",
    "Herval d'Oeste", "Ibiam", "Ibicaré", "Ibirama", "Içara", "Ilhota", "Imaruí",
    "Imbituba", "Imbuia", "Indaial", "Iomerê", "Ipira", "Iporã do Oeste",
    "Ipuaçu", "Ipumirim", "Iraceminha", "Irani", "Irati", "Irineópolis",
    "Itá", "Itaiópolis", "Itajaí", "Itapema", "Itapiranga", "Itapoá", "Ituporanga", "Jaborá", "Jacinto Machado", "Jaguaruna",
    "Jaraguá do Sul", "Jardinópolis", "Joaçaba", "Joinville", "José Boiteux", "Jupiá",
    "Lacerdópolis", "Lages", "Laguna", "Lajeado Grande", "Laurentino", "Lauro Muller", "Lebon Régis",
    "Leoberto Leal", "Lindóia do Sul", "Lontras", "Luiz Alves", "Luzerna", "Macieira", "Mafra",
    "Major Gercino", "Major Vieira", "Maracajá", "Maravilha", "Marema", "Massaranduba", "Matos Costa",
    "Meleiro", "Mirim Doce", "Modelo", "Mondaí", "Monte Carlo", "Monte Castelo", "Morro da Fumaça", "Morro Grande",
    "Navegantes", "Nova Erechim", "Nova Itaberaba", "Nova Trento", "Nova Veneza", "Novo Horizonte", "Orleans",
    "Otacílio Costa", "Ouro", "Ouro Verde", "Paial", "Painel", "Palhoça", "Palma Sola", "Palmeira",
    "Palmitos", "Papanduva", "Paraíso", "Passo de Torres", "Passos Maia", "Paulo Lopes", "Pedras Grandes",
    "Penha", "Peritiba", "Pescaria Brava", "Petrolândia", "Pinhalzinho", "Pinheiro Preto",
    "Piratuba", "Planalto Alegre", "Pomerode", "Ponte Alta", "Ponte Alta do Norte", "Ponte Serrada",
    "Porto Belo", "Porto União", "Pouso Redondo", "Praia Grande", "Presidente Castello Branco", "Presidente Getúlio",
    "Presidente Nereu", "Princesa", "Quilombo", "Rancho Queimado", "Rio das Antas", "Rio do Campo", "Rio do Oeste",
    "Rio do Sul", "Rio dos Cedros", "Rio Fortuna", "Rio Negrinho", "Rio Rufino", "Riqueza", "Rodeio", "Romelândia",
    "Salete", "Saltinho", "Salto Veloso", "Sangão", "Santa Cecília", "Santa Helena", "Santa Rosa de Lima", "Santa Rosa do Sul",
    "Santa Terezinha", "Santa Terezinha do Progresso", "Santiago do Sul",
    "Santo Amaro da Imperatriz", "São Bento do Sul", "São Bernardino", "São Bonifácio",
    "São Carlos", "São Cristovão do Sul", "São Domingos",
    "São Francisco do Sul", "São João Batista", "São João do Itaperiú", "São João do Oeste",
    "São João do Sul", "São Joaquim", "São José", "São José do Cedro", "São José do Cerrito", "São Lourenço do Oeste", "São Ludgero",
    "São Martinho", "São Miguel da Boa Vista", "São Miguel do Oeste", "São Pedro de Alcântara", "Saudades", "Schroeder",
    "Seara", "Serra Alta", "Siderópolis", "Sombrio", "Sul Brasil", "Taió",
    "Tangará", "Tigrinhos", "Tijucas", "Timbé do Sul", "Timbó", "Timbó Grande", "Três Barras",
    "Treviso", "Treze de Maio", "Treze Tílias", "Trombudo Central", "Tubarão", "Tunápolis", "Turvo",
    "União do Oeste", "Urubici", "Urupema", "Urussanga", "Vargeão", "Vargem",
    "Vargem Bonita", "Vidal Ramos", "Videira", "Vitor Meireles", "Witmarsum",
    "Xanxerê", "Xavantina", "Xaxim", "Zortéa"
)

@dataclass
class Resultado:
    arquivo: str
    status: str
    campos: dict[str, Any] = field(default_factory=dict)
    origem_campos: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


# ── Normalização ─────────────────────────────────────────────────────


def _normalizar(texto: Any) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", str(texto))
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


def _contem(agulha: str, palheiro: str) -> bool:
    """Busca por palavra inteira, evitando casar 'sul' dentro de 'consultoria'."""
    return re.search(rf"\b{re.escape(agulha)}\b", palheiro) is not None


# ── Extração determinística ──────────────────────────────────────────


def extrair_titulo(corpo: str, nome_arquivo: str) -> str:
    for linha in corpo.splitlines():
        cabecalho = _HEADING_RE.match(linha.strip())
        if cabecalho and len(cabecalho.group(1)) == 1:
            return cabecalho.group(2).strip()

    return nome_arquivo.replace("-", " ").replace("_", " ").strip().capitalize()


def extrair_ano(nome_arquivo: str, titulo: str, corpo: str) -> int | None:
    for origem in (nome_arquivo, titulo, corpo[:2000]):
        encontrados = [int(m.group(0)) for m in _ANO_RE.finditer(str(origem))]
        plausiveis = [a for a in encontrados if 1990 <= a <= 2100]
        if plausiveis:
            return max(plausiveis)
    return None


def extrair_data_referencia(
    nome_arquivo: str, titulo: str, ano: int | None
) -> str | None:
    base = f"{nome_arquivo} {titulo}"

    mes = _MES_ANO_RE.search(base)
    if mes:
        sigla = mes.group(1).lower()[:3]
        return f"{mes.group(2)}-{MESES.get(sigla, '01')}"

    trimestre = _TRIMESTRE_RE.search(base)
    if trimestre and ano:
        numero = trimestre.group(1) or trimestre.group(2)
        return f"{ano}-Q{numero}"

    return str(ano) if ano else None


def extrair_regional(texto_normalizado: str) -> str | None:
    """Detecta a regional citada, priorizando nomes mais específicos.

    A ordem por tamanho evita que 'Extremo Oeste' case como 'Oeste'
    e 'Centro Norte' case como 'Norte'.
    """
    for regional in sorted(REGIONAIS, key=len, reverse=True):
        if _contem(_normalizar(regional), texto_normalizado):
            return regional
    return None


def extrair_municipios(texto_normalizado: str, limite: int = 5) -> list[str]:
    """Detecta municípios citados, filtrando dois falsos positivos comuns.

    - Nomes contidos em regionais: 'Itajaí' aparece em 'Vale do Itajaí'.
    - Nomes contidos em outros municípios: 'Camboriú' em 'Balneário Camboriú'.
    """
    texto = texto_normalizado

    # Neutraliza as regionais antes de procurar municípios.
    for regional in REGIONAIS:
        texto = re.sub(
            rf"\b{re.escape(_normalizar(regional))}\b", " ", texto
        )

    # # Neutralizar os territorios antes de procurar municípios.
    # for territorio in TERRITORIOS:
    #     texto = re.sub(
    #         rf"\b{re.escape(_normalizar(territorio))}\b", " ", texto
    #     )
    
    encontrados = [
        municipio
        for municipio in MUNICIPIOS
        if _contem(_normalizar(municipio), texto)
    ]

    # Descarta o nome curto quando o composto também foi detectado.
    filtrados = [
        municipio
        for municipio in encontrados
        if not any(
            outro != municipio
            and _normalizar(municipio) in _normalizar(outro)
            for outro in encontrados
        )
    ]

    return filtrados[:limite]


def _pontuar_vocabulario(
    texto_normalizado: str,
    vocabulario: dict[str, tuple[str, ...]],
) -> list[tuple[str, int]]:
    pontuados: list[tuple[str, int]] = []

    for rotulo, termos in vocabulario.items():
        pontos = sum(
            len(re.findall(rf"\b{re.escape(_normalizar(t))}\b", texto_normalizado))
            for t in termos
        )
        if pontos:
            pontuados.append((rotulo, pontos))

    pontuados.sort(key=lambda item: (-item[1], item[0]))

    return pontuados


def extrair_temas(texto_normalizado: str, limite: int = 3) -> list[str]:
    return [t for t, _ in _pontuar_vocabulario(texto_normalizado, TEMAS)[:limite]]


def extrair_setores(texto_normalizado: str, limite: int = 3) -> list[str]:
    pontuados = _pontuar_vocabulario(texto_normalizado, SETORES)
    # Exige menção recorrente: uma citação isolada não define o setor.
    return [s for s, pontos in pontuados if pontos >= 2][:limite]


# ── Enriquecimento por LLM ───────────────────────────────────────────

_SYSTEM_LLM = """Você classifica estudos do Observatório de Negócios do \
Sebrae/SC para catalogação.

Baseie-se EXCLUSIVAMENTE no texto fornecido. Não infira e não invente.

Responda apenas com JSON válido:

{{
  "tema": ["<1 a 3 de: {temas}>"],
  "setor": ["<0 a 2 de: {setores}>"],
  "regional": "<uma de: {regionais}, ou 'Santa Catarina' se o estudo for \
estadual, ou null>",
  "municipio": ["<municípios catarinenses centrais no estudo, ou vazio>"]
}}

Use listas vazias ou null quando o texto não permitir determinar."""


def enriquecer_com_llm(
    titulo: str,
    corpo: str,
    campos: dict[str, Any],
) -> dict[str, Any]:
    """Preenche apenas os campos que a heurística não resolveu."""
    faltantes = [
        campo
        for campo in ("tema", "setor", "regional", "municipio")
        if not campos.get(campo)
    ]

    if not faltantes:
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("Pacote 'openai' indisponível; --llm ignorado.")
        return {}

    endpoint = os.getenv("OPEN_AI_ENDPOINT")
    api_key = os.getenv("FOUNDRY_API_KEY")

    if not endpoint or not api_key:
        log.warning("OPEN_AI_ENDPOINT/FOUNDRY_API_KEY ausentes; --llm ignorado.")
        return {}

    cliente = OpenAI(api_key=api_key, base_url=endpoint)
    modelo = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4-mini")

    instrucao = _SYSTEM_LLM.format(
        temas=", ".join(TEMAS),
        setores=", ".join(SETORES),
        regionais=", ".join(REGIONAIS),
    )

    try:
        resposta = cliente.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": instrucao},
                {
                    "role": "user",
                    "content": f"Título: {titulo}\n\nTexto:\n{corpo[:6000]}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        dados = json.loads(resposta.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Falha ao consultar o LLM: %s", exc)
        return {}

    return {campo: dados.get(campo) for campo in faltantes if dados.get(campo)}


# ── Montagem do bloco ────────────────────────────────────────────────

ORDEM_CAMPOS = (
    "title",
    "ano",
    "tema",
    "setor",
    "municipio",
    "regional",
    "fonte",
    "data_referencia",
)


def _escalar_ou_lista(valor: Any) -> Any:
    """Lista com um único item vira escalar, para manter o YAML enxuto."""
    if isinstance(valor, (list, tuple)):
        itens = [str(v).strip() for v in valor if str(v).strip()]
        if not itens:
            return None
        return itens[0] if len(itens) == 1 else itens
    return valor


def montar_bloco(campos: dict[str, Any]) -> str:
    ordenado = {
        chave: _escalar_ou_lista(campos.get(chave))
        for chave in ORDEM_CAMPOS
        if campos.get(chave) not in (None, "", [], {})
    }

    corpo = yaml.safe_dump(
        ordenado,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()

    return f"---\n{corpo}\n---\n\n"


def analisar(
    caminho: Path,
    raiz: Path,
    usar_llm: bool = False,
) -> Resultado:
    relativo = str(caminho.relative_to(raiz)).replace("\\", "/")

    bruto = caminho.read_bytes()

    try:
        texto = bruto.decode("utf-8")
    except UnicodeDecodeError:
        texto = bruto.decode("utf-8-sig", errors="replace")

    texto = texto.lstrip("\ufeff")

    ja_tem = bool(_FRONTMATTER_RE.match(texto))
    corpo = _FRONTMATTER_RE.sub("", texto, count=1) if ja_tem else texto

    nome_base = caminho.stem
    titulo = extrair_titulo(corpo, nome_base)

    # Cabeçalho do documento concentra os termos mais relevantes.
    amostra = _normalizar(f"{nome_base} {titulo} {corpo[:8000]}")

    ano = extrair_ano(nome_base, titulo, corpo)

    campos: dict[str, Any] = {
        "title": titulo,
        "ano": ano,
        "tema": extrair_temas(amostra),
        "setor": extrair_setores(amostra),
        "municipio": extrair_municipios(amostra),
        "regional": extrair_regional(amostra),
        "fonte": FONTE_PADRAO,
        "data_referencia": extrair_data_referencia(nome_base, titulo, ano),
    }

    origem = {
        chave: "heuristica" if valor else "ausente"
        for chave, valor in campos.items()
    }

    if usar_llm:
        complemento = enriquecer_com_llm(titulo, corpo, campos)
        for chave, valor in complemento.items():
            campos[chave] = valor
            origem[chave] = "llm"

    avisos: list[str] = []

    if not campos["ano"]:
        avisos.append("Ano não identificado; informe manualmente.")

    if not campos["regional"]:
        # Estudos estaduais não citam regional — é o caso mais comum.
        campos["regional"] = "Santa Catarina"
        origem["regional"] = "padrao"

    if not campos["tema"]:
        avisos.append("Tema não identificado pelo vocabulário controlado.")

    return Resultado(
        arquivo=relativo,
        status="pronto" if not ja_tem else "ja_possui",
        campos=campos,
        origem_campos=origem,
        avisos=avisos,
    )


def aplicar(caminho: Path, campos: dict[str, Any], sobrescrever: bool) -> None:
    bruto = caminho.read_bytes()

    try:
        texto = bruto.decode("utf-8")
    except UnicodeDecodeError:
        texto = bruto.decode("utf-8-sig", errors="replace")

    texto = texto.lstrip("\ufeff")

    if _FRONTMATTER_RE.match(texto):
        if not sobrescrever:
            return
        texto = _FRONTMATTER_RE.sub("", texto, count=1)

    caminho.write_text(
        montar_bloco(campos) + texto.lstrip("\n"),
        encoding="utf-8",
    )


# ── Execução ─────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Gera frontmatter YAML para os estudos do Observatório."
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path(os.getenv("ESTUDOS_PATH", _RAIZ / "estudos")),
        help="Diretório dos arquivos Markdown.",
    )
    parser.add_argument(
        "--simular",
        action="store_true",
        help="Mostra o que seria gerado, sem alterar arquivos.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Usa o LLM para completar campos que a heurística não resolveu.",
    )
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="Substitui o frontmatter de arquivos que já possuem um.",
    )
    parser.add_argument(
        "--relatorio",
        type=Path,
        default=None,
        help="Salva o resultado em JSON.",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_RAIZ / ".env")
    except ImportError:
        pass

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

    print(f"\n{len(arquivos)} arquivo(s) em {raiz}")
    print("Modo: " + ("simulação" if args.simular else "aplicação"))
    if args.llm:
        print("LLM: ativo para campos não resolvidos pela heurística")
    print()

    resultados: list[Resultado] = []

    for caminho in arquivos:
        resultado = analisar(caminho, raiz, usar_llm=args.llm)

        if resultado.status == "ja_possui" and not args.sobrescrever:
            print(f"IGNORADO (já possui) — {resultado.arquivo}")
            resultados.append(resultado)
            continue

        print(f"--- {resultado.arquivo}")
        print(montar_bloco(resultado.campos).rstrip())

        marcados = [
            f"{c}={o}"
            for c, o in resultado.origem_campos.items()
            if o in ("llm", "padrao")
        ]
        if marcados:
            print(f"    origem: {', '.join(marcados)}")

        for aviso in resultado.avisos:
            print(f"    aviso: {aviso}")

        if not args.simular:
            aplicar(caminho, resultado.campos, args.sobrescrever)
            print("    gravado")

        print()
        resultados.append(resultado)

    totais: dict[str, int] = {}
    for r in resultados:
        totais[r.status] = totais.get(r.status, 0) + 1

    print("Resumo:")
    for status, quantidade in sorted(totais.items()):
        print(f"  {status:14} {quantidade}")

    com_aviso = sum(1 for r in resultados if r.avisos)
    if com_aviso:
        print(f"  {'com avisos':14} {com_aviso}")

    if args.simular:
        print("\nNenhum arquivo foi alterado. Remova --simular para aplicar.")

    if args.relatorio:
        args.relatorio.parent.mkdir(parents=True, exist_ok=True)
        args.relatorio.write_text(
            json.dumps(
                [
                    {
                        "arquivo": r.arquivo,
                        "status": r.status,
                        "campos": r.campos,
                        "origem_campos": r.origem_campos,
                        "avisos": r.avisos,
                    }
                    for r in resultados
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nRelatório: {args.relatorio.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
