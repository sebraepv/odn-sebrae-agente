"""Leitor do Vault do Obsidian (regras de negócio do Observatório).

Responsabilidade única: ler notas Markdown com frontmatter YAML e
devolver o contexto de regras de negócio relevante para a pergunta.

Não faz embeddings, não faz LLM, não faz IO de rede.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from stopwordsiso import stopwords

# ── Localização do Vault ─────────────────────────────────────────────

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PACKAGE_ROOT / ".env")

_vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
# O Vault complementa a resposta com regras de negócio, mas não é necessário
# para a fonte quantitativa estática derivada de ``etl_empresas.df_final``.
VAULT_PATH = Path(_vault_path).expanduser() if _vault_path else None

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)\Z",
    re.DOTALL,
)

# ── Modelo da nota ───────────────────────────────────────────────────


@dataclass(frozen=True)
class VaultNote:
    """Uma nota do Vault já parseada."""

    path: Path
    title: str
    metadata: dict[str, Any]
    body: str
    keywords: frozenset[str] = field(default_factory=frozenset)

    @property
    def indicador(self) -> str | None:
        return self.metadata.get("indicador")

    @property
    def fonte(self) -> str | None:
        return self.metadata.get("fonte")

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "approved")).lower()

    def to_context(self) -> str:
        """Renderiza a nota como bloco de contexto para o LLM."""
        meta = "\n".join(
            f"- {k}: {v}"
            for k, v in self.metadata.items()
            if v is not None
        )
        return (
            f"### {self.title}\n"
            f"(origem: {self.path.name})\n\n"
            f"Metadados:\n{meta}\n\n"
            f"{self.body.strip()}"
        )


# ── Normalização e tokenização ───────────────────────────────────────


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


_STOPWORDS = {
    _strip_accents(word.lower())
    for language in ("pt", "en")
    for word in stopwords(language)
}


def _variants(token: str) -> set[str]:
    """Gera variações simples para tolerar plural em português."""
    forms = {token}
    if len(token) > 3:
        if token.endswith("es"):
            forms.add(token[:-2])
        if token.endswith("s"):
            forms.add(token[:-1])
        if token.endswith("is"):
            forms.add(token[:-2] + "l")
    return forms


def _tokenize(text: str) -> set[str]:
    normalized = _strip_accents(str(text).lower())
    tokens = re.findall(r"[a-z0-9_]{3,}", normalized)

    result: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        result |= {f for f in _variants(token) if len(f) >= 3}
    return result


# ── Parsing ──────────────────────────────────────────────────────────


def _parse_note(path: Path) -> VaultNote | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    match = _FRONTMATTER_RE.match(raw)

    if match:
        try:
            metadata = yaml.safe_load(match.group("yaml")) or {}
        except yaml.YAMLError:
            metadata = {}
        body = match.group("body")
    else:
        metadata = {}
        body = raw

    if not isinstance(metadata, dict):
        metadata = {}

    title = (
        metadata.get("title")
        or metadata.get("indicador")
        or path.stem.replace("_", " ")
    )

    keywords = _tokenize(title) | _tokenize(path.stem)

    for key in ("indicador", "fonte", "dominio", "tags", "aliases"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                keywords |= _tokenize(item)
        elif value is not None:
            keywords |= _tokenize(value)

    return VaultNote(
        path=path,
        title=str(title),
        metadata=metadata,
        body=body,
        keywords=frozenset(keywords),
    )


@lru_cache(maxsize=1)
def _load_all(vault: Path, mtime_signature: float) -> tuple[VaultNote, ...]:
    notes = []
    for path in sorted(vault.rglob("*.md")):
        if any(part.startswith(".") for part in path.parts):
            continue
        note = _parse_note(path)
        if note is not None and note.status != "deprecated":
            notes.append(note)
    return tuple(notes)


def _signature(vault: Path) -> float:
    """Invalida o cache quando qualquer nota é alterada."""
    try:
        return max(
            (p.stat().st_mtime for p in vault.rglob("*.md")),
            default=0.0,
        )
    except OSError:
        return 0.0


def load_notes(vault: Path | None = None) -> tuple[VaultNote, ...]:
    """Carrega notas válidas do Vault; retorna vazio quando ele não é configurado."""
    target = Path(vault) if vault else VAULT_PATH
    if target is None or not target.exists():
        return ()
    return _load_all(target, _signature(target))


# ── Seleção por relevância ───────────────────────────────────────────


def find_relevant_notes(
    query: str,
    limit: int = 5,
    vault: Path | None = None,
) -> list[VaultNote]:
    """Retorna as notas mais relevantes para a pergunta.

    Ranking lexical simples: interseção de tokens entre a pergunta e as
    palavras-chave da nota, com peso extra para menção no corpo.
    """
    notes = load_notes(vault)
    if not notes:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[int, VaultNote]] = []

    for note in notes:
        score = 3 * len(query_tokens & note.keywords)

        body_tokens = _tokenize(note.body)
        score += len(query_tokens & body_tokens)

        if score > 0:
            scored.append((score, note))

    scored.sort(key=lambda item: (-item[0], item[1].path.name))

    return [note for _, note in scored[:limit]]


def build_business_context(
    query: str,
    limit: int = 5,
    vault: Path | None = None,
) -> str:
    """Monta o bloco de contexto de regras de negócio para o prompt."""
    notes = find_relevant_notes(query, limit=limit, vault=vault)

    if not notes:
        return ""

    blocks = [note.to_context() for note in notes]

    return (
        "## Regras de negócio do Observatório de Negócios\n\n"
        + "\n\n---\n\n".join(blocks)
    )


def load_business_rules(vault: Path | None = None) -> str:
    """Fallback: carrega o Vault inteiro. Use apenas em vaults pequenos."""
    notes = load_notes(vault)
    return "\n\n---\n\n".join(note.to_context() for note in notes)
