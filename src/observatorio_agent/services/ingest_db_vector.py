"""Ingere os estudos Markdown como chunks e embeddings no Azure SQL Database.

Defina ``AZURE_SQL_CONNECTION_STRING`` no ambiente. Para embeddings, defina
``OPEN_AI_ENDPOINT``, ``FOUNDRY_API_KEY``,
``AZURE_AI_EMBEDDING_DEPLOYMENT_NAME`` e
``AZURE_SQL_EMBEDDING_DIMENSIONS`` (a dimensão do deployment configurado).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import pyodbc
import truststore
from dotenv import load_dotenv
from openai import OpenAI

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PACKAGE_ROOT / ".env")

TABLE_NAME = "dbo.rag_chunks"
CHUNK_SIZE = int(os.getenv("AZURE_SQL_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("AZURE_SQL_CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("AZURE_SQL_BATCH_SIZE", "100"))
EMBEDDING_BATCH_SIZE = int(os.getenv("AZURE_SQL_EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_SQL_EMBEDDING_DIMENSIONS", "1536"))
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDINGS_MODEL_DEPLOYMENT_NAME")

_CREATE_TABLE_SQL = f"""
IF OBJECT_ID(N'{TABLE_NAME}', N'U') IS NULL
BEGIN
    CREATE TABLE {TABLE_NAME} (
        id BIGINT IDENTITY(1, 1) NOT NULL PRIMARY KEY,
        source_name NVARCHAR(260) NOT NULL,
        source_path NVARCHAR(1024) NOT NULL,
        chunk_index INT NOT NULL,
        content NVARCHAR(MAX) NOT NULL,
        content_hash CHAR(64) NOT NULL,
        embedding VECTOR({EMBEDDING_DIMENSIONS}) NULL,
        created_at DATETIME2(7) NOT NULL CONSTRAINT DF_rag_chunks_created_at DEFAULT SYSUTCDATETIME(),
        updated_at DATETIME2(7) NOT NULL CONSTRAINT DF_rag_chunks_updated_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT UQ_rag_chunks_source_chunk UNIQUE (source_path, chunk_index)
    );
END;
"""

_UPSERT_SQL = f"""
UPDATE {TABLE_NAME}
SET source_name = ?, content = ?, content_hash = ?,
    embedding = CASE
        WHEN content_hash <> ? OR source_name <> ? THEN NULL
        ELSE embedding
    END,
    updated_at = SYSUTCDATETIME()
WHERE source_path = ? AND chunk_index = ?
  AND (content_hash <> ? OR source_name <> ?);

IF @@ROWCOUNT = 0 AND NOT EXISTS (
    SELECT 1 FROM {TABLE_NAME} WITH (UPDLOCK, HOLDLOCK)
    WHERE source_path = ? AND chunk_index = ?
)
BEGIN
    INSERT INTO {TABLE_NAME} (
        source_name, source_path, chunk_index, content, content_hash
    ) VALUES (?, ?, ?, ?, ?);
END;
"""

_ADD_EMBEDDING_COLUMN_SQL = f"""
IF COL_LENGTH(N'{TABLE_NAME}', N'embedding') IS NULL
BEGIN
    ALTER TABLE {TABLE_NAME}
    ADD embedding VECTOR({EMBEDDING_DIMENSIONS}) NULL;
END;
"""

_SELECT_CHUNKS_WITHOUT_EMBEDDING_SQL = f"""
SELECT id, content
FROM {TABLE_NAME}
WHERE embedding IS NULL
ORDER BY id;
"""

_UPDATE_EMBEDDING_SQL = f"""
UPDATE {TABLE_NAME}
SET embedding = CAST(CONVERT(NVARCHAR(MAX), ?) AS VECTOR({EMBEDDING_DIMENSIONS})),
    updated_at = SYSUTCDATETIME()
WHERE id = ?;
"""

def get_connection() -> pyodbc.Connection:
    """Abre conexão segura com o Azure SQL Database."""
    # conn_str = (
    #     "Driver={ODBC Driver 18 for SQL Server};"
    #     "Server=tcp:sc-sandbox-sqlsrv.database.windows.net,1433;"
    #     "Database=odn-database;"
    #     "Authentication=ActiveDirectoryInteractive;"
    #     "Encrypt=yes;"
    #     "TrustServerCertificate=no;"
    # )

    conn_str = os.getenv("SQL_CONNECTION_STRING")

    print(conn_str)

    conn = pyodbc.connect(conn_str)
    print(f'Conexão OK: {conn}')

    return conn


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[str]:
    """Divide texto preservando parágrafos sempre que possível."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("AZURE_SQL_CHUNK_SIZE deve ser maior que o overlap, que deve ser não negativo.")

    normalized = text.strip()
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            paragraph_break = normalized.rfind("\n\n", start, end)
            line_break = normalized.rfind("\n", start, end)
            word_break = normalized.rfind(" ", start, end)
            boundary = max(paragraph_break, line_break, word_break)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            yield chunk

        if end == text_length:
            break
        start = max(end - overlap, start + 1)


def ensure_table(connection: pyodbc.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(_CREATE_TABLE_SQL)
    connection.commit()


def ensure_embedding_column(connection: pyodbc.Connection) -> None:
    """Adiciona a coluna VECTOR de embeddings caso ela ainda não exista."""
    with connection.cursor() as cursor:
        cursor.execute(_ADD_EMBEDDING_COLUMN_SQL)
    connection.commit()


def get_embedding_client() -> OpenAI:
    """Cria o cliente de embeddings com as credenciais carregadas do .env."""
    endpoint = os.getenv("OPEN_AI_ENDPOINT")
    api_key = os.getenv("FOUNDRY_API_KEY")
    missing = [
        name
        for name, value in {
            "OPEN_AI_ENDPOINT": endpoint,
            "FOUNDRY_API_KEY": api_key,
            "AZURE_AI_EMBEDDING_DEPLOYMENT_NAME": EMBEDDING_DEPLOYMENT,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Variáveis de ambiente obrigatórias ausentes: {', '.join(missing)}."
        )

    # Usa o repositório de certificados do Windows, incluindo a CA corporativa
    # usada por proxies de inspeção TLS, sem desabilitar a verificação HTTPS.
    truststore.inject_into_ssl()
    return OpenAI(base_url=endpoint, api_key=api_key)


def populate_embeddings(connection: pyodbc.Connection) -> int:
    """Gera e persiste embeddings para chunks novos ou alterados."""
    if EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError("AZURE_SQL_EMBEDDING_BATCH_SIZE deve ser maior que zero.")

    with connection.cursor() as cursor:
        pending_chunks = cursor.execute(_SELECT_CHUNKS_WITHOUT_EMBEDDING_SQL).fetchall()

    if not pending_chunks:
        return 0

    client = get_embedding_client()
    updated = 0
    with connection.cursor() as cursor:
        for start in range(0, len(pending_chunks), EMBEDDING_BATCH_SIZE):
            batch = pending_chunks[start:start + EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT,
                input=[row.content for row in batch],
            )
            rows = [
                (str(item.embedding), row.id)
                for item, row in zip(response.data, batch, strict=True)
            ]
            cursor.executemany(_UPDATE_EMBEDDING_SQL, rows)
            updated += len(rows)
        connection.commit()

    return updated


def chunk_parameters(study: Path, studies_path: Path) -> list[tuple[object, ...]]:
    """Monta os parâmetros de upsert para todos os chunks de um estudo."""
    source_path = str(study.relative_to(studies_path.parent))
    rows: list[tuple[object, ...]] = []

    for chunk_index, content in enumerate(split_text(study.read_text(encoding="utf-8"))):
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rows.append((
            study.name, content, content_hash, content_hash, study.name,
            source_path, chunk_index, content_hash, study.name,
            source_path, chunk_index,
            study.name, source_path, chunk_index, content, content_hash,
        ))
    return rows


def ingest_studies() -> int:
    """Cria a tabela e faz upsert dos chunks de todos os estudos Markdown."""
    studies_path = _PACKAGE_ROOT / "estudos"
    studies = sorted(studies_path.glob("*.md"))
    if not studies:
        raise RuntimeError(f"Nenhum estudo Markdown encontrado em {studies_path}.")

    total_chunks = 0
    with get_connection() as connection:
        ensure_table(connection)
        ensure_embedding_column(connection)
        with connection.cursor() as cursor:
            cursor.fast_executemany = True
            for study in studies:
                rows = chunk_parameters(study, studies_path)
                for start in range(0, len(rows), BATCH_SIZE):
                    cursor.executemany(_UPSERT_SQL, rows[start:start + BATCH_SIZE])
                total_chunks += len(rows)
        connection.commit()
        total_embeddings = populate_embeddings(connection)

    print(f"Embeddings atualizados: {total_embeddings}.")
    return total_chunks


if __name__ == "__main__":
    count = ingest_studies()
    print(f"Ingestão concluída: {count} chunks enviados para {TABLE_NAME}.")
