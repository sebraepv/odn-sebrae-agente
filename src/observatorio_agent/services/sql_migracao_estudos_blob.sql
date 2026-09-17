/* =====================================================================
   Migração: origem dos estudos no Azure Blob Storage

   Acrescenta rastreabilidade da origem e habilita a detecção de
   alterações por ETag, evitando o download de estudos inalterados.
   ===================================================================== */

SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF COL_LENGTH('dbo.documentos', 'origem_uri') IS NULL
    ALTER TABLE dbo.documentos ADD origem_uri NVARCHAR(1000) NULL;
GO

IF COL_LENGTH('dbo.documentos', 'origem_etag') IS NULL
    ALTER TABLE dbo.documentos ADD origem_etag NVARCHAR(200) NULL;
GO

IF COL_LENGTH('dbo.documentos', 'origem_tamanho') IS NULL
    ALTER TABLE dbo.documentos ADD origem_tamanho BIGINT NULL;
GO

IF COL_LENGTH('dbo.documentos', 'ingerido_em') IS NULL
    ALTER TABLE dbo.documentos ADD ingerido_em DATETIME2(3) NULL;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_documentos_etag'
      AND object_id = OBJECT_ID('dbo.documentos')
)
    CREATE INDEX IX_documentos_etag
        ON dbo.documentos (documento_chave, origem_etag);
GO

/* ---------------------------------------------------------------------
   View de acompanhamento da ingestão
   --------------------------------------------------------------------- */

CREATE OR ALTER VIEW dbo.vw_status_ingestao
AS
SELECT
    d.documento_id,
    d.documento_chave,
    d.titulo,
    d.ano,
    d.regional,
    d.origem_uri,
    d.origem_etag,
    d.origem_tamanho,
    d.ingerido_em,
    d.atualizado_em,
    COUNT(c.chunk_id)                              AS total_chunks,
    SUM(CASE WHEN c.metadados_tfidf IS NOT NULL
             THEN 1 ELSE 0 END)                    AS chunks_enriquecidos,
    AVG(CAST(c.n_caracteres AS FLOAT))             AS media_caracteres,
    MAX(c.modelo_embedding)                        AS modelo_embedding,
    MAX(c.modelo_llm)                              AS modelo_llm
FROM dbo.documentos AS d
LEFT JOIN dbo.chunks AS c
       ON c.documento_id = d.documento_id
GROUP BY
    d.documento_id, d.documento_chave, d.titulo, d.ano, d.regional,
    d.origem_uri, d.origem_etag, d.origem_tamanho,
    d.ingerido_em, d.atualizado_em;
GO

/* ---------------------------------------------------------------------
   Procedure de remoção — documentos retirados do container
   --------------------------------------------------------------------- */

CREATE OR ALTER PROCEDURE dbo.sp_remover_documento
    @documento_chave NVARCHAR(400)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @documento_id INT;

    SELECT @documento_id = documento_id
    FROM dbo.documentos
    WHERE documento_chave = @documento_chave;

    IF @documento_id IS NULL
    BEGIN
        RETURN;
    END;

    -- A FK de chunks usa ON DELETE CASCADE.
    DELETE FROM dbo.documentos WHERE documento_id = @documento_id;
END;
GO
