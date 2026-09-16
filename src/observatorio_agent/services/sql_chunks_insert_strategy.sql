/* =====================================================================
   Observatório de Negócios — Sebrae/SC
   Esquema RAG simplificado (configuração única)

   Baseado em Mishra et al. (2026), arXiv:2512.05411, IEEE CAI 2026,
   restrito à configuração de maior precisão reportada (82,5%):

       Chunking   : recursive
       Metadados  : gerados por LLM (estrutural, técnico, contextual)
       Enrichment : TF-IDF weighted (90:10), renormalizado L2
       Embedding  : OpenAI text-embedding-3-small (1536 dimensões)

   Um único vetor por chunk. Sem matriz experimental.
   ===================================================================== */

SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

/* ---------------------------------------------------------------------
   1. DOCUMENTOS
   --------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.documentos', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.documentos (
        documento_id       INT IDENTITY(1,1) PRIMARY KEY,
        documento_chave    NVARCHAR(400)  NOT NULL,
        titulo             NVARCHAR(500)  NOT NULL,
        conteudo_markdown  NVARCHAR(MAX)  NOT NULL,
        conteudo_hash      CHAR(64)       NOT NULL,

        ano                INT            NULL,
        tema               NVARCHAR(400)  NULL,
        setor              NVARCHAR(400)  NULL,
        municipio          NVARCHAR(400)  NULL,
        regional           NVARCHAR(400)  NULL,
        fonte              NVARCHAR(400)  NULL,
        data_referencia    NVARCHAR(100)  NULL,
        classificacao      NVARCHAR(50)   NOT NULL
            CONSTRAINT DF_documentos_classificacao DEFAULT N'interno',

        criado_em          DATETIME2(3)   NOT NULL
            CONSTRAINT DF_documentos_criado DEFAULT SYSUTCDATETIME(),
        atualizado_em      DATETIME2(3)   NOT NULL
            CONSTRAINT DF_documentos_atualizado DEFAULT SYSUTCDATETIME(),

        CONSTRAINT UQ_documentos_chave UNIQUE (documento_chave)
    );

    CREATE INDEX IX_documentos_filtros
        ON dbo.documentos (ano, regional);
END;
GO

/* ---------------------------------------------------------------------
   2. CHUNKS — conteúdo, metadados do LLM e vetor em uma única tabela
   --------------------------------------------------------------------- */

IF OBJECT_ID(N'dbo.chunks', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.chunks (
        chunk_id          BIGINT IDENTITY(1,1) PRIMARY KEY,
        documento_id      INT      NOT NULL,
        chunk_indice      INT      NOT NULL,

        -- Conteúdo
        chunk_texto       NVARCHAR(MAX)  NOT NULL,
        chunk_hash        CHAR(64)       NOT NULL,
        n_caracteres      INT            NOT NULL,
        heading_path      NVARCHAR(1000) NULL,

        -- Metadados gerados por LLM
        tipo_conteudo       NVARCHAR(100)  NULL,
        palavras_chave      NVARCHAR(1000) NULL,
        resumo              NVARCHAR(1000) NULL,
        entidades           NVARCHAR(1000) NULL,
        indicadores         NVARCHAR(1000) NULL,
        fontes_dados        NVARCHAR(1000) NULL,
        intencao            NVARCHAR(500)  NULL,
        perguntas_atendidas NVARCHAR(MAX)  NULL,
        metadados_json      NVARCHAR(MAX)  NULL,

        -- Texto de metadados priorizado por TF-IDF (o que foi embedado)
        metadados_tfidf   NVARCHAR(MAX)  NULL,

        -- Vetor final (conteúdo 90% + metadados 10%, renormalizado)
        embedding         VECTOR(1536)   NOT NULL,

        -- Rastreabilidade
        modelo_llm        NVARCHAR(100)  NULL,
        modelo_embedding  NVARCHAR(100)  NOT NULL,
        peso_conteudo     DECIMAL(4,3)   NOT NULL
            CONSTRAINT DF_chunks_peso_conteudo DEFAULT 0.900,
        prompt_versao     NVARCHAR(20)   NOT NULL
            CONSTRAINT DF_chunks_prompt_versao DEFAULT N'v1',
        criado_em         DATETIME2(3)   NOT NULL
            CONSTRAINT DF_chunks_criado DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_chunks_documento
            FOREIGN KEY (documento_id)
            REFERENCES dbo.documentos (documento_id) ON DELETE CASCADE,
        CONSTRAINT UQ_chunks_documento_indice
            UNIQUE (documento_id, chunk_indice)
    );

    CREATE INDEX IX_chunks_hash ON dbo.chunks (chunk_hash);
END;
GO

/* ---------------------------------------------------------------------
   3. VIEW DE CONSULTA — superfície única para o agente
   --------------------------------------------------------------------- */

CREATE OR ALTER VIEW dbo.vw_rag_chunks
AS
SELECT
    c.chunk_id,
    c.chunk_indice,
    c.chunk_texto,
    c.heading_path,
    c.n_caracteres,
    c.embedding,

    c.tipo_conteudo,
    c.palavras_chave,
    c.indicadores,
    c.entidades,
    c.intencao,
    c.resumo,

    d.documento_chave,
    d.titulo AS documento_titulo,
    d.ano,
    d.tema,
    d.setor,
    d.municipio,
    d.regional,
    d.fonte,
    d.data_referencia
FROM dbo.chunks AS c
INNER JOIN dbo.documentos AS d
        ON d.documento_id = c.documento_id;
GO

/* ---------------------------------------------------------------------
   4. PROCEDURE DE BUSCA — consulta fixa e parametrizada
   --------------------------------------------------------------------- */

CREATE OR ALTER PROCEDURE dbo.sp_buscar_chunks
    @embedding_consulta NVARCHAR(MAX),
    @top_k              INT   = 6,
    @distancia_maxima   FLOAT = 0.65,
    @ano_minimo         INT   = NULL,
    @regional           NVARCHAR(400) = NULL,
    @tema               NVARCHAR(400) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @v VECTOR(1536) = CAST(@embedding_consulta AS VECTOR(1536));

    SELECT TOP (@top_k)
        v.chunk_id,
        v.documento_titulo,
        v.heading_path,
        v.chunk_texto,
        v.ano,
        v.regional,
        v.fonte,
        v.palavras_chave,
        v.indicadores,
        VECTOR_DISTANCE('cosine', v.embedding, @v) AS distancia
    FROM dbo.vw_rag_chunks AS v
    WHERE (@ano_minimo IS NULL OR v.ano >= @ano_minimo)
      AND (@regional   IS NULL OR v.regional = @regional OR v.regional IS NULL)
      AND (@tema       IS NULL OR v.tema LIKE '%' + @tema + '%')
      AND VECTOR_DISTANCE('cosine', v.embedding, @v) <= @distancia_maxima
    ORDER BY distancia;
END;
GO
