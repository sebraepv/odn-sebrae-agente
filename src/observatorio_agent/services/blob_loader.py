"""Origem dos estudos do Observatório — Azure Blob Storage ou disco local.

Responsabilidades:
    - Publicar os estudos Markdown no container (carga inicial e sincronização).
    - Listar os estudos disponíveis sem baixar conteúdo.
    - Identificar o que mudou desde a última ingestão.
    - Entregar o conteúdo ao pipeline em formato padronizado.

O pipeline de ingestão não conhece a origem: recebe sempre ``DocumentoBruto``.

Detecção de alterações em duas camadas:
    - Upload  : SHA-256 do conteúdo, gravado como metadado do blob. Evita
                reenviar arquivo idêntico, o que geraria ETag novo e
                dispararia reprocessamento desnecessário na ingestão.
    - Ingestão: ETag do blob, comparado com o valor gravado no Azure SQL.
                Retornado pela listagem, sem custo de download.

Variáveis de ambiente:
    AZURE_STORAGE_ACCOUNT             conta de armazenamento (Entra ID)
    AZURE_STORAGE_CONTAINER           container (default: estudos)
    AZURE_STORAGE_PREFIX              subpasta lógica dentro do container    ## Não definido
    AZURE_STORAGE_CONNECTION_STRING   alternativa à Managed Identity
    ESTUDOS_PATH                      diretório local dos Markdown

Uso como script (publicação):
    python blob_loader.py
    python blob_loader.py --origem ./estudos --relatorio ./logs/sync.json
    python blob_loader.py --forcar
    python blob_loader.py --remover-remotos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

log = logging.getLogger("blob-loader")

EXTENSOES = (".md", ".markdown")

BLOCO_LEITURA = 1024 * 1024


# ── Estruturas de dados ──────────────────────────────────────────────


@dataclass(frozen=True)
class DocumentoBruto:
    """Documento recuperado da origem, antes do parsing."""

    chave: str
    conteudo: str
    origem_uri: str
    etag: str | None = None
    tamanho: int | None = None

    @property
    def conteudo_hash(self) -> str:
        return hashlib.sha256(self.conteudo.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReferenciaDocumento:
    """Metadados da listagem, sem o conteúdo baixado."""

    chave: str
    origem_uri: str
    etag: str | None = None
    tamanho: int | None = None
    conteudo_hash: str | None = None


@dataclass(frozen=True)
class ResultadoUpload:
    """Resultado da sincronização de um arquivo."""

    chave: str
    status: str
    origem_local: str = ""
    destino_blob: str | None = None


class FonteDocumentos(Protocol):
    """Contrato comum entre blob e sistema de arquivos."""

    def listar(self) -> list[ReferenciaDocumento]: ...

    def baixar(self, referencia: ReferenciaDocumento) -> DocumentoBruto: ...


# ── Utilidades ───────────────────────────────────────────────────────


def _normalizar_chave(chave: str) -> str:
    """Padroniza separadores e remove barra inicial."""
    return chave.replace("\\", "/").lstrip("/")


def _decodificar(bruto: bytes) -> str:
    """Decodifica UTF-8 tolerando BOM de arquivos gerados no Windows.

    O BOM precisa ser removido: se sobrar antes do frontmatter, a regex
    ``\\A---`` do parser não reconhece o bloco YAML.
    """
    try:
        conteudo = bruto.decode("utf-8")
    except UnicodeDecodeError:
        conteudo = bruto.decode("utf-8-sig", errors="replace")

    return conteudo.lstrip("\ufeff")


def _hash_arquivo(caminho: Path) -> str:
    """SHA-256 do arquivo, lido em blocos."""
    digest = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(BLOCO_LEITURA), b""):
            digest.update(bloco)

    return digest.hexdigest()


def _hash_texto(conteudo: str) -> str:
    return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()


def listar_markdown_local(raiz: Path) -> list[Path]:
    """Arquivos Markdown de um diretório, incluindo subpastas."""
    return sorted(
        caminho
        for caminho in raiz.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES
    )


# ── Origem: sistema de arquivos ──────────────────────────────────────


class FonteLocal:
    """Lê os estudos de um diretório local."""

    def __init__(self, raiz: Path) -> None:
        # resolve() é necessário: as_uri() falha em caminho relativo.
        self.raiz = Path(raiz).resolve()

        if not self.raiz.exists():
            raise RuntimeError(f"Diretório não encontrado: {self.raiz}")

    def listar(self) -> list[ReferenciaDocumento]:
        referencias: list[ReferenciaDocumento] = []

        for caminho in listar_markdown_local(self.raiz):
            estado = caminho.stat()
            chave = _normalizar_chave(str(caminho.relative_to(self.raiz)))

            referencias.append(
                ReferenciaDocumento(
                    chave=chave,
                    origem_uri=caminho.as_uri(),
                    # mtime como proxy de ETag: muda a cada alteração.
                    etag=f"mtime-{int(estado.st_mtime)}",
                    tamanho=estado.st_size,
                )
            )

        return referencias

    def baixar(self, referencia: ReferenciaDocumento) -> DocumentoBruto:
        caminho = self.raiz / referencia.chave

        return DocumentoBruto(
            chave=referencia.chave,
            conteudo=_decodificar(caminho.read_bytes()),
            origem_uri=referencia.origem_uri,
            etag=referencia.etag,
            tamanho=referencia.tamanho,
        )


# ── Origem: Azure Blob Storage ───────────────────────────────────────


class FonteBlob:
    """Lê e publica os estudos em um container do Azure Blob Storage.

    Autenticação por Managed Identity / Entra ID sempre que possível;
    a connection string é aceita apenas como alternativa de contorno.
    """

    def __init__(
        self,
        conta: str | None = None,
        container: str | None = None,
        prefixo: str | None = None,
        connection_string: str | None = None,
    ) -> None:
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Pacote 'azure-storage-blob' não instalado. Execute: "
                "pip install azure-storage-blob azure-identity"
            ) from exc

        self.container_nome = container or os.getenv(
            "AZURE_STORAGE_CONTAINER", "estudos"
        )
        self.prefixo = (
            prefixo or os.getenv("AZURE_STORAGE_PREFIX", "")
        ).strip("/")

        conexao = connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )

        if conexao:
            log.warning(
                "Usando connection string. Prefira Managed Identity "
                "(AZURE_STORAGE_ACCOUNT com DefaultAzureCredential)."
            )
            self._cliente = ContainerClient.from_connection_string(
                conexao, container_name=self.container_nome
            )
            return

        nome_conta = conta or os.getenv("AZURE_STORAGE_ACCOUNT")

        if not nome_conta:
            raise RuntimeError(
                "Defina AZURE_STORAGE_ACCOUNT ou "
                "AZURE_STORAGE_CONNECTION_STRING."
            )

        from azure.identity import DefaultAzureCredential

        self._cliente = ContainerClient(
            account_url=f"https://{nome_conta}.blob.core.windows.net",
            container_name=self.container_nome,
            credential=DefaultAzureCredential(),
        )

    # ── Caminhos ─────────────────────────────────────────────────────

    @property
    def base_uri(self) -> str:
        return self._cliente.url.rstrip("/")

    def _nome_blob(self, chave: str) -> str:
        """Converte a chave lógica no nome completo dentro do container."""
        chave = _normalizar_chave(chave)
        return f"{self.prefixo}/{chave}" if self.prefixo else chave

    def _chave_de(self, nome_blob: str) -> str:
        """Remove o prefixo do nome do blob, devolvendo a chave lógica."""
        if self.prefixo and nome_blob.startswith(f"{self.prefixo}/"):
            return nome_blob[len(self.prefixo) + 1 :]
        return nome_blob

    def uri_de(self, chave: str) -> str:
        return f"{self.base_uri}/{self._nome_blob(chave)}"

    # ── Leitura ──────────────────────────────────────────────────────

    def listar(self) -> list[ReferenciaDocumento]:
        """Lista os estudos sem baixar conteúdo.

        A listagem devolve ETag, tamanho e o hash gravado como metadado,
        o que permite decidir o que processar antes de qualquer download.
        """
        referencias: list[ReferenciaDocumento] = []

        itens = self._cliente.list_blobs(
            name_starts_with=self.prefixo or None,
            include=["metadata"],
        )

        for blob in itens:
            nome = blob.name

            if not nome.lower().endswith(EXTENSOES):
                continue

            metadata = getattr(blob, "metadata", None) or {}

            referencias.append(
                ReferenciaDocumento(
                    chave=self._chave_de(nome),
                    origem_uri=f"{self.base_uri}/{nome}",
                    etag=(blob.etag or "").strip('"') or None,
                    tamanho=blob.size,
                    conteudo_hash=metadata.get("content_hash") or None,
                )
            )

        referencias.sort(key=lambda r: r.chave)

        return referencias

    def baixar(self, referencia: ReferenciaDocumento) -> DocumentoBruto:
        blob = self._cliente.get_blob_client(
            self._nome_blob(referencia.chave)
        )

        return DocumentoBruto(
            chave=referencia.chave,
            conteudo=_decodificar(blob.download_blob().readall()),
            origem_uri=referencia.origem_uri,
            etag=referencia.etag,
            tamanho=referencia.tamanho,
        )

    # ── Escrita ──────────────────────────────────────────────────────

    def enviar(
        self,
        chave: str,
        conteudo: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Publica ou atualiza um estudo Markdown no container.

        O SHA-256 do conteúdo é gravado nos metadados do blob para que a
        próxima sincronização possa detectar arquivos idênticos.
        """
        from azure.storage.blob import ContentSettings

        nome = self._nome_blob(chave)
        blob = self._cliente.get_blob_client(nome)

        propriedades = {"content_hash": _hash_texto(conteudo)}

        if metadata:
            propriedades.update(metadata)

        blob.upload_blob(
            conteudo.encode("utf-8"),
            overwrite=True,
            metadata=propriedades,
            content_settings=ContentSettings(
                content_type="text/markdown",
                content_encoding="utf-8",
            ),
        )

        return f"{self.base_uri}/{nome}"

    def remover(self, chave: str) -> None:
        """Exclui um blob do container."""
        self._cliente.delete_blob(
            self._nome_blob(chave),
            delete_snapshots="include",
        )


# ── Seleção automática da origem ─────────────────────────────────────


def criar_fonte() -> FonteDocumentos:
    """Escolhe blob ou disco conforme as variáveis de ambiente."""
    conta = os.getenv("AZURE_STORAGE_ACCOUNT")
    conexao = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if conta or conexao:
        fonte = FonteBlob()
        log.info(
            "Origem: Azure Blob Storage — container '%s'%s",
            fonte.container_nome,
            f", prefixo '{fonte.prefixo}'" if fonte.prefixo else "",
        )
        return fonte

    raiz = Path(os.getenv("ESTUDOS_PATH", "estudos"))
    log.info("Origem: sistema de arquivos — %s", raiz.resolve())

    return FonteLocal(raiz)


# ── Comparação com o que já foi ingerido ─────────────────────────────


def selecionar_pendentes(
    referencias: list[ReferenciaDocumento],
    etags_no_banco: dict[str, str | None],
    forcar: bool = False,
) -> tuple[list[ReferenciaDocumento], list[str]]:
    """Separa o que precisa ser processado do que já está atualizado.

    A comparação por ETag evita o download dos estudos inalterados — e,
    com isso, evita também a geração de metadados por LLM e os embeddings,
    que são as etapas caras do pipeline.

    Retorna (pendentes, inalterados).
    """
    pendentes: list[ReferenciaDocumento] = []
    inalterados: list[str] = []

    for referencia in referencias:
        gravado = etags_no_banco.get(referencia.chave)

        if forcar or not referencia.etag or gravado != referencia.etag:
            pendentes.append(referencia)
        else:
            inalterados.append(referencia.chave)

    return pendentes, inalterados


def identificar_removidos(
    referencias: list[ReferenciaDocumento],
    chaves_no_banco: Iterable[str],
) -> list[str]:
    """Documentos que existem no banco mas não estão mais na origem."""
    atuais = {r.chave for r in referencias}
    return sorted(chave for chave in chaves_no_banco if chave not in atuais)


# ── Publicação: disco local → Blob ───────────────────────────────────


def sincronizar_com_blob(
    diretorio_local: Path,
    fonte: FonteBlob | None = None,
    forcar: bool = False,
    remover_remotos: bool = False,
) -> list[ResultadoUpload]:
    """Publica os Markdown locais no container, preservando subpastas.

    Arquivos cujo SHA-256 já corresponde ao metadado do blob não são
    reenviados: sobrescrever um blob idêntico geraria ETag novo e faria
    a ingestão reprocessar o estudo sem necessidade.

    Args:
        diretorio_local: diretório com os estudos Markdown.
        fonte: cliente do container; criado automaticamente se omitido.
        forcar: reenvia todos os arquivos, mesmo sem alteração.
        remover_remotos: exclui do container o que não existe localmente.

    Returns:
        Resultado de cada arquivo processado.
    """
    raiz = Path(diretorio_local).resolve()

    if not raiz.exists():
        raise RuntimeError(f"Diretório local não encontrado: {raiz}")

    arquivos = listar_markdown_local(raiz)

    if not arquivos:
        raise RuntimeError(f"Nenhum arquivo Markdown em {raiz}")

    destino = fonte or FonteBlob()

    remotos = {r.chave: r for r in destino.listar()}

    resultados: list[ResultadoUpload] = []
    chaves_locais: set[str] = set()

    for caminho in arquivos:
        chave = _normalizar_chave(str(caminho.relative_to(raiz)))
        chaves_locais.add(chave)

        remoto = remotos.get(chave)
        hash_local = _hash_arquivo(caminho)

        # O hash do blob é do conteúdo decodificado; o local é do arquivo
        # em disco. Diferem quando há BOM — por isso a segunda comparação.
        conteudo = _decodificar(caminho.read_bytes())
        hash_normalizado = _hash_texto(conteudo)

        inalterado = remoto is not None and remoto.conteudo_hash in (
            hash_local,
            hash_normalizado,
        )

        if inalterado and not forcar:
            resultados.append(
                ResultadoUpload(
                    chave=chave,
                    status="inalterado",
                    origem_local=str(caminho),
                    destino_blob=destino.uri_de(chave),
                )
            )
            log.info("Inalterado: %s", chave)
            continue

        uri = destino.enviar(
            chave=chave,
            conteudo=conteudo,
            metadata={"origem": "observatorio"},
        )

        status = "atualizado" if remoto else "enviado"

        resultados.append(
            ResultadoUpload(
                chave=chave,
                status=status,
                origem_local=str(caminho),
                destino_blob=uri,
            )
        )
        log.info("%s: %s", status.capitalize(), chave)

    if remover_remotos:
        for chave in sorted(set(remotos) - chaves_locais):
            destino.remover(chave)

            resultados.append(
                ResultadoUpload(
                    chave=chave,
                    status="removido",
                    destino_blob=destino.uri_de(chave),
                )
            )
            log.info("Removido do container: %s", chave)

    return resultados


# ── Execução como script ─────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Publica os estudos Markdown do Observatório no "
            "Azure Blob Storage."
        )
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=Path(os.getenv("ESTUDOS_PATH", "estudos")),
        help="Diretório local dos Markdown (default: ESTUDOS_PATH).",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Reenvia todos os arquivos, mesmo sem alteração.",
    )
    parser.add_argument(
        "--remover-remotos",
        action="store_true",
        help="Exclui do container os arquivos ausentes localmente.",
    )
    parser.add_argument(
        "--relatorio",
        type=Path,
        default=None,
        help="Caminho para salvar o relatório da sincronização em JSON.",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    resultados = sincronizar_com_blob(
        diretorio_local=args.origem,
        forcar=args.forcar,
        remover_remotos=args.remover_remotos,
    )

    totais: dict[str, int] = {}
    for resultado in resultados:
        totais[resultado.status] = totais.get(resultado.status, 0) + 1

    log.info("Sincronização concluída:")
    for status, quantidade in sorted(totais.items()):
        log.info("  %-12s %d", status, quantidade)

    if args.relatorio:
        args.relatorio.parent.mkdir(parents=True, exist_ok=True)
        args.relatorio.write_text(
            json.dumps(
                [asdict(item) for item in resultados],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("Relatório: %s", args.relatorio.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
