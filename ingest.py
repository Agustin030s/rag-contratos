"""Fase 1 — Ingesta: documentos -> chunks -> embeddings -> ChromaDB.

Uso:
    python ingest.py             # indexa solo los documentos nuevos o modificados
    python ingest.py --reindex   # borra la colección y vuelve a indexar todo

Este módulo también expone `open_vectorstore()`, que usa rag.py para abrir la
MISMA base con el MISMO modelo de embeddings.
"""

import argparse
import hashlib
import logging
import re
import sys
from functools import lru_cache

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

EXTENSIONES = (".md", ".txt")


class EmbeddingModelMismatchError(Exception):
    """La base se creó con un modelo de embeddings distinto al configurado."""


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # cl100k_base es un tokenizador estándar para medir el tamaño de los chunks.
    # El modelo de embeddings usa su propio tokenizador, que en español genera
    # una cantidad de tokens parecida, así que los chunks de 500 entran en su
    # límite de 512.
    return tiktoken.get_encoding("cl100k_base")


def contar_tokens(texto: str) -> int:
    return len(_encoding().encode(texto))


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
# Los contratos traen líneas del tipo "**Proveedor:** OndaPlay". Los asteriscos
# son opcionales para aceptar también .txt sin formato Markdown.
_RE_PROVEEDOR = re.compile(r"^\**Proveedor:\**\s*(.+?)\s*$", re.MULTILINE)
_RE_NUMERO = re.compile(r"^\**N[úu]mero de contrato:\**\s*(\S+)", re.MULTILINE)


def _extraer(patron: re.Pattern, texto: str, campo: str, archivo: str) -> str:
    coincidencia = patron.search(texto)
    if coincidencia is None:
        logger.warning("No se encontró '%s' en %s; se usa 'desconocido'.", campo, archivo)
        return "desconocido"
    return coincidencia.group(1)


def load_documents() -> list[Document]:
    """Lee los .md y .txt de data/ y les agrega archivo, proveedor y número."""
    if not config.DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta {config.DATA_DIR}. Ejecuta: python generate_contracts.py"
        )

    rutas = sorted(p for p in config.DATA_DIR.iterdir() if p.suffix.lower() in EXTENSIONES)
    documentos = []
    for ruta in rutas:
        texto = ruta.read_text(encoding="utf-8")
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "archivo": ruta.name,
                    "proveedor": _extraer(_RE_PROVEEDOR, texto, "Proveedor", ruta.name),
                    "numero_contrato": _extraer(_RE_NUMERO, texto, "Número de contrato", ruta.name),
                    # Huella del contenido: si cambia una sola letra, cambia el
                    # hash y la ingesta sabe que hay que reindexar este documento.
                    "hash_documento": hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16],
                },
            )
        )
    logger.info("Documentos leídos: %d (en %s)", len(documentos), config.DATA_DIR.name)
    return documentos


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _chunk_id(archivo: str, chunk_index: int) -> str:
    # ID estable: el mismo archivo y la misma posición dan siempre el mismo ID.
    # Chroma hace "upsert" por ID, así que indexar dos veces el mismo chunk lo
    # reemplaza en lugar de duplicarlo.
    hash_archivo = hashlib.sha256(archivo.encode("utf-8")).hexdigest()[:12]
    return f"{hash_archivo}-{chunk_index:03d}"


def split_documents(documentos: list[Document]) -> list[Document]:
    # Medimos en TOKENS y no en caracteres (como en la Unidad 2): los límites
    # del modelo y el costo se cuentan en tokens, y en español un carácter no
    # equivale a un token. Con length_function=contar_tokens, chunk_size=500
    # significa "hasta 500 tokens" de verdad.
    # Los separadores van de mayor a menor: primero intenta cortar entre
    # párrafos (cláusulas), después entre líneas, frases y palabras.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=contar_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documentos:
        meta = doc.metadata
        # Encabezado de contexto: los fragmentos del medio de un contrato dicen
        # "EL PROVEEDOR", nunca su nombre. Sin este encabezado, una pregunta como
        # "¿qué preaviso tiene OndaPlay?" no se parecería a la cláusula correcta.
        encabezado = f"[Contrato {meta['numero_contrato']} | Proveedor: {meta['proveedor']}]\n"
        for i, texto in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Document(
                    id=_chunk_id(meta["archivo"], i),
                    page_content=encabezado + texto,
                    metadata={**meta, "chunk_index": i},
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))


def _existe_coleccion(client) -> bool:
    # chromadb >= 0.6 devuelve objetos Collection; versiones anteriores, nombres.
    nombres = [getattr(c, "name", c) for c in client.list_collections()]
    return config.COLLECTION_NAME in nombres


def collection_count() -> int | None:
    """Cantidad de chunks de la colección, o None si todavía no existe.

    Antes de devolver nada comprueba que la base se haya creado con el mismo
    modelo de embeddings que el configurado ahora (error #1 de un RAG: indexar
    con un modelo y consultar con otro devuelve resultados sin sentido o falla
    por dimensiones distintas).
    """
    if not config.VECTORSTORE_DIR.exists():
        return None

    client = _client()
    if not _existe_coleccion(client):
        return None

    coleccion = client.get_collection(config.COLLECTION_NAME, embedding_function=None)
    guardado = (coleccion.metadata or {}).get("embedding_model")
    if guardado and guardado != config.EMBEDDING_MODEL:
        raise EmbeddingModelMismatchError(
            f"La base en '{config.VECTORSTORE_DIR.name}' se creó con '{guardado}', pero "
            f"EMBEDDING_MODEL es '{config.EMBEDDING_MODEL}'. "
            "Ejecuta: python ingest.py --reindex"
        )
    return coleccion.count()


@lru_cache(maxsize=1)
def get_embeddings():
    """Modelo de embeddings local. Se carga una sola vez (tarda unos segundos)."""
    # Import diferido: así este módulo se puede importar (por ejemplo, para
    # usar load_documents) aunque falte instalar algún paquete.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Cargando el modelo de embeddings local '%s'...", config.EMBEDDING_MODEL)
    # Los modelos E5 se entrenaron con prefijos: "passage: " para los textos que
    # se guardan y "query: " para las preguntas. Sin ellos la búsqueda empeora.
    # normalize_embeddings=True deja los vectores con largo 1, así la distancia
    # coseno es comparable entre consultas.
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "},
    )


def _crear_vectorstore():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.VECTORSTORE_DIR),
        # Guardamos el modelo en los metadatos de la colección para poder
        # detectar después un cambio de modelo. "cosine" hace que la distancia
        # quede entre 0 (idénticos) y 2, más fácil de interpretar en los logs.
        # OJO: Chroma solo aplica esta metadata al CREAR la colección; si ya
        # existe conserva la anterior, por eso la verificamos en collection_count().
        collection_metadata={"embedding_model": config.EMBEDDING_MODEL, "hnsw:space": "cosine"},
    )


def open_vectorstore():
    """Abre la base existente para consultar (la usa rag.py)."""
    cantidad = collection_count()  # puede lanzar EmbeddingModelMismatchError
    if not cantidad:
        raise RuntimeError("La base vectorial está vacía o no existe. Ejecuta: python ingest.py")
    return _crear_vectorstore()


# ---------------------------------------------------------------------------
# Ingesta
# ---------------------------------------------------------------------------
def run_ingest(reindex: bool = False) -> int:
    """Indexa los documentos de data/. Devuelve la cantidad de chunks en la base."""
    if reindex:
        # Con --reindex NO verificamos el modelo: existe justamente para
        # arreglar una base creada con otro modelo de embeddings.
        client = _client()
        if _existe_coleccion(client):
            client.delete_collection(config.COLLECTION_NAME)
            logger.info("Colección '%s' borrada (--reindex).", config.COLLECTION_NAME)

    cantidad = collection_count()  # verifica el modelo de embeddings
    documentos = load_documents()
    if not documentos:
        raise FileNotFoundError(f"No hay archivos {EXTENSIONES} en {config.DATA_DIR}")

    # Comparamos el hash de cada documento con el guardado en la base. Así no
    # alcanza con que la colección exista: si un contrato cambió, se reindexa
    # SOLO ese contrato (y no los 10).
    indexados = _documentos_indexados() if cantidad else {}
    actuales = {d.metadata["archivo"]: d for d in documentos}
    nuevos = [d for a, d in actuales.items() if a not in indexados]
    modificados = [
        d for a, d in actuales.items()
        if a in indexados and indexados[a]["hash"] != d.metadata["hash_documento"]
    ]
    eliminados = [a for a in indexados if a not in actuales]

    if not (nuevos or modificados or eliminados):
        logger.info("La base ya existe con %d chunks, se omite la ingesta (ningún documento cambió).", cantidad)
        return cantidad
    logger.info(
        "Cambios detectados: %d nuevos, %d modificados, %d eliminados.",
        len(nuevos), len(modificados), len(eliminados),
    )

    vectorstore = _crear_vectorstore()
    # Primero borramos los chunks viejos de los documentos modificados o
    # eliminados: si un contrato ahora tiene menos chunks, los sobrantes no
    # quedan huérfanos en la base.
    ids_viejos = [
        id_ for archivo in [d.metadata["archivo"] for d in modificados] + eliminados
        for id_ in indexados[archivo]["ids"]
    ]
    if ids_viejos:
        vectorstore.delete(ids=ids_viejos)
        logger.info("Borrados %d chunks desactualizados.", len(ids_viejos))

    a_indexar = nuevos + modificados
    chunks = split_documents(a_indexar)
    if chunks:
        logger.info("Generando embeddings de %d chunks en local...", len(chunks))
        vectorstore.add_documents(chunks, ids=[c.id for c in chunks])
        tokens = [contar_tokens(c.page_content) for c in chunks]
        logger.info(
            "Resumen: %d documentos leídos, %d indexados, %d chunks creados, "
            "%.0f tokens promedio por chunk (máx. %d).",
            len(documentos), len(a_indexar), len(chunks), sum(tokens) / len(tokens), max(tokens),
        )
    return collection_count() or 0


def _documentos_indexados() -> dict[str, dict]:
    """{archivo: {"hash": ..., "ids": [...]}} de lo que ya está en la base.

    Lee solo los metadatos con el cliente de chromadb, sin cargar el modelo de
    embeddings: si nada cambió, la ingesta termina en un instante.
    """
    coleccion = _client().get_collection(config.COLLECTION_NAME, embedding_function=None)
    datos = coleccion.get(include=["metadatas"])
    indexados: dict[str, dict] = {}
    for id_, meta in zip(datos["ids"], datos["metadatas"]):
        # Los chunks de una base vieja no tienen hash: hash=None nunca coincide,
        # así que esos documentos se reindexan una vez y quedan al día.
        entrada = indexados.setdefault(meta["archivo"], {"hash": meta.get("hash_documento"), "ids": []})
        entrada["ids"].append(id_)
    return indexados


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa los contratos de data/ en ChromaDB.")
    parser.add_argument(
        "--reindex", action="store_true", help="borra la colección y vuelve a indexar"
    )
    args = parser.parse_args()

    try:
        run_ingest(reindex=args.reindex)
    # La ingesta no llama a ninguna API ni necesita key: los embeddings se
    # calculan en local. Claude recién interviene al responder (rag.py).
    except (EmbeddingModelMismatchError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
    except ImportError as exc:
        logger.error("Falta instalar un paquete [%s]: %s", type(exc).__name__, exc)
    except OSError as exc:
        # Por ejemplo, sin internet la primera vez: el modelo no se puede descargar.
        logger.error("No se pudo cargar el modelo de embeddings [%s]: %s", type(exc).__name__, exc)
    else:
        return 0
    return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
