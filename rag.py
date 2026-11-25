"""Fases 2 y 3 — Recuperación y generación (RAG) con una cadena LCEL asíncrona.

Flujo:
    pregunta -> búsqueda con distancia (top_k) -> umbral de distancia
             -> format_docs -> prompt grounded -> LLM
             -> PydanticOutputParser (+ with_retry) -> RAGResponse
             -> validación de referencias

Dos puntos de entrada:
    get_rag_response(query)        búsqueda normal (top_k fragmentos)
    compare_contracts(query, ...)  cruza contratos (2 fragmentos por proveedor)
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from operator import itemgetter
from pathlib import Path

import anthropic
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

import config
from ingest import EmbeddingModelMismatchError, open_vectorstore
from schemas import RAGResponse

logger = logging.getLogger(__name__)

NO_LO_SE = "No lo sé"
RESPUESTA_ERROR = "No se pudo responder por un error técnico."


def _no_lo_se() -> RAGResponse:
    return RAGResponse(respuesta=NO_LO_SE, encontrado=False, referencias=[])


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    """Une los fragmentos con un encabezado de fuente para que el modelo pueda citar."""
    return "\n\n---\n\n".join(
        f"[Fuente: {d.metadata['archivo']} | Proveedor: {d.metadata['proveedor']}]\n{d.page_content}"
        for d in docs
    )


def filtrar_por_umbral(resultados: list[tuple[Document, float]]) -> list[Document]:
    """Registra cada fragmento con su distancia y descarta los que superan el umbral."""
    docs = []
    for doc, distancia in resultados:
        aceptado = distancia <= config.UMBRAL_DISTANCIA
        logger.info(
            "  chunk recuperado: %s #%d (distancia=%.3f)%s",
            doc.metadata["archivo"], doc.metadata["chunk_index"], distancia,
            "" if aceptado else " -> descartado, supera el umbral",
        )
        if aceptado:
            docs.append(doc)
    if resultados and not docs:
        logger.info(
            "Ningún fragmento queda bajo el umbral (%.2f): se responde '%s' sin llamar al LLM.",
            config.UMBRAL_DISTANCIA, NO_LO_SE,
        )
    return docs


# ---------------------------------------------------------------------------
# Prompt: "filtro de veracidad"
# ---------------------------------------------------------------------------
parser = PydanticOutputParser(pydantic_object=RAGResponse)

_prompt_base = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente legal que analiza contratos. Responde SOLO con la "
            "información del CONTEXTO.\n"
            "Reglas:\n"
            f"- Si la respuesta no está en el contexto, responde exactamente '{NO_LO_SE}', "
            "marca encontrado=false y deja referencias vacía.\n"
            "- Si encontrado=true, incluye al menos una referencia.\n"
            "- No inventes cifras, fechas ni cláusulas. Cita el archivo de cada dato.\n"
            "- Atención a las negaciones: que un contrato mencione la inflación no "
            "significa que tenga ajuste. Lee la frase completa.\n"
            "- Un contrato puede remitir a un ANEXO: si el anexo está en el contexto, úsalo.\n"
            "- Usa solo los nombres de archivo que aparecen en los encabezados [Fuente: ...].\n\n"
            "{format_instructions}",
        ),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
)
# partial() fija las instrucciones de formato (salen del propio esquema
# Pydantic) y deja libres solo {contexto} y {pregunta}.
prompt = _prompt_base.partial(format_instructions=parser.get_format_instructions())


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def _crear_llm():
    # Import diferido, como en el resto del proyecto: importar rag.py no
    # falla aunque falte el paquete o la API key.
    from langchain_anthropic import ChatAnthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Falta ANTHROPIC_API_KEY en el archivo .env")

    # NO pasamos `temperature`: los modelos actuales de Anthropic la rechazan
    # con un error 400. Los embeddings no usan ninguna API: son locales (config.py).
    return ChatAnthropic(model=config.LLM_MODEL, max_tokens=config.LLM_MAX_TOKENS)


# Se reintenta un JSON mal formado (o que rompe las reglas del esquema) y los
# errores transitorios de red. AuthenticationError NO está a propósito: una key
# inválida no se arregla reintentando.
ERRORES_REINTENTABLES: tuple[type[BaseException], ...] = (
    OutputParserException,
    anthropic.APIConnectionError,  # incluye APITimeoutError
    anthropic.RateLimitError,
)


# ---------------------------------------------------------------------------
# Cadenas LCEL
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Componentes:
    vectorstore: object
    responder: Runnable  # {"docs", "pregunta"} -> RAGResponse
    rag_chain: Runnable  # pregunta -> {"resultados", "docs", "pregunta", "respuesta"}


@lru_cache(maxsize=1)
def _get_componentes() -> _Componentes:
    # Se crean la primera vez que se usan (y no al importar el módulo): así
    # importar rag.py no falla si todavía no hay base o API key. lru_cache no
    # guarda las excepciones, así que si falla se vuelve a intentar.
    # OJO: es una función BLOQUEANTE (carga el modelo de embeddings, ~470 MB).
    # Desde código async se llama con asyncio.to_thread para no frenar el event loop.
    vectorstore = open_vectorstore()

    # Búsqueda CON distancia como primer paso de la cadena. as_retriever() no
    # devuelve la distancia, y la necesitamos para el log y para el umbral;
    # así se busca una sola vez por consulta.
    async def buscar(pregunta: str) -> list[tuple[Document, float]]:
        return await vectorstore.asimilarity_search_with_score(pregunta, k=config.TOP_K)

    # .with_retry() envuelve a llm | parser: si el modelo devuelve un JSON mal
    # formado, el parser lanza OutputParserException DENTRO del bloque
    # reintentado y se vuelve a pedir la respuesta. El prompt queda afuera
    # porque formatear el template no falla de forma transitoria.
    generar = (_crear_llm() | parser).with_retry(
        stop_after_attempt=2, retry_if_exception_type=ERRORES_REINTENTABLES
    )
    respuesta_llm = (
        {
            "contexto": itemgetter("docs") | RunnableLambda(format_docs),
            "pregunta": itemgetter("pregunta"),
        }
        | prompt
        | generar
    )
    # Sin documentos (ninguno pasó el umbral) no hay contexto: respondemos
    # "No lo sé" sin llamar al LLM.
    responder = RunnableBranch(
        (lambda x: not x["docs"], RunnableLambda(lambda _: _no_lo_se())),
        respuesta_llm,
    )
    # RunnableParallel conserva lo recuperado junto a la pregunta; los pasos
    # siguientes agregan los documentos filtrados y la respuesta. Al final
    # tenemos todo junto para validar las referencias contra lo que se usó.
    rag_chain = (
        RunnableParallel(resultados=RunnableLambda(buscar), pregunta=RunnablePassthrough())
        | RunnablePassthrough.assign(docs=itemgetter("resultados") | RunnableLambda(filtrar_por_umbral))
        | RunnablePassthrough.assign(respuesta=responder)
    )

    return _Componentes(vectorstore=vectorstore, responder=responder, rag_chain=rag_chain)


# ---------------------------------------------------------------------------
# Validaciones y errores
# ---------------------------------------------------------------------------
def _validar_respuesta(respuesta: RAGResponse, docs: list[Document]) -> RAGResponse:
    """Filtro anti-alucinación aplicado DESPUÉS del modelo."""
    if not respuesta.encontrado:
        # Garantizamos el contrato de la API: "no encontrado" siempre es
        # exactamente "No lo sé" (el esquema ya impide que traiga referencias).
        if respuesta.respuesta != NO_LO_SE:
            logger.info("encontrado=false: se normaliza la respuesta a '%s'.", NO_LO_SE)
        return _no_lo_se()

    # El modelo podría citar un archivo que NO estaba en el contexto (inventado
    # o recordado de otra pregunta). Solo aceptamos los realmente recuperados.
    recuperados = {d.metadata["archivo"] for d in docs}
    validas = []
    for ref in respuesta.referencias:
        archivo = Path(ref.archivo).name  # por si el modelo agrega "data/"
        if archivo in recuperados:
            validas.append(ref.model_copy(update={"archivo": archivo}))
        else:
            logger.warning("Referencia descartada: '%s' no está entre los documentos recuperados.", ref.archivo)

    if not validas:
        # Una respuesta "encontrada" sin ninguna fuente real no se puede
        # verificar: es preferible abstenerse que devolver algo sin respaldo.
        logger.warning("encontrado=true pero ninguna referencia es válida: se responde '%s'.", NO_LO_SE)
        return _no_lo_se()

    # Construimos un objeto nuevo (y no model_copy) para que el model_validator
    # del esquema vuelva a comprobar la coherencia.
    return RAGResponse(respuesta=respuesta.respuesta, encontrado=True, referencias=validas)


def _respuesta_error(exc: Exception) -> RAGResponse:
    """Registra el error y devuelve una respuesta controlada (nunca crashea).

    El detalle va en el campo `error`, así quien llama distingue un fallo
    técnico de un "No lo sé" legítimo.
    """
    if isinstance(exc, anthropic.AuthenticationError):
        logger.error("Error de autenticación con Anthropic: revisa ANTHROPIC_API_KEY (no se reintenta).")
        mensaje = "No se pudo autenticar con Anthropic: revisa ANTHROPIC_API_KEY."
    elif isinstance(exc, anthropic.APIError):  # incluye conexión, timeout y rate limit
        logger.error("Error de la API de Anthropic [%s]: %s", type(exc).__name__, exc)
        mensaje = f"La API de Anthropic no está disponible ({type(exc).__name__}). Intenta más tarde."
    elif isinstance(exc, OutputParserException):
        logger.error("El modelo devolvió un JSON inválido tras los reintentos: %s", exc)
        mensaje = "El modelo no devolvió una respuesta con el formato esperado."
    elif isinstance(exc, (EmbeddingModelMismatchError, RuntimeError, ValueError, ImportError, OSError)):
        logger.error("Error de configuración [%s]: %s", type(exc).__name__, exc)
        mensaje = f"Error de configuración: {exc}"
    else:
        logger.exception("Error inesperado [%s]: %s", type(exc).__name__, exc)
        mensaje = f"Error inesperado ({type(exc).__name__})."
    return RAGResponse(respuesta=RESPUESTA_ERROR, encontrado=False, referencias=[], error=mensaje)


def _log_resultado(respuesta: RAGResponse) -> None:
    logger.info(
        "Resultado: encontrado=%s, referencias=%s",
        respuesta.encontrado,
        [r.archivo for r in respuesta.referencias],
    )


# ---------------------------------------------------------------------------
# Puntos de entrada
# ---------------------------------------------------------------------------
async def get_rag_response(query: str) -> RAGResponse:
    """Responde una pregunta con los TOP_K fragmentos más parecidos."""
    if not query or not query.strip():
        logger.warning("Consulta vacía.")
        return _no_lo_se()

    logger.info("Consulta: %s", query)
    try:
        comp = await asyncio.to_thread(_get_componentes)
        # a. búsqueda de similitud (con distancia) + umbral
        # b. prompt con los fragmentos  c. LLM asíncrono  d. parseo a RAGResponse
        salida = await comp.rag_chain.ainvoke(query)
    except Exception as exc:  # red de seguridad: la función nunca rompe el programa
        return _respuesta_error(exc)

    respuesta = _validar_respuesta(salida["respuesta"], salida["docs"])
    _log_resultado(respuesta)
    return respuesta


def _listar_proveedores(vectorstore) -> list[str]:
    metadatos = vectorstore.get(include=["metadatas"])["metadatas"]
    return sorted({m["proveedor"] for m in metadatos})


async def compare_contracts(query: str, proveedores: list[str] | None = None) -> RAGResponse:
    """Responde preguntas que cruzan VARIOS contratos.

    ¿Por qué existe? Con TOP_K=4, una pregunta como "¿qué contratos tienen
    ajuste por inflación?" recibe solo 4 fragmentos: como mucho ve 4 de los 10
    contratos y la respuesta queda incompleta. Subir TOP_K a 20 tampoco sirve:
    los más parecidos pueden ser varios fragmentos del mismo contrato y se cae
    en el "Lost in the Middle". Aquí hacemos UNA búsqueda por proveedor (con
    filtro de metadatos), así cada contrato aporta sus fragmentos más relevantes.
    """
    logger.info("Comparación: %s | proveedores=%s", query, proveedores or "todos")
    try:
        comp = await asyncio.to_thread(_get_componentes)
        if proveedores is None:
            proveedores = _listar_proveedores(comp.vectorstore)

        # Las búsquedas son independientes: las lanzamos en paralelo.
        por_proveedor = await asyncio.gather(
            *(
                comp.vectorstore.asimilarity_search_with_score(
                    query, k=config.COMPARE_K_POR_CONTRATO, filter={"proveedor": p}
                )
                for p in proveedores
            )
        )
        for proveedor, encontrados in zip(proveedores, por_proveedor):
            if not encontrados:
                logger.warning("  sin fragmentos para el proveedor '%s'.", proveedor)

        # Mismo umbral que la búsqueda normal: un contrato que no tiene nada
        # parecido a la pregunta no aporta contexto.
        docs = filtrar_por_umbral([r for encontrados in por_proveedor for r in encontrados])

        # UNA sola llamada al LLM, con el mismo prompt y el mismo parser.
        respuesta = await comp.responder.ainvoke({"docs": docs, "pregunta": query})
    except Exception as exc:
        return _respuesta_error(exc)

    respuesta = _validar_respuesta(respuesta, docs)
    _log_resultado(respuesta)
    return respuesta
