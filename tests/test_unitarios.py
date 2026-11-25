"""Tests sin API key: LLM y embeddings simulados (ver conftest.py).

Uso:  pytest -v
"""

import asyncio
from collections import Counter

import anthropic
import pytest
from pydantic import ValidationError

import config
import ingest
import rag
from schemas import MAX_EXTRACTO, RAGResponse, Referencia

REF = {"archivo": "contrato_04_ondaplay.md", "proveedor": "OndaPlay", "extracto": "preaviso de 60 días"}


# ---------------------------------------------------------------------------
# Esquema
# ---------------------------------------------------------------------------
def test_encontrado_true_exige_referencias():
    with pytest.raises(ValidationError, match="exige al menos una referencia"):
        RAGResponse(respuesta="60 días", encontrado=True, referencias=[])


def test_encontrado_false_no_admite_referencias():
    with pytest.raises(ValidationError, match="no admite referencias"):
        RAGResponse(respuesta="No lo sé", encontrado=False, referencias=[REF])


def test_respuestas_validas():
    assert RAGResponse(respuesta="60 días", encontrado=True, referencias=[REF]).error is None
    assert RAGResponse(respuesta="No lo sé", encontrado=False).referencias == []


def test_campo_error_no_llega_al_llm():
    # `error` lo completa el sistema: no debe figurar en las instrucciones de formato.
    instrucciones = rag.parser.get_format_instructions()
    assert '"error"' not in instrucciones
    assert '"referencias"' in instrucciones


def test_extracto_largo_se_recorta():
    ref = Referencia(archivo="a.md", proveedor="X", extracto="x" * 500)
    assert len(ref.extracto) == MAX_EXTRACTO


# ---------------------------------------------------------------------------
# Ingesta
# ---------------------------------------------------------------------------
def test_chunks_con_metadatos_e_ids_estables(entorno):
    documentos = ingest.load_documents()
    primera, segunda = ingest.split_documents(documentos), ingest.split_documents(documentos)

    assert len(documentos) == 10
    assert [c.id for c in primera] == [c.id for c in segunda]  # mismos IDs al reindexar
    assert len({c.id for c in primera}) == len(primera)  # sin repetidos
    for chunk in primera:
        assert {"archivo", "proveedor", "numero_contrato", "chunk_index", "hash_documento"} <= chunk.metadata.keys()
        assert chunk.page_content.startswith(f"[Contrato {chunk.metadata['numero_contrato']}")
    # 500 tokens del splitter + el encabezado de contexto (~11 tokens)
    assert max(ingest.contar_tokens(c.page_content) for c in primera) <= config.CHUNK_SIZE + 20


def test_ingesta_incremental(entorno):
    total = ingest.run_ingest()
    embebidos = entorno.textos_embebidos
    assert total == embebidos > 0

    # Sin cambios: no se recalcula ningún embedding.
    assert ingest.run_ingest() == total
    assert entorno.textos_embebidos == embebidos

    # Un contrato modificado: se reindexa SOLO ese contrato.
    ruta = config.DATA_DIR / "contrato_04_ondaplay.md"
    ruta.write_text(ruta.read_text(encoding="utf-8") + "\nNota agregada.\n", encoding="utf-8")
    chunks_ondaplay = len(ingest.split_documents([d for d in ingest.load_documents() if d.metadata["proveedor"] == "OndaPlay"]))
    assert ingest.run_ingest() == total
    assert entorno.textos_embebidos == embebidos + chunks_ondaplay

    # Un contrato eliminado: sus chunks desaparecen de la base.
    (config.DATA_DIR / "contrato_10_brisa_tv.md").unlink()
    assert ingest.run_ingest() < total
    assert "contrato_10_brisa_tv.md" not in ingest._documentos_indexados()


def test_modelo_de_embeddings_distinto_pide_reindex(entorno, monkeypatch):
    ingest.run_ingest()
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "otro-modelo")
    with pytest.raises(ingest.EmbeddingModelMismatchError, match="--reindex"):
        ingest.collection_count()


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
PREGUNTA = "¿Cuál es el plazo de preaviso para rescindir el contrato con OndaPlay?"


def test_respuesta_con_referencia_valida_y_una_sola_busqueda(entorno, usar_llm):
    ingest.run_ingest()
    llm = usar_llm()
    consultas_antes = entorno.consultas

    r = asyncio.run(rag.get_rag_response(PREGUNTA))

    assert r.encontrado and r.error is None
    assert r.referencias[0].archivo == "contrato_04_ondaplay.md"  # la búsqueda encontró el contrato
    assert llm.llamadas == 1
    assert entorno.consultas - consultas_antes == 1  # se busca UNA sola vez por consulta


def test_referencia_inventada_se_convierte_en_no_lo_se(entorno, usar_llm):
    ingest.run_ingest()
    usar_llm(lambda fuentes, _: {
        "respuesta": "60 días",
        "encontrado": True,
        "referencias": [{"archivo": "contrato_99_inventado.md", "proveedor": "X", "extracto": "..."}],
    })

    r = asyncio.run(rag.get_rag_response(PREGUNTA))

    assert (r.respuesta, r.encontrado, r.referencias, r.error) == ("No lo sé", False, [], None)


def test_umbral_se_abstiene_sin_llamar_al_llm(entorno, usar_llm):
    ingest.run_ingest()
    llm = usar_llm()

    # Ninguna palabra en común con los contratos -> distancia 1 > umbral.
    r = asyncio.run(rag.get_rag_response("Receta empanadas tucumanas"))

    assert (r.respuesta, r.encontrado, r.error) == ("No lo sé", False, None)
    assert llm.llamadas == 0


def test_json_invalido_se_reintenta(entorno, usar_llm):
    ingest.run_ingest()

    def generar(fuentes, prompt):
        if llm.llamadas == 1:
            return "esto no es JSON"  # primer intento: salida rota
        archivo, proveedor = fuentes[0]
        return {
            "respuesta": "60 días", "encontrado": True,
            "referencias": [{"archivo": archivo, "proveedor": proveedor, "extracto": "cita"}],
        }

    llm = usar_llm(generar)

    r = asyncio.run(rag.get_rag_response(PREGUNTA))

    assert r.encontrado
    assert llm.llamadas == 2  # el primer JSON roto disparó el reintento


def test_error_de_api_va_en_el_campo_error(entorno, usar_llm):
    ingest.run_ingest()

    def caida(fuentes, prompt):
        raise anthropic.APIConnectionError(message="sin conexión", request=None)

    usar_llm(caida)
    r = asyncio.run(rag.get_rag_response(PREGUNTA))

    assert r.error is not None and "Anthropic" in r.error
    assert not r.encontrado and r.referencias == []
    assert r.respuesta != "No lo sé"  # un fallo técnico no se confunde con una abstención


def test_base_inexistente_devuelve_error_controlado(entorno, usar_llm):
    usar_llm()
    r = asyncio.run(rag.get_rag_response(PREGUNTA))  # no se ejecutó la ingesta
    assert r.error is not None and "ingest.py" in r.error


def test_compare_contracts_cubre_todos_los_contratos(entorno, usar_llm):
    ingest.run_ingest()
    llm = usar_llm()

    r = asyncio.run(rag.compare_contracts("¿Qué contratos tienen ajuste de precio por inflación?"))

    por_archivo = Counter(archivo for archivo, _ in llm.ultimas_fuentes)
    assert r.encontrado
    assert llm.llamadas == 1  # una sola llamada al LLM para los 10 contratos
    assert len(por_archivo) == 10
    assert max(por_archivo.values()) <= config.COMPARE_K_POR_CONTRATO
