"""Simulaciones compartidas por los tests.

Los tests NO usan la API de Anthropic ni el modelo de embeddings real:
- `EmbeddingsPalabras`: embeddings de "bolsa de palabras" (determinísticos,
  instantáneos y sin descargas). Textos con palabras en común quedan cerca, así
  que la búsqueda igual encuentra el contrato correcto.
- `LLMSimulado`: lee las fuentes del prompt y devuelve el JSON que cada test
  necesita (válido, con una referencia inventada, mal formado, etc.).
Cada test trabaja sobre una copia de data/ y una base en una carpeta temporal.
"""

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
import ingest
import rag

DIMENSIONES = 16384  # muchas posiciones: casi sin colisiones entre palabras


class EmbeddingsPalabras(Embeddings):
    """Cada palabra suma 1 en una posición (por hash). Vectores normalizados."""

    def __init__(self) -> None:
        self.textos_embebidos = 0  # para comprobar cuánto se indexa
        self.consultas = 0  # para comprobar que cada pregunta se busca una sola vez

    def _vector(self, texto: str) -> list[float]:
        vector = [0.0] * DIMENSIONES
        for palabra in re.findall(r"\w+", texto.lower()):
            indice = int(hashlib.md5(palabra.encode()).hexdigest(), 16) % DIMENSIONES
            vector[indice] += 1.0
        norma = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norma for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.textos_embebidos += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.consultas += 1
        return self._vector(text)


class LLMSimulado:
    """Reemplaza a ChatAnthropic. `generar(fuentes, prompt)` arma la respuesta.

    `fuentes` es la lista de (archivo, proveedor) de los encabezados
    [Fuente: ... | Proveedor: ...] del contexto que recibió el "modelo".
    """

    def __init__(self, generar: Callable[[list[tuple[str, str]], str], dict | str]) -> None:
        self.generar = generar
        self.llamadas = 0
        self.ultimas_fuentes: list[tuple[str, str]] = []

    def _invocar(self, prompt_value) -> AIMessage:
        self.llamadas += 1
        texto = prompt_value.to_string()
        self.ultimas_fuentes = re.findall(r"\[Fuente: (\S+) \| Proveedor: ([^\]]+)\]", texto)
        salida = self.generar(self.ultimas_fuentes, texto)
        return AIMessage(content=salida if isinstance(salida, str) else json.dumps(salida))

    def runnable(self) -> RunnableLambda:
        return RunnableLambda(self._invocar)


def citar_primera_fuente(fuentes: list[tuple[str, str]], _prompt: str) -> dict:
    """Respuesta 'buena': cita el primer fragmento del contexto."""
    archivo, proveedor = fuentes[0]
    return {
        "respuesta": "Dato encontrado en el contrato.",
        "encontrado": True,
        "referencias": [{"archivo": archivo, "proveedor": proveedor, "extracto": "cita"}],
    }


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Copia de data/ + base temporal + embeddings simulados. Devuelve los embeddings."""
    data = tmp_path / "data"
    shutil.copytree(config.DATA_DIR, data)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path / "vectorstore")
    # Con bolsa de palabras, una pregunta sin palabras en común con los
    # contratos queda a distancia 1. El umbral real (0,18) está calibrado para
    # multilingual-e5-small; aquí usamos uno acorde a estos embeddings.
    monkeypatch.setattr(config, "UMBRAL_DISTANCIA", 0.95)

    embeddings = EmbeddingsPalabras()
    monkeypatch.setattr(ingest, "get_embeddings", lambda: embeddings)
    rag._get_componentes.cache_clear()
    yield embeddings
    rag._get_componentes.cache_clear()


@pytest.fixture
def usar_llm(monkeypatch):
    """Instala un LLMSimulado en rag.py: usar_llm(generar) -> LLMSimulado."""

    def _instalar(generar=citar_primera_fuente) -> LLMSimulado:
        llm = LLMSimulado(generar)
        monkeypatch.setattr(rag, "_crear_llm", llm.runnable)
        rag._get_componentes.cache_clear()
        return llm

    return _instalar
