"""Configuración central del proyecto.

Todo se lee del `.env` con un valor por defecto, así el código nunca tiene
valores "mágicos" repartidos por los módulos. La única API key del proyecto
(ANTHROPIC_API_KEY) NO se lee aquí: la toma directamente el SDK de Anthropic.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Cargamos el .env al importar la configuración: cualquier script que use
# config.py ve las mismas variables, sin depender del orden de los imports.
load_dotenv()

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR") or BASE_DIR / "vectorstore")
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "contratos"

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
# El modelo de embeddings de la ingesta y el de las consultas TIENEN que ser
# el mismo: cada modelo genera vectores en su propio "espacio" y compararlos
# entre sí no tiene sentido. Por eso se define en un solo lugar.
# Anthropic no ofrece un modelo de embeddings, así que los calculamos en local
# con sentence-transformers (gratis y sin API key). multilingual-e5-small
# entiende español y acepta textos de hasta 512 tokens: alcanza para los
# chunks de 500. La primera vez se descarga (~470 MB) y queda en caché.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-small"
LLM_MODEL = os.getenv("LLM_MODEL") or "claude-sonnet-5-5"
# Límite de tokens de la respuesta. No conviene ponerlo bajo: si el modelo
# llega al límite, el JSON queda cortado y el parser falla.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or 16000)

# ---------------------------------------------------------------------------
# Chunking (medido en TOKENS, no en caracteres)
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 500)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)

# ---------------------------------------------------------------------------
# Recuperación
# ---------------------------------------------------------------------------
# TOP_K debe quedar entre 3 y 5. Con menos, es fácil que el fragmento correcto
# quede afuera; con más, el contexto se llena de fragmentos poco relevantes y
# el modelo tiende a ignorar lo que queda en el medio ("Lost in the Middle":
# los LLM usan mejor la información del principio y del final del contexto).
TOP_K = int(os.getenv("TOP_K") or 4)
if not 3 <= TOP_K <= 5:
    logger.warning("TOP_K=%d está fuera del rango recomendado (3 a 5).", TOP_K)

# Modo comparación: cuántos fragmentos se recuperan de CADA contrato.
COMPARE_K_POR_CONTRATO = 2
