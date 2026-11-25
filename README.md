# rag-contratos

Sistema RAG local para consultar y cruzar información de contratos. Indexa los contratos de `data/` en ChromaDB y, con **Claude (Anthropic)** como LLM, responde preguntas en español con un objeto validado (`RAGResponse`): la respuesta, si se encontró en los documentos y las referencias (archivo, proveedor y cita) que la respaldan. Si la información no está en los contratos, responde **"No lo sé"**.

```
data/*.md → chunking (500 tokens/50 overlap) → embeddings → ChromaDB (./vectorstore)
          → búsqueda con distancia (top_k) → umbral → prompt grounded → LLM
          → PydanticOutputParser → validación de referencias → RAGResponse
```

| Etapa | Dónde corre | Modelo | ¿Necesita API key? |
|---|---|---|---|
| LLM (redacta la respuesta) | API de Anthropic | `claude-sonnet-5-5` | Sí: `ANTHROPIC_API_KEY` |
| Embeddings (ingesta y consultas) | En tu computadora (`sentence-transformers`) | `intfloat/multilingual-e5-small` | No |

La única API key del proyecto es la de Anthropic. Anthropic no ofrece un modelo de embeddings, así que los embeddings se calculan en local con un modelo gratuito y multilingüe. La ingesta no llama a ninguna API.

> **Dataset ficticio.** Los 10 contratos de `data/` son inventados y se generaron con fines educativos. Las empresas (el cliente ConectaNet S.A. y los proveedores de streaming) y todos los datos no son reales.

## Estructura

```
rag-contratos/
├── data/                    # 10 contratos .md generados (dataset de ejemplo)
├── eval/
│   ├── ground_truth.json    # respuestas correctas de cada contrato (NO se indexa)
│   └── resultados/          # evidencia de cada corrida de test_rag.py (.json + .log)
├── generate_contracts.py    # Fase 0: genera los contratos y el ground truth
├── config.py                # rutas, modelos, top_k, chunk_size (desde el .env)
├── schemas.py               # modelos Pydantic: Referencia y RAGResponse
├── ingest.py                # Fase 1: chunking + ChromaDB
├── rag.py                   # Fases 2 y 3: retriever + cadena LCEL asíncrona
├── test_rag.py              # pruebas con la API real, contra el ground truth
├── tests/                   # tests con pytest, con LLM y embeddings simulados (sin API key)
│   ├── conftest.py
│   └── test_unitarios.py
├── pytest.ini
├── requirements.txt         # dependencias
├── .env.example
├── .gitignore
└── README.md
```

### El dataset

| Archivo | Proveedor | Contenido | Ajuste de precio | Caso especial |
|---|---|---|---|---|
| `contrato_01_streamvista.md` | StreamVista | Películas | CPI de EE. UU., anual, sin tope | — |
| `contrato_02_cinenova.md` | CineNova | Series | IPC del INDEC, trimestral, tope 15 % | — |
| `contrato_03_pantallasur.md` | PantallaSur | Deportes | IPC del INDEC, semestral, sin tope | — |
| `contrato_04_ondaplay.md` | OndaPlay | Infantil | Precio fijo | — |
| `contrato_05_mirador_tv.md` | Mirador TV | Documentales | Precio fijo | **Trampa:** menciona la inflación para negar el ajuste |
| `contrato_06_faro_media.md` | Faro Media | Noticias | CPI de EE. UU., semestral, tope 4 % | — |
| `contrato_07_lumen_plus.md` | Lumen+ | Series premium | Precio fijo | — |
| `contrato_08_kairos_play.md` | Kairos Play | Anime | IPC del INDEC, anual, tope 30 % | **Trampa:** el ajuste está en el ANEXO I |
| `contrato_09_atlas_streaming.md` | Atlas Streaming | Música | Precio fijo | — |
| `contrato_10_brisa_tv.md` | Brisa TV | Telenovelas | IPC del INDEC, trimestral, tope 10 % | — |

Cada contrato tiene entre 1327 y 1477 tokens y 11 cláusulas (objeto, vigencia, renovación, precio, ajuste, pago, exclusividad, SLA, rescisión, datos personales y jurisdicción). Con chunks de 500 tokens se divide en 3 o 4 fragmentos (34 en total). El resto de los valores de cada contrato está en `eval/ground_truth.json`.

## Instalación

Requiere **Python 3.12**.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Luego completa `ANTHROPIC_API_KEY` en `.env`.

> La primera vez que ejecutes `ingest.py` se descarga el modelo de embeddings (unos 470 MB, una sola vez; queda en la caché de Hugging Face). `pip install` también instala `torch`, que es grande: puede tardar unos minutos.

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `ANTHROPIC_API_KEY` | API key de Anthropic (la única del proyecto) | — |
| `LLM_MODEL` | Modelo de Claude | `claude-sonnet-5-5` |
| `LLM_MAX_TOKENS` | Máximo de tokens de la respuesta | `16000` |
| `EMBEDDING_MODEL` | Modelo de embeddings local (Hugging Face) | `intfloat/multilingual-e5-small` |
| `TOP_K` | Fragmentos recuperados por consulta (3 a 5) | `4` |
| `UMBRAL_DISTANCIA` | Distancia máxima de un fragmento para usarlo; si ninguno la cumple, "No lo sé" sin llamar al LLM | `0.18` |

## Ejecución

```bash
python generate_contracts.py   # opcional: los contratos ya están en data/
python ingest.py               # indexa en local, sin API key: solo los contratos nuevos o modificados
python ingest.py --reindex     # borra la colección y vuelve a indexar todo
pytest -v                      # 15 tests con LLM y embeddings simulados: sin API key ni descargas
python test_rag.py             # 7 pruebas con Claude real (antes ejecuta la ingesta incremental)
```

`test_rag.py` ejecuta 7 pruebas y compara las respuestas con `eval/ground_truth.json`:

| # | Prueba | Qué verifica |
|---|---|---|
| 1 | Preaviso de OndaPlay | `encontrado=True`, que diga 60 días y que cite el archivo de OndaPlay |
| 2.1 | Precio de Netflix (no hay contrato) | `encontrado=False` y "No lo sé" |
| 2.2 | Criptomonedas en StreamVista (el contrato no lo menciona) | `encontrado=False` y "No lo sé" |
| 3 | ¿Mirador TV se ajusta por inflación? | Que responda que **no** (trampa de negación) |
| 4 | Ajuste de Kairos Play | Índice, frecuencia y tope tomados del **anexo** |
| 5 | ¿Qué contratos tienen ajuste y con qué índice? | `compare_contracts`: lista contrato por contrato con aciertos y errores |
| 6 | SLA y penalización de CineNova vs. Faro Media | `compare_contracts` con dos proveedores |

Al final muestra un resumen del tipo `X de 7 pruebas pasaron` y **guarda la evidencia** en `eval/resultados/`: un `.json` con la fecha, los modelos, cada pregunta, la `RAGResponse` completa y las verificaciones, y un `.log` con el log de la corrida (chunks recuperados, distancias, referencias descartadas).

**Resultado de la corrida real** (24/11/2026, `claude-sonnet-5-5`, `intfloat/multilingual-e5-small`, `TOP_K=4`, umbral 0,18): **7 de 7 pruebas pasaron** y el cruce acertó los 10 contratos. Evidencia: `eval/resultados/2026-11-24_2001_claude-sonnet-5-5.json` y su `.log`.

### Tests sin API key (`pytest`)

`tests/` tiene 15 tests que reemplazan el LLM por un `RunnableLambda` que cita las fuentes del prompt y los embeddings por una "bolsa de palabras" determinística. No necesitan API key ni descargan el modelo, y cada uno trabaja sobre una copia de `data/` en una carpeta temporal (nunca toca `vectorstore/`). Cubren:

- **Esquema:** `encontrado=true` exige referencias y `encontrado=false` no las admite; el campo `error` no llega a las instrucciones del LLM; los extractos largos se recortan.
- **Ingesta:** metadatos, IDs estables, ingesta incremental (sin cambios no recalcula nada, un contrato modificado reindexa solo ese y uno borrado desaparece) y detección de un modelo de embeddings distinto.
- **RAG:** una sola búsqueda por consulta; una referencia inventada se convierte en "No lo sé"; el umbral se abstiene **sin llamar al LLM**; un JSON roto se reintenta; un error de API va en el campo `error`; `compare_contracts` cubre los 10 contratos con máximo 2 fragmentos cada uno y una sola llamada al LLM.

### Uso desde otro archivo

```python
import asyncio
from rag import compare_contracts, get_rag_response

async def main():
    r = await get_rag_response("¿Qué jurisdicción tiene el contrato con Brisa TV?")
    print(r.respuesta, [ref.archivo for ref in r.referencias])

    r = await compare_contracts("¿Qué contratos tienen exclusividad territorial?")
    print(r.respuesta)

asyncio.run(main())
```

### Ejemplos de salida (`RAGResponse`) de una corrida real

Salidas copiadas de `eval/resultados/2026-11-24_2001_claude-sonnet-5-5.json` (`python test_rag.py` del 24/11/2026, con `claude-sonnet-5-5` y `intfloat/multilingual-e5-small`).

Respuesta encontrada (prueba 1):

```json
{
  "respuesta": "Según el contrato con OndaPlay (CN-2026-004), cualquiera de las partes puede rescindirlo sin expresión de causa con un preaviso mínimo de sesenta (60) días, notificado en forma fehaciente. Además, si EL CLIENTE rescinde antes de cumplirse la mitad de la vigencia, debe pagar una penalidad equivalente a tres (3) meses de facturación calculada sobre el mínimo garantizado de suscriptores.",
  "encontrado": true,
  "referencias": [
    {
      "archivo": "contrato_04_ondaplay.md",
      "proveedor": "OndaPlay",
      "extracto": "Cualquiera de LAS PARTES podrá rescindir el presente contrato sin expresión de causa, notificando su decisión a la otra parte en forma fehaciente con un preaviso mínimo de sesenta (60) días."
    }
  ],
  "error": null
}
```

Información que no está en los contratos (prueba 2.1, "¿Cuál es el precio del contrato con Netflix?"):

```json
{
  "respuesta": "No lo sé",
  "encontrado": false,
  "referencias": [],
  "error": null
}
```

Un error técnico **no** se confunde con un "No lo sé": la respuesta lo dice y el detalle va en `error`. Este ejemplo sale del test `test_error_de_api_va_en_el_campo_error`, que simula una caída de la API:

```json
{
  "respuesta": "No se pudo responder por un error técnico.",
  "encontrado": false,
  "referencias": [],
  "error": "La API de Anthropic no está disponible (APIConnectionError). Intenta más tarde."
}
```

## Decisiones de diseño

- **Embeddings locales.** Anthropic no tiene un modelo de embeddings, y usar otro proveedor obligaría a tener una segunda API key. Por eso los embeddings se calculan en local con `HuggingFaceEmbeddings` y el modelo `intfloat/multilingual-e5-small`: gratuito, entrenado en español (entre otros idiomas), con vectores de 384 dimensiones y textos de hasta 512 tokens, que alcanza para los chunks de 500. Los modelos E5 se entrenaron con prefijos (`"passage: "` para los chunks y `"query: "` para las preguntas) y `ingest.py` los agrega. Los vectores se normalizan para que la distancia coseno sea comparable.
- **Tokens y no caracteres** (como en la Unidad 2). El `RecursiveCharacterTextSplitter` mide con `tiktoken` (`length_function=contar_tokens`), así `chunk_size=500` son 500 tokens reales. Los límites de los modelos (incluido el de 512 tokens del modelo de embeddings) y el costo se cuentan en tokens, y en español un carácter no equivale a un token. Los separadores (`"\n\n"`, `"\n"`, `". "`, `" "`, `""`) intentan cortar primero entre cláusulas y solo en último caso dentro de una palabra.
- **Encabezado en cada chunk.** Los fragmentos del medio de un contrato dicen "EL PROVEEDOR", nunca su nombre. Por eso `ingest.py` agrega `[Contrato CN-2026-004 | Proveedor: OndaPlay]` al inicio de cada chunk antes de generar el embedding. Sin esto, "¿qué preaviso tiene OndaPlay?" se parecería por igual a la cláusula de rescisión de los 10 contratos. El encabezado suma unos 11 tokens (máximo real por chunk: 511).
- **`TOP_K=4`.** Entre 3 y 5 es el rango recomendado. Con menos, es fácil que el fragmento correcto quede afuera; con más, el contexto se llena de fragmentos poco relevantes y aparece el *Lost in the Middle*: los LLM aprovechan peor la información que queda en el medio de un contexto largo.
- **Ingesta incremental (cómo se evita reindexar).** Cada chunk guarda el `hash_documento` (SHA-256 del contenido del contrato). Al ejecutar `python ingest.py`, se compara el hash de cada archivo de `data/` con el guardado en la base, leyendo solo los metadatos, sin cargar el modelo de embeddings:
  - si nada cambió, se registra `La base ya existe con N chunks, se omite la ingesta` y termina en un instante;
  - un contrato **nuevo** se indexa; uno **modificado** se reindexa (primero se borran sus chunks viejos, así no quedan huérfanos si ahora tiene menos); uno **eliminado** desaparece de la base. Los demás no se tocan.

  `--reindex` borra todo y vuelve a indexar. Además, cada chunk tiene un **ID estable** (hash del nombre del archivo + `chunk_index`), así indexar dos veces el mismo chunk lo reemplaza en lugar de duplicarlo.
- **Embeddings no coincidentes (error #1).** El nombre del modelo de embeddings se guarda en los metadatos de la colección. Al abrir la base, si no coincide con `EMBEDDING_MODEL`, se lanza `EmbeddingModelMismatchError` con el mensaje "Ejecuta: python ingest.py --reindex". La verificación se hace **antes** de abrir la colección con LangChain, porque Chroma conserva la metadata original si la colección ya existe. Ingesta y consultas usan la misma función (`open_vectorstore`) y el modelo sale de un único lugar (`config.py`).
- **Cómo se evitan las alucinaciones**, en cuatro capas:
  1. **Umbral de distancia (antes del LLM):** los fragmentos con distancia mayor a `UMBRAL_DISTANCIA` se descartan, y si no queda ninguno se responde "No lo sé" **sin llamar al LLM** (`RunnableBranch`). Sin umbral esa rama nunca se activaba, porque la base siempre devuelve `k` fragmentos. El valor 0,18 se calibró con el modelo real: las preguntas sobre los contratos quedaron entre 0,105 y 0,168, y las preguntas fuera de tema ("receta de empanadas", "mundial de 1986") desde 0,187. Las trampas que sí hablan de contratos ("precio con Netflix", 0,147) pasan el umbral y las filtra el prompt.
  2. **Prompt como filtro de veracidad:** responder solo con el contexto; "No lo sé" si no está; no inventar cifras, fechas ni cláusulas; citar el archivo; atención a las negaciones y a los anexos.
  3. **Reglas en el esquema (`model_validator`):** `encontrado=true` exige al menos una referencia y `encontrado=false` no admite ninguna. Si el LLM rompe la regla, el parser falla y se reintenta. La regla también vale para cualquier `RAGResponse` que arme el código.
  4. **Validación de referencias:** después del LLM se descarta (con un warning en el log) toda referencia cuyo `archivo` no esté entre los documentos realmente recuperados. Si no queda **ninguna** referencia válida, la respuesta se convierte en "No lo sé": una respuesta sin fuentes reales no se puede verificar.
- **Una sola búsqueda por consulta.** El primer paso de la cadena es `asimilarity_search_with_score`, y no `as_retriever()`, porque este último no devuelve la distancia. Con ese único resultado se registran las distancias en el log, se aplica el umbral y se arma el contexto. `RunnableParallel` conserva lo recuperado junto a la pregunta, y `RunnablePassthrough.assign` agrega los documentos filtrados y la respuesta, para validar las referencias al final.
- **La carga del modelo no bloquea el event loop.** La primera consulta carga el modelo de embeddings (~470 MB), una operación bloqueante. Por eso `get_rag_response` y `compare_contracts` llaman a `_get_componentes()` con `await asyncio.to_thread(...)`: la carga corre en otro hilo y el event loop sigue libre para otras tareas.
- **`with_retry` sobre `llm | parser`.** Si el modelo devuelve un JSON mal formado, el parser lanza `OutputParserException` dentro del bloque reintentado y se pide otra respuesta (máximo 2 intentos). También se reintentan los errores de red y de rate limit de Anthropic; los de autenticación no, porque una key inválida no se arregla reintentando.
- **Claude sin `temperature`.** `ChatAnthropic` se crea sin `temperature`, porque los modelos actuales de Anthropic la rechazan con un error 400. `max_tokens` queda alto (16000) a propósito: si la respuesta se corta, el JSON queda incompleto y el parser falla.
- **Errores controlados y distinguibles.** `get_rag_response` y `compare_contracts` nunca lanzan excepciones: los errores de la API de Anthropic, de red o de configuración se registran en el log y se devuelve una `RAGResponse` con `respuesta="No se pudo responder por un error técnico."`, `encontrado=false` y el detalle en el campo **`error`**. Así quien llama distingue un fallo técnico (`error` con texto) de un "No lo sé" legítimo (`error=null`). El campo `error` lo completa solo el sistema: con `SkipJsonSchema` queda fuera de las instrucciones de formato que recibe el LLM. `ingest.py` también registra los errores y termina con código de salida 1.
- **Modo comparación (`compare_contracts`).** Con `top_k=4`, una pregunta como "¿qué contratos tienen ajuste por inflación?" solo ve 4 fragmentos, es decir, como mucho 4 de los 10 contratos. Subir `top_k` tampoco sirve: los fragmentos más parecidos pueden ser varios del mismo contrato. Por eso este modo hace **una búsqueda por proveedor** con filtro de metadatos (`filter={"proveedor": p}`, `k=2`), las ejecuta en paralelo con `asyncio.gather` y hace **una sola llamada** al LLM con todos los fragmentos (máximo 2 por contrato), usando el mismo prompt y el mismo parser.
- **Dataset generado sin LLM.** `generate_contracts.py` arma los contratos con una plantilla a partir de una lista de diccionarios. El mismo diccionario produce el texto y la entrada de `ground_truth.json`, así las respuestas correctas se conocen con certeza.
- **Evidencia y tests.** `test_rag.py` guarda cada corrida real en `eval/resultados/` (fecha y modelo en el nombre). `pytest` prueba la lógica sin API key, con LLM y embeddings simulados, así los tests corren en cualquier máquina sin gastar créditos.

> **Dónde se configura cada modelo:** Claude se crea en `_crear_llm()` (`rag.py`) y los embeddings en `get_embeddings()` (`ingest.py`). Para cambiar de modelo de Claude, alcanza con modificar `LLM_MODEL` en el `.env`. Si cambias `EMBEDDING_MODEL`, hay que reindexar (`python ingest.py --reindex`); si te olvidas, el propio sistema lo avisa.
