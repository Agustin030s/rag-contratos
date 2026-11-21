# rag-contratos

Sistema RAG local para consultar y cruzar información de contratos. Indexa los contratos de `data/` en ChromaDB y, con **Claude (Anthropic)** como LLM, responde preguntas en español con un objeto validado (`RAGResponse`): la respuesta, si se encontró en los documentos y las referencias (archivo, proveedor y cita) que la respaldan. Si la información no está en los contratos, responde **"No lo sé"**.

```
data/*.md → chunking (500 tokens/50 overlap) → embeddings → ChromaDB (./vectorstore)
          → retriever (top_k) → prompt grounded → LLM → PydanticOutputParser → RAGResponse
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
│   └── ground_truth.json    # respuestas correctas de cada contrato (NO se indexa)
├── generate_contracts.py    # Fase 0: genera los contratos y el ground truth
├── config.py                # rutas, modelos, top_k, chunk_size (desde el .env)
├── schemas.py               # modelos Pydantic: Referencia y RAGResponse
├── ingest.py                # Fase 1: chunking + ChromaDB
├── rag.py                   # Fases 2 y 3: retriever + cadena LCEL asíncrona
├── test_rag.py              # pruebas contra el ground truth
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

## Ejecución

```bash
python generate_contracts.py   # opcional: los contratos ya están en data/
python ingest.py               # indexa en local, sin API key (si la base ya existe, no hace nada)
python ingest.py --reindex     # borra la colección y vuelve a indexar
python test_rag.py             # pruebas (si no hay base, primero ejecuta la ingesta)
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

Al final muestra un resumen del tipo `X de 7 pruebas pasaron`.

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

### Ejemplos de salida (`RAGResponse`)

Respuesta encontrada:

```json
{
  "respuesta": "El plazo de preaviso para rescindir el contrato con OndaPlay es de sesenta (60) días.",
  "encontrado": true,
  "referencias": [
    {
      "archivo": "contrato_04_ondaplay.md",
      "proveedor": "OndaPlay",
      "extracto": "notificando su decisión a la otra parte en forma fehaciente con un preaviso mínimo de sesenta (60) días."
    }
  ]
}
```

Información que no está en los contratos:

```json
{
  "respuesta": "No lo sé",
  "encontrado": false,
  "referencias": []
}
```

## Decisiones de diseño

- **Embeddings locales.** Anthropic no tiene un modelo de embeddings, y usar otro proveedor obligaría a tener una segunda API key. Por eso los embeddings se calculan en local con `HuggingFaceEmbeddings` y el modelo `intfloat/multilingual-e5-small`: gratuito, entrenado en español (entre otros idiomas), con vectores de 384 dimensiones y textos de hasta 512 tokens, que alcanza para los chunks de 500. Los modelos E5 se entrenaron con prefijos (`"passage: "` para los chunks y `"query: "` para las preguntas) y `ingest.py` los agrega. Los vectores se normalizan para que la distancia coseno sea comparable.
- **Tokens y no caracteres** (como en la Unidad 2). El `RecursiveCharacterTextSplitter` mide con `tiktoken` (`length_function=contar_tokens`), así `chunk_size=500` son 500 tokens reales. Los límites de los modelos (incluido el de 512 tokens del modelo de embeddings) y el costo se cuentan en tokens, y en español un carácter no equivale a un token. Los separadores (`"\n\n"`, `"\n"`, `". "`, `" "`, `""`) intentan cortar primero entre cláusulas y solo en último caso dentro de una palabra.
- **Encabezado en cada chunk.** Los fragmentos del medio de un contrato dicen "EL PROVEEDOR", nunca su nombre. Por eso `ingest.py` agrega `[Contrato CN-2026-004 | Proveedor: OndaPlay]` al inicio de cada chunk antes de generar el embedding. Sin esto, "¿qué preaviso tiene OndaPlay?" se parecería por igual a la cláusula de rescisión de los 10 contratos. El encabezado suma unos 11 tokens (máximo real por chunk: 511).
- **`TOP_K=4`.** Entre 3 y 5 es el rango recomendado. Con menos, es fácil que el fragmento correcto quede afuera; con más, el contexto se llena de fragmentos poco relevantes y aparece el *Lost in the Middle*: los LLM aprovechan peor la información que queda en el medio de un contexto largo.
- **Cómo se evita reindexar.** Si la colección ya existe y tiene chunks, `python ingest.py` lo registra (`La base ya existe con N chunks, se omite la ingesta`) y no vuelve a calcular los embeddings. Para forzarlo está `--reindex`. Además, cada chunk tiene un **ID estable** (hash del nombre del archivo + `chunk_index`): como Chroma hace *upsert* por ID, indexar dos veces el mismo chunk lo reemplaza en lugar de duplicarlo.
- **Embeddings no coincidentes (error #1).** El nombre del modelo de embeddings se guarda en los metadatos de la colección. Al abrir la base, si no coincide con `EMBEDDING_MODEL`, se lanza `EmbeddingModelMismatchError` con el mensaje "Ejecuta: python ingest.py --reindex". La verificación se hace **antes** de abrir la colección con LangChain, porque Chroma conserva la metadata original si la colección ya existe. Ingesta y consultas usan la misma función (`open_vectorstore`) y el modelo sale de un único lugar (`config.py`).
- **Cómo se evitan las alucinaciones**, en tres capas:
  1. **Prompt como filtro de veracidad:** responder solo con el contexto; "No lo sé" si no está; no inventar cifras, fechas ni cláusulas; citar el archivo; atención a las negaciones y a los anexos.
  2. **Campo `encontrado`:** el modelo tiene que declarar si la respuesta está en el contexto. Si es `false`, el código normaliza la salida a exactamente `"No lo sé"` y sin referencias.
  3. **Validación de referencias:** después del LLM se descarta (con un warning en el log) toda referencia cuyo `archivo` no esté entre los documentos realmente recuperados. Si no se recupera ningún documento, se responde "No lo sé" **sin llamar al LLM** (`RunnableBranch`).
- **`RunnableParallel` en la cadena.** `{"docs": retriever, "pregunta": RunnablePassthrough()}` conserva los documentos recuperados junto a la respuesta. Sin ellos no se podrían validar las referencias.
- **`with_retry` sobre `llm | parser`.** Si el modelo devuelve un JSON mal formado, el parser lanza `OutputParserException` dentro del bloque reintentado y se pide otra respuesta (máximo 2 intentos). También se reintentan los errores de red y de rate limit de Anthropic; los de autenticación no, porque una key inválida no se arregla reintentando.
- **Claude sin `temperature`.** `ChatAnthropic` se crea sin `temperature`, porque los modelos actuales de Anthropic la rechazan con un error 400. `max_tokens` queda alto (16000) a propósito: si la respuesta se corta, el JSON queda incompleto y el parser falla.
- **Errores controlados.** `get_rag_response` y `compare_contracts` nunca lanzan excepciones: los errores de la API de Anthropic, de red o de configuración se registran en el log y se devuelve una `RAGResponse` con un mensaje de error y `encontrado=false`. `ingest.py` hace lo mismo y termina con código de salida 1.
- **Modo comparación (`compare_contracts`).** Con `top_k=4`, una pregunta como "¿qué contratos tienen ajuste por inflación?" solo ve 4 fragmentos, es decir, como mucho 4 de los 10 contratos. Subir `top_k` tampoco sirve: los fragmentos más parecidos pueden ser varios del mismo contrato. Por eso este modo hace **una búsqueda por proveedor** con filtro de metadatos (`filter={"proveedor": p}`, `k=2`), las ejecuta en paralelo con `asyncio.gather` y hace **una sola llamada** al LLM con todos los fragmentos (máximo 2 por contrato), usando el mismo prompt y el mismo parser.
- **Dataset generado sin LLM.** `generate_contracts.py` arma los contratos con una plantilla a partir de una lista de diccionarios. El mismo diccionario produce el texto y la entrada de `ground_truth.json`, así las respuestas correctas se conocen con certeza.
- **Distancia en los logs.** `as_retriever()` no devuelve la distancia de cada fragmento. Para registrarla, `get_rag_response` repite la búsqueda con `asimilarity_search_with_score`, lo que solo implica calcular otra vez el embedding de la consulta (es local y gratis). La colección usa distancia coseno: 0 = idénticos.

> **Dónde se configura cada modelo:** Claude se crea en `_crear_llm()` (`rag.py`) y los embeddings en `get_embeddings()` (`ingest.py`). Para cambiar de modelo de Claude, alcanza con modificar `LLM_MODEL` en el `.env`. Si cambias `EMBEDDING_MODEL`, hay que reindexar (`python ingest.py --reindex`); si te olvidas, el propio sistema lo avisa.
