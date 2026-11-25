"""Pruebas del RAG contra el ground truth.

Uso:  python test_rag.py

Antes de las pruebas ejecuta la ingesta (incremental: si la base existe y
ningún contrato cambió, no hace nada). Cada corrida deja evidencia en
eval/resultados/: un .json con las respuestas y verificaciones, y un .log con
el log completo, ambos con la fecha y el modelo en el nombre.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime

from dotenv import load_dotenv

import config
from schemas import RAGResponse

GROUND_TRUTH: dict[str, dict] = {}
resultados: list[dict] = []

RESULTADOS_DIR = config.EVAL_DIR / "resultados"
INICIO = datetime.now()
# Ej.: eval/resultados/2026-11-24_1530_claude-sonnet-5-5.json (y .log)
NOMBRE_CORRIDA = f"{INICIO:%Y-%m-%d_%H%M}_{config.LLM_MODEL}"


# ---------------------------------------------------------------------------
# Ayudas
# ---------------------------------------------------------------------------
def separador(titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def normalizar(texto: str) -> str:
    # "99.5" y "99,5" valen lo mismo; comparamos en minúsculas.
    return texto.lower().replace(",", ".")


def es_no_lo_se(r: RAGResponse) -> bool:
    # Un error técnico NO cuenta como "No lo sé", aunque tenga encontrado=false.
    return r.error is None and r.respuesta.strip().rstrip(".") == "No lo sé"


def cita(r: RAGResponse, proveedor: str) -> bool:
    return GROUND_TRUTH[proveedor]["archivo"] in {ref.archivo for ref in r.referencias}


def verificar(nombre: str, pregunta: str, r: RAGResponse, checks: dict[str, bool]) -> None:
    checks = {"sin error técnico": r.error is None, **checks}
    print(f"Pregunta: {pregunta}")
    print(r.model_dump_json(indent=2))
    for descripcion, ok in checks.items():
        print(f"  {'✔' if ok else '✘'} {descripcion}")
    resultados.append({
        "prueba": nombre,
        "ok": all(checks.values()),
        "pregunta": pregunta,
        "respuesta": r.model_dump(),
        "verificaciones": checks,
    })


def guardar_evidencia() -> None:
    """Guarda las respuestas reales de esta corrida, con fecha y modelos."""
    reporte = {
        "fecha": INICIO.isoformat(timespec="seconds"),
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.TOP_K,
        "umbral_distancia": config.UMBRAL_DISTANCIA,
        "resumen": f"{sum(r['ok'] for r in resultados)} de {len(resultados)} pruebas pasaron",
        "pruebas": resultados,
    }
    ruta = RESULTADOS_DIR / f"{NOMBRE_CORRIDA}.json"
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEvidencia guardada en {ruta.relative_to(config.BASE_DIR)} (y el log en .log)")


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
async def prueba_respuesta_en_documentos() -> None:
    from rag import get_rag_response

    separador("1. Pregunta con respuesta en los documentos")
    pregunta = "¿Cuál es el plazo de preaviso para rescindir el contrato con OndaPlay?"
    esperado = GROUND_TRUTH["OndaPlay"]["preaviso_dias"]
    r = await get_rag_response(pregunta)
    verificar("Preaviso OndaPlay", pregunta, r, {
        "encontrado=True": r.encontrado,
        f"la respuesta menciona {esperado} días (ground truth)": bool(
            re.search(rf"\b{esperado}\b", r.respuesta)
        ),
        "las referencias incluyen el archivo de OndaPlay": cita(r, "OndaPlay"),
    })


async def prueba_trampas_no_esta() -> None:
    from rag import get_rag_response

    preguntas = [
        ("Precio Netflix (no existe)", "¿Cuál es el precio del contrato con Netflix?"),
        ("Criptomonedas StreamVista", "¿Qué dice el contrato de StreamVista sobre pagos en criptomonedas?"),
    ]
    for i, (nombre, pregunta) in enumerate(preguntas, start=1):
        separador(f"2.{i}. Pregunta trampa (la información no está)")
        r = await get_rag_response(pregunta)
        verificar(nombre, pregunta, r, {
            "encontrado=False": not r.encontrado,
            "respuesta = 'No lo sé'": es_no_lo_se(r),
        })


async def prueba_negacion() -> None:
    from rag import get_rag_response

    separador("3. Trampa de negación (menciona la inflación para negar el ajuste)")
    gt = next(g for g in GROUND_TRUTH.values() if g["caso_trampa"] == "negacion_inflacion")
    pregunta = f"¿El precio del contrato con {gt['proveedor']} se ajusta por inflación?"
    r = await get_rag_response(pregunta)
    texto = normalizar(r.respuesta)
    verificar("Negación de ajuste", pregunta, r, {
        "encontrado=True": r.encontrado,
        "responde que NO tiene ajuste (ground truth: sin ajuste)": (
            re.search(r"\bno\b", texto) is not None and not es_no_lo_se(r)
        ),
        f"cita el archivo de {gt['proveedor']}": cita(r, gt["proveedor"]),
    })


async def prueba_anexo() -> None:
    from rag import get_rag_response

    separador("4. Dato en un anexo")
    gt = next(g for g in GROUND_TRUTH.values() if g["caso_trampa"] == "ajuste_en_anexo")
    pregunta = (
        f"¿El contrato con {gt['proveedor']} tiene ajuste de precio? "
        "¿Con qué índice, frecuencia y tope?"
    )
    r = await get_rag_response(pregunta)
    texto = normalizar(r.respuesta)
    indice = gt["indice"].split()[0].lower()  # "ipc" o "cpi"
    verificar("Ajuste en el anexo", pregunta, r, {
        "encontrado=True": r.encontrado,
        f"índice {gt['indice']}": indice in texto,
        f"frecuencia {gt['frecuencia']}": gt["frecuencia"] in texto,
        f"tope {gt['tope']} %": re.search(rf"\b{gt['tope']}\s*%", texto) is not None,
    })


async def prueba_cruce() -> None:
    from rag import compare_contracts

    separador("5. Cruce de contratos (compare_contracts)")
    pregunta = "¿Qué contratos tienen ajuste por inflación y con qué índice?"
    # Pedimos un formato de una línea por contrato para poder compararlo con
    # el ground truth de forma automática.
    consulta = (
        pregunta + " Responde con una línea por cada contrato que tenga ajuste, con el "
        "formato 'Proveedor: índice'. No incluyas los contratos sin ajuste."
    )
    r = await compare_contracts(consulta)
    print(f"Pregunta: {pregunta}")
    print(r.model_dump_json(indent=2))
    comparacion = []

    # Para cada proveedor: ¿aparece en alguna línea de la respuesta y con qué índice?
    lineas = [normalizar(linea) for linea in r.respuesta.splitlines() if linea.strip()]
    aciertos = 0
    print("\n  Comparación con el ground truth:")
    for proveedor, gt in GROUND_TRUTH.items():
        linea = next((l for l in lineas if proveedor.lower() in l), None)
        predicho = None
        if linea is not None:
            # El CPI se revisa PRIMERO: el modelo a veces escribe "IPC de EE. UU. (CPI)",
            # que también contiene "ipc".
            if any(s in linea for s in ("cpi", "ee. uu", "ee.uu", "estados unidos")):
                predicho = "CPI de EE. UU."
            elif "ipc" in linea or "indec" in linea:
                predicho = "IPC del INDEC"
            else:
                predicho = "?"
        ok = predicho == gt["indice"]
        aciertos += ok
        comparacion.append({"proveedor": proveedor, "esperado": gt["indice"], "obtenido": predicho, "ok": ok})
        print(
            f"  {'✔' if ok else '✘'} {proveedor:<16} esperado: {gt['indice'] or 'sin ajuste':<16}"
            f" obtenido: {predicho or 'sin ajuste'}"
        )
    print(f"  Aciertos: {aciertos} de {len(GROUND_TRUTH)}")
    resultados.append({
        "prueba": "Cruce: ajuste por inflación",
        "ok": r.error is None and aciertos == len(GROUND_TRUTH),
        "pregunta": consulta,
        "respuesta": r.model_dump(),
        "verificaciones": {f"aciertos: {aciertos} de {len(GROUND_TRUTH)}": aciertos == len(GROUND_TRUTH)},
        "comparacion_ground_truth": comparacion,
    })


async def prueba_comparacion() -> None:
    from rag import compare_contracts

    separador("6. Comparación de dos contratos (compare_contracts)")
    pregunta = "Compara el SLA y la penalización de CineNova y Faro Media"
    r = await compare_contracts(pregunta, proveedores=["CineNova", "Faro Media"])
    texto = normalizar(r.respuesta)
    checks = {"encontrado=True": r.encontrado}
    for proveedor in ("CineNova", "Faro Media"):
        gt = GROUND_TRUTH[proveedor]
        checks[f"{proveedor}: SLA {gt['sla']} %"] = str(gt["sla"]) in texto
        checks[f"{proveedor}: penalización {gt['penalizacion_sla_pct']} %"] = (
            re.search(rf"\b{gt['penalizacion_sla_pct']}\s*%", texto) is not None
        )
        checks[f"cita el archivo de {proveedor}"] = cita(r, proveedor)
    verificar("Comparación SLA", pregunta, r, checks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    # La única key es la de Anthropic: los embeddings se calculan en local.
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("No hay ANTHROPIC_API_KEY configurada. Copia .env.example a .env y complétala.")
        return

    GROUND_TRUTH.update(
        {g["proveedor"]: g for g in json.loads(config.GROUND_TRUTH_PATH.read_text(encoding="utf-8"))}
    )

    from ingest import run_ingest

    try:
        separador("Ingesta (solo indexa lo que no está o cambió)")
        run_ingest()
    except Exception as exc:
        logging.error("No se pudo preparar la base [%s]: %s", type(exc).__name__, exc)
        return

    await prueba_respuesta_en_documentos()
    await prueba_trampas_no_esta()
    await prueba_negacion()
    await prueba_anexo()
    await prueba_cruce()
    await prueba_comparacion()

    separador("RESUMEN")
    for r in resultados:
        print(f"  {'✔' if r['ok'] else '✘'} {r['prueba']}")
    pasaron = sum(r["ok"] for r in resultados)
    print(f"\n{pasaron} de {len(resultados)} pruebas pasaron")
    guardar_evidencia()


if __name__ == "__main__":
    load_dotenv()
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    # El log va a la consola y, además, a un archivo: queda como evidencia.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTADOS_DIR / f"{NOMBRE_CORRIDA}.log", encoding="utf-8"),
        ],
    )
    # Bajamos el ruido de las librerías HTTP para que se lean los logs del RAG.
    for ruidoso in ("httpx", "httpx2", "sentence_transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    asyncio.run(main())
