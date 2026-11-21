"""Fase 0 — Genera el dataset de contratos ficticios.

Uso:  python generate_contracts.py

Los contratos se arman con una plantilla a partir de la lista CONTRATOS, SIN
usar un LLM. Así los datos son 100 % consistentes y conocemos la respuesta
correcta de cada pregunta: el mismo diccionario genera el texto del contrato y
su entrada en eval/ground_truth.json.
"""

import json
import logging
from datetime import date

import tiktoken

import config

logger = logging.getLogger(__name__)

LEYENDA = (
    "DOCUMENTO FICTICIO GENERADO CON FINES EDUCATIVOS. "
    "Las empresas y los datos no son reales."
)
CLIENTE = "ConectaNet S.A."
MIN_TOKENS, MAX_TOKENS = 900, 1500

# ---------------------------------------------------------------------------
# Datos de cada contrato
# ---------------------------------------------------------------------------
# ajuste = None -> precio fijo. Si hay ajuste: índice "IPC" (INDEC) o "CPI"
# (EE. UU.), frecuencia y tope porcentual (None = sin tope).
# caso_trampa:
#   "negacion_inflacion" -> precio fijo, pero el texto MENCIONA la inflación
#                           para negar el ajuste.
#   "ajuste_en_anexo"    -> el ajuste no está en la cláusula de precio, sino
#                           en un ANEXO al final del documento.
CONTRATOS: list[dict] = [
    {
        "numero": "CN-2026-001", "proveedor": "StreamVista", "razon_social": "StreamVista S.R.L.",
        "domicilio": "Av. Imaginaria 450, Ciudad Autónoma de Buenos Aires",
        "fecha_firma": date(2026, 1, 20), "fecha_inicio": date(2026, 2, 1),
        "contenido": "películas",
        "detalle": "estrenos de cine internacional, clásicos restaurados y películas de producción nacional, con un catálogo mínimo de 1.200 títulos",
        "vigencia_meses": 24, "renovacion_automatica": True,
        "moneda": "USD", "precio": 2.10, "minimo_suscriptores": 50_000,
        "ajuste": {"indice": "CPI", "frecuencia": "anual", "tope": None},
        "plazo_pago_dias": 30, "exclusividad": True,
        "sla": 99.9, "penalizacion_sla_pct": 10,
        "preaviso_dias": 60, "penalidad_rescision_meses": 3, "jurisdiccion": "CABA",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-002", "proveedor": "CineNova", "razon_social": "CineNova Contenidos S.A.",
        "domicilio": "Calle Inventada 1020, Córdoba, provincia de Córdoba",
        "fecha_firma": date(2026, 2, 16), "fecha_inicio": date(2026, 3, 1),
        "contenido": "series",
        "detalle": "series originales, miniseries y temporadas completas de ficción internacional, con un mínimo de 300 temporadas disponibles",
        "vigencia_meses": 12, "renovacion_automatica": False,
        "moneda": "ARS", "precio": 1850.00, "minimo_suscriptores": 20_000,
        "ajuste": {"indice": "IPC", "frecuencia": "trimestral", "tope": 15},
        "plazo_pago_dias": 45, "exclusividad": False,
        "sla": 99.5, "penalizacion_sla_pct": 5,
        "preaviso_dias": 30, "penalidad_rescision_meses": 2, "jurisdiccion": "Córdoba",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-003", "proveedor": "PantallaSur", "razon_social": "PantallaSur Deportes S.A.",
        "domicilio": "Ruta Ficticia 8 km 30, Pilar, provincia de Buenos Aires",
        "fecha_firma": date(2026, 1, 5), "fecha_inicio": date(2026, 1, 15),
        "contenido": "deportes",
        "detalle": "transmisiones en vivo de fútbol, básquet y tenis, resúmenes de cada fecha y programas de análisis deportivo",
        "vigencia_meses": 36, "renovacion_automatica": True,
        "moneda": "ARS", "precio": 2400.00, "minimo_suscriptores": 80_000,
        "ajuste": {"indice": "IPC", "frecuencia": "semestral", "tope": None},
        "plazo_pago_dias": 60, "exclusividad": True,
        "sla": 99.9, "penalizacion_sla_pct": 15,
        "preaviso_dias": 90, "penalidad_rescision_meses": 6, "jurisdiccion": "arbitraje",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-004", "proveedor": "OndaPlay", "razon_social": "OndaPlay Kids S.R.L.",
        "domicilio": "Pasaje Supuesto 77, Rosario, provincia de Santa Fe",
        "fecha_firma": date(2026, 3, 18), "fecha_inicio": date(2026, 4, 1),
        "contenido": "contenido infantil",
        "detalle": "dibujos animados, series educativas y películas para toda la familia, con controles parentales integrados",
        "vigencia_meses": 24, "renovacion_automatica": False,
        "moneda": "USD", "precio": 1.20, "minimo_suscriptores": 30_000,
        "ajuste": None,
        "plazo_pago_dias": 30, "exclusividad": False,
        "sla": 99.5, "penalizacion_sla_pct": 5,
        "preaviso_dias": 60, "penalidad_rescision_meses": 3, "jurisdiccion": "CABA",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-005", "proveedor": "Mirador TV", "razon_social": "Mirador TV S.A.",
        "domicilio": "Av. de los Ejemplos 2300, Mendoza, provincia de Mendoza",
        "fecha_firma": date(2026, 2, 2), "fecha_inicio": date(2026, 2, 15),
        "contenido": "documentales",
        "detalle": "documentales de naturaleza, historia, ciencia y sociedad, incluidas producciones originales de EL PROVEEDOR",
        "vigencia_meses": 12, "renovacion_automatica": True,
        "moneda": "ARS", "precio": 1500.00, "minimo_suscriptores": 15_000,
        "ajuste": None,
        "plazo_pago_dias": 45, "exclusividad": False,
        "sla": 99.5, "penalizacion_sla_pct": 8,
        "preaviso_dias": 30, "penalidad_rescision_meses": 2, "jurisdiccion": "Córdoba",
        "caso_trampa": "negacion_inflacion",
    },
    {
        "numero": "CN-2026-006", "proveedor": "Faro Media", "razon_social": "Faro Media Group S.A.",
        "domicilio": "Calle Hipotética 515, Ciudad Autónoma de Buenos Aires",
        "fecha_firma": date(2026, 4, 14), "fecha_inicio": date(2026, 5, 1),
        "contenido": "noticias",
        "detalle": "señales de noticias en vivo las 24 horas, programas periodísticos y contenido informativo a demanda",
        "vigencia_meses": 36, "renovacion_automatica": False,
        "moneda": "USD", "precio": 0.90, "minimo_suscriptores": 40_000,
        "ajuste": {"indice": "CPI", "frecuencia": "semestral", "tope": 4},
        "plazo_pago_dias": 60, "exclusividad": True,
        "sla": 99.9, "penalizacion_sla_pct": 12,
        "preaviso_dias": 90, "penalidad_rescision_meses": 6, "jurisdiccion": "arbitraje",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-007", "proveedor": "Lumen+", "razon_social": "Lumen Plus Entertainment S.R.L.",
        "domicilio": "Av. Simulada 990, La Plata, provincia de Buenos Aires",
        "fecha_firma": date(2026, 3, 2), "fecha_inicio": date(2026, 3, 15),
        "contenido": "series premium",
        "detalle": "series de alto presupuesto, estrenos simultáneos con su lanzamiento internacional y contenido en resolución 4K",
        "vigencia_meses": 24, "renovacion_automatica": True,
        "moneda": "USD", "precio": 2.80, "minimo_suscriptores": 25_000,
        "ajuste": None,
        "plazo_pago_dias": 30, "exclusividad": False,
        "sla": 99.9, "penalizacion_sla_pct": 10,
        "preaviso_dias": 60, "penalidad_rescision_meses": 3, "jurisdiccion": "CABA",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-008", "proveedor": "Kairos Play", "razon_social": "Kairos Play S.A.",
        "domicilio": "Boulevard Ficción 140, Córdoba, provincia de Córdoba",
        "fecha_firma": date(2026, 5, 19), "fecha_inicio": date(2026, 6, 1),
        "contenido": "anime",
        "detalle": "series y películas de animación japonesa, con subtítulos y doblaje al español",
        "vigencia_meses": 12, "renovacion_automatica": True,
        "moneda": "ARS", "precio": 1200.00, "minimo_suscriptores": 10_000,
        "ajuste": {"indice": "IPC", "frecuencia": "anual", "tope": 30},
        "plazo_pago_dias": 45, "exclusividad": False,
        "sla": 99.5, "penalizacion_sla_pct": 5,
        "preaviso_dias": 30, "penalidad_rescision_meses": 2, "jurisdiccion": "Córdoba",
        "caso_trampa": "ajuste_en_anexo",
    },
    {
        "numero": "CN-2026-009", "proveedor": "Atlas Streaming", "razon_social": "Atlas Streaming S.A.",
        "domicilio": "Calle Imaginada 3300, Ciudad Autónoma de Buenos Aires",
        "fecha_firma": date(2025, 12, 15), "fecha_inicio": date(2026, 1, 1),
        "contenido": "música y conciertos",
        "detalle": "conciertos en vivo, festivales, videoclips y documentales musicales",
        "vigencia_meses": 36, "renovacion_automatica": False,
        "moneda": "ARS", "precio": 900.00, "minimo_suscriptores": 35_000,
        "ajuste": None,
        "plazo_pago_dias": 45, "exclusividad": True,
        "sla": 99.5, "penalizacion_sla_pct": 7,
        "preaviso_dias": 90, "penalidad_rescision_meses": 6, "jurisdiccion": "arbitraje",
        "caso_trampa": None,
    },
    {
        "numero": "CN-2026-010", "proveedor": "Brisa TV", "razon_social": "Brisa TV Producciones S.R.L.",
        "domicilio": "Av. Costanera Ficticia 60, Mar del Plata, provincia de Buenos Aires",
        "fecha_firma": date(2026, 6, 17), "fecha_inicio": date(2026, 7, 1),
        "contenido": "telenovelas",
        "detalle": "telenovelas latinoamericanas, series dramáticas y programas de entretenimiento",
        "vigencia_meses": 24, "renovacion_automatica": True,
        "moneda": "ARS", "precio": 1650.00, "minimo_suscriptores": 20_000,
        "ajuste": {"indice": "IPC", "frecuencia": "trimestral", "tope": 10},
        "plazo_pago_dias": 60, "exclusividad": False,
        "sla": 99.5, "penalizacion_sla_pct": 6,
        "preaviso_dias": 30, "penalidad_rescision_meses": 3, "jurisdiccion": "CABA",
        "caso_trampa": None,
    },
]

# ---------------------------------------------------------------------------
# Ayudas de redacción
# ---------------------------------------------------------------------------
_NUMEROS = {
    2: "dos", 3: "tres", 6: "seis", 12: "doce", 24: "veinticuatro", 30: "treinta",
    36: "treinta y seis", 45: "cuarenta y cinco", 60: "sesenta", 90: "noventa",
}
_MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
_INDICES = {
    "IPC": "Índice de Precios al Consumidor (IPC) publicado por el Instituto "
           "Nacional de Estadística y Censos (INDEC)",
    "CPI": "Índice de Precios al Consumidor de los Estados Unidos de América "
           "(CPI, Consumer Price Index) publicado por la Oficina de Estadísticas "
           "Laborales de ese país",
}
INDICES_GT = {"IPC": "IPC del INDEC", "CPI": "CPI de EE. UU."}
_PERIODOS = {"trimestral": 3, "semestral": 6, "anual": 12}
_MONEDAS = {"USD": "dólares estadounidenses", "ARS": "pesos argentinos"}
_JURISDICCIONES = {
    "CABA": (
        "Para cualquier controversia derivada del presente "
        "contrato, LAS PARTES se someten a la jurisdicción de los tribunales "
        "ordinarios en lo comercial de la Ciudad Autónoma de Buenos Aires, con "
        "renuncia expresa a cualquier otro fuero o jurisdicción."
    ),
    "Córdoba": (
        "Para cualquier controversia derivada del presente "
        "contrato, LAS PARTES se someten a la jurisdicción de los tribunales "
        "ordinarios de la ciudad de Córdoba, provincia de Córdoba, con renuncia "
        "expresa a cualquier otro fuero o jurisdicción."
    ),
    "arbitraje": (
        "Toda controversia derivada del presente "
        "contrato será resuelta en forma definitiva mediante arbitraje de derecho, a "
        "cargo de un tribunal de tres (3) árbitros con sede en la Ciudad Autónoma de "
        "Buenos Aires. El laudo será definitivo e inapelable, y LAS PARTES renuncian "
        "a someter la controversia a los tribunales ordinarios."
    ),
}


def con_letras(n: int) -> str:
    """60 -> 'sesenta (60)', como se escribe en los contratos."""
    return f"{_NUMEROS[n]} ({n})"


def fecha_larga(d: date) -> str:
    return f"{d.day} de {_MESES[d.month - 1]} de {d.year}"


def numero_es(valor: float, decimales: int = 2) -> str:
    """Formato argentino: 1850.5 -> '1.850,50'."""
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def porcentaje_es(valor: float) -> str:
    """99.9 -> '99,9'; 10 -> '10'."""
    return numero_es(valor, 1).removesuffix(",0")


def archivo_de(c: dict) -> str:
    slug = c["proveedor"].lower().replace("+", " plus").replace(" ", "_")
    return f"contrato_{c['numero'][-2:]}_{slug}.md"


def texto_ajuste(ajuste: dict) -> str:
    meses = _PERIODOS[ajuste["frecuencia"]]
    if ajuste["tope"] is None:
        tope = "El ajuste no tendrá tope máximo."
    else:
        tope = (
            f"El ajuste de cada período no podrá superar un tope del "
            f"{porcentaje_es(ajuste['tope'])} %; el excedente no se trasladará a "
            "períodos posteriores."
        )
    return (
        f"El precio por suscriptor se ajustará en forma {ajuste['frecuencia']}, es "
        f"decir, cada {con_letras(meses)} meses, de acuerdo con la variación del "
        f"{_INDICES[ajuste['indice']]}, registrada en el período inmediatamente "
        f"anterior. {tope}"
    )


def clausula_ajuste(c: dict) -> str:
    if c["caso_trampa"] == "ajuste_en_anexo":
        # Trampa: aquí NO está el dato; el RAG tiene que encontrar el anexo.
        return (
            "Las condiciones de actualización del precio se rigen por lo establecido "
            "en el ANEXO I del presente contrato, que forma parte integrante de él."
        )
    if c["caso_trampa"] == "negacion_inflacion":
        # Trampa: el texto habla de inflación... para NEGAR el ajuste.
        return (
            "LAS PARTES declaran conocer el contexto inflacionario de la economía "
            "argentina y, aun así, acuerdan expresamente que el precio NO se ajustará "
            "por inflación ni por ningún índice durante toda la vigencia del contrato, "
            "incluidas sus eventuales renovaciones. El precio pactado en la cláusula 4 "
            "es fijo y cualquier modificación requerirá un nuevo acuerdo escrito."
        )
    if c["ajuste"] is None:
        return (
            "El precio pactado en la cláusula 4 es fijo durante toda la vigencia del "
            "contrato y no estará sujeto a actualizaciones. Cualquier modificación "
            "requerirá un acuerdo escrito firmado por ambas PARTES."
        )
    return texto_ajuste(c["ajuste"])


# ---------------------------------------------------------------------------
# Plantilla del contrato
# ---------------------------------------------------------------------------
def render_contrato(c: dict) -> str:
    moneda = c["moneda"]
    precio = numero_es(c["precio"])
    minimo = numero_es(c["minimo_suscriptores"], 0)

    if c["renovacion_automatica"]:
        renovacion = (
            "Al vencimiento del plazo de vigencia, el contrato se renovará "
            "automáticamente por períodos sucesivos de doce (12) meses, en las mismas "
            "condiciones, salvo que alguna de LAS PARTES comunique por escrito su "
            "decisión de no renovarlo con la anticipación prevista en la cláusula 9."
        )
    else:
        renovacion = (
            "El presente contrato NO se renovará automáticamente. Al finalizar la "
            "vigencia, cualquier prórroga deberá ser acordada por escrito y firmada por "
            "ambas PARTES; en caso contrario, el contrato se extinguirá de pleno "
            "derecho, sin necesidad de notificación previa."
        )

    if moneda == "USD":
        moneda_pago = (
            "Los pagos se realizarán en dólares estadounidenses o, a opción de EL "
            "CLIENTE, en pesos argentinos al tipo de cambio vendedor del día hábil "
            "anterior al pago."
        )
    else:
        moneda_pago = "Los pagos se realizarán en pesos argentinos."

    if c["exclusividad"]:
        exclusividad = (
            "EL PROVEEDOR otorga a EL CLIENTE exclusividad territorial en la República "
            "Argentina: durante la vigencia, EL PROVEEDOR no podrá licenciar el mismo "
            "catálogo a otros proveedores de internet o de televisión paga que operen "
            "en el país. El incumplimiento de esta obligación facultará a EL CLIENTE a "
            "rescindir el contrato con causa."
        )
    else:
        exclusividad = (
            "La licencia NO es exclusiva. EL PROVEEDOR podrá ofrecer el mismo catálogo "
            "a otros proveedores de internet o de televisión paga dentro del territorio "
            "de la República Argentina."
        )

    texto = f"""# CONTRATO DE LICENCIA DE CONTENIDOS AUDIOVISUALES

> **{LEYENDA}**

**Número de contrato:** {c["numero"]}
**Fecha de firma:** {fecha_larga(c["fecha_firma"])}
**Proveedor:** {c["proveedor"]}
**Cliente:** {CLIENTE}

## Partes

En la Ciudad Autónoma de Buenos Aires, el {fecha_larga(c["fecha_firma"])}, se celebra el presente contrato entre:

- **{CLIENTE}**, proveedor de internet con domicilio en Av. Ficticia 1234, Ciudad Autónoma de Buenos Aires, en adelante "EL CLIENTE"; y
- **{c["razon_social"]}**, titular del servicio de streaming "{c["proveedor"]}", con domicilio en {c["domicilio"]}, en adelante "EL PROVEEDOR".

En conjunto, "LAS PARTES", que acuerdan las siguientes cláusulas.

## Cláusulas

### CLÁUSULA 1. OBJETO

EL PROVEEDOR otorga a EL CLIENTE una licencia para incluir en sus paquetes de internet el servicio de streaming de {c["contenido"]} de EL PROVEEDOR, que comprende {c["detalle"]}. EL PROVEEDOR es el único responsable por la titularidad de los derechos del contenido.

### CLÁUSULA 2. VIGENCIA

El presente contrato tendrá una vigencia de {con_letras(c["vigencia_meses"])} meses, contados a partir del {fecha_larga(c["fecha_inicio"])}.

### CLÁUSULA 3. RENOVACIÓN

{renovacion}

### CLÁUSULA 4. PRECIO

Como contraprestación por la licencia, EL CLIENTE abonará a EL PROVEEDOR la suma de {moneda} {precio} ({_MONEDAS[moneda]}) por cada suscriptor activo y por mes. EL CLIENTE garantiza un mínimo de {minimo} suscriptores activos por mes: si la cantidad real fuera inferior, abonará igualmente el precio correspondiente a ese mínimo garantizado. Los precios no incluyen el Impuesto al Valor Agregado (IVA), que se facturará por separado.

### CLÁUSULA 5. AJUSTE DE PRECIO

{clausula_ajuste(c)}

### CLÁUSULA 6. FORMA Y PLAZO DE PAGO

EL PROVEEDOR facturará mensualmente y EL CLIENTE pagará cada factura dentro de los {con_letras(c["plazo_pago_dias"])} días corridos desde su recepción, mediante transferencia bancaria a la cuenta que EL PROVEEDOR indique por escrito. {moneda_pago} La falta de pago en término producirá la mora automática.

### CLÁUSULA 7. EXCLUSIVIDAD TERRITORIAL

{exclusividad}

### CLÁUSULA 8. NIVEL DE SERVICIO (SLA)

EL PROVEEDOR garantiza una disponibilidad mensual del servicio de streaming del {porcentaje_es(c["sla"])} %, excluidas las ventanas de mantenimiento programado. Si la disponibilidad de un mes fuera inferior a ese nivel, EL PROVEEDOR otorgará a EL CLIENTE una bonificación del {porcentaje_es(c["penalizacion_sla_pct"])} % sobre la factura de ese mes, que se descontará de la facturación siguiente.

### CLÁUSULA 9. RESCISIÓN

Cualquiera de LAS PARTES podrá rescindir el presente contrato sin expresión de causa, notificando su decisión a la otra parte en forma fehaciente con un preaviso mínimo de {con_letras(c["preaviso_dias"])} días. Si EL CLIENTE rescindiera antes de cumplirse la mitad de la vigencia, deberá abonar a EL PROVEEDOR, en concepto de penalidad por rescisión anticipada, el equivalente a {con_letras(c["penalidad_rescision_meses"])} meses de facturación calculada sobre el mínimo garantizado de suscriptores.

### CLÁUSULA 10. PROTECCIÓN DE DATOS PERSONALES

LAS PARTES se obligan a cumplir la Ley N.º 25.326 de Protección de los Datos Personales y su normativa complementaria. EL PROVEEDOR tratará los datos de los suscriptores solo para prestar el servicio. EL PROVEEDOR no podrá ceder ni vender esos datos y notificará cualquier incidente de seguridad dentro de las setenta y dos (72) horas.

### CLÁUSULA 11. JURISDICCIÓN

{_JURISDICCIONES[c["jurisdiccion"]]}

En prueba de conformidad, se firman dos (2) ejemplares de un mismo tenor.

**Firmas (ficticias):** por EL CLIENTE, {CLIENTE}; por EL PROVEEDOR, {c["razon_social"]}.
"""

    if c["caso_trampa"] == "ajuste_en_anexo":
        texto += f"""
## ANEXO I. ACTUALIZACIÓN DEL PRECIO

Conforme a lo previsto en la cláusula 5, LAS PARTES acuerdan el siguiente mecanismo de actualización del precio por suscriptor fijado en la cláusula 4. {texto_ajuste(c["ajuste"])} Para el cálculo se tomará el último índice publicado a la fecha de cada ajuste.
"""
    return texto


def ground_truth_de(c: dict) -> dict:
    """Valores correctos del contrato, para verificar las respuestas del RAG."""
    ajuste = c["ajuste"]
    return {
        "proveedor": c["proveedor"],
        "archivo": archivo_de(c),
        "numero_contrato": c["numero"],
        "tiene_ajuste_inflacion": ajuste is not None,
        "indice": INDICES_GT[ajuste["indice"]] if ajuste else None,
        "frecuencia": ajuste["frecuencia"] if ajuste else None,
        "tope": ajuste["tope"] if ajuste else None,
        "moneda": c["moneda"],
        "precio": c["precio"],
        "vigencia_meses": c["vigencia_meses"],
        "renovacion_automatica": c["renovacion_automatica"],
        "preaviso_dias": c["preaviso_dias"],
        "exclusividad": c["exclusividad"],
        "sla": c["sla"],
        "penalizacion_sla_pct": c["penalizacion_sla_pct"],
        "plazo_pago_dias": c["plazo_pago_dias"],
        "jurisdiccion": c["jurisdiccion"],
        "caso_trampa": c["caso_trampa"],
    }


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    config.EVAL_DIR.mkdir(exist_ok=True)
    # Mismo tokenizador con el que ingest.py mide los chunks (cl100k_base).
    encoding = tiktoken.get_encoding("cl100k_base")

    for c in CONTRATOS:
        texto = render_contrato(c)
        ruta = config.DATA_DIR / archivo_de(c)
        ruta.write_text(texto, encoding="utf-8")

        tokens = len(encoding.encode(texto))
        if MIN_TOKENS <= tokens <= MAX_TOKENS:
            logger.info("%-32s %5d tokens", ruta.name, tokens)
        else:
            logger.warning(
                "%-32s %5d tokens (fuera del rango %d-%d)", ruta.name, tokens, MIN_TOKENS, MAX_TOKENS
            )

    ground_truth = [ground_truth_de(c) for c in CONTRATOS]
    config.GROUND_TRUTH_PATH.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Generados %d contratos en %s y el ground truth en %s",
        len(CONTRATOS), config.DATA_DIR.name, config.GROUND_TRUTH_PATH.relative_to(config.BASE_DIR),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
