"""Contrato de datos del RAG.

`PydanticOutputParser` convierte estos modelos en instrucciones de formato
para el LLM, así que las `description` también funcionan como parte del prompt:
cuanto más precisas, mejor rellena el modelo cada campo.
"""

import logging
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)

MAX_EXTRACTO = 200


class Referencia(BaseModel):
    """Fragmento de un contrato que respalda la respuesta."""

    archivo: str = Field(
        description=(
            "Nombre del archivo del contrato, copiado tal cual del encabezado "
            "[Fuente: ...] del contexto. Ejemplo: contrato_04_ondaplay.md"
        )
    )
    proveedor: str = Field(
        description="Proveedor del contrato, copiado del encabezado [Proveedor: ...]."
    )
    extracto: str = Field(
        max_length=MAX_EXTRACTO,
        description=(
            f"Cita textual y breve (hasta {MAX_EXTRACTO} caracteres) del "
            "fragmento del contexto que contiene el dato."
        ),
    )

    @field_validator("extracto", mode="before")
    @classmethod
    def recortar_extracto(cls, valor: str) -> str:
        # Red de seguridad: si el modelo copia una cita más larga, la recortamos
        # en vez de rechazar toda la respuesta (y gastar un reintento) por un
        # detalle de longitud. mode="before" corre ANTES de validar max_length.
        if isinstance(valor, str) and len(valor) > MAX_EXTRACTO:
            logger.info("Extracto recortado de %d a %d caracteres.", len(valor), MAX_EXTRACTO)
            return valor[: MAX_EXTRACTO - 1].rstrip() + "…"
        return valor


class RAGResponse(BaseModel):
    """Respuesta del asistente, siempre respaldada por el contexto recuperado."""

    respuesta: str = Field(
        description=(
            "Respuesta en español basada SOLO en el contexto. Si la información "
            "no está en el contexto, exactamente: No lo sé"
        )
    )
    encontrado: bool = Field(
        description=(
            "true si la respuesta se apoya en el contexto; false si la "
            "información no está en el contexto."
        )
    )
    referencias: list[Referencia] = Field(
        default_factory=list,
        description=(
            "Fragmentos del contexto usados para responder, uno por dato citado. "
            "Obligatoria (al menos una) si encontrado es true; vacía si es false."
        ),
    )
    # Lo completa el SISTEMA, nunca el modelo: separa un fallo técnico (API caída,
    # base sin crear...) de un "No lo sé" legítimo. SkipJsonSchema lo deja fuera
    # del JSON Schema, así no aparece en las instrucciones de formato del LLM.
    error: SkipJsonSchema[str | None] = Field(
        default=None,
        description="Mensaje del error técnico, si lo hubo. None si la consulta se procesó.",
    )

    @model_validator(mode="after")
    def coherencia_referencias(self) -> Self:
        # La regla vive en el esquema (y no solo en rag.py): cualquier
        # RAGResponse, venga del LLM o del código, tiene que cumplirla. Si el
        # LLM la rompe, el parser lanza OutputParserException y se reintenta.
        if self.encontrado and not self.referencias:
            raise ValueError("encontrado=true exige al menos una referencia.")
        if not self.encontrado and self.referencias:
            raise ValueError("encontrado=false no admite referencias.")
        if self.error is not None and self.encontrado:
            raise ValueError("Una respuesta con error no puede tener encontrado=true.")
        return self
