from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass(slots=True)
class ExperimentResult:
    """
    Representa una solución candidata encontrada por el algoritmo.
    Contiene el horario, coste y validaciones de restricciones.
    """
    schedule: List[List[int]]  # Matriz TxN (0 o 1)
    total_cost: float
    final_volume_ok: bool
    min_usage_ok: bool
    volumes: List[float]       # Evolución del volumen del tanque

    @property
    def is_feasible(self) -> bool:
        """Devuelve True si cumple todas las restricciones obligatorias."""
        return self.final_volume_ok and self.min_usage_ok

    @property
    def utility(self) -> float | None:
        """
        Utilidad para la cola de prioridad.
        Queremos MINIMIZAR coste, por lo que la utilidad es el negativo del coste.
        Retorna None si la solución es infactible.
        """
        if not self.is_feasible:
            return None
        return -self.total_cost