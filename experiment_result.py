from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Any


@dataclass(slots=True)
class ExperimentResult:
    """
    Representa el resultado de una ejecución (sea exitosa o no).
    """
    # --- Datos Matemáticos de la Solución (Pueden ser vacíos si falla) ---
    schedule: List[List[int]] | None  # Matriz TxN
    total_cost: float | None
    volumes: List[float] | None

    # --- Metadatos del Experimento ---
    problem_id: str = field(default="")
    algorithm: str = field(default="")
    execution_time: float = field(default=0.0)

    # Valores posibles: "Success", "Infeasible", "Exception", "Timeout"
    status: str = field(default="Unknown")

    @property
    def utility(self) -> float | None:
        """
        Retorna la métrica de calidad para la TopKQueue.
        Utility = -1 * Total Cost.
        """
        if self.total_cost is None:
            return None
        return -self.total_cost

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resultado a diccionario para guardar en JSON."""
        return asdict(self)