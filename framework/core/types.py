# framework/core/types.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, Optional


class RunStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"


@dataclass
class ExperimentResult:
    """
    Resultado estándar de la ejecución de un algoritmo sobre un problema.
    """
    problem_id: str
    algo_name: str
    status: RunStatus

    start_time: str
    end_time: str
    wall_time_sec: float

    # Métricas de la solución
    solution: Optional[Dict[str, int]] = None          # asignación de variables
    energy: Optional[float] = None                     # energía del BQM (si aplica)
    utility: Optional[float] = None                    # utilidad (p.ej. -coste)
    cost: Optional[float] = None                       # coste real del dominio

    # Metadatos variados
    meta: Dict[str, Any] = field(default_factory=dict)

    # Info de error (si aplica)
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializable a JSON fácilmente."""
        return asdict(self)
