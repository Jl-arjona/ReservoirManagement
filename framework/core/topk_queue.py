# framework/core/topk_queue.py
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple

from .types import ExperimentResult


@dataclass
class TopKQueue:
    """
    Cola de prioridad acotada que mantiene las K mejores soluciones
    según 'utility' (cuanto más alta, mejor).
    """
    k: int
    _heap: List[Tuple[float, ExperimentResult]] = field(default_factory=list)

    def push(self, result: ExperimentResult) -> None:
        """Inserta un resultado en la cola si tiene utilidad."""
        if result.utility is None:
            return

        entry = (result.utility, result)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        else:
            # Montón mínimo: el peor está en _heap[0]
            if entry[0] > self._heap[0][0]:
                heapq.heapreplace(self._heap, entry)

    def best_first(self) -> List[ExperimentResult]:
        """
        Devuelve la lista de resultados, ordenada de mayor a menor utilidad.
        """
        return [r for _, r in sorted(self._heap, key=lambda e: e[0], reverse=True)]
