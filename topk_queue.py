from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Callable

@dataclass
class TopKQueue:
    """
    Cola de prioridad acotada genérica.
    Mantiene los K mejores elementos según una función de puntuación (utility).
    """
    k: int

    # Función para extraer la utilidad. Por defecto busca el atributo .utility
    key: Callable[[Any], float | None] = lambda x: getattr(x, "utility", None)

    # Estructuras internas (no modificar manualmente)
    _heap: List[Tuple[float, int, Any]] = field(default_factory=list)
    _counter: itertools.count = field(default_factory=itertools.count)

    def push(self, item: Any) -> None:
        """
        Inserta un item si su utilidad es válida y mejora lo existente.
        """
        u = self.key(item)

        # Si la utilidad es None (ej. solución infactible), ignoramos
        if u is None:
            return

        # Desempate usando contador para evitar comparar objetos complejos
        count = next(self._counter)
        entry = (u, count, item)

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        else:
            # En min-heap, [0] es el peor de los K mejores.
            # Si nuestra utilidad es mayor, reemplazamos al peor.
            if u > self._heap[0][0]:
                heapq.heapreplace(self._heap, entry)

    def best_first(self) -> List[Any]:
        """Devuelve los items ordenados de MAYOR a MENOR utilidad."""
        return [item for _, _, item in sorted(self._heap, key=lambda e: e[0], reverse=True)]