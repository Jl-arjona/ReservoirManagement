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

    Comportamiento de la función 'key':
      - Debe retornar un `float` representando la utilidad (mayor es mejor).
      - Debe retornar `None` si el elemento es inválido o infactible.
        En tal caso, el elemento será DESCARTADO silenciosamente.
    """
    k: int

    # Función para extraer la utilidad del item.
    # Por defecto intenta leer el atributo .utility (compatible con ExperimentResult).
    # Contrato: Retornar None implica que el elemento se ignora.
    key: Callable[[Any], float | None] = lambda x: getattr(x, "utility", None)

    # Estructuras internas (Min-Heap para mantener el Top-K)
    _heap: List[Tuple[float, int, Any]] = field(default_factory=list)
    _counter: itertools.count = field(default_factory=itertools.count)

    def push(self, item: Any) -> None:
        """
        Intenta insertar un item en la cola.

        Si self.key(item) retorna None (indicando infactibilidad),
        el item se ignora inmediatamente.
        """
        u = self.key(item)

        # Contrato explícito: None significa "ignorar / infactible"
        if u is None:
            return

        # Desempate usando contador para evitar comparar objetos complejos
        # entre sí cuando tienen la misma utilidad.
        count = next(self._counter)
        entry = (u, count, item)

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        else:
            # En un min-heap, el elemento [0] es el peor de los K mejores.
            # Si nuestra utilidad es mayor que el peor actual, lo reemplazamos.
            if u > self._heap[0][0]:
                heapq.heapreplace(self._heap, entry)

    def best_first(self) -> List[Any]:
        """Devuelve los items ordenados de MAYOR a MENOR utilidad."""
        # El heap guarda tuplas (utilidad, contador, item)
        # Ordenamos descendente por utilidad (e[0])
        return [item for _, _, item in sorted(self._heap, key=lambda e: e[0], reverse=True)]