# problems/reservoir/model.py
from __future__ import annotations
# problems/reservoir/model.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Any
import json
import random


@dataclass
class ReservoirInstance:
    """Instancia de problema de embalse.

    Esta estructura debe ser lo suficientemente genérica como para:
    - guardarse en JSON
    - reconstruirse fácilmente
    - usarse por tus algoritmos (brute-force, QUBO, etc.)
    """
    problem_id: str

    num_pumps: int
    horizon: int  # número de slots de tiempo

    power: List[float]
    costs: List[float]
    flow: List[float]
    demand: List[float]

    v_init: float
    v_min: float
    v_max: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "num_pumps": self.num_pumps,
            "horizon": self.horizon,
            "power": self.power,
            "costs": self.costs,
            "flow": self.flow,
            "demand": self.demand,
            "v_init": self.v_init,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReservoirInstance":
        return cls(
            problem_id=data["problem_id"],
            num_pumps=int(data["num_pumps"]),
            horizon=int(data["horizon"]),
            power=list(map(float, data["power"])),
            costs=list(map(float, data["costs"])),
            flow=list(map(float, data["flow"])),
            demand=list(map(float, data["demand"])),
            v_init=float(data["v_init"]),
            v_min=float(data["v_min"]),
            v_max=float(data["v_max"]),
        )

    @property
    def time_indices(self) -> List[int]:
        return list(range(self.horizon))


# -----------------------------
#  IO helpers
# -----------------------------


def save_instance(instance: ReservoirInstance, path: Path) -> None:
    """Guarda una instancia en JSON (crea directorios si no existen)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(instance.to_dict(), f, indent=2)


def load_instance(path: Path) -> ReservoirInstance:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ReservoirInstance.from_dict(data)


def list_instance_files(directory: Path) -> List[Path]:
    """Lista todos los JSON de instancia en un directorio."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.json") if p.is_file())


# -----------------------------
#  Generación aleatoria
# -----------------------------


def generate_random_instance(
    problem_id: str,
    num_pumps: int,
    horizon: int,
    power_range: Sequence[float] = (10.0, 40.0),
    flow_range: Sequence[float] = (50.0, 200.0),
    cost_range: Sequence[float] = (100.0, 400.0),
    demand_range: Sequence[float] = (30.0, 160.0),
    v_init_range: Sequence[float] = (500.0, 800.0),
    v_min_margin: float = 0.9,
    v_max_factor: float = 3.0,
    rng: random.Random | None = None,
) -> ReservoirInstance:
    """Genera una instancia aleatoria 'razonable' de embalse.

    Todos los rangos son parametrizables. La idea es que las magnitudes sean
    del mismo orden y no haya locuras de escalas.
    """
    rng = rng or random.Random()

    power = [rng.uniform(*power_range) for _ in range(num_pumps)]
    flow = [rng.uniform(*flow_range) for _ in range(num_pumps)]
    costs = [rng.uniform(*cost_range) for _ in range(horizon)]

    demand = [rng.uniform(*demand_range) for _ in range(horizon)]

    # Nivel inicial y límites
    v_init = rng.uniform(*v_init_range)

    # Volumen mínimo un poco por debajo de v_init
    v_min = v_init * v_min_margin

    # Volumen máximo proporcional a v_init (más holgura)
    v_max = v_init * v_max_factor

    return ReservoirInstance(
        problem_id=problem_id,
        num_pumps=num_pumps,
        horizon=horizon,
        power=power,
        costs=costs,
        flow=flow,
        demand=demand,
        v_init=v_init,
        v_min=v_min,
        v_max=v_max,
    )
