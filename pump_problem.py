from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

# Constantes del paper para generación aleatoria realista
DEFAULT_PRICE_LEVELS = (169.0, 283.0, 336.0)  # Tabla 2 [cite: 92]


@dataclass(slots=True)
class WaterPumpProblem:
    """
    Instancia del problema de programación de bombas.
    [cite_start]Basado en el modelo de Sustainability 13, 3470[cite: 8].
    """
    problem_id: str

    # --- Dimensiones ---
    num_pumps: int
    horizon: int

    # --- Parámetros de Bombas ---
    pump_power: list[float]  # P_j (kW) [cite: 76]
    pump_flows: list[float]  # b_j (m³/h) [cite: 76]

    # --- Parámetros Temporales ---

    electricity_prices: list[float]  # p_i (PLN/MWh) [cite: 92]

    demand: list[float]  # d_i (m³) [cite: 100]

    # --- Parámetros del Tanque ---

    v_init: float  # [cite: 119]

    v_min: float  # [cite: 87]

    v_max: float  # [cite: 74]

    # --- Restricciones Operativas ---

    max_active_pumps: int = 6  # Eq(1) [cite: 162]

    min_pump_usage: float = 1.0  # Eq(2) [cite: 165]
    time_step_hours: float = 1.0
    power_limits: list[float] | None = None  # L_i (kW) opcional [cite: 173]

    def __post_init__(self) -> None:
        """Validación estricta de dimensiones y coherencia."""
        if len(self.pump_power) != self.num_pumps:
            raise ValueError("Longitud incorrecta en pump_power")
        if len(self.pump_flows) != self.num_pumps:
            raise ValueError("Longitud incorrecta en pump_flows")
        if len(self.electricity_prices) != self.horizon:
            raise ValueError("Longitud incorrecta en electricity_prices")
        if len(self.demand) != self.horizon:
            raise ValueError("Longitud incorrecta en demand")

        if not (0 <= self.v_min < self.v_max):
            raise ValueError(f"v_min ({self.v_min}) debe ser menor que v_max ({self.v_max})")
        if not (self.v_min <= self.v_init <= self.v_max):
            raise ValueError(f"v_init fuera de rango: {self.v_init}")

    @property
    def total_max_flow(self) -> float:
        """Caudal máximo teórico (N mejores bombas)."""
        best_pumps = sorted(self.pump_flows, reverse=True)[:self.max_active_pumps]
        return sum(best_pumps)

    def is_feasible_basic(self) -> bool:
        """Chequeo rápido de viabilidad física."""
        total_demand = sum(self.demand)
        max_production = self.total_max_flow * self.time_step_hours * self.horizon

        if max_production < total_demand:
            return False

        required_slots = self.min_pump_usage * self.num_pumps
        total_slots = self.horizon * self.max_active_pumps
        if required_slots > total_slots:
            return False

        return True

    # --- IO ---
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaterPumpProblem:
        return cls(**data)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> WaterPumpProblem:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --- Generadores ---

def make_reference_instance(problem_id: str = "kowalik_2021_reference") -> WaterPumpProblem:
    """Instancia exacta del paper."""
    return WaterPumpProblem(
        problem_id=problem_id,
        num_pumps=7,
        horizon=24,
        pump_flows=[75.0, 133.0, 157.0, 176.0, 59.0, 69.0, 120.0],
        pump_power=[15.0, 37.0, 33.0, 33.0, 22.0, 33.0, 22.0],
        electricity_prices=[
            283.0 if 7 <= h < 13 else 336.0 if 16 <= h < 21 else 169.0
            for h in range(24)
        ],
        demand=[
            44.62, 31.27, 26.22, 27.51, 31.50, 46.18, 69.47, 100.36,
            131.85, 148.51, 149.89, 142.21, 132.09, 129.29, 124.06, 114.68,
            109.33, 115.76, 126.95, 131.48, 138.86, 131.91, 111.53, 70.43
        ],
        v_init=550.0, v_min=523.5, v_max=1500.0,
        max_active_pumps=6, min_pump_usage=1.0
    )


def generate_random_instance(
        problem_id: str, num_pumps: int = 7, horizon: int = 24, rng: random.Random | None = None
) -> WaterPumpProblem:
    rng = rng or random.Random()

    # Lógica simplificada para ejemplo
    pump_power = [round(rng.uniform(15, 40), 1) for _ in range(num_pumps)]
    pump_flows = [round(p * rng.uniform(2.5, 5.5), 1) for p in pump_power]

    prices = [DEFAULT_PRICE_LEVELS[0]] * horizon
    # Añadir picos simples
    for h in range(horizon):
        if 7 <= h < 13:
            prices[h] = DEFAULT_PRICE_LEVELS[1]
        elif 16 <= h < 21:
            prices[h] = DEFAULT_PRICE_LEVELS[2]

    max_flow = sum(sorted(pump_flows, reverse=True)[:min(num_pumps, 6)])
    avg_demand = max_flow * rng.uniform(0.4, 0.6)
    demand = [round(avg_demand * rng.uniform(0.7, 1.3), 2) for _ in range(horizon)]

    v_init = round(sum(demand) * 0.25, 1)

    instance = WaterPumpProblem(
        problem_id=problem_id, num_pumps=num_pumps, horizon=horizon,
        pump_power=pump_power, pump_flows=pump_flows, electricity_prices=prices,
        demand=demand, v_init=v_init, v_min=v_init * 0.9, v_max=v_init * 3.5
    )

    if not instance.is_feasible_basic():
        return generate_random_instance(problem_id, num_pumps, horizon, rng)
    return instance