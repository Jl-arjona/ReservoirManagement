from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

# Constantes del paper para generación aleatoria realista
# Precios de la Tabla 2 del artículo [cite: 92]
DEFAULT_PRICE_LEVELS = (169.0, 283.0, 336.0)


@dataclass(slots=True)
class WaterPumpProblem:
    """
    Instancia del problema de programación de bombas.
    Basado en el modelo de Sustainability 13, 3470[cite: 8].
    """
    problem_id: str

    # --- Dimensiones ---
    num_pumps: int
    horizon: int

    # --- Parámetros de Bombas ---
    pump_power: list[float]  # P_j (kW)
    pump_flows: list[float]  # b_j (m³/h)

    # --- Parámetros Temporales ---
    electricity_prices: list[float]  # p_i (PLN/MWh)
    demand: list[float]  # d_i (m³)

    # --- Parámetros del Tanque ---
    v_init: float
    v_min: float
    v_max: float

    # --- Restricciones Operativas ---

    max_active_pumps: int = 6  # Restricción Eq(1) [cite: 150]

    min_pump_usage: float = 1.0  # Restricción Eq(2) [cite: 151]
    time_step_hours: float = 1.0
    power_limits: list[float] | None = None  # L_i (kW) opcional [cite: 124, 155]

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

        if self.power_limits is not None and len(self.power_limits) != self.horizon:
            raise ValueError("Longitud incorrecta en power_limits")

        if not (0 <= self.v_min < self.v_max):
            raise ValueError(f"v_min ({self.v_min}) debe ser menor que v_max ({self.v_max})")
        if not (self.v_min <= self.v_init <= self.v_max):
            raise ValueError(f"v_init fuera de rango: {self.v_init}")

    @property
    def cost_matrix(self) -> list[list[float]]:
        """
        Matriz de costes operativos C_ij (PLN).
        Calculado como: p_i * P_j / 1000 * time_step
        Referencia: Ecuación para c_ij[cite: 111].
        """
        factor = self.time_step_hours / 1000.0
        return [
            [
                price * power * factor
                for power in self.pump_power
            ]
            for price in self.electricity_prices
        ]

    @property
    def total_max_flow(self) -> float:
        """Caudal máximo teórico (usando las N mejores bombas)."""
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

    def simulate_volumes(self, schedule: Iterable[Iterable[float]]) -> list[float]:
        """
        Simula la evolución del tanque dado un horario de bombeo.
        Necesario para validar soluciones externas (ej. QUBO).

        Args:
            schedule: Matriz (T x J) donde schedule[i][j] es 1 (on) o 0 (off).
        Returns:
            Lista con el volumen del tanque al final de cada periodo.
        """
        volumes = []
        current_v = self.v_init

        for t, pump_status_t in enumerate(schedule):
            # Agua bombeada en este periodo (suma de b_j * estado_j)
            inflow = sum(
                flow * status * self.time_step_hours
                for flow, status in zip(self.pump_flows, pump_status_t)
            )

            # Balance de masas: V_t = V_{t-1} + Entrada - Salida
            current_v = current_v + inflow - self.demand[t]
            volumes.append(current_v)

        return volumes

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
    """
    Crea la instancia exacta del paper Sustainability 13, 03470.
    Útil para validar el solver contra resultados conocidos (aprox 82 PLN).
    """
    # Datos Tabla 1 [cite: 76]
    pump_flows = [75.0, 133.0, 157.0, 176.0, 59.0, 69.0, 120.0]
    pump_power = [15.0, 37.0, 33.0, 33.0, 22.0, 33.0, 22.0]

    # Datos Tabla 3 (Demanda) [cite: 100]
    demand = [
        44.62, 31.27, 26.22, 27.51, 31.50, 46.18, 69.47, 100.36,
        131.85, 148.51, 149.89, 142.21, 132.09, 129.29, 124.06, 114.68,
        109.33, 115.76, 126.95, 131.48, 138.86, 131.91, 111.53, 70.43
    ]

    # Datos Tabla 2 (Precios por franja horaria) [cite: 92]
    prices = []
    for h in range(24):
        if 7 <= h < 13:  # 07:00 a 12:59
            prices.append(283.0)
        elif 16 <= h < 21:  # 16:00 a 20:59
            prices.append(336.0)
        else:  # Resto (Valle)
            prices.append(169.0)

    return WaterPumpProblem(
        problem_id=problem_id,
        num_pumps=7,
        horizon=24,
        pump_power=pump_power,
        pump_flows=pump_flows,
        electricity_prices=prices,
        demand=demand,
        v_init = 550.0,  # [cite: 120]
    
        v_min = 523.5,  # [cite: 87]
    
        v_max = 1500.0,  # [cite: 74]
    
        max_active_pumps = 6,  # [cite: 150]
    
        min_pump_usage = 1.0  # [cite: 84]
    )

    def generate_random_instance(
            problem_id: str,
            num_pumps: int = 7,
            horizon: int = 24,
            max_active_pumps: int | None = None,
            max_retries: int = 100,
            rng: random.Random | None = None
    ) -> WaterPumpProblem:
        """
        Genera una instancia aleatoria factible.
        Usa un bucle de reintento para garantizar viabilidad física.
        """
        rng = rng or random.Random()

        # Si no se especifica, usamos la regla del paper (N-1 o tope 6)
        if max_active_pumps is None:
            max_active_pumps = min(num_pumps, 6)

        for attempt in range(max_retries):
            # 1. Bombas
            pump_power = []
            pump_flows = []
            for _ in range(num_pumps):
                p = round(rng.uniform(15.0, 40.0), 1)
                efficiency = rng.uniform(2.5, 5.5)
                f = round(p * efficiency, 1)
                pump_power.append(p)
                pump_flows.append(f)

            # 2. Precios
            prices = [0.0] * horizon
            peak_start_morning = rng.randint(6, 8)
            peak_end_morning = peak_start_morning + rng.randint(4, 6)
            peak_start_evening = rng.randint(16, 18)
            peak_end_evening = peak_start_evening + rng.randint(3, 5)

            for h in range(horizon):
                if peak_start_morning <= h < peak_end_morning:
                    prices[h] = DEFAULT_PRICE_LEVELS[1]
                elif peak_start_evening <= h < peak_end_evening:
                    prices[h] = DEFAULT_PRICE_LEVELS[2]
                else:
                    prices[h] = DEFAULT_PRICE_LEVELS[0]

            # 3. Demanda y Tanque
            # Calcular capacidad basado en el límite REAL de bombas activas
            real_max_flow = sum(sorted(pump_flows, reverse=True)[:max_active_pumps])

            # Generamos demanda promedio al 40-65% de esa capacidad real
            avg_demand = real_max_flow * rng.uniform(0.4, 0.65)

            demand = []
            for _ in range(horizon):
                d = avg_demand * rng.uniform(0.7, 1.3)
                demand.append(round(d, 2))

            total_demand = sum(demand)
            v_init = round(total_demand * 0.25, 1)

            instance = WaterPumpProblem(
                problem_id=problem_id,
                num_pumps=num_pumps,
                horizon=horizon,
                pump_power=pump_power,
                pump_flows=pump_flows,
                electricity_prices=prices,
                demand=demand,
                v_init=v_init,
                v_min=round(v_init * 0.9, 1),
                v_max=round(v_init * 3.5, 1),
                max_active_pumps=max_active_pumps  # IMPORTANTE: Pasar la restricción real
            )

            if instance.is_feasible_basic():
                return instance

        raise RuntimeError(
            f"No se pudo generar instancia factible para {problem_id} "
            f"con {max_active_pumps} bombas activas tras {max_retries} intentos."
        )