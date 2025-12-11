from __future__ import annotations

import itertools
from typing import List
from pathlib import Path

# Imports corregidos de módulos locales
from pump_problem import WaterPumpProblem
from topk_queue import TopKQueue
from experiment_result import ExperimentResult


class BruteForceSolver:
    def __init__(self, problem: WaterPumpProblem):
        self.problem = problem
        # [cite_start]Precalcular estados: Filtra Eq(1) (máximo bombas activas) [cite: 162]
        self.possible_configs = self._precompute_configurations()

    def _precompute_configurations(self) -> List[dict]:
        """Genera combinaciones válidas [0,1]^N respetando max_active_pumps."""
        configs = []
        for combination in itertools.product([0, 1], repeat=self.problem.num_pumps):
            if sum(combination) > self.problem.max_active_pumps:
                continue

            flow = sum(b * f for b, f in zip(combination, self.problem.pump_flows))
            power = sum(b * p for b, p in zip(combination, self.problem.pump_power))

            configs.append({
                "mask": combination,
                "flow": flow,
                "power_kw": power
            })
        return configs

    def solve(self, top_k: int = 10) -> List[ExperimentResult]:
        queue = TopKQueue(k=top_k)

        # Iniciamos Backtracking
        # Usamos listas mutables para usage_counts, schedule_acc y volumes_acc
        # para evitar copias masivas de memoria.
        self._dfs(
            t=0,
            current_vol=self.problem.v_init,
            current_cost=0.0,
            usage_counts=[0] * self.problem.num_pumps,
            schedule_acc=[],
            volumes_acc=[],
            queue=queue
        )
        return queue.best_first()

    def _dfs(self,
             t: int,
             current_vol: float,
             current_cost: float,
             usage_counts: List[int],
             schedule_acc: List[List[int]],
             volumes_acc: List[float],
             queue: TopKQueue):

        # --- CASO BASE: Fin del horizonte ---
        if t == self.problem.horizon:
            # [cite_start]Validar Eq(2): Uso mínimo [cite: 165]
            if all(u >= self.problem.min_pump_usage for u in usage_counts):
                # Copia profunda necesaria porque las listas se reciclan en backtracking
                res = ExperimentResult(
                    schedule=[list(s) for s in schedule_acc],
                    total_cost=current_cost,
                    final_volume_ok=True,
                    min_usage_ok=True,
                    volumes=list(volumes_acc)
                )
                queue.push(res)
            return

        # --- RECURSIÓN ---
        # Cacheo de variables locales para rendimiento
        price = self.problem.electricity_prices[t]
        demand = self.problem.demand[t]
        dt = self.problem.time_step_hours
        v_min, v_max = self.problem.v_min, self.problem.v_max
        p_limit = self.problem.power_limits[t] if self.problem.power_limits else None

        TOL = 1e-5

        for config in self.possible_configs:
            # [cite_start]Eq(6) Opcional: Límite de potencia [cite: 173]
            if p_limit is not None and config["power_kw"] > p_limit:
                continue

            # [cite_start]Eq(3) Balance de masas [cite: 167]
            new_vol = current_vol + (config["flow"] * dt) - demand

            # [cite_start]Eq(4) Restricciones de tanque [cite: 169]
            if not (v_min - TOL <= new_vol <= v_max + TOL):
                continue

            # --- AVANZAR (Modificar estado) ---
            step_cost = (price * config["power_kw"] * dt) / 1000.0

            # Actualizar contadores in-place
            for i, on in enumerate(config["mask"]):
                if on: usage_counts[i] += 1

            schedule_acc.append(config["mask"])
            volumes_acc.append(new_vol)

            # --- RECURSIÓN ---
            self._dfs(t + 1, new_vol, current_cost + step_cost, usage_counts,
                      schedule_acc, volumes_acc, queue)

            # --- RETROCEDER (Restaurar estado) ---
            volumes_acc.pop()
            schedule_acc.pop()
            for i, on in enumerate(config["mask"]):
                if on: usage_counts[i] -= 1


# =============================================================================
#  MAIN DE DEMOSTRACIÓN
# =============================================================================

def main():
    # 1. Buscar fichero de test generado
    test_dir = Path("data/test")
    test_files = list(test_dir.glob("problem_3_10_*.json"))

    if not test_files:
        print("❌ No hay ficheros en data/test/. Ejecuta primero generate_test_data.py")
        return

    # Tomamos el primero
    target_file = test_files[0]
    print(f"🔄 Cargando problema: {target_file.name}...")
    problem = WaterPumpProblem.load(target_file)

    print(f"   Bombas: {problem.num_pumps} | Horizonte: {problem.horizon}h")
    print(f"   V_init: {problem.v_init} | V_min: {problem.v_min} | V_max: {problem.v_max}")

    # 2. Resolver
    print("🚀 Ejecutando Fuerza Bruta (DFS con Backtracking)...")
    solver = BruteForceSolver(problem)
    solutions = solver.solve(top_k=3)

    # 3. Mostrar resultados
    if not solutions:
        print("⚠️ No se encontró ninguna solución factible.")
        return

    print(f"\n✅ Se encontraron {len(solutions)} soluciones óptimas.")

    best = solutions[0]
    print(f"\n🏆 MEJOR SOLUCIÓN (Coste: {best.total_cost:.4f} PLN)")
    print(f"   Volumen Final: {best.volumes[-1]:.2f} m³")

    print("\n   📅 Horario Detallado:")
    for t, (pumps_state, vol) in enumerate(zip(best.schedule, best.volumes)):
        active_indices = [i + 1 for i, on in enumerate(pumps_state) if on]
        # Formato visual: h00 | Vol: 500.0 | Bombas: [1, 3]
        print(f"     h{t:02d} | Vol: {vol:7.2f} | Activas: {active_indices if active_indices else '(Ninguna)'}")


if __name__ == "__main__":
    main()