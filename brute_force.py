from __future__ import annotations

import itertools
import time
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

# Librerías para paralelismo y visualización
from joblib import Parallel, delayed
from tqdm import tqdm

# Imports locales
from pump_problem import WaterPumpProblem
from topk_queue import TopKQueue
from experiment_result import ExperimentResult


class OptimizationTimeoutError(Exception):
    """Excepción lanzada cuando el solver excede el tiempo límite."""
    pass


class BruteForceSolver:
    def __init__(self, problem: WaterPumpProblem):
        self.problem = problem
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

    def solve(self, top_k: int = 10, timeout: float = 60.0) -> List[ExperimentResult]:
        """
        Ejecuta el solver.
        Args:
            top_k: Número de soluciones a guardar.
            timeout: Tiempo máximo en segundos antes de abortar.
        """
        # 1. Poda Global
        if not self.problem.is_feasible_basic():
            return []

        queue = TopKQueue(k=top_k)
        start_time = time.perf_counter()

        # Iniciamos Backtracking con control de tiempo
        self._dfs(
            t=0,
            current_vol=self.problem.v_init,
            current_cost=0.0,
            usage_counts=[0] * self.problem.num_pumps,
            schedule_acc=[],
            volumes_acc=[],
            queue=queue,
            start_time=start_time,
            timeout_limit=timeout
        )
        return queue.best_first()

    def _dfs(self,
             t: int,
             current_vol: float,
             current_cost: float,
             usage_counts: List[int],
             schedule_acc: List[List[int]],
             volumes_acc: List[float],
             queue: TopKQueue,
             start_time: float,
             timeout_limit: float):

        # --- CONTROL DE TIMEOUT ---
        if (time.perf_counter() - start_time) > timeout_limit:
            raise OptimizationTimeoutError("Tiempo límite excedido")

        # --- PODA ANTICIPADA (Eq. 2: Uso mínimo) ---
        remaining_slots = self.problem.horizon - t
        min_req = self.problem.min_pump_usage

        for u in usage_counts:
            if u + remaining_slots < min_req:
                return

        # --- CASO BASE: Fin del horizonte ---
        if t == self.problem.horizon:
            if all(u >= min_req for u in usage_counts):
                # --- CORRECCIÓN AQUÍ: Instanciación limpia ---
                res = ExperimentResult(
                    schedule=[list(s) for s in schedule_acc],
                    total_cost=current_cost,
                    volumes=list(volumes_acc),
                    status="Success"
                )
                queue.push(res)
            return

        # --- RECURSIÓN ---
        # Variables locales
        price = self.problem.electricity_prices[t]
        demand = self.problem.demand[t]
        dt = self.problem.time_step_hours
        v_min, v_max = self.problem.v_min, self.problem.v_max
        p_limit = self.problem.power_limits[t] if self.problem.power_limits else None
        TOL = 1e-5

        for config in self.possible_configs:
            # Filtros
            if p_limit is not None and config["power_kw"] > p_limit:
                continue

            new_vol = current_vol + (config["flow"] * dt) - demand

            if not (v_min - TOL <= new_vol <= v_max + TOL):
                continue

            # Avanzar
            step_cost = (price * config["power_kw"] * dt) / 1000.0

            for i, on in enumerate(config["mask"]):
                if on: usage_counts[i] += 1

            schedule_acc.append(config["mask"])
            volumes_acc.append(new_vol)

            self._dfs(t + 1, new_vol, current_cost + step_cost, usage_counts,
                      schedule_acc, volumes_acc, queue, start_time, timeout_limit)

            # Retroceder
            volumes_acc.pop()
            schedule_acc.pop()
            for i, on in enumerate(config["mask"]):
                if on: usage_counts[i] -= 1


# =============================================================================
#  LÓGICA DE EJECUCIÓN PARALELA Y REPORTING
# =============================================================================

def process_single_file(file_path: Path, result_dir: Path, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Procesa un archivo y determina el status: Success, Infeasible, Timeout, Exception.
    Siempre guarda un JSON con el resultado.
    """
    t_start = time.perf_counter()
    problem_id = file_path.stem

    # Objeto resultado base (vacío / fallido por defecto)
    result_obj = ExperimentResult(
        schedule=None,
        total_cost=None,
        volumes=None,
        problem_id=problem_id,
        algorithm="BruteForceSolver",
        status="Unknown"
    )

    try:
        # 1. Cargar
        problem = WaterPumpProblem.load(file_path)
        problem_id = problem.problem_id
        result_obj.problem_id = problem_id

        # 2. Resolver con Timeout
        solver = BruteForceSolver(problem)
        solutions = solver.solve(top_k=1, timeout=timeout)

        elapsed = time.perf_counter() - t_start
        result_obj.execution_time = elapsed

        if solutions:
            # --- SUCCESS ---
            best = solutions[0]
            result_obj.status = "Success"
            result_obj.schedule = best.schedule
            result_obj.total_cost = best.total_cost
            result_obj.volumes = best.volumes
        else:
            # --- INFEASIBLE ---
            result_obj.status = "Infeasible"

    except OptimizationTimeoutError:
        # --- TIMEOUT ---
        elapsed = time.perf_counter() - t_start
        result_obj.execution_time = elapsed
        result_obj.status = "Timeout"

    except Exception as e:
        # --- EXCEPTION ---
        elapsed = time.perf_counter() - t_start
        result_obj.execution_time = elapsed
        result_obj.status = "Exception"

    # 3. Guardar JSON (Siempre)
    output_file = result_dir / f"result_{problem_id}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result_obj.to_dict(), f, indent=2)

    # 4. Retornar estadísticas para el reporte global
    return {
        "id": problem_id,
        "status": result_obj.status,
        "time": result_obj.execution_time,
        "cost": result_obj.total_cost
    }


def main():
    test_dir = Path("data/test")
    result_dir = Path("data/result")
    result_dir.mkdir(parents=True, exist_ok=True)

    files = list(test_dir.glob("*.json"))
    if not files:
        print("❌ No se encontraron archivos en data/test")
        return

    # CONFIGURACIÓN
    TIMEOUT_SECONDS = 100.0

    print(f"🚀 Procesando {len(files)} instancias (Timeout={TIMEOUT_SECONDS}s)...")

    t_global_start = time.perf_counter()

    # Generador de tareas
    tasks = (delayed(process_single_file)(f, result_dir, TIMEOUT_SECONDS) for f in files)

    # Ejecutor paralelo
    parallel_runner = Parallel(n_jobs=-1, return_as="generator")

    results_stats = []

    # Consumimos el generador con tqdm
    for res in tqdm(parallel_runner(tasks), total=len(files), unit="prob"):
        results_stats.append(res)

    t_global_end = time.perf_counter()

    # --- Estadísticas ---
    total_time = t_global_end - t_global_start

    # Agrupar por status
    by_status = {"Success": [], "Infeasible": [], "Timeout": [], "Exception": []}
    for r in results_stats:
        s = r.get("status", "Exception")
        if s not in by_status: s = "Exception"
        by_status[s].append(r)

    print("\n" + "=" * 50)
    print("📊 REPORTE FINAL")
    print("=" * 50)
    print(f"⏱️  Tiempo Global:   {total_time:.2f} s")
    print(f"✅ Success:         {len(by_status['Success'])}")
    print(f"⚠️ Infeasible:      {len(by_status['Infeasible'])}")
    print(f"🐢 Timeout:         {len(by_status['Timeout'])}")
    print(f"❌ Exception:       {len(by_status['Exception'])}")

    if by_status['Success']:
        # Filtramos costos None por seguridad
        valid_costs = [r['cost'] for r in by_status['Success'] if r['cost'] is not None]
        if valid_costs:
            avg_cost = sum(valid_costs) / len(valid_costs)
            print(f"💰 Coste Medio:     {avg_cost:.2f} PLN")

    # Listar Timeouts o Excepciones
    if by_status['Timeout']:
        print("\n🐢 Timeouts:")
        for r in by_status['Timeout'][:5]:  # Mostrar max 5
            print(f"   - {r['id']} (> {TIMEOUT_SECONDS}s)")

    if by_status['Exception']:
        print("\n🔥 Exceptions:")
        for r in by_status['Exception'][:5]:  # Mostrar max 5
            print(f"   - {r['id']}")


if __name__ == "__main__":
    main()