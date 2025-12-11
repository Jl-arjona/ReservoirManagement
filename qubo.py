from __future__ import annotations

import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# D-Wave Imports
from dwave.samplers import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

# Local Project Imports
from pump_problem import WaterPumpProblem
from experiment_result import ExperimentResult


class QuboSolver:
    """
    Solver basado en D-Wave LeapHybridSampler para el problema de programación de bombas.
    Adapta la formulación QUBO/BQM a las estructuras de datos del proyecto.
    """

    def __init__(self, problem: WaterPumpProblem, c3_gamma: float = 0.00052):
        self.problem = problem
        # Multiplicador de Lagrange para la restricción de volumen.
        # Puede requerir ajuste fino (tuning) según la escala del problema.
        self.c3_gamma = c3_gamma

    def build_bqm(self) -> Tuple[BinaryQuadraticModel, List[List[str]]]:
        """
        Construye el modelo cuadrático binario (BQM) basado en la instancia del problema.
        """
        num_pumps = self.problem.num_pumps
        horizon = self.problem.horizon
        time_indices = list(range(horizon))

        # Nombres de variables: P{bomba}_{tiempo}
        # x[p][t] accede al nombre de la variable
        x = [[f'P{p}_{t}' for t in time_indices] for p in range(num_pumps)]

        bqm = BinaryQuadraticModel('BINARY')

        # --- FUNCIÓN OBJETIVO: MINIMIZAR COSTES ---
        # Cost = Sum(Potencia_j * Precio_i * x_ij / 1000)
        # Gamma para el objetivo: Ajusta la prioridad del coste frente a las restricciones.
        # En el script original era 10000.
        obj_gamma = 10000

        for p in range(num_pumps):
            for t in time_indices:
                # Coste en este slot si la bomba se enciende
                # (kW * PLN/MWh / 1000) * dt (dt=1h asumido en formula original, pero usaremos time_step)
                cost_coeff = (self.problem.pump_power[p] * self.problem.electricity_prices[
                    t] / 1000.0 * self.problem.time_step_hours)

                bqm.add_variable(x[p][t], obj_gamma * cost_coeff)

        # --- RESTRICCIÓN 1: USO MÍNIMO (Mantenimiento) ---
        # Cada bomba debe funcionar al menos 'min_pump_usage' horas (slots)
        # LB = min_pump_usage, UB = horizon
        for p in range(num_pumps):
            c1_vars = [(x[p][t], 1) for t in time_indices]
            bqm.add_linear_inequality_constraint(
                c1_vars,
                lb=self.problem.min_pump_usage,
                ub=horizon,
                lagrange_multiplier=obj_gamma * 2,  # Penalización alta
                label=f'c1_min_usage_P{p}'
            )

        # --- RESTRICCIÓN 2: MÁXIMO BOMBAS SIMULTÁNEAS (Seguridad) ---
        # Sum(x_ij) <= max_active_pumps para cada tiempo t
        for t in time_indices:
            c2_vars = [(x[p][t], 1) for p in range(num_pumps)]
            bqm.add_linear_inequality_constraint(
                c2_vars,
                constant=0,  # No hay constante sumada a las variables
                lb=0,  # Puede haber 0 bombas encendidas
                ub=self.problem.max_active_pumps,
                lagrange_multiplier=obj_gamma * 2,
                label=f'c2_max_active_T{t}'
            )

        # --- RESTRICCIÓN 3: NIVELES DEL TANQUE (Física) ---
        # V_min <= V_t <= V_max
        # V_t = V_init + Sum_{k=0..t}(Bombeo_k) - Sum_{k=0..t}(Demanda_k)
        #
        # Reordenando para BQM (Variables a la izquierda, Constantes a la derecha):
        # V_min - V_init + CumDemand_t <= CumPump_t <= V_max - V_init + CumDemand_t

        # Factor de escala (100) del script original para manejar coeficientes flotantes
        scale = 100.0

        current_cum_demand = 0.0

        for t in time_indices:
            current_cum_demand += self.problem.demand[t]

            # Variables de bombeo acumulado hasta el tiempo t
            # Coeficiente = Flujo * time_step * scale
            c3_vars = []
            for k in range(t + 1):
                for p in range(num_pumps):
                    coeff = self.problem.pump_flows[p] * self.problem.time_step_hours * scale
                    c3_vars.append((x[p][k], coeff))

            # Límites calculados
            # V_t = V_init + Pumped - Demand
            # Pumped = V_t - V_init + Demand

            lower_bound_pumped = (self.problem.v_min - self.problem.v_init + current_cum_demand) * scale
            upper_bound_pumped = (self.problem.v_max - self.problem.v_init + current_cum_demand) * scale

            bqm.add_linear_inequality_constraint(
                c3_vars,
                lb=lower_bound_pumped,
                ub=upper_bound_pumped,
                lagrange_multiplier=self.c3_gamma * obj_gamma,  # Usamos el gamma ajustado
                label=f'c3_volume_T{t}'
            )

        return bqm, x

    def solve(self) -> ExperimentResult:
        """
        Ejecuta el problema en D-Wave y retorna un ExperimentResult.
        """
        t_start = time.perf_counter()

        # 1. Construir modelo
        bqm, x_vars = self.build_bqm()

        # 2. Enviar a D-Wave (Hybrid Sampler)
        # Nota: Requiere DWAVE_API_TOKEN en variables de entorno o configuración local
        sampler = SimulatedAnnealingSampler()

        # Ejecutamos
        sampleset = sampler.sample(bqm)

        # Tomamos la mejor muestra
        best_sample = sampleset.first.sample

        elapsed = time.perf_counter() - t_start

        # 3. Procesar resultados y convertirlos a ExperimentResult
        schedule = []

        # Reconstruir la matriz de horario
        for t in range(self.problem.horizon):
            row = []
            for p in range(self.problem.num_pumps):
                var_name = x_vars[p][t]
                # D-Wave devuelve 1.0 o 0.0, convertimos a int
                val = int(best_sample[var_name])
                row.append(val)
            schedule.append(row)

        # 4. Validar y Calcular métricas usando la lógica de ExperimentResult
        # Simulamos los volúmenes para obtener la evolución real basada en el horario obtenido
        volumes = self.problem.simulate_volumes(schedule)

        # Calcular coste total real
        total_cost = 0.0
        cost_matrix = self.problem.cost_matrix
        for t in range(self.problem.horizon):
            for p in range(self.problem.num_pumps):
                if schedule[t][p] == 1:
                    total_cost += cost_matrix[t][p]

        # Verificar restricciones manualmente para el status
        # (El sampler híbrido suele respetar restricciones fuertes, pero no siempre al 100%)

        # Check volúmenes
        vol_ok = all(self.problem.v_min - 1e-3 <= v <= self.problem.v_max + 1e-3 for v in volumes)

        # Check uso mínimo
        usage_counts = [sum(schedule[t][p] for t in range(self.problem.horizon))
                        for p in range(self.problem.num_pumps)]
        usage_ok = all(u >= self.problem.min_pump_usage for u in usage_counts)

        # Determinar status final
        status = "Success"
        if not vol_ok or not usage_ok:
            status = "Infeasible"  # La solución cuántica violó alguna restricción

        return ExperimentResult(
            schedule=schedule,
            total_cost=total_cost,
            volumes=volumes,
            problem_id=self.problem.problem_id,
            algorithm="QuboSolver",
            execution_time=elapsed,
            status=status
        )


# =============================================================================
#  EJECUCIÓN POR LOTES (MAIN)
# =============================================================================

def main():
    test_dir = Path("data/test")
    result_dir = Path("data/result")
    result_dir.mkdir(parents=True, exist_ok=True)

    files = list(test_dir.glob("*.json"))
    if not files:
        print("❌ No se encontraron archivos en data/test")
        return

    # Verificar Token de D-Wave
    if not os.environ.get("DWAVE_API_TOKEN") and not os.path.exists(os.path.expanduser("~/.dwave_config")):
        print("⚠️ ADVERTENCIA: No se detectó configuración de D-Wave.")
        print("   Asegúrate de configurar tu API Token con 'dwave config create'")
        print("   o exportar la variable DWAVE_API_TOKEN.")

    print(f"🚀 Iniciando QuboSolver para {len(files)} instancias...")
    print(f"📂 Entrada: {test_dir}")
    print(f"📂 Salida:  {result_dir}")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for file_path in files:
        print(f"\nProcessing {file_path.name}...")

        try:
            # 1. Cargar Problema
            problem = WaterPumpProblem.load(file_path)

            # 2. Resolver
            solver = QuboSolver(problem)
            result = solver.solve()

            # 3. Guardar Resultado con sufijo de algoritmo
            # Ejemplo: result_problem_3_10_001_QuboSolver.json
            output_filename = f"result_{problem.problem_id}_{result.algorithm}.json"
            output_path = result_dir / output_filename

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

            print(f"   Status: {result.status}")
            print(f"   Time:   {result.execution_time:.2f}s")
            if result.total_cost is not None:
                print(f"   Cost:   {result.total_cost:.2f} PLN")

            if result.status == "Success":
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            fail_count += 1

            # Intentar guardar log de error
            error_filename = f"error_{file_path.stem}_QuboSolver.json"
            with (result_dir / error_filename).open("w", encoding="utf-8") as f:
                json.dump({
                    "problem_id": file_path.stem,
                    "algorithm": "QuboSolver",
                    "status": "Exception",
                    "error": str(e)
                }, f, indent=2)

    print("\n" + "=" * 50)
    print("📊 REPORTE FINAL QUBO")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed/Infeasible: {fail_count}")


if __name__ == "__main__":
    main()