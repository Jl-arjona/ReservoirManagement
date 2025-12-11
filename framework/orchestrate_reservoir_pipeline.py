# framework/orchestrate_reservoir_pipeline.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from dwave.samplers import SimulatedAnnealingSampler

from .core.experiment_runner import run_with_timeout
from .core.topk_queue import TopKQueue
from .core.types import ExperimentResult
from .penalties import OptunaPenaltyOptimizer, PenaltyParams
from .reservoir_problem import (
    ReservoirInstance,
    random_reservoir_instance,
    reservoir_penalty_config,
    brute_force_reservoir,
    algo_reservoir_bruteforce,
    algo_reservoir_qubo,
    build_reservoir_bqm,
    evaluate_reservoir_solution,
    reservoir_feature_vector,
)
from .ml.penalty_models import train_best_penalty_model


logger = logging.getLogger(__name__)


def sampler_factory():
    return SimulatedAnnealingSampler()


# -------------------------------------------------------------
#  FASE 1: PARA CADA PROBLEMA, OBTENER ÓPTIMO Y PENALIZACIONES
# -------------------------------------------------------------

def optimize_penalties_for_instance(
    inst: ReservoirInstance,
    n_trials: int,
    reads_per_trial: int,
) -> Tuple[ReservoirInstance, float, PenaltyParams]:
    """
    Para un problema concreto:
      - calcula el coste óptimo por fuerza bruta,
      - optimiza penalizaciones con Optuna solo para ese problema.
    """
    bf = brute_force_reservoir(inst)
    opt_cost = bf["best_cost"]
    if bf["best_sample"] is None:
        raise RuntimeError(f"Problema {inst.problem_id}: fuerza bruta sin solución factible")

    param_cfgs = reservoir_penalty_config()
    optimizer = OptunaPenaltyOptimizer(
        param_configs=param_cfgs,
        training_instances=[inst],
        optimal_costs={inst.problem_id: opt_cost},
        build_bqm_fn=build_reservoir_bqm,
        eval_solution_fn=evaluate_reservoir_solution,
        sampler_factory=sampler_factory,
        n_trials=n_trials,
        n_jobs=1,            # dentro de cada problema, Optuna en 1 hilo
        reads_per_trial=reads_per_trial,
    )
    best_params = optimizer.fit(verbose=False)
    logger.info(
        f"[Optuna] {inst.problem_id}: better score={optimizer.study.best_value:.4f}, "
        f"gamma={best_params['gamma']:.4g}"
    )
    return inst, opt_cost, best_params


# -------------------------------------------------------------
#  ORQUESTACIÓN COMPLETA
# -------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_dir = Path("reservoir_results")
    out_dir.mkdir(exist_ok=True)

    # ---------------------------
    # 1) Generar problemas train
    # ---------------------------
    train_instances: List[ReservoirInstance] = []
    problem_idx = 0
    for num_pumps in [2, 3, 4]:
        for horizon in [3, 4, 5]:
            pid = f"train_{problem_idx}"
            inst = random_reservoir_instance(pid, num_pumps=num_pumps, horizon=horizon, seed=problem_idx)
            train_instances.append(inst)
            problem_idx += 1

    logger.info(f"Generados {len(train_instances)} problemas de entrenamiento")

    # ---------------------------
    # 2) Para cada problema:
    #    fuerza bruta + Optuna
    # ---------------------------
    n_trials_per_problem = 40
    reads_per_trial = 40

    results_phase1: List[Tuple[ReservoirInstance, float, PenaltyParams]] = Parallel(
        n_jobs=-1, backend="loky"
    )(
        delayed(optimize_penalties_for_instance)(
            inst, n_trials=n_trials_per_problem, reads_per_trial=reads_per_trial
        )
        for inst in train_instances
    )

    # Dataset para ML: X (features), y (penalizaciones)
    feature_names = list(reservoir_feature_vector(train_instances[0]).keys())
    param_names = list(reservoir_penalty_config().keys())

    X_list: List[List[float]] = []
    y_list: List[List[float]] = []

    # Para info adicional
    per_instance_opt_cost: Dict[str, float] = {}
    per_instance_penalties: Dict[str, Dict[str, float]] = {}

    for inst, opt_cost, penalties in results_phase1:
        feats = reservoir_feature_vector(inst)
        X_list.append([feats[name] for name in feature_names])
        y_list.append([penalties[name] for name in param_names])
        per_instance_opt_cost[inst.problem_id] = opt_cost
        per_instance_penalties[inst.problem_id] = penalties.to_dict()

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    # ---------------------------
    # 3) Entrenar modelo de ML
    # ---------------------------
    logger.info("Entrenando modelos de ML para penalizaciones...")
    penalty_model = train_best_penalty_model(
        X=X,
        y=y,
        feature_names=feature_names,
        param_names=param_names,
        cv=3,
    )

    # ---------------------------
    # 4) Problemas de test
    # ---------------------------
    test_instances: List[ReservoirInstance] = []
    for num_pumps in [3, 4]:
        for horizon in [4, 5, 6]:
            pid = f"test_{num_pumps}_{horizon}"
            inst = random_reservoir_instance(pid, num_pumps=num_pumps, horizon=horizon, seed=1000 + num_pumps * 10 + horizon)
            test_instances.append(inst)

    logger.info(f"Generados {len(test_instances)} problemas de test")

    # Cola de top-K soluciones QUBO por utilidad
    topk = TopKQueue(k=10)

    all_results: List[ExperimentResult] = []

    # ---------------------------
    # 5) Evaluar en test: fuerza bruta (si posible) + QUBO con penaliz. ML
    # ---------------------------
    for inst in test_instances:
        print(f"\n=== Test problem {inst.problem_id}: P={inst.num_pumps}, T={inst.horizon} ===")

        # (a) Fuerza bruta (con timeout razonable)
        bf_timeout = 60.0  # segundos, ajusta según tamaño
        bf_result = run_with_timeout(
            inst,
            algo_name="bruteforce",
            algo_fn=algo_reservoir_bruteforce,
            timeout_sec=bf_timeout,
        )
        all_results.append(bf_result)

        if bf_result.status == bf_result.status.OK and bf_result.cost is not None:
            print(f"  Fuerza bruta: coste={bf_result.cost:.3f}, tiempo={bf_result.wall_time_sec:.2f}s")
        else:
            print(f"  Fuerza bruta: status={bf_result.status.value}")

        # (b) Penalizaciones por ML
        feats = reservoir_feature_vector(inst)
        predicted_params = penalty_model.predict_params(feats)
        penalties = PenaltyParams(predicted_params)

        qubo_timeout = 30.0
        qubo_result = run_with_timeout(
            inst,
            algo_name="qubo_ml_penalties",
            algo_fn=algo_reservoir_qubo,
            timeout_sec=qubo_timeout,
            algo_kwargs={
                "penalties": penalties,
                "num_reads": 300,
                "sampler": sampler_factory(),
            },
        )
        all_results.append(qubo_result)
        topk.push(qubo_result)

        if qubo_result.status == qubo_result.status.OK and qubo_result.cost is not None:
            print(
                f"  QUBO+ML: coste={qubo_result.cost:.3f}, "
                f"violaciones={qubo_result.meta.get('violations')}, "
                f"tiempo={qubo_result.wall_time_sec:.2f}s"
            )
        else:
            print(f"  QUBO+ML: status={qubo_result.status.value}")

        # Comparar si tenemos BF válido
        if bf_result.status == bf_result.status.OK and bf_result.cost is not None and qubo_result.cost is not None:
            gap = qubo_result.cost - bf_result.cost
            print(f"  GAP coste (QUBO - BF): {gap:.3f}")

    # ---------------------------
    # 6) Guardar resultados + top-K
    # ---------------------------
    results_path = out_dir / "experiment_results.json"
    topk_path = out_dir / "topk_solutions.json"
    penalties_path = out_dir / "train_penalties.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    with topk_path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in topk.best_first()], f, indent=2)

    with penalties_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "per_instance_opt_cost": per_instance_opt_cost,
                "per_instance_penalties": per_instance_penalties,
                "feature_names": feature_names,
                "param_names": param_names,
            },
            f,
            indent=2,
        )

    print(f"\nResultados guardados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
