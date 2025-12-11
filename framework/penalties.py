# framework/penalties.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, Any, List, Optional

import optuna


@dataclass
class PenaltyParams:
    """
    Contenedor genérico: nombre_parametro -> valor.
    Ej: {"gamma": 0.1, "lam_c1": 5.0, "lam_c2": 2.0, "lam_c3": 0.3}
    """
    values: Dict[str, float]

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def __setitem__(self, name: str, value: float) -> None:
        self.values[name] = float(value)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)


@dataclass
class PenaltyParamConfig:
    """
    Configuración de un parámetro de penalización para Optuna.
    """
    name: str
    min_value: float
    max_value: float
    log_scale: bool = True


class OptunaPenaltyOptimizer:
    """
    Optimizador de penalizaciones (gamma, lambdas, etc.) usando Optuna.

    Pensado para ser flexible:
      - lista de configs de parámetros (nombre + rango),
      - lista de problemas de entrenamiento,
      - coste óptimo (por fuerza bruta) para cada problema,
      - callbacks específicos del problema:
          * build_bqm(instance, penalties) -> (bqm, meta)
          * evaluate_solution(instance, sample_dict, meta) -> dict con:
              {
                "utility": float,
                "cost": float,
                "violations": int
              }
    """

    def __init__(
        self,
        param_configs: Dict[str, PenaltyParamConfig],
        training_instances: List[Any],
        optimal_costs: Dict[str, float],
        build_bqm_fn: Callable[[Any, PenaltyParams], Any],
        eval_solution_fn: Callable[[Any, Dict[str, int], Any], Dict[str, Any]],
        sampler_factory: Callable[[], Any],
        n_trials: int = 100,
        n_jobs: int = 1,
        reads_per_trial: int = 50,
    ) -> None:
        self.param_configs = param_configs
        self.training_instances = training_instances
        self.optimal_costs = optimal_costs
        self.build_bqm_fn = build_bqm_fn
        self.eval_solution_fn = eval_solution_fn
        self.sampler_factory = sampler_factory
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.reads_per_trial = reads_per_trial

        self._study: Optional[optuna.Study] = None
        self._best_params: Optional[PenaltyParams] = None

    # ---------------------------------------------------------
    #   OBJETIVO OPTUNA
    # ---------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        # 1) Samplear un conjunto de parámetros de penalización.
        values: Dict[str, float] = {}
        for name, cfg in self.param_configs.items():
            if cfg.log_scale:
                v = trial.suggest_float(name, cfg.min_value, cfg.max_value, log=True)
            else:
                v = trial.suggest_float(name, cfg.min_value, cfg.max_value)
            values[name] = v

        penalties = PenaltyParams(values)

        total_score = 0.0

        # 2) Evaluar en todos los problemas de entrenamiento.
        for inst in self.training_instances:
            bqm, meta = self.build_bqm_fn(inst, penalties)
            sampler = self.sampler_factory()
            sampleset = sampler.sample(bqm, num_reads=self.reads_per_trial)

            best = sampleset.first  # mejor energía QUBO
            sample_dict = {k: int(v) for k, v in best.sample.items()}

            eval_res = self.eval_solution_fn(inst, sample_dict, meta)
            true_opt_cost = self.optimal_costs[inst.problem_id]

            # GAP de coste real
            gap = max(eval_res["cost"] - true_opt_cost, 0.0)

            # Penalizar violaciones fuertemente
            penalty_viol = eval_res["violations"] * 1e4

            total_score += gap + penalty_viol

        # Media sobre todos los problemas usados en este optimizador
        return total_score / max(len(self.training_instances), 1)

    # ---------------------------------------------------------
    #   INTERFAZ PÚBLICA
    # ---------------------------------------------------------

    def fit(self, verbose: bool = True) -> PenaltyParams:
        sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        self._study = study

        best = study.best_params
        self._best_params = PenaltyParams(best)

        if verbose:
            print("\n[OptunaPenaltyOptimizer] Mejores parámetros encontrados:")
            for k, v in best.items():
                print(f"  {k} = {v:.6g}")
            print(f"[OptunaPenaltyOptimizer] Mejor score objetivo: {study.best_value:.4f}")

        return self._best_params

    @property
    def best_params(self) -> Optional[PenaltyParams]:
        return self._best_params

    @property
    def study(self) -> Optional[optuna.Study]:
        return self._study
