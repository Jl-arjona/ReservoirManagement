# orchestrate_reservoir_pipeline.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Any

import json
import logging

from framework.core.types import ExperimentResult  # asumo que ya lo tienes
from framework.core.topk_queue import TopKQueue   # idem
# from framework.ml.penalty_models import ...     # si ya tienes algo aquí

from problems.reservoir.model import (
    ReservoirInstance,
    generate_random_instance,
    save_instance,
    load_instance,
    list_instance_files,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# -------------------------------------------------------------------
# Configuración de rutas
# -------------------------------------------------------------------

BASE_DATA_DIR = Path("data") / "reservoir"
TRAIN_DIR = BASE_DATA_DIR / "train"
TEST_DIR = BASE_DATA_DIR / "test"

BASE_RESULTS_DIR = Path("results") / "reservoir"


def make_run_dir(tag: str = "run") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_RESULTS_DIR / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Creado directorio de resultados: %s", run_dir)
    return run_dir


# -------------------------------------------------------------------
# Helpers para generación de instancias
# -------------------------------------------------------------------


def ensure_train_instances(
    n_train: int = 20,
    pumps_options=(2, 3, 4, 5),
    horizon_options=(3, 4, 5, 6),
) -> List[Path]:
    """Si no hay instancias en TRAIN_DIR, genera n_train al azar."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_instance_files(TRAIN_DIR)
    if existing:
        logger.info("Encontradas %d instancias de entrenamiento en %s", len(existing), TRAIN_DIR)
        return existing

    logger.info("No hay instancias en %s, generando %d...", TRAIN_DIR, n_train)

    paths: List[Path] = []
    for i in range(n_train):
        num_pumps = int(pumps_options[i % len(pumps_options)])
        horizon = int(horizon_options[i % len(horizon_options)])
        problem_id = f"train_{i:03d}_P{num_pumps}_T{horizon}"

        inst = generate_random_instance(
            problem_id=problem_id,
            num_pumps=num_pumps,
            horizon=horizon,
        )
        path = TRAIN_DIR / f"{problem_id}.json"
        save_instance(inst, path)
        paths.append(path)

    logger.info("Generadas %d instancias de entrenamiento en %s", len(paths), TRAIN_DIR)
    return paths


def ensure_test_instances(
    n_test: int = 10,
    pumps_options=(2, 3, 4, 5),
    horizon_options=(3, 4, 5, 6),
) -> List[Path]:
    """Si no hay instancias en TEST_DIR, genera n_test al azar."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_instance_files(TEST_DIR)
    if existing:
        logger.info("Encontradas %d instancias de test en %s", len(existing), TEST_DIR)
        return existing

    logger.info("No hay instancias de test en %s, generando %d...", TEST_DIR, n_test)

    paths: List[Path] = []
    for i in range(n_test):
        num_pumps = int(pumps_options[i % len(pumps_options)])
        horizon = int(horizon_options[i % len(horizon_options)])
        problem_id = f"test_{i:03d}_P{num_pumps}_T{horizon}"

        inst = generate_random_instance(
            problem_id=problem_id,
            num_pumps=num_pumps,
            horizon=horizon,
        )
        path = TEST_DIR / f"{problem_id}.json"
        save_instance(inst, path)
        paths.append(path)

    logger.info("Generadas %d instancias de test en %s", len(paths), TEST_DIR)
    return paths


# -------------------------------------------------------------------
# Hooks a TU lógica de experimentación
# -------------------------------------------------------------------

def run_phase1_for_instance(instance: ReservoirInstance) -> tuple[
    List[ExperimentResult],
    Dict[str, float] | None,
]:
    """Hook donde enchufas tu lógica de Fase 1 para un problema.

    Implementación por defecto segura: no lanza excepciones y devuelve
    resultados vacíos, permitiendo que el pipeline se ejecute de extremo a
    extremo aunque aún no hayas conectado tu algoritmo.
    """

    logger.warning(
        "run_phase1_for_instance() usando implementación por defecto sin efectos para %s",
        instance.problem_id,
    )
    return [], None


def run_phase2_for_instance(
    instance: ReservoirInstance,
    penalty_model: Any,
) -> List[ExperimentResult]:
    """Hook para Fase 2: aplicar modelo de penalizaciones a problemas nuevos."""
    logger.warning(
        "run_phase2_for_instance() usando implementación por defecto sin efectos para %s",
        instance.problem_id,
    )
    return []


# -------------------------------------------------------------------
# Serialización de resultados
# -------------------------------------------------------------------

def save_experiment_results(
    run_dir: Path,
    all_results: List[ExperimentResult],
    topk: TopKQueue | None = None,
    penalties_by_problem: Dict[str, Dict[str, float]] | None = None,
) -> None:
    """Guarda resultados agregados de una ejecución en JSON."""

    # 1) Todos los resultados de experimentos
    experiments_path = run_dir / "experiments.json"
    with experiments_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    logger.info("Guardados resultados de experimentos en %s", experiments_path)

    # 2) Top-k soluciones (si hay cola)
    if topk is not None:
        # Asumo que topk almacena ExperimentResult; ajusta si usas otro tipo.
        topk_path = run_dir / "topk.json"
        with topk_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in topk.best_first()], f, indent=2)
        logger.info("Guardadas mejores soluciones (top-k) en %s", topk_path)

    # 3) Penalizaciones por problema
    if penalties_by_problem is not None:
        penalties_path = run_dir / "penalties.json"
        with penalties_path.open("w", encoding="utf-8") as f:
            json.dump(penalties_by_problem, f, indent=2)
        logger.info("Guardadas penalizaciones por problema en %s", penalties_path)


# -------------------------------------------------------------------
# Orquestación global
# -------------------------------------------------------------------

def main() -> None:
    logger.info("=== Orquestación del pipeline de RESERVOIR ===")

    # 1) Asegurar datasets (train/test) en disco
    train_files = ensure_train_instances()
    test_files = ensure_test_instances()

    # 2) Fase 1: entrenamiento / búsqueda de penalizaciones por problema
    logger.info("Iniciando Fase 1 (entrenamiento / penalizaciones)...")

    # Si quieres hacerlo en paralelo, aquí puedes usar joblib.Parallel,
    # pluggeando run_phase1_for_instance().
    all_results_phase1: List[ExperimentResult] = []
    penalties_by_problem: Dict[str, Dict[str, float]] = {}
    topk = TopKQueue(k=10)

    for path in train_files:
        inst = load_instance(path)
        logger.info("Fase1 - problema %s", inst.problem_id)

        results, best_penalties = run_phase1_for_instance(inst)

        all_results_phase1.extend(results)

        # Actualizar top-k con cualquier resultado que tenga utilidad
        for r in results:
            # asumo que ExperimentResult tiene atributo 'utility'
            if getattr(r, "utility", None) is not None:
                topk.push(r)

        if best_penalties is not None:
            penalties_by_problem[inst.problem_id] = best_penalties

    # 3) Guardar resultados Fase 1 en un directorio con timestamp
    run_dir_phase1 = make_run_dir(tag="phase1")
    save_experiment_results(
        run_dir=run_dir_phase1,
        all_results=all_results_phase1,
        topk=topk,
        penalties_by_problem=penalties_by_problem or None,
    )

    logger.info("Fase 1 completada. Resultados en %s", run_dir_phase1)

    # --------------------------------------------------------------
    # 4) (Opcional) Fase 2 - usar un modelo de ML para generalizar
    #     penalizaciones a nuevos problemas.
    #     Aquí cargarías penalty_model entrenado y aplicarías a test_files.
    # --------------------------------------------------------------

    # Ejemplo (esqueleto):
    #
    # penalty_model = train_penalty_model_from_phase1(penalties_by_problem, train_files)
    #
    # all_results_phase2: List[ExperimentResult] = []
    # topk_phase2 = TopKQueue(k=10)
    #
    # for path in test_files:
    #     inst = load_instance(path)
    #     results = run_phase2_for_instance(inst, penalty_model)
    #     all_results_phase2.extend(results)
    #     for r in results:
    #         if getattr(r, "utility", None) is not None:
    #             topk_phase2.push(r)
    #
    # run_dir_phase2 = make_run_dir(tag="phase2")
    # save_experiment_results(
    #     run_dir=run_dir_phase2,
    #     all_results=all_results_phase2,
    #     topk=topk_phase2,
    # )
    # logger.info("Fase 2 completada. Resultados en %s", run_dir_phase2)


if __name__ == "__main__":
    main()
