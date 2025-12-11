# framework/core/experiment_runner.py
from __future__ import annotations

import concurrent.futures as cf
import logging
from datetime import datetime, timezone
from time import perf_counter
import traceback as tb
from typing import Any, Callable, Dict, TypeAlias

from .types import ExperimentResult, RunStatus

logger = logging.getLogger(__name__)


# (solution, energy, utility, cost, meta)
RawAlgoResult: TypeAlias = tuple[Dict[str, int], float, float, float, Dict[str, Any]]



def run_with_timeout(
    problem: Any,
    algo_name: str,
    algo_fn: Callable[..., RawAlgoResult],
    timeout_sec: float,
    algo_kwargs: Dict[str, Any] | None = None,
) -> ExperimentResult:
    """
  Ejecuta un algoritmo con timeout, captura excepciones y tiempos,
    y empaqueta todo en un ExperimentResult.
    """
    algo_kwargs = algo_kwargs or {}
    problem_id = getattr(problem, "problem_id", "unknown")

    start_dt = datetime.now(timezone.utc)
    logger.info("[%s] Starting on problem %s at %s", algo_name, problem_id, start_dt.isoformat())

    result_data: Dict[str, Any] = {
        "solution": None,
        "energy": None,
        "utility": None,
        "cost": None,
        "meta": {},
        "exception_type": None,
        "exception_message": None,
        "exception_traceback": None,
    }


    status = RunStatus.OK
    future: cf.Future[RawAlgoResult] | None = None
    start_perf = perf_counter()


    try:
        with cf.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(algo_fn, problem, **algo_kwargs)
            solution, energy, utility, cost, meta = future.result(timeout=timeout_sec)
            result_data.update(
                {
                    "solution": solution,
                    "energy": energy,
                    "utility": utility,
                    "cost": cost,
                    "meta": meta,
                }
            )
    except cf.TimeoutError:
        status = RunStatus.TIMEOUT
        if future:
            future.cancel()
        logger.warning("[%s] Timeout after %.1fs on problem %s", algo_name, timeout_sec, problem_id)

    except Exception as exc:  # noqa: BLE001
        status = RunStatus.EXCEPTION
        result_data.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "exception_traceback": tb.format_exc(),
            }
        )
        logger.exception("[%s] Exception on problem %s", algo_name, problem_id)

    wall = perf_counter() - start_perf
    end_dt = datetime.now(timezone.utc)

    result = ExperimentResult(
        problem_id=problem_id,
        algo_name=algo_name,
        status=status,
        start_time=start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        wall_time_sec=wall,
        **result_data,
    )

    logger.info(
        "[%s] Finished on problem %s status=%s, wall_time=%.2fs, utility=%s, cost=%s",
        algo_name,
        problem_id,
        result.status.value,
        wall,
        result.utility,
        result.cost,
    )
    return result
