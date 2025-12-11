# framework/core/experiment_runner.py
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import time
import traceback as tb
from typing import Any, Callable, Dict, Tuple

from .types import ExperimentResult, RunStatus

logger = logging.getLogger(__name__)


def _run_algo_inner(
    algo_fn: Callable[..., Tuple[Dict[str, int], float, float, float, Dict[str, Any]]],
    problem: Any,
    algo_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, int], float, float, float, Dict[str, Any]]:
    """
    Ejecuta el algoritmo y retorna:
      (solution_dict, energy, utility, cost, meta)
    """
    return algo_fn(problem, **algo_kwargs)


def run_with_timeout(
    problem: Any,
    algo_name: str,
    algo_fn: Callable[..., Tuple[Dict[str, int], float, float, float, Dict[str, Any]]],
    timeout_sec: float,
    algo_kwargs: Dict[str, Any] | None = None,
) -> ExperimentResult:
    """
    Ejecuta un algoritmo con timeout, captura excepciones, tiempos,
    y empaqueta todo en un ExperimentResult.
    """
    algo_kwargs = algo_kwargs or {}

    start_dt = dt.datetime.utcnow()
    start_time = start_dt.isoformat()
    problem_id = getattr(problem, "problem_id", "?")

    logger.info(f"[{algo_name}] Starting on problem {problem_id} at {start_time}")

    solution = None
    energy = None
    utility = None
    cost = None
    meta: Dict[str, Any] = {}

    status = RunStatus.OK
    exc_type = None
    exc_msg = None
    exc_tb_str = None

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_algo_inner, algo_fn, problem, algo_kwargs)
            try:
                solution, energy, utility, cost, meta = future.result(timeout=timeout_sec)
            except cf.TimeoutError:
                status = RunStatus.TIMEOUT
                future.cancel()
                logger.warning(f"[{algo_name}] Timeout after {timeout_sec:.1f}s on problem {problem_id}")
            except Exception as e:  # noqa: BLE001
                status = RunStatus.EXCEPTION
                exc_type = type(e).__name__
                exc_msg = str(e)
                exc_tb_str = tb.format_exc()
                logger.exception(f"[{algo_name}] Exception on problem {problem_id}: {exc_type}: {exc_msg}")
    except Exception as e:  # noqa: BLE001
        status = RunStatus.EXCEPTION
        exc_type = type(e).__name__
        exc_msg = str(e)
        exc_tb_str = tb.format_exc()
        logger.exception(f"[{algo_name}] Outer exception on problem {problem_id}: {exc_type}: {exc_msg}")

    t1 = time.time()
    end_dt = dt.datetime.utcnow()
    end_time = end_dt.isoformat()
    wall = t1 - t0

    result = ExperimentResult(
        problem_id=problem_id,
        algo_name=algo_name,
        status=status,
        start_time=start_time,
        end_time=end_time,
        wall_time_sec=wall,
        solution=solution,
        energy=energy,
        utility=utility,
        cost=cost,
        meta=meta,
        exception_type=exc_type,
        exception_message=exc_msg,
        exception_traceback=exc_tb_str,
    )

    logger.info(
        f"[{algo_name}] Finished on problem {problem_id} "
        f"status={result.status.value}, wall_time={wall:.2f}s, "
        f"utility={result.utility}, cost={result.cost}"
    )
    return result
