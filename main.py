"""
Extended reservoir-management QUBO model (based on Sustainability 13, 3470).

- Discretized y_ij in [0, 1] using K bits (QUBO).
- Constraints implemented as quadratic penalties via
  dimod.BinaryQuadraticModel.add_linear_inequality_constraint.
- Solvable with:
    - neal.SimulatedAnnealingSampler (classical)
    - D-Wave LeapHybridSampler (if available and token configured)

Main entry point: run_example_from_article()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import dimod

from dwave.samplers import SimulatedAnnealingSampler


try:
    from dwave.system import LeapHybridSampler
    HAVE_DWAVE = True
except ImportError:
    HAVE_DWAVE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReservoirInstance:
    """Problem data for the reservoir-management model."""
    num_pumps: int
    time: List[int]           # list of time slots (e.g. [1..24])
    power: List[float]        # power[p] (kW)
    costs: List[float]        # costs[t] (tariff)
    flow: List[float]         # flow[p] (m^3/h)
    demand: List[float]       # demand[t] (m^3)
    v_init: float             # initial volume
    v_min: float              # minimum reservoir level
    v_max: float              # maximum reservoir level
    min_time_per_pump: float  # minimum total operating time per pump (h)
    max_pumps_per_slot: int   # max simultaneous pumps per time slot


@dataclass
class PenaltyConfig:
    """Penalty weights for the QUBO formulation."""
    K: int = 5            # bits for y_ij
    gamma: float = 10.0   # objective scale (cost terms)
    lam_min_time: float = 50.0
    lam_max_pumps: float = 50.0
    lam_reservoir: float = 50.0
    lam_couple: float = 10.0


# ---------------------------------------------------------------------------
# Build extended BQM
# ---------------------------------------------------------------------------

def build_extended_bqm(
    inst: ReservoirInstance,
    pen: PenaltyConfig,
) -> Tuple[dimod.BinaryQuadraticModel, List[List[str]], List[List[List[str]]]]:
    """
    Build BQM for the extended reservoir model with discretized y_ij.

    Variables:
        x[p][t]  in {0,1}  pump p allowed/active at time t (binary)
        y_bits[p][t][k]  bits encoding y_{p,t} in [0,1], using K bits.

    Encoding:
        y_{p,t} ≈ Delta * sum_{k=0}^{K-1} 2^k * Y_{p,t,k}
        Delta = 1 / (2^K - 1)

    Objective (discretized):
        min sum_{p,t} (power[p] * costs[t] / 1000) * y_{p,t}
        scaled by pen.gamma.

    Constraints (as penalties):
        1) min time per pump: sum_t y_{p,t} >= min_time_per_pump
        2) max pumps per slot: sum_p x_{p,t} <= max_pumps_per_slot
        3) reservoir bounds: v_min <= v_t <= v_max
        4) coupling: y_{p,t} <= x_{p,t}
    """

    print("\nBuilding extended BQM...")

    num_pumps = inst.num_pumps
    time = inst.time
    num_times = len(time)
    K = pen.K
    Delta = 1.0 / (2**K - 1)

    # x[p][t] variables
    x = [
        [f"P{p}_{time[t_idx]}" for t_idx in range(num_times)]
        for p in range(num_pumps)
    ]

    # y bits variables
    y_bits = [
        [
            [f"Y{p}_{time[t_idx]}_b{k}" for k in range(K)]
            for t_idx in range(num_times)
        ]
        for p in range(num_pumps)
    ]

    bqm = dimod.BinaryQuadraticModel('BINARY')

    # ----------------------------------------------------------------------
    # Objective: sum_{p,t} gamma * c_{p,t} * y_{p,t}
    # c_{p,t} = power[p] * costs[t] / 1000
    # y_{p,t} encoded with bits
    # ----------------------------------------------------------------------
    for p in range(num_pumps):
        for t_idx in range(num_times):
            c_pt = pen.gamma * inst.power[p] * inst.costs[t_idx] / 1000.0
            for k in range(K):
                var = y_bits[p][t_idx][k]
                coeff = c_pt * Delta * (2**k)
                bqm.add_variable(var, coeff)

    # ----------------------------------------------------------------------
    # Constraint 1: minimum time per pump
    # sum_t y_{p,t} >= min_time_per_pump
    # ----------------------------------------------------------------------
    for p in range(num_pumps):
        terms = []
        for t_idx in range(num_times):
            for k in range(K):
                var = y_bits[p][t_idx][k]
                coef = Delta * (2**k)
                terms.append((var, coef))

        bqm.add_linear_inequality_constraint(
            terms,
            lb=inst.min_time_per_pump,
            ub=float(num_times),
            lagrange_multiplier=pen.lam_min_time,
            label=f"c1_min_time_p{p}"
        )

    # ----------------------------------------------------------------------
    # Constraint 2: max pumps per slot
    # sum_p x_{p,t} <= max_pumps_per_slot
    # ----------------------------------------------------------------------
    for t_idx in range(num_times):
        c2_terms = [(x[p][t_idx], 1.0) for p in range(num_pumps)]
        bqm.add_linear_inequality_constraint(
            c2_terms,
            constant=-float(inst.max_pumps_per_slot),
            lagrange_multiplier=pen.lam_max_pumps,
            label=f"c2_max_pumps_t{time[t_idx]}"
        )

    # ----------------------------------------------------------------------
    # Constraint 3: reservoir bounds
    # v_t = v_{t-1} + sum_p flow[p]*y_{p,t} - demand[t]
    # v_min <= v_t <= v_max  for all t
    #
    # We express this as inequality constraints on sum flow[p]*y_{p,τ<=t}
    # and scale by "scale" to get integer-like coefficients.
    # ----------------------------------------------------------------------
    scale = 100

    for t_idx in range(num_times):
        c3_terms = []
        for p in range(num_pumps):
            for tau_idx in range(t_idx + 1):
                for k in range(K):
                    var = y_bits[p][tau_idx][k]
                    coef = int(inst.flow[p] * Delta * (2**k) * scale)
                    c3_terms.append((var, coef))

        const = inst.v_init - sum(inst.demand[:t_idx + 1])
        bqm.add_linear_inequality_constraint(
            c3_terms,
            constant=int(const * scale),
            lb=int(inst.v_min * scale),
            ub=int(inst.v_max * scale),
            lagrange_multiplier=pen.lam_reservoir,
            label=f"c3_reservoir_t{time[t_idx]}"
        )

    # ----------------------------------------------------------------------
    # Constraint 4: coupling y_{p,t} <= x_{p,t}
    # Delta sum_k 2^k Y_{p,t,k} - x_{p,t} <= 0
    # ----------------------------------------------------------------------
    for p in range(num_pumps):
        for t_idx in range(num_times):
            terms = []
            for k in range(K):
                var = y_bits[p][t_idx][k]
                coef = Delta * (2**k)
                terms.append((var, coef))
            terms.append((x[p][t_idx], -1.0))

            bqm.add_linear_inequality_constraint(
                terms,
                ub=0.0,
                lagrange_multiplier=pen.lam_couple,
                label=f"c4_couple_p{p}_t{time[t_idx]}"
            )

    return bqm, x, y_bits


# ---------------------------------------------------------------------------
# Utility functions: reconstruct y, reservoir, cost, constraints
# ---------------------------------------------------------------------------

def reconstruct_y(sample: dict, y_bits, K: int) -> np.ndarray:
    """Rebuild y_{p,t} from bit variables y_bits[p][t][k]."""
    num_pumps = len(y_bits)
    num_times = len(y_bits[0])
    Delta = 1.0 / (2**K - 1)

    y_vals = np.zeros((num_pumps, num_times))
    for p in range(num_pumps):
        for t_idx in range(num_times):
            val = 0.0
            for k in range(K):
                bit_name = y_bits[p][t_idx][k]
                val += sample[bit_name] * Delta * (2**k)
            y_vals[p, t_idx] = val
    return y_vals


def compute_reservoir_levels(
    inst: ReservoirInstance,
    y_vals: np.ndarray
) -> List[float]:
    """Compute reservoir levels v_t for t = 0..T."""
    levels = [inst.v_init]
    num_times = len(inst.time)
    num_pumps = inst.num_pumps

    for t_idx in range(num_times):
        inflow = 0.0
        for p in range(num_pumps):
            inflow += inst.flow[p] * y_vals[p, t_idx]
        v_next = levels[-1] + inflow - inst.demand[t_idx]
        levels.append(v_next)

    return levels


def compute_cost_pln(
    inst: ReservoirInstance,
    y_vals: np.ndarray
) -> float:
    """Compute real cost in PLN (no penalties)."""
    total_cost = 0.0
    num_pumps = inst.num_pumps
    num_times = len(inst.time)

    for p in range(num_pumps):
        for t_idx in range(num_times):
            c_pt = inst.power[p] * inst.costs[t_idx] / 1000.0
            total_cost += c_pt * y_vals[p, t_idx]

    return total_cost


def check_constraints(
    inst: ReservoirInstance,
    sample: dict,
    x,
    y_bits,
    pen: PenaltyConfig,
    tol: float = 1e-3
) -> Dict[str, List]:
    """Check main constraints and print a short report."""
    time = inst.time
    num_pumps = inst.num_pumps
    num_times = len(time)

    y_vals = reconstruct_y(sample, y_bits, K=pen.K)
    levels = compute_reservoir_levels(inst, y_vals)

    violations = {
        "max_pumps": [],      # (t, pumps_on)
        "min_time": [],       # (p, total_time)
        "reservoir_low": [],  # (t, v_t)
        "reservoir_high": [], # (t, v_t)
        "coupling": []        # (p, t, y_pt, x_pt)
    }

    # R2: max pumps per slot
    for t_idx in range(num_times):
        pumps_on = sum(sample[x[p][t_idx]] for p in range(num_pumps))
        if pumps_on > inst.max_pumps_per_slot + tol:
            violations["max_pumps"].append((time[t_idx], pumps_on))

    # R1: min time per pump
    for p in range(num_pumps):
        total_time = y_vals[p, :].sum()
        if total_time < inst.min_time_per_pump - tol:
            violations["min_time"].append((p + 1, total_time))

    # R3: reservoir bounds
    for t_idx in range(1, num_times + 1):
        v_t = levels[t_idx]
        if v_t < inst.v_min - tol:
            violations["reservoir_low"].append((time[t_idx - 1], v_t))
        if v_t > inst.v_max + tol:
            violations["reservoir_high"].append((time[t_idx - 1], v_t))

    # Coupling y_{p,t} <= x_{p,t}
    for p in range(num_pumps):
        for t_idx in range(num_times):
            x_pt = sample[x[p][t_idx]]
            y_pt = y_vals[p, t_idx]
            if y_pt > x_pt + tol:
                violations["coupling"].append((p + 1, time[t_idx], y_pt, x_pt))

    # Print short report
    print("\n=== Constraint check ===")

    if violations["max_pumps"]:
        print("- Max pumps per slot VIOLATED at:")
        for t, pumps_on in violations["max_pumps"]:
            print(f"  t={t}: {pumps_on} pumps (> {inst.max_pumps_per_slot})")
    else:
        print("- Max pumps per slot: OK")

    if violations["min_time"]:
        print("- Min time per pump VIOLATED for:")
        for p, tt in violations["min_time"]:
            print(f"  Pump {p}: total time = {tt:.3f} h (< {inst.min_time_per_pump})")
    else:
        print("- Min time per pump: OK")

    if violations["reservoir_low"] or violations["reservoir_high"]:
        print("- Reservoir bounds VIOLATED:")
        for t, v in violations["reservoir_low"]:
            print(f"  t={t}: v={v:.2f} m³ (< v_min={inst.v_min})")
        for t, v in violations["reservoir_high"]:
            print(f"  t={t}: v={v:.2f} m³ (> v_max={inst.v_max})")
    else:
        print("- Reservoir bounds: OK")

    if violations["coupling"]:
        print("- Coupling VIOLATED at:")
        for p, t, y_pt, x_pt in violations["coupling"]:
            print(f"  Pump {p}, t={t}: y={y_pt:.3f} > x={x_pt}")
    else:
        print("- Coupling y <= x: OK")

    return violations


def summarize_solution(
    inst: ReservoirInstance,
    sample: dict,
    x,
    y_bits,
    pen: PenaltyConfig
) -> None:
    """Print schedule x/y and reservoir levels + per-pump summary."""
    num_pumps = inst.num_pumps
    time = inst.time
    num_times = len(time)

    y_vals = reconstruct_y(sample, y_bits, K=pen.K)
    levels = compute_reservoir_levels(inst, y_vals)
    cost_pln = compute_cost_pln(inst, y_vals)

    print("\n=== SOLUTION (extended QUBO) ===\n")

    # Table x/y
    header = (
        "t | " +
        "  ".join(f"x{p + 1}" for p in range(num_pumps)) +
        "  ||  " +
        "  ".join(f"y{p + 1}(h)" for p in range(num_pumps))
    )
    print(header)
    print("-" * len(header))

    for t_idx, tt in enumerate(time):
        xs = [sample[x[p][t_idx]] for p in range(num_pumps)]
        ys = [f"{y_vals[p, t_idx]:.2f}" for p in range(num_pumps)]
        print(f"{tt:2d}| " + "  ".join(str(v) for v in xs) + "  ||  " + "  ".join(ys))

    print("\nReservoir levels:")
    for t_idx, v in enumerate(levels):
        if t_idx == 0:
            print(f"  t=0 (init): {v:.2f} m³")
        else:
            print(f"  t={time[t_idx - 1]}: {v:.2f} m³")

    print(f"\nApprox. energy cost (PLN): {cost_pln:.3f}")

    # Per-pump summary
    print("\n=== Per-pump summary ===\n")
    for p in range(num_pumps):
        total_time = y_vals[p, :].sum()
        total_volume = (y_vals[p, :] * inst.flow[p]).sum()
        print(f"Pump {p + 1}:")
        print(f"  Total running time: {total_time:.2f} h")
        print(f"  Total pumped volume: {total_volume:.1f} m³")
        print(f"  Detailed schedule:")
        for t_idx, tt in enumerate(time):
            if y_vals[p, t_idx] > 1e-6:
                print(f"    t={tt}: {y_vals[p, t_idx]:.2f} h")
        print("")


# ---------------------------------------------------------------------------
# Example from the article
# ---------------------------------------------------------------------------

def example_instance_from_article() -> ReservoirInstance:
    """Build the 7-pump, 24-hour instance from the article."""
    num_pumps = 7
    time = list(range(1, 25))

    flow = [75, 133, 157, 176, 59, 69, 120]
    power = [15, 37, 33, 33, 22, 33, 22]

    demand = [
        44.62, 31.27, 26.22, 27.51, 31.50, 46.18,
        69.47, 100.36, 131.85, 148.51, 149.89, 142.21,
        132.09, 129.29, 124.06, 114.68, 109.33, 115.76,
        126.95, 131.48, 138.86, 131.91, 111.53, 70.43
    ]

    costs = (
        [169] * 7 +
        [283] * 6 +
        [169] * 3 +
        [336] * 5 +
        [169] * 3
    )

    v_init = 550.0
    v_min = 523.5
    v_max = 1500.0

    # from the article: each pump must work at least 1 hour per day
    min_time_per_pump = 1.0
    # at most 6 pumps simultaneously
    max_pumps_per_slot = 6

    return ReservoirInstance(
        num_pumps=num_pumps,
        time=time,
        power=power,
        costs=costs,
        flow=flow,
        demand=demand,
        v_init=v_init,
        v_min=v_min,
        v_max=v_max,
        min_time_per_pump=min_time_per_pump,
        max_pumps_per_slot=max_pumps_per_slot
    )


def run_example_from_article(
    use_dwave: bool = False,
    num_reads: int = 100
) -> None:
    """
    Build and solve the example instance using QUBO.

    Arguments:
        use_dwave: if True and LeapHybridSampler is available, use it.
                   otherwise, use neal.SimulatedAnnealingSampler.
        num_reads: number of reads (for SA); ignored by hybrid sampler.
    """
    inst = example_instance_from_article()
    pen = PenaltyConfig(
        K=5,
        gamma=10.0,
        lam_min_time=50.0,
        lam_max_pumps=50.0,
        lam_reservoir=50.0,
        lam_couple=10.0
    )

    bqm, x, y_bits = build_extended_bqm(inst, pen)

    # Choose sampler
    if use_dwave and HAVE_DWAVE:
        print("\nUsing LeapHybridSampler...")
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm)
    else:
        print("\nUsing SimulatedAnnealingSampler (neal)...")
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)

    best = sampleset.first
    sample = best.sample
    print("\nBest energy:", best.energy)

    summarize_solution(inst, sample, x, y_bits, pen)
    check_constraints(inst, sample, x, y_bits, pen, tol=1e-3)


if __name__ == "__main__":
    # Change use_dwave=True if you want to use LeapHybrid and have a token.
    run_example_from_article(use_dwave=False, num_reads=50)
