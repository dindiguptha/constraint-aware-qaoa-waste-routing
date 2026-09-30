"""
Stage 3: QUBO Formulation
Encodes the TSP-style routing (single truck, capacity satisfied by
construction since total demand < capacity, so capacity constraint
is NOT yet binding -- we formulate route-finding as position-based
TSP QUBO first; capacity penalty term added in Stage 3b if needed).

Variables: x[i,p] = 1 if bin i is visited at position p in the route
Uses PyQUBO's Array + constraint helpers to keep the encoding readable.
"""

import numpy as np
from pyqubo import Array, Constraint, Placeholder
import neal  # classical QUBO sampler, useful for validating the QUBO
             # is correctly formulated BEFORE trusting QAOA results

from stage1_setup import generate_instance, build_distance_matrix, NUM_BINS
from stage2_baselines import route_cost, exact_solver


def build_qubo(dist_matrix, num_bins, A=10.0, B=1.0):
    """
    A = constraint penalty weight (must dominate cost term)
    B = cost term weight
    We only encode the `num_bins` actual bins as position variables;
    the depot (node 0) is fixed as start/end and added back afterward.
    """
    n = num_bins  # positions 0..n-1 correspond to bins 1..n in some order
    x = Array.create('x', shape=(n, n), vartype='BINARY')

    # ---- Constraint 1: each bin visited exactly once ----
    bin_once = sum(
        Constraint((sum(x[i, p] for p in range(n)) - 1) ** 2, label=f'bin_{i}_once')
        for i in range(n)
    )

    # ---- Constraint 2: each position filled exactly once ----
    pos_once = sum(
        Constraint((sum(x[i, p] for i in range(n)) - 1) ** 2, label=f'pos_{p}_once')
        for p in range(n)
    )

    # ---- Cost term: total travel distance ----
    # bin index i (0..n-1) maps to actual node id i+1 (since node 0 = depot)
    cost = 0
    for p in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dist_matrix[i + 1][j + 1]
                    cost += d * x[i, p] * x[j, p + 1]

    # depot -> first bin, and last bin -> depot
    for i in range(n):
        cost += dist_matrix[0][i + 1] * x[i, 0]
        cost += dist_matrix[i + 1][0] * x[i, n - 1]

    H = A * (bin_once + pos_once) + B * cost
    model = H.compile()
    qubo, offset = model.to_qubo()
    return model, qubo, offset


def decode_solution(sample, num_bins):
    """Convert PyQUBO binary sample back into a route [0, bin,...,bin, 0]."""
    order = [None] * num_bins
    for key, val in sample.items():
        if val == 1 and key.startswith('x['):
            i, p = key[2:-1].split('][')
            order[int(p)] = int(i) + 1  # map back to actual bin id
    if None in order:
        return None  # infeasible decode
    return [0] + order + [0]


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)

    print(f"Building QUBO for {NUM_BINS} bins ({NUM_BINS**2} binary variables)...")
    model, qubo, offset = build_qubo(dist_matrix, NUM_BINS)
    print(f"QUBO has {len(qubo)} nonzero terms, offset={offset:.2f}")

    # Sanity check: solve the QUBO with a classical simulated annealer
    # (neal) BEFORE trusting QAOA -- this validates the formulation itself
    sampler = neal.SimulatedAnnealingSampler()
    bqm = model.to_bqm()
    sampleset = sampler.sample(bqm, num_reads=200)
    best = sampleset.first.sample

    route = decode_solution(best, NUM_BINS)
    print(f"\nSimulated-annealing decoded route: {route}")

    if route is not None and len(set(route[1:-1])) == NUM_BINS:
        cost = route_cost(route, dist_matrix)
        print(f"Route cost: {cost:.3f}  (feasible: visits all bins exactly once)")
    else:
        print("INFEASIBLE decode -- constraint penalty (A) likely needs tuning")

    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"Known optimal cost (from Stage 2): {exact_cost:.3f}")
