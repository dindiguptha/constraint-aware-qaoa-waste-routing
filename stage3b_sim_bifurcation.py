"""
Stage 3b: Quantum-Inspired Baseline (Simulated Bifurcation)
Solves the SAME QUBO (from Stage 3) using Simulated Bifurcation --
a quantum-inspired classical algorithm that mimics quantum annealing
dynamics but runs entirely on classical hardware (CPU/GPU, no qubits).

This gives us a 3-way comparison for the paper:
  Classical (nearest-neighbor / exact)  vs
  Quantum-Inspired (Simulated Bifurcation)  vs
  True Quantum (QAOA, Stage 4)
"""

import numpy as np
import torch
import simulated_bifurcation as sb

from stage1_setup import generate_instance, build_distance_matrix, NUM_BINS
from stage2_baselines import route_cost, exact_solver
from stage3_qubo import build_qubo, decode_solution


def qubo_dict_to_matrix(qubo, num_vars):
    """Convert PyQUBO's sparse dict QUBO {(var1,var2): coeff} into a dense
    matrix indexed 0..num_vars-1, plus a mapping from var name -> index."""
    var_names = sorted(set(k for pair in qubo.keys() for k in pair))
    idx = {name: i for i, name in enumerate(var_names)}
    n = len(var_names)
    Q = np.zeros((n, n))
    for (v1, v2), coeff in qubo.items():
        i, j = idx[v1], idx[v2]
        Q[i, j] += coeff
    return Q, idx


def run_simulated_bifurcation(qubo, num_bins):
    Q, idx = qubo_dict_to_matrix(qubo, num_bins * num_bins)
    Q_tensor = torch.tensor(Q, dtype=torch.float32)

    best_vector, best_value = sb.minimize(
        Q_tensor, domain="binary", agents=64, max_steps=5000, verbose=False
    )

    # decode back into var_name -> 0/1
    inv_idx = {i: name for name, i in idx.items()}
    sample = {inv_idx[i]: int(best_vector[i].item()) for i in range(len(inv_idx))}
    return sample


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)
    model, qubo, offset = build_qubo(dist_matrix, NUM_BINS)

    print("Running Simulated Bifurcation on the QUBO...")
    sample = run_simulated_bifurcation(qubo, NUM_BINS)
    route = decode_solution(sample, NUM_BINS)

    print(f"\nSimulated Bifurcation decoded route: {route}")

    if route is not None and len(set(route[1:-1])) == NUM_BINS:
        cost = route_cost(route, dist_matrix)
        print(f"Route cost: {cost:.3f}  (feasible)")
    else:
        print("INFEASIBLE decode -- may need more agents/steps or penalty retuning")

    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"Known optimal cost: {exact_cost:.3f}")
