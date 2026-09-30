"""
Stage 4c: Feasibility Rate Analysis
Instead of only looking at the single most-frequent bitstring, this
decodes EVERY sampled bitstring from QAOA and checks:
  - is it a valid route (each bin visited exactly once)?
  - if valid, what's its cost vs the known optimal?

This gives the actual metrics the paper needs: feasibility rate,
average cost of feasible samples, and best feasible cost found.
"""

from collections import Counter
import numpy as np

from stage1_setup import generate_instance, build_distance_matrix
from stage3_qubo import build_qubo, decode_solution
from stage4a_qubo_to_ising import qubo_to_ising
from stage4b_qaoa import run_qaoa, NUM_BINS
from stage2_baselines import exact_solver, route_cost


def analyze_samples(samples, var_to_wire, num_bins, dist_matrix):
    wire_to_var = {v: k for k, v in var_to_wire.items()}
    bitstrings = ["".join(str(b) for b in row) for row in samples]

    total = len(bitstrings)
    feasible_costs = []
    feasible_count = 0

    for bstr in bitstrings:
        sample_dict = {wire_to_var[i]: int(bit) for i, bit in enumerate(bstr)}
        route = decode_solution(sample_dict, num_bins)
        if route is not None and len(set(route[1:-1])) == num_bins:
            feasible_count += 1
            feasible_costs.append(route_cost(route, dist_matrix))

    feasibility_rate = feasible_count / total * 100

    result = {
        "total_samples": total,
        "feasible_count": feasible_count,
        "feasibility_rate_pct": feasibility_rate,
        "unique_bitstrings": len(set(bitstrings)),
    }
    if feasible_costs:
        result["best_feasible_cost"] = min(feasible_costs)
        result["avg_feasible_cost"] = float(np.mean(feasible_costs))
        result["worst_feasible_cost"] = max(feasible_costs)
    return result


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)
    model, qubo, qubo_offset = build_qubo(dist_matrix, NUM_BINS)
    h, J, offset = qubo_to_ising(qubo, qubo_offset)

    var_names = sorted(set(h.keys()) | set(v for pair in J.keys() for v in pair))
    var_to_wire = {name: i for i, name in enumerate(var_names)}
    n_wires = len(var_to_wire)

    params, cost_history, samples = run_qaoa(h, J, var_to_wire, n_wires)

    result = analyze_samples(samples, var_to_wire, NUM_BINS, dist_matrix)

    print("\n=== Feasibility Analysis (all 500 shots) ===")
    print(f"Total samples:        {result['total_samples']}")
    print(f"Unique bitstrings:    {result['unique_bitstrings']}")
    print(f"Feasible count:       {result['feasible_count']}")
    print(f"Feasibility rate:     {result['feasibility_rate_pct']:.2f}%")

    if result['feasible_count'] > 0:
        print(f"Best feasible cost:   {result['best_feasible_cost']:.3f}")
        print(f"Avg feasible cost:    {result['avg_feasible_cost']:.3f}")
        print(f"Worst feasible cost:  {result['worst_feasible_cost']:.3f}")
    else:
        print("No feasible solutions found in this sample set.")

    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"\nKnown optimal cost:   {exact_cost:.3f}")

    if result['feasible_count'] > 0:
        gap = (result['best_feasible_cost'] - exact_cost) / exact_cost * 100
        print(f"Best QAOA feasible sample is {gap:.1f}% above optimal")
