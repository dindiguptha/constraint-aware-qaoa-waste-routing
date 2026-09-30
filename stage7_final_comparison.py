"""
Stage 7: Final Consolidated Comparison (4 bins, single truck)
Runs every validated method on the SAME problem instance and produces
one summary table + one bar chart for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt

from stage1_setup import generate_instance, build_distance_matrix, plot_instance
from stage2_baselines import nearest_neighbor, exact_solver, route_cost
from stage3_qubo import build_qubo, decode_solution
from stage3b_sim_bifurcation import run_simulated_bifurcation
from stage4a_qubo_to_ising import qubo_to_ising
from stage4b_qaoa import run_qaoa
from stage5_qaoansatz import (
    build_cost_only_qubo, build_var_to_wire, run_qaoansatz, decode_bitstring
)

NUM_BINS = 4

def feasibility_and_best(bitstrings_or_samples, decode_fn, dist_matrix, num_bins):
    costs = []
    feasible = 0
    for item in bitstrings_or_samples:
        route = decode_fn(item)
        if route is not None and len(set(route[1:-1])) == num_bins:
            feasible += 1
            costs.append(route_cost(route, dist_matrix))
    rate = feasible / len(bitstrings_or_samples) * 100
    best = min(costs) if costs else None
    return rate, best


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)
    plot_instance(nodes, save_path="/home/dmacs/Downloads/new/code_package/final_instance.png")

    results = {}

# 1. Exact
_, exact_cost = exact_solver(dist_matrix, NUM_BINS)
results['Exact (brute-force)'] = {'cost': exact_cost, 'feasibility': 100.0}

# 2. Nearest-neighbor
nn_route = nearest_neighbor(dist_matrix, NUM_BINS)
nn_cost = route_cost(nn_route, dist_matrix)
results['Nearest-Neighbor'] = {'cost': nn_cost, 'feasibility': 100.0}

# 3. Simulated Bifurcation (quantum-inspired)
model, qubo, qubo_offset = build_qubo(dist_matrix, NUM_BINS)
sb_sample = run_simulated_bifurcation(qubo, NUM_BINS)
sb_route = decode_solution(sb_sample, NUM_BINS)
sb_cost = route_cost(sb_route, dist_matrix)
results['Simulated Bifurcation'] = {'cost': sb_cost, 'feasibility': 100.0}

# 4. Standard QAOA (penalty-based)
h, J, offset = qubo_to_ising(qubo, qubo_offset)
var_names = sorted(set(h.keys()) | set(v for pair in J.keys() for v in pair))
var_to_wire = {name: i for i, name in enumerate(var_names)}
_, _, samples_std = run_qaoa(h, J, var_to_wire, len(var_to_wire))
wire_to_var = {v: k for k, v in var_to_wire.items()}
bitstrings_std = ["".join(str(b) for b in row) for row in samples_std]

def decode_std(bstr):
    sample_dict = {wire_to_var[i]: int(bit) for i, bit in enumerate(bstr)}
    return decode_solution(sample_dict, NUM_BINS)

rate_std, best_std = feasibility_and_best(bitstrings_std, decode_std, dist_matrix, NUM_BINS)
results['Standard QAOA'] = {'cost': best_std, 'feasibility': rate_std}

# 5. QAOAnsatz (XY-mixer)
model2, qubo2, offset2 = build_cost_only_qubo(dist_matrix, NUM_BINS)
h2, J2, off2 = qubo_to_ising(qubo2, offset2)
var_to_wire2 = build_var_to_wire(NUM_BINS)
wire_to_var2 = {v: k for k, v in var_to_wire2.items()}
_, samples_ansatz = run_qaoansatz(h2, J2, var_to_wire2, NUM_BINS)
bitstrings_ansatz = ["".join(str(b) for b in row) for row in samples_ansatz]

def decode_ansatz(bstr):
    return decode_bitstring(bstr, wire_to_var2, NUM_BINS)

rate_ansatz, best_ansatz = feasibility_and_best(bitstrings_ansatz, decode_ansatz, dist_matrix, NUM_BINS)
results['QAOAnsatz (XY-mixer)'] = {'cost': best_ansatz, 'feasibility': rate_ansatz}

# ---- Print final table ----
print(f"\n{'Method':<25}{'Best Cost':>12}{'Gap %':>10}{'Feasibility %':>16}")
print("-" * 63)
for name, r in results.items():
    gap = (r['cost'] - exact_cost) / exact_cost * 100 if r['cost'] else float('nan')
    print(f"{name:<25}{r['cost']:>12.3f}{gap:>10.2f}{r['feasibility']:>16.2f}")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

names = list(results.keys())
costs = [results[n]['cost'] for n in names]
feas = [results[n]['feasibility'] for n in names]

axes[0].bar(names, costs, color=['gray', 'gray', 'orange', 'steelblue', 'green'])
axes[0].axhline(exact_cost, color='red', linestyle='--', label='Optimal')
axes[0].set_ylabel('Route Cost')
axes[0].set_title('Best Route Cost by Method')
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend()

axes[1].bar(names, feas, color=['gray', 'gray', 'orange', 'steelblue', 'green'])
axes[1].set_ylabel('Feasibility Rate (%)')
axes[1].set_title('Feasibility Rate by Method')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
