"""
Stage 11: Full Multi-Instance x Multi-Trial Study (100 measurements/method)

For each of 10 random problem instances, train each circuit ONCE
(deterministic), then draw 10 independent 500-shot batches -> 100 total
feasibility measurements per method. Decomposes variance into:
  1. Shot-noise variance (batch-to-batch, within a fixed trained circuit)
  2. Instance-to-instance variance (between different problem layouts)
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from stage1_setup import generate_instance, build_distance_matrix
from stage2_baselines import exact_solver, route_cost
from stage3_qubo import build_qubo, decode_solution
from stage4a_qubo_to_ising import qubo_to_ising
from stage4b_qaoa import build_hamiltonians, NUM_LAYERS as NUM_LAYERS_STD, STEPS, LEARNING_RATE
from stage5_qaoansatz import (
    build_cost_only_qubo, build_var_to_wire, decode_bitstring,
    initial_feasible_permutation, NUM_LAYERS as NUM_LAYERS_ANSATZ
)

NUM_BINS = 4
N_INSTANCES = 10
N_BATCHES_PER_INSTANCE = 10
SHOTS = 500


def train_standard(dist_matrix):
    model, qubo, qubo_offset = build_qubo(dist_matrix, NUM_BINS)
    h, J, offset = qubo_to_ising(qubo, qubo_offset)
    var_names = sorted(set(h.keys()) | set(v for pair in J.keys() for v in pair))
    var_to_wire = {name: i for i, name in enumerate(var_names)}
    wire_to_var = {v: k for k, v in var_to_wire.items()}
    n_wires = len(var_to_wire)

    cost_h, mixer_h = build_hamiltonians(h, J, var_to_wire)
    dev = qml.device("lightning.qubit", wires=n_wires)

    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        for w in range(n_wires):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, NUM_LAYERS_STD, params[0], params[1])
        return qml.expval(cost_h)

    params = pnp.array([[0.5]*NUM_LAYERS_STD, [0.5]*NUM_LAYERS_STD], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    for _ in range(STEPS):
        params, _ = opt.step_and_cost(circuit, params)

    return params, qaoa_layer, n_wires, wire_to_var


def sample_standard(params, qaoa_layer, n_wires, wire_to_var, dist_matrix, seed):
    dev_shots = qml.device("lightning.qubit", wires=n_wires, shots=SHOTS, seed=seed)

    @qml.qnode(dev_shots)
    def sample_circuit(params):
        for w in range(n_wires):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, NUM_LAYERS_STD, params[0], params[1])
        return qml.sample(wires=range(n_wires))

    samples = sample_circuit(params)
    bitstrings = ["".join(str(b) for b in row) for row in samples]
    feasible_costs = []
    for bstr in bitstrings:
        sample_dict = {wire_to_var[i]: int(bit) for i, bit in enumerate(bstr)}
        route = decode_solution(sample_dict, NUM_BINS)
        if route is not None and len(set(route[1:-1])) == NUM_BINS:
            feasible_costs.append(route_cost(route, dist_matrix))
    rate = len(feasible_costs) / len(bitstrings) * 100
    return rate


def train_ansatz(dist_matrix):
    model2, qubo2, offset2 = build_cost_only_qubo(dist_matrix, NUM_BINS)
    h2, J2, off2 = qubo_to_ising(qubo2, offset2)
    var_to_wire2 = build_var_to_wire(NUM_BINS)
    wire_to_var2 = {v: k for k, v in var_to_wire2.items()}
    n_wires = NUM_BINS * NUM_BINS

    coeffs, obs = [], []
    for var, coeff in h2.items():
        coeffs.append(coeff); obs.append(qml.PauliZ(var_to_wire2[var]))
    for (v1, v2), coeff in J2.items():
        coeffs.append(coeff)
        obs.append(qml.PauliZ(var_to_wire2[v1]) @ qml.PauliZ(var_to_wire2[v2]))
    cost_h = qml.Hamiltonian(coeffs, obs, grouping_type=None)

    init_wires = initial_feasible_permutation(NUM_BINS)
    dev = qml.device("lightning.qubit", wires=n_wires)

    def ring_xy_mixer(beta):
        for p in range(NUM_BINS):
            group = [p * NUM_BINS + i for i in range(NUM_BINS)]
            for k in range(NUM_BINS):
                w1, w2 = group[k], group[(k + 1) % NUM_BINS]
                qml.IsingXY(beta, wires=[w1, w2])

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        for w in init_wires:
            qml.PauliX(wires=w)
        for layer in range(NUM_LAYERS_ANSATZ):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            ring_xy_mixer(params[1][layer])
        return qml.expval(cost_h)

    params = pnp.array([[0.5]*NUM_LAYERS_ANSATZ, [0.5]*NUM_LAYERS_ANSATZ], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    for _ in range(STEPS):
        params, _ = opt.step_and_cost(circuit, params)

    return params, cost_h, init_wires, ring_xy_mixer, n_wires, wire_to_var2


def sample_ansatz(params, cost_h, init_wires, ring_xy_mixer, n_wires, wire_to_var2, dist_matrix, seed):
    dev_shots = qml.device("lightning.qubit", wires=n_wires, shots=SHOTS, seed=seed)

    @qml.qnode(dev_shots)
    def sample_circuit(params):
        for w in init_wires:
            qml.PauliX(wires=w)
        for layer in range(NUM_LAYERS_ANSATZ):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            ring_xy_mixer(params[1][layer])
        return qml.sample(wires=range(n_wires))

    samples = sample_circuit(params)
    bitstrings = ["".join(str(b) for b in row) for row in samples]
    feasible_costs = []
    for bstr in bitstrings:
        route = decode_bitstring(bstr, wire_to_var2, NUM_BINS)
        if route is not None and len(set(route[1:-1])) == NUM_BINS:
            feasible_costs.append(route_cost(route, dist_matrix))
    rate = len(feasible_costs) / len(bitstrings) * 100
    return rate


if __name__ == "__main__":
    std_all_rates, ansatz_all_rates = [], []
    std_instance_means, ansatz_instance_means = [], []
    std_within_stds, ansatz_within_stds = [], []

    for i in range(N_INSTANCES):
        inst_seed = 100 + i
        nodes = generate_instance(NUM_BINS, seed=inst_seed)
        dist_matrix = build_distance_matrix(nodes)
        _, exact_cost = exact_solver(dist_matrix, NUM_BINS)

        std_params, std_layer, std_wires, std_w2v = train_standard(dist_matrix)
        ans_params, ans_cost_h, ans_init, ans_mixer, ans_wires, ans_w2v = train_ansatz(dist_matrix)

        std_batch_rates, ansatz_batch_rates = [], []
        for b in range(N_BATCHES_PER_INSTANCE):
            batch_seed = 1000 * (i + 1) + b
            r_std = sample_standard(std_params, std_layer, std_wires, std_w2v, dist_matrix, batch_seed)
            r_ans = sample_ansatz(ans_params, ans_cost_h, ans_init, ans_mixer, ans_wires, ans_w2v, dist_matrix, batch_seed)
            std_batch_rates.append(r_std)
            ansatz_batch_rates.append(r_ans)
            std_all_rates.append(r_std)
            ansatz_all_rates.append(r_ans)

        std_instance_means.append(np.mean(std_batch_rates))
        ansatz_instance_means.append(np.mean(ansatz_batch_rates))
        std_within_stds.append(np.std(std_batch_rates))
        ansatz_within_stds.append(np.std(ansatz_batch_rates))

        print(f"Instance {i+1}/{N_INSTANCES} (seed={inst_seed}, optimal={exact_cost:.2f}): "
              f"StdQAOA mean={np.mean(std_batch_rates):.2f}% (within-SD={np.std(std_batch_rates):.2f}%) | "
              f"QAOAnsatz mean={np.mean(ansatz_batch_rates):.2f}% (within-SD={np.std(ansatz_batch_rates):.2f}%)",
              flush=True)

    print(f"\n{'='*75}")
    print(f"FULL STUDY: {N_INSTANCES} instances x {N_BATCHES_PER_INSTANCE} batches = "
          f"{N_INSTANCES*N_BATCHES_PER_INSTANCE} total measurements per method")
    print(f"{'='*75}")

    print(f"\n--- Standard QAOA ---")
    print(f"Pooled (all 100): mean={np.mean(std_all_rates):.2f}%  SD={np.std(std_all_rates):.2f}%")
    print(f"Instance-to-instance: mean of instance-means={np.mean(std_instance_means):.2f}%  "
          f"SD of instance-means={np.std(std_instance_means):.2f}%")
    print(f"Shot-noise (within-instance): avg within-instance SD={np.mean(std_within_stds):.2f}%")

    print(f"\n--- QAOAnsatz (XY-mixer) ---")
    print(f"Pooled (all 100): mean={np.mean(ansatz_all_rates):.2f}%  SD={np.std(ansatz_all_rates):.2f}%")
    print(f"Instance-to-instance: mean of instance-means={np.mean(ansatz_instance_means):.2f}%  "
          f"SD of instance-means={np.std(ansatz_instance_means):.2f}%")
    print(f"Shot-noise (within-instance): avg within-instance SD={np.mean(ansatz_within_stds):.2f}%")

    print(f"\nImprovement factor (pooled mean): {np.mean(ansatz_all_rates)/np.mean(std_all_rates):.2f}x")

    import json
    with open("stage11_raw_results.json", "w") as f:
        json.dump({
            "std_all_rates": std_all_rates,
            "ansatz_all_rates": ansatz_all_rates,
            "std_instance_means": std_instance_means,
            "ansatz_instance_means": ansatz_instance_means,
        }, f, indent=2)
    print("\nRaw results saved to stage11_raw_results.json")
