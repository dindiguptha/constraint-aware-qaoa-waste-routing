"""
Stage 10: Multi-Instance Generalization Test (streamlined, 1 sample/instance)

Runs the validated 4-bin pipeline across N_INSTANCES different random
problem layouts (different seeds), training QAOA/QAOAnsatz once per
instance and sampling 500 shots once per instance -- to answer "does
this result generalize across problems, or was one map lucky?"

NOTE: superseded by stage11_full_study.py, which repeats this design
with 10 shot-batches per instance (100 total measurements/method) and
decomposes variance into shot-noise vs. instance-to-instance components.
Kept here as the original, faster single-sample-per-instance version.
"""

import numpy as np

from stage1_setup import generate_instance, build_distance_matrix
from stage2_baselines import exact_solver, route_cost
from stage3_qubo import build_qubo, decode_solution
from stage4a_qubo_to_ising import qubo_to_ising
from stage4b_qaoa import build_hamiltonians, NUM_LAYERS as NUM_LAYERS_STD, STEPS, LEARNING_RATE
from stage5_qaoansatz import (
    build_cost_only_qubo, build_var_to_wire, decode_bitstring,
    initial_feasible_permutation, NUM_LAYERS as NUM_LAYERS_ANSATZ
)
import pennylane as qml
from pennylane import numpy as pnp

NUM_BINS = 4
N_INSTANCES = 10
SHOTS = 500


def run_standard_qaoa_once(dist_matrix, seed):
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
    best = min(feasible_costs) if feasible_costs else None
    return rate, best


def run_qaoansatz_once(dist_matrix, seed):
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
    best = min(feasible_costs) if feasible_costs else None
    return rate, best


if __name__ == "__main__":
    std_rates, std_gaps = [], []
    ansatz_rates, ansatz_gaps = [], []

    for i in range(N_INSTANCES):
        seed = 100 + i
        nodes = generate_instance(NUM_BINS, seed=seed)
        dist_matrix = build_distance_matrix(nodes)
        _, exact_cost = exact_solver(dist_matrix, NUM_BINS)

        std_rate, std_best = run_standard_qaoa_once(dist_matrix, seed=seed)
        ansatz_rate, ansatz_best = run_qaoansatz_once(dist_matrix, seed=seed)

        std_gap = (std_best - exact_cost) / exact_cost * 100 if std_best else None
        ansatz_gap = (ansatz_best - exact_cost) / exact_cost * 100 if ansatz_best else None

        std_rates.append(std_rate)
        ansatz_rates.append(ansatz_rate)
        if std_gap is not None: std_gaps.append(std_gap)
        if ansatz_gap is not None: ansatz_gaps.append(ansatz_gap)

        print(f"Instance {i+1}/{N_INSTANCES} (seed={seed}): "
              f"optimal={exact_cost:.2f} | "
              f"StdQAOA feas={std_rate:.1f}% gap={std_gap if std_gap is not None else 'N/A'} | "
              f"QAOAnsatz feas={ansatz_rate:.1f}% gap={ansatz_gap if ansatz_gap is not None else 'N/A'}")

    print(f"\n{'='*70}")
    print(f"SUMMARY ACROSS {N_INSTANCES} RANDOM INSTANCES")
    print(f"{'='*70}")
    print(f"Standard QAOA  : feasibility = {np.mean(std_rates):.2f}% +/- {np.std(std_rates):.2f}%")
    if std_gaps:
        print(f"                 optimality gap (when feasible) = {np.mean(std_gaps):.2f}% +/- {np.std(std_gaps):.2f}%")
    print(f"QAOAnsatz      : feasibility = {np.mean(ansatz_rates):.2f}% +/- {np.std(ansatz_rates):.2f}%")
    if ansatz_gaps:
        print(f"                 optimality gap (when feasible) = {np.mean(ansatz_gaps):.2f}% +/- {np.std(ansatz_gaps):.2f}%")
    print(f"\nImprovement factor (mean feasibility): {np.mean(ansatz_rates)/np.mean(std_rates):.2f}x")
