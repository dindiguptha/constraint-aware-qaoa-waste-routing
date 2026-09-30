"""
Stage 8: Multi-Trial Feasibility Analysis

WHY THIS EXISTS: earlier runs of QAOAnsatz gave different feasibility rates
(15.2% vs 10.80%) across separate executions, even on the identical problem
instance. The cause: QAOA TRAINING (the parameter optimization loop) is
fully deterministic here (adjoint differentiation, fixed initial params,
Adam optimizer -- no randomness). The variance comes entirely from the
final SHOT-SAMPLING step: measuring the trained circuit 500 times draws
random outcomes from its output probability distribution, and we never
fixed a seed for that random draw.

FIX: train the circuit ONCE (deterministic), then draw N_TRIALS independent
batches of 500 shots each, using a different explicit seed per batch. This
correctly reports the sampling variance as mean +/- std, instead of
presenting one arbitrary draw as "the" result.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from stage1_setup import generate_instance, build_distance_matrix
from stage3_qubo import build_qubo, decode_solution
from stage4a_qubo_to_ising import qubo_to_ising
from stage2_baselines import exact_solver, route_cost
from stage5_qaoansatz import (
    build_cost_only_qubo, build_var_to_wire, decode_bitstring,
    initial_feasible_permutation, NUM_LAYERS as NUM_LAYERS_ANSATZ
)
from stage4b_qaoa import build_hamiltonians, NUM_LAYERS as NUM_LAYERS_STD, STEPS, LEARNING_RATE

NUM_BINS = 4
N_TRIALS = 10        # number of independent 500-shot sampling batches
SHOTS_PER_TRIAL = 500


# ---------------------------------------------------------------
# Standard QAOA: train once, sample N_TRIALS times with different seeds
# ---------------------------------------------------------------
def train_and_multisample_standard(h, J, var_to_wire, n_wires):
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

    params = pnp.array([[0.5] * NUM_LAYERS_STD, [0.5] * NUM_LAYERS_STD], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    print("Training standard QAOA (once, deterministic)...")
    for step in range(STEPS):
        params, cost = opt.step_and_cost(circuit, params)
    print(f"  final cost = {cost:.4f}")

    all_samples = []
    for trial in range(N_TRIALS):
        dev_shots = qml.device("lightning.qubit", wires=n_wires,
                                shots=SHOTS_PER_TRIAL, seed=1000 + trial)

        @qml.qnode(dev_shots)
        def sample_circuit(params):
            for w in range(n_wires):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, NUM_LAYERS_STD, params[0], params[1])
            return qml.sample(wires=range(n_wires))

        samples = sample_circuit(params)
        all_samples.append(samples)
    return all_samples


# ---------------------------------------------------------------
# QAOAnsatz: train once, sample N_TRIALS times with different seeds
# ---------------------------------------------------------------
def train_and_multisample_ansatz(h, J, var_to_wire, n_bins):
    n_wires = n_bins * n_bins
    coeffs, obs = [], []
    for var, coeff in h.items():
        coeffs.append(coeff); obs.append(qml.PauliZ(var_to_wire[var]))
    for (v1, v2), coeff in J.items():
        coeffs.append(coeff)
        obs.append(qml.PauliZ(var_to_wire[v1]) @ qml.PauliZ(var_to_wire[v2]))
    cost_h = qml.Hamiltonian(coeffs, obs, grouping_type=None)

    init_wires = initial_feasible_permutation(n_bins)
    dev = qml.device("lightning.qubit", wires=n_wires)

    def ring_xy_mixer(beta):
        for p in range(n_bins):
            group = [p * n_bins + i for i in range(n_bins)]
            for k in range(n_bins):
                w1, w2 = group[k], group[(k + 1) % n_bins]
                qml.IsingXY(beta, wires=[w1, w2])

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        for w in init_wires:
            qml.PauliX(wires=w)
        for layer in range(NUM_LAYERS_ANSATZ):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            ring_xy_mixer(params[1][layer])
        return qml.expval(cost_h)

    params = pnp.array([[0.5] * NUM_LAYERS_ANSATZ, [0.5] * NUM_LAYERS_ANSATZ], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    print("Training QAOAnsatz (once, deterministic)...")
    for step in range(STEPS):
        params, cost = opt.step_and_cost(circuit, params)
    print(f"  final cost = {cost:.4f}")

    all_samples = []
    for trial in range(N_TRIALS):
        dev_shots = qml.device("lightning.qubit", wires=n_wires,
                                shots=SHOTS_PER_TRIAL, seed=2000 + trial)

        @qml.qnode(dev_shots)
        def sample_circuit(params):
            for w in init_wires:
                qml.PauliX(wires=w)
            for layer in range(NUM_LAYERS_ANSATZ):
                qml.qaoa.cost_layer(params[0][layer], cost_h)
                ring_xy_mixer(params[1][layer])
            return qml.sample(wires=range(n_wires))

        samples = sample_circuit(params)
        all_samples.append(samples)
    return all_samples, var_to_wire


def feasibility_per_trial(all_samples, decode_fn, dist_matrix, num_bins):
    rates, bests = [], []
    for samples in all_samples:
        bitstrings = ["".join(str(b) for b in row) for row in samples]
        feasible_costs = []
        for bstr in bitstrings:
            route = decode_fn(bstr)
            if route is not None and len(set(route[1:-1])) == num_bins:
                feasible_costs.append(route_cost(route, dist_matrix))
        rate = len(feasible_costs) / len(bitstrings) * 100
        rates.append(rate)
        bests.append(min(feasible_costs) if feasible_costs else None)
    return rates, bests


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)
    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)

    # ---- Standard QAOA ----
    model, qubo, qubo_offset = build_qubo(dist_matrix, NUM_BINS)
    h, J, offset = qubo_to_ising(qubo, qubo_offset)
    var_names = sorted(set(h.keys()) | set(v for pair in J.keys() for v in pair))
    var_to_wire = {name: i for i, name in enumerate(var_names)}
    wire_to_var = {v: k for k, v in var_to_wire.items()}

    std_samples = train_and_multisample_standard(h, J, var_to_wire, len(var_to_wire))

    def decode_std(bstr):
        sample_dict = {wire_to_var[i]: int(bit) for i, bit in enumerate(bstr)}
        return decode_solution(sample_dict, NUM_BINS)

    std_rates, std_bests = feasibility_per_trial(std_samples, decode_std, dist_matrix, NUM_BINS)

    # ---- QAOAnsatz ----
    model2, qubo2, offset2 = build_cost_only_qubo(dist_matrix, NUM_BINS)
    h2, J2, off2 = qubo_to_ising(qubo2, offset2)
    var_to_wire2 = build_var_to_wire(NUM_BINS)
    wire_to_var2 = {v: k for k, v in var_to_wire2.items()}

    ansatz_samples, _ = train_and_multisample_ansatz(h2, J2, var_to_wire2, NUM_BINS)

    def decode_ansatz(bstr):
        return decode_bitstring(bstr, wire_to_var2, NUM_BINS)

    ansatz_rates, ansatz_bests = feasibility_per_trial(ansatz_samples, decode_ansatz, dist_matrix, NUM_BINS)

    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"Multi-trial results ({N_TRIALS} independent {SHOTS_PER_TRIAL}-shot batches)")
    print(f"{'='*60}")

    print(f"\nStandard QAOA feasibility rates per trial: {[f'{r:.2f}' for r in std_rates]}")
    print(f"  Mean: {np.mean(std_rates):.2f}%   Std Dev: {np.std(std_rates):.2f}%")
    valid_std_bests = [b for b in std_bests if b is not None]
    if valid_std_bests:
        print(f"  Best cost found across trials: {min(valid_std_bests):.3f} "
              f"(optimal = {exact_cost:.3f})")

    print(f"\nQAOAnsatz feasibility rates per trial: {[f'{r:.2f}' for r in ansatz_rates]}")
    print(f"  Mean: {np.mean(ansatz_rates):.2f}%   Std Dev: {np.std(ansatz_rates):.2f}%")
    valid_ansatz_bests = [b for b in ansatz_bests if b is not None]
    if valid_ansatz_bests:
        print(f"  Best cost found across trials: {min(valid_ansatz_bests):.3f} "
              f"(optimal = {exact_cost:.3f})")

    print(f"\nImprovement factor (mean): {np.mean(ansatz_rates)/np.mean(std_rates):.2f}x")
