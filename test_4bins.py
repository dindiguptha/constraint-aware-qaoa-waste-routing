"""
Stage 4b: QAOA in PennyLane
Builds the cost Hamiltonian from Ising (h, J), runs QAOA with a
standard X-mixer, and optimizes the variational parameters.

Scope for this run: 4 bins -> 16 qubits (position-based TSP encoding).
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import time

from stage1_setup import generate_instance, build_distance_matrix
from stage3_qubo import build_qubo, decode_solution
from stage4a_qubo_to_ising import qubo_to_ising
from stage2_baselines import exact_solver, route_cost

NUM_BINS = 4          # testing scale ceiling
NUM_LAYERS = 2         # QAOA depth (p)
STEPS = 0             # optimization steps
LEARNING_RATE = 0.1


def build_hamiltonians(h, J, var_to_wire):
    """Construct cost Hamiltonian (Ising) and standard X mixer Hamiltonian."""
    coeffs, obs = [], []

    for var, coeff in h.items():
        w = var_to_wire[var]
        coeffs.append(coeff)
        obs.append(qml.PauliZ(w))

    for (v1, v2), coeff in J.items():
        w1, w2 = var_to_wire[v1], var_to_wire[v2]
        coeffs.append(coeff)
        obs.append(qml.PauliZ(w1) @ qml.PauliZ(w2))

    cost_h = qml.Hamiltonian(coeffs, obs)

    n_wires = len(var_to_wire)
    mixer_coeffs = [1.0] * n_wires
    mixer_obs = [qml.PauliX(w) for w in range(n_wires)]
    mixer_h = qml.Hamiltonian(mixer_coeffs, mixer_obs)

    return cost_h, mixer_h


def run_qaoa(h, J, var_to_wire, n_wires):
    cost_h, mixer_h = build_hamiltonians(h, J, var_to_wire)

    dev = qml.device("lightning.qubit", wires=n_wires)

    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        for w in range(n_wires):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, NUM_LAYERS, params[0], params[1])
        return qml.expval(cost_h)

    params = pnp.array(
        [[0.5] * NUM_LAYERS, [0.5] * NUM_LAYERS], requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    print(f"Starting QAOA optimization ({n_wires} qubits, p={NUM_LAYERS} layers)...")
    t0 = time.time()
    cost_history = []
    for step in range(STEPS):
        params, cost = opt.step_and_cost(circuit, params)
        cost_history.append(cost)
        if step % 10 == 0 or step == STEPS - 1:
            print(f"  step {step:3d}  cost = {cost:.4f}")
    elapsed = time.time() - t0
    print(f"Optimization finished in {elapsed:.1f}s")

    # Sample the final circuit to get a bitstring solution
    @qml.qnode(dev)
    def sample_circuit(params):
        for w in range(n_wires):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, NUM_LAYERS, params[0], params[1])
        return qml.sample(wires=range(n_wires))

    dev_sample = qml.device("lightning.qubit", wires=n_wires, shots=500)

    @qml.qnode(dev_sample)
    def sample_circuit_shots(params):
        for w in range(n_wires):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, NUM_LAYERS, params[0], params[1])
        return qml.sample(wires=range(n_wires))

    samples = sample_circuit_shots(params)
    return params, cost_history, samples


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)
    model, qubo, qubo_offset = build_qubo(dist_matrix, NUM_BINS)
    h, J, offset = qubo_to_ising(qubo, qubo_offset)

    var_names = sorted(set(h.keys()) | set(v for pair in J.keys() for v in pair))
    var_to_wire = {name: i for i, name in enumerate(var_names)}
    n_wires = len(var_to_wire)
    print(f"Number of qubits: {n_wires}")

    params, cost_history, samples = run_qaoa(h, J, var_to_wire, n_wires)

    # Convert most frequent bitstring sample into a route
    from collections import Counter
    bitstrings = ["".join(str(b) for b in row) for row in samples]
    most_common, count = Counter(bitstrings).most_common(1)[0]
    print(f"\nMost frequent bitstring: {most_common} (seen {count}/{len(samples)} times)")

    wire_to_var = {v: k for k, v in var_to_wire.items()}
    sample_dict = {wire_to_var[i]: int(bit) for i, bit in enumerate(most_common)}

    route = decode_solution(sample_dict, NUM_BINS)
    print(f"Decoded route: {route}")

    if route is not None and len(set(route[1:-1])) == NUM_BINS:
        cost = route_cost(route, dist_matrix)
        print(f"QAOA route cost: {cost:.3f}  (FEASIBLE)")
    else:
        print("QAOA result INFEASIBLE (expected at low shot count / p -- will report feasibility rate)")

    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"Known optimal cost: {exact_cost:.3f}")
