"""
Stage 5: XY-Mixer QAOAnsatz (following Huang, Matsuyama & Yamashiro, arXiv:2503.17051)

KEY IDEA:
Standard QAOA (Stage 4) used PENALTY terms to *discourage* the circuit from
landing on infeasible states (bin not visited exactly once, etc). This only
discourages -- it doesn't prevent -- infeasible outputs, which is why we saw
only 2% feasibility.

QAOAnsatz instead designs the MIXER Hamiltonian so the circuit can ONLY ever
move between feasible states for the "each position holds exactly one bin"
(row) constraint. This is done using a ring of IsingXY gates within each
position's group of qubits, which preserves the total excitation count
(number of 1s) within that group -- i.e. if a position starts with exactly
one bin "on", it can only ever have exactly one bin "on", for any mixer angle.

NOTE (honest scoping): the ring-XY mixer guarantees the ROW constraint
(each position -> exactly one bin) by construction. It does NOT by itself
guarantee the COLUMN constraint (each bin used at MOST once across all
positions) -- the original paper's CVRP subproblem doesn't strictly need
this either, since revisiting a customer doesn't help the objective. To be
safe/correct for our TSP-style single-truck case (every bin MUST be visited
exactly once), we keep a lighter penalty term for the column constraint in
the cost Hamiltonian. This is a reasonable, documented adaptation of the
paper's technique to our slightly different (full-tour) problem setting.
"""

import time
from collections import Counter
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pyqubo import Array, Constraint

from stage1_setup import generate_instance, build_distance_matrix
from stage4a_qubo_to_ising import qubo_to_ising
from stage2_baselines import exact_solver, route_cost

NUM_BINS = 4          # confirmed practical ceiling on this hardware
NUM_LAYERS = 2
STEPS = 60
LEARNING_RATE = 0.1


def build_cost_only_qubo(dist_matrix, num_bins, B_col=8.0):
    """
    Cost Hamiltonian = travel distance term (same as Stage 3)
                      + light penalty for COLUMN constraint only
                        (each bin used at most/exactly once across positions)
    Row constraint (each position -> exactly one bin) is NOT penalized here
    -- it's structurally guaranteed by the ring-XY mixer instead.
    """
    n = num_bins
    x = Array.create('x', shape=(n, n), vartype='BINARY')

    col_once = sum(
        Constraint((sum(x[i, p] for i in range(n)) - 1) ** 2, label=f'col_{p}_once')
        for p in range(n)
    )

    cost = 0
    for p in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dist_matrix[i + 1][j + 1]
                    cost += d * x[i, p] * x[j, p + 1]
    for i in range(n):
        cost += dist_matrix[0][i + 1] * x[i, 0]
        cost += dist_matrix[i + 1][0] * x[i, n - 1]

    H = cost + B_col * col_once
    model = H.compile()
    qubo, offset = model.to_qubo()
    return model, qubo, offset


def var_name(i, p):
    return f'x[{i}][{p}]'


def build_var_to_wire(n):
    """Wires grouped by POSITION so each group of n consecutive wires is
    one ring for the XY-mixer: wires[p*n : (p+1)*n] = bin-choices for position p."""
    var_to_wire = {}
    for p in range(n):
        for i in range(n):
            var_to_wire[var_name(i, p)] = p * n + i
    return var_to_wire


def initial_feasible_permutation(n):
    """Identity permutation: position p starts assigned to bin p.
    Returns list of wire indices that should be set to |1>."""
    wires_to_flip = []
    var_to_wire = build_var_to_wire(n)
    for p in range(n):
        i = p  # identity assignment, guaranteed feasible for the row constraint
        wires_to_flip.append(var_to_wire[var_name(i, p)])
    return wires_to_flip


def run_qaoansatz(h, J, var_to_wire, n_bins):
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
        """One ring-XY mixer sweep per position-group."""
        for p in range(n_bins):
            group = [p * n_bins + i for i in range(n_bins)]
            for k in range(n_bins):
                w1, w2 = group[k], group[(k + 1) % n_bins]
                qml.IsingXY(beta, wires=[w1, w2])

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        for w in init_wires:
            qml.PauliX(wires=w)
        for layer in range(NUM_LAYERS):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            ring_xy_mixer(params[1][layer])
        return qml.expval(cost_h)

    params = pnp.array([[0.5] * NUM_LAYERS, [0.5] * NUM_LAYERS], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    print(f"Starting QAOAnsatz optimization ({n_wires} qubits, p={NUM_LAYERS})...")
    t0 = time.time()
    for step in range(STEPS):
        params, cost = opt.step_and_cost(circuit, params)
        if step % 10 == 0 or step == STEPS - 1:
            print(f"  step {step:3d}  cost = {cost:.4f}")
    print(f"Done in {time.time()-t0:.1f}s")

    dev_shots = qml.device("lightning.qubit", wires=n_wires, shots=500)

    @qml.qnode(dev_shots)
    def sample_circuit(params):
        for w in init_wires:
            qml.PauliX(wires=w)
        for layer in range(NUM_LAYERS):
            qml.qaoa.cost_layer(params[0][layer], cost_h)
            ring_xy_mixer(params[1][layer])
        return qml.sample(wires=range(n_wires))

    samples = sample_circuit(params)
    return params, samples


def decode_bitstring(bstr, wire_to_var, num_bins):
    order = [None] * num_bins
    for w, bit in enumerate(bstr):
        if bit == '1':
            i, p = wire_to_var[w][2:-1].split('][')
            order[int(p)] = int(i) + 1
    if None in order:
        return None
    return [0] + order + [0]


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)

    model, qubo, qubo_offset = build_cost_only_qubo(dist_matrix, NUM_BINS)
    h, J, offset = qubo_to_ising(qubo, qubo_offset)

    var_to_wire = build_var_to_wire(NUM_BINS)
    wire_to_var = {v: k for k, v in var_to_wire.items()}

    params, samples = run_qaoansatz(h, J, var_to_wire, NUM_BINS)

    bitstrings = ["".join(str(b) for b in row) for row in samples]

    feasible_costs = []
    row_ok_count = 0  # row constraint should ALWAYS hold by construction
    for bstr in bitstrings:
        # sanity check: verify row constraint (mixer guarantee) actually holds
        row_ok = True
        for p in range(NUM_BINS):
            group_bits = [bstr[p * NUM_BINS + i] for i in range(NUM_BINS)]
            if group_bits.count('1') != 1:
                row_ok = False
                break
        if row_ok:
            row_ok_count += 1

        route = decode_bitstring(bstr, wire_to_var, NUM_BINS)
        if route is not None and len(set(route[1:-1])) == NUM_BINS:
            feasible_costs.append(route_cost(route, dist_matrix))

    total = len(bitstrings)
    print(f"\n=== QAOAnsatz Results ===")
    print(f"Row constraint held (mixer guarantee): {row_ok_count}/{total} "
          f"({row_ok_count/total*100:.1f}%)")
    print(f"Fully feasible routes (row AND column): {len(feasible_costs)}/{total} "
          f"({len(feasible_costs)/total*100:.2f}%)")

    if feasible_costs:
        print(f"Best feasible cost:  {min(feasible_costs):.3f}")
        print(f"Avg feasible cost:   {np.mean(feasible_costs):.3f}")

    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"Known optimal cost:  {exact_cost:.3f}")
