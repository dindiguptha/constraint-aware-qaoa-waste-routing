"""
Stage 4a: QUBO -> Ising Conversion
QUBO objective: sum_ij Q_ij * x_i * x_j   (x_i in {0,1})
Substitution:   x_i = (1 - z_i) / 2        (z_i in {-1,+1})

For i == j (diagonal, since x_i^2 = x_i for binary vars):
    Q_ii * x_i = Q_ii * (1 - z_i)/2
               = Q_ii/2 - (Q_ii/2) * z_i

For i != j (off-diagonal pair, combining Q_ij and Q_ji into one term
since PyQUBO's qubo dict already gives each unordered pair once):
    Q_ij * x_i * x_j = Q_ij * (1-z_i)(1-z_j)/4
                     = Q_ij/4 * (1 - z_i - z_j + z_i*z_j)

Collecting terms gives linear coefficients h_i (coefficient of z_i),
quadratic coefficients J_ij (coefficient of z_i*z_j), and a constant
offset -- exactly what PennyLane's qml.Hamiltonian / QAOA needs.
"""

import numpy as np
from stage1_setup import generate_instance, build_distance_matrix, NUM_BINS
from stage3_qubo import build_qubo, decode_solution
from stage2_baselines import exact_solver, route_cost


def qubo_to_ising(qubo, offset_qubo=0.0):
    """
    qubo: dict {(var1, var2): coeff}  (var1==var2 for diagonal/linear terms)
    Returns: h (dict var->coeff), J (dict (var1,var2)->coeff), offset (float)
    """
    h = {}
    J = {}
    offset = offset_qubo

    for (v1, v2), coeff in qubo.items():
        if v1 == v2:
            # diagonal term: Q_ii * x_i
            h[v1] = h.get(v1, 0.0) - coeff / 2
            offset += coeff / 2
        else:
            # off-diagonal term: Q_ij * x_i * x_j
            key = tuple(sorted((v1, v2)))
            J[key] = J.get(key, 0.0) + coeff / 4
            h[v1] = h.get(v1, 0.0) - coeff / 4
            h[v2] = h.get(v2, 0.0) - coeff / 4
            offset += coeff / 4

    return h, J, offset


def verify_conversion(qubo, h, J, offset, sample_bits, qubo_offset=0.0):
    """
    sample_bits: dict var_name -> 0/1
    Confirms QUBO energy == Ising energy for the same assignment,
    which validates the algebra above.
    qubo_offset: PyQUBO's own constant offset (from model.to_qubo()),
    must be added since it's separate from the dict terms.
    """
    # QUBO energy (raw dict terms + PyQUBO's own constant offset)
    qubo_energy = qubo_offset
    for (v1, v2), coeff in qubo.items():
        qubo_energy += coeff * sample_bits[v1] * sample_bits[v2]

    # Ising energy (convert bits -> spins)
    spins = {v: (1 - 2 * b) for v, b in sample_bits.items()}  # x=0->z=1, x=1->z=-1
    ising_energy = offset
    for v, coeff in h.items():
        ising_energy += coeff * spins[v]
    for (v1, v2), coeff in J.items():
        ising_energy += coeff * spins[v1] * spins[v2]

    return qubo_energy, ising_energy


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)
    model, qubo, qubo_offset = build_qubo(dist_matrix, NUM_BINS)

    h, J, offset = qubo_to_ising(qubo, qubo_offset)
    print(f"Ising model: {len(h)} linear terms, {len(J)} quadratic terms")

    # Verify using the known-optimal decoded route's bit assignment
    # Reconstruct the bit assignment for the known optimal route via
    # simulated annealing sample from Stage 3 (re-derive quickly here)
    import neal
    sampler = neal.SimulatedAnnealingSampler()
    bqm = model.to_bqm()
    sampleset = sampler.sample(bqm, num_reads=200)
    best_sample = sampleset.first.sample

    qubo_e, ising_e = verify_conversion(qubo, h, J, offset, best_sample, qubo_offset)
    print(f"\nQUBO energy:  {qubo_e:.6f}")
    print(f"Ising energy: {ising_e:.6f}")
    print(f"Match: {np.isclose(qubo_e, ising_e)}")
