"""
Stage 6: Logarithmic (Binary) Position Encoding

Instead of one-hot x[i,p] (n^2 qubits), each position p gets its own
binary NUMBER using k = ceil(log2(n)) qubits, representing which bin
(0..n-1) is visited at that position.

KEY CHALLENGE (explained honestly in our discussion): enforcing that no
two positions decode to the SAME bin requires an equality-check between
two k-bit binary numbers, which is naturally higher-degree (not quadratic).
We rely on PyQUBO's automatic degree-reduction (verified above) to convert
these into valid QUBO with auxiliary variables.

VALIDATION STRATEGY: build this for a tiny n (3-4 bins) FIRST, solve with
classical simulated annealing, and confirm it agrees with the known exact
optimal (from Stage 2) before trusting it at larger n. This mirrors the
same validate-before-trust approach used for the one-hot QUBO in Stage 3.
"""

import math
import numpy as np
from pyqubo import Array, Constraint, Placeholder
import neal

from stage1_setup import generate_instance, build_distance_matrix
from stage2_baselines import exact_solver, route_cost


def bits_needed(n):
    return max(1, math.ceil(math.log2(n)))


def build_log_qubo(dist_matrix, num_bins, A_range=15.0, A_distinct=15.0, B=1.0):
    """
    Variables: b[p][k] = k-th bit of the bin-index assigned to position p.
    Decoded value at position p: sum_k b[p][k] * 2^k  (must be in [0, n-1])
    """
    n = num_bins
    k = bits_needed(n)
    b = Array.create('b', shape=(n, k), vartype='BINARY')

    def value_expr(p):
        return sum(b[p, bit] * (2 ** bit) for bit in range(k))

    # ---- Constraint A: decoded value must be in valid range [0, n-1] ----
    # Penalize any value above n-1 (only matters when n is not a power of 2).
    range_penalty = 0
    max_val = (2 ** k) - 1
    if max_val > n - 1:
        # Soft penalty: for values > n-1, distance-from-valid-range squared.
        # Simplify by penalizing each invalid integer value explicitly.
        for invalid_val in range(n, 2 ** k):
            # indicator that value == invalid_val: product of matching bits
            bits_of_val = [(invalid_val >> bit) & 1 for bit in range(k)]
            indicator = 1
            for bit in range(k):
                bit_var = b_p_bit = None
            # build indicator per position p later (needs p in scope)
        # We'll fold this into the per-position loop below instead.

    range_penalty = 0
    for p in range(n):
        for invalid_val in range(n, 2 ** k):
            bits_of_val = [(invalid_val >> bit) & 1 for bit in range(k)]
            indicator = 1
            for bit in range(k):
                if bits_of_val[bit] == 1:
                    indicator = indicator * b[p, bit]
                else:
                    indicator = indicator * (1 - b[p, bit])
            range_penalty += Constraint(indicator, label=f'range_p{p}_v{invalid_val}')

    # ---- Constraint B: distinctness -- no two positions share the same bin ----
    distinct_penalty = 0
    for p in range(n):
        for q in range(p + 1, n):
            # equal_bits[bit] = 1 if b[p,bit] == b[q,bit], via XNOR:
            # XNOR(x,y) = 1 - x - y + 2xy
            equal_indicator = 1
            for bit in range(k):
                xnor = 1 - b[p, bit] - b[q, bit] + 2 * b[p, bit] * b[q, bit]
                equal_indicator = equal_indicator * xnor
            # equal_indicator == 1 only if ALL bits match (same bin chosen) -> penalize
            distinct_penalty += Constraint(equal_indicator, label=f'distinct_{p}_{q}')

    # ---- Cost term: distance between consecutive positions' bins ----
    # Since bin index is now encoded as a binary number rather than one-hot,
    # we express cost via indicator variables per (position, actual bin)
    # pair, reusing the same per-bit indicator trick.
    def bin_indicator(p, bin_id):
        """QUBO expression that equals 1 iff position p decodes to bin_id."""
        bits_of_val = [(bin_id >> bit) & 1 for bit in range(k)]
        indicator = 1
        for bit in range(k):
            if bits_of_val[bit] == 1:
                indicator = indicator * b[p, bit]
            else:
                indicator = indicator * (1 - b[p, bit])
        return indicator

    cost = 0
    for p in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dist_matrix[i + 1][j + 1]
                    cost += d * bin_indicator(p, i) * bin_indicator(p + 1, j)
    for i in range(n):
        cost += dist_matrix[0][i + 1] * bin_indicator(0, i)
        cost += dist_matrix[i + 1][0] * bin_indicator(n - 1, i)

    H = B * cost + A_range * range_penalty + A_distinct * distinct_penalty
    # CRITICAL FIX: PyQUBO's automatic degree-reduction (needed for our
    # degree-3/4 equality-check terms) introduces auxiliary variables with
    # their own enforcement penalty, controlled by `strength` (default=5).
    # Our own penalty/cost coefficients (up to ~50) were LARGER than this
    # default, so the solver could violate aux=product relationships to
    # fake artificially low energy for infeasible solutions. Fix: set
    # strength well above our own largest coefficient.
    model = H.compile(strength=500.0)
    qubo, offset = model.to_qubo()
    return model, qubo, offset, k


def decode_log_solution(sample, num_bins, k):
    route = [None] * num_bins
    for p in range(num_bins):
        val = 0
        for bit in range(k):
            key = f'b[{p}][{bit}]'
            if sample.get(key, 0) == 1:
                val += 2 ** bit
        route[p] = val + 1  # map 0..n-1 -> bin ids 1..n
    return [0] + route + [0]


if __name__ == "__main__":
    NUM_BINS = 4  # VALIDATE at small scale first, same size as our known-good result
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)

    print(f"Building LOG-encoded QUBO for {NUM_BINS} bins "
          f"({NUM_BINS} x {bits_needed(NUM_BINS)} = {NUM_BINS * bits_needed(NUM_BINS)} qubits "
          f"before auxiliary variables)...")

    model, qubo, offset, k = build_log_qubo(dist_matrix, NUM_BINS, A_range=50.0, A_distinct=50.0)
    all_vars = set(v for pair in qubo.keys() for v in pair)
    print(f"After PyQUBO's automatic degree-reduction: {len(all_vars)} total qubits "
          f"({len(qubo)} QUBO terms)")

    sampler = neal.SimulatedAnnealingSampler()
    bqm = model.to_bqm()
    sampleset = sampler.sample(bqm, num_reads=500)
    best = sampleset.first.sample

    route = decode_log_solution(best, NUM_BINS, k)
    print(f"\nDecoded route: {route}")

    if len(set(route[1:-1])) == NUM_BINS:
        cost = route_cost(route, dist_matrix)
        print(f"Route cost: {cost:.3f}  (FEASIBLE - visits all bins exactly once)")
    else:
        print(f"INFEASIBLE decode: {route}  -- penalty weights likely need tuning")

    _, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"Known optimal cost: {exact_cost:.3f}")
    print(f"\nMatch: {np.isclose(cost, exact_cost) if len(set(route[1:-1])) == NUM_BINS else 'N/A (infeasible)'}")
