"""
Stage 2: Classical Baselines
1) Nearest-neighbor heuristic (greedy, not ML)
2) Exact brute-force solver (ground-truth optimal for small n)

Both operate on the same distance matrix from Stage 1, so results
are directly comparable to the QAOA solution later.
"""

import itertools
import numpy as np
from stage1_setup import generate_instance, build_distance_matrix, NUM_BINS


def route_cost(route, dist_matrix):
    """route is a list of node indices starting and ending at depot (0)."""
    cost = 0.0
    for i in range(len(route) - 1):
        cost += dist_matrix[route[i]][route[i + 1]]
    return cost


def nearest_neighbor(dist_matrix, num_bins):
    """Greedy heuristic: always go to the closest unvisited bin."""
    unvisited = set(range(1, num_bins + 1))
    route = [0]  # start at depot
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda j: dist_matrix[current][j])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    route.append(0)  # return to depot
    return route


def exact_solver(dist_matrix, num_bins):
    """Brute-force optimal: try all permutations of bins.
    Only tractable for small num_bins (<= ~9-10)."""
    bins = list(range(1, num_bins + 1))
    best_route, best_cost = None, float("inf")
    for perm in itertools.permutations(bins):
        route = [0] + list(perm) + [0]
        cost = route_cost(route, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_route = route
    return best_route, best_cost


if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)

    nn_route = nearest_neighbor(dist_matrix, NUM_BINS)
    nn_cost = route_cost(nn_route, dist_matrix)
    print(f"Nearest-Neighbor route: {nn_route}")
    print(f"Nearest-Neighbor cost:  {nn_cost:.3f}")

    exact_route, exact_cost = exact_solver(dist_matrix, NUM_BINS)
    print(f"\nExact optimal route:   {exact_route}")
    print(f"Exact optimal cost:    {exact_cost:.3f}")

    gap = ((nn_cost - exact_cost) / exact_cost) * 100
    print(f"\nNearest-Neighbor is {gap:.1f}% above optimal")
