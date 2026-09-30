"""
Stage 1: Problem Setup
Defines the depot + bins as a small CVRP instance.
Coordinates are loosely modeled on a compact campus-like layout.
Demand simulates IoT-sensor fill-level readings (0-1 scaled to a load unit).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducibility - important to mention in the paper

# ---- Configuration ----
NUM_BINS = 4          # validated scope (16 qubits) -- see paper Section 6.1
                       # for why 5-6 bins exceed available memory
VEHICLE_CAPACITY = 10  # arbitrary unit matching demand scale
AREA_SIZE = 10          # coordinate grid size (km or campus-units)
# NOTE: demand range is deliberately kept low so total demand stays
# under VEHICLE_CAPACITY -> feasible as a SINGLE route (no depot return
# mid-route). Multi-trip / larger demand is future-work scope.

def generate_instance(num_bins=NUM_BINS, seed=42):
    rng = np.random.default_rng(seed)

    # Depot at origin-ish (e.g. garage location)
    depot = {"id": 0, "x": 0.0, "y": 0.0, "demand": 0.0}

    bins = []
    for i in range(1, num_bins + 1):
        bins.append({
            "id": i,
            "x": rng.uniform(0, AREA_SIZE),
            "y": rng.uniform(0, AREA_SIZE),
            # simulated fill-level demand (IoT sensor reading), scaled
            "demand": round(rng.uniform(0.5, 1.5), 2)
        })

    nodes = [depot] + bins
    return nodes

def build_distance_matrix(nodes):
    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = nodes[i]["x"] - nodes[j]["x"]
                dy = nodes[i]["y"] - nodes[j]["y"]
                dist[i][j] = np.sqrt(dx**2 + dy**2)
    return dist

def plot_instance(nodes, save_path="instance_plot.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    depot = nodes[0]
    ax.scatter(depot["x"], depot["y"], c="red", s=150, marker="s", label="Depot")
    for n in nodes[1:]:
        ax.scatter(n["x"], n["y"], c="green", s=80)
        ax.annotate(f'Bin{n["id"]}\n(d={n["demand"]})', (n["x"], n["y"]), fontsize=8)
    ax.set_title(f"CVRP Instance: {len(nodes)-1} bins")
    ax.legend()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved instance plot to {save_path}")

if __name__ == "__main__":
    nodes = generate_instance(NUM_BINS)
    dist_matrix = build_distance_matrix(nodes)

    print("Nodes:")
    for n in nodes:
        print(f"  id={n['id']:2d}  x={n['x']:.2f}  y={n['y']:.2f}  demand={n['demand']}")

    print("\nTotal demand:", sum(n["demand"] for n in nodes))
    print("Vehicle capacity:", VEHICLE_CAPACITY)
    print("\nDistance matrix:\n", np.round(dist_matrix, 2))

    plot_instance(nodes)
