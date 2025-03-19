import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import gudhi as gd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


# --- Step 1: Extract Ordinal Patterns ---
def ordinal_patterns(data, m):
    patterns = []
    for i in range(len(data) - m + 1):
        pattern = tuple(np.argsort(data[i:i + m]))
        patterns.append(pattern)
    return np.array(patterns)


# --- Step 2: Build the OPN with Weighted Edges ---
def build_opn(data, m):
    patterns = ordinal_patterns(data, m)
    G = nx.DiGraph()
    weights = defaultdict(int)

    for i in range(len(patterns) - 1):
        weights[(tuple(patterns[i]), tuple(patterns[i + 1]))] += 1

    for (u, v), weight in weights.items():
        G.add_edge(u, v, weight=weight)

    return G, patterns


# --- Step 3: Compute Permutation Entropy ---
def compute_permutation_entropy(patterns):
    unique_patterns, counts = np.unique(patterns, return_counts=True, axis=0)
    probabilities = counts / np.sum(counts)
    return entropy(probabilities)


# --- Step 4: Visualize the OPN with Weighted Edges ---
def visualize_opn(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')

    # Add edge labels (weights)
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Ordinal Partition Network (OPN) with Weights")
    plt.show()


# --- Step 5: Compute Persistent Homology ---
def compute_persistent_homology(G):
    nodes = list(G.nodes())
    dist_matrix = np.full((len(nodes), len(nodes)), np.inf)

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if G.has_edge(u, v):
                dist_matrix[i, j] = 1 / G[u][v]['weight']

    rips = gd.RipsComplex(distance_matrix=dist_matrix)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence()

    gd.plot_persistence_diagram(diag)
    plt.title("Persistence Diagram of Weighted OPN")
    plt.show()


# --- Step 6: Example Usage ---
np.random.seed(42)
# Dataset 1: Noisy sine wave
point_cloud_1 = np.sin(np.linspace(0, 20, 100)) + 0.2 * np.random.randn(100)
# Dataset 2: Pure noise
point_cloud_2 = np.random.randn(100)

dimension = 5

# Build and visualize OPN for both datasets
G1, patterns1 = build_opn(point_cloud_1, dimension)
G2, patterns2 = build_opn(point_cloud_2, dimension)

visualize_opn(G1)
visualize_opn(G2)

compute_persistent_homology(G1)
compute_persistent_homology(G2)

# Compute and display Permutation Entropy for both datasets
pe_value_1 = compute_permutation_entropy(patterns1)
pe_value_2 = compute_permutation_entropy(patterns2)

print(f"Permutation Entropy (Noisy Sine Wave): {pe_value_1:.4f}")
print(f"Permutation Entropy (Pure Noise): {pe_value_2:.4f}")

print(f"Number of nodes (unique patterns) in Dataset 1: {G1.number_of_nodes()}")
print(f"Number of edges (transitions) in Dataset 1: {G1.number_of_edges()}")

print(f"Number of nodes (unique patterns) in Dataset 2: {G2.number_of_nodes()}")
print(f"Number of edges (transitions) in Dataset 2: {G2.number_of_edges()}")
