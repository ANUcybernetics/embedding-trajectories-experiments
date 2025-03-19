import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist, squareform

# --- Step 1: Takens Embedding ---
def takens_embedding(data, m, tau):
    """
    Perform Takens embedding on multivariate time series data.

    Parameters:
    - data (np.array): Shape (N, d), original time series data.
    - m (int): Embedding dimension (number of delays).
    - tau (int): Time delay step size.

    Returns:
    - np.array: Embedded data of shape (N - (m-1)*tau, m*d)
    """
    N, d = data.shape
    embedded_data = np.zeros((N - (m - 1) * tau, m * d))

    for i in range(m):
        embedded_data[:, i * d: (i + 1) * d] = data[i * tau: N - (m - 1) * tau + i * tau]

    return embedded_data


# --- Step 2: Persistence Entropy ---
def compute_persistence_entropy(point_cloud):
    distance_matrix = squareform(pdist(point_cloud))
    rips = gd.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()
    lifetimes = [death - birth for birth, death in simplex_tree.persistence_intervals_in_dimension(1) if np.isfinite(death)]
    if len(lifetimes) == 0:
        return 0
    lifetimes = np.array(lifetimes)
    p_i = lifetimes / np.sum(lifetimes)
    return -np.sum(p_i * np.log(p_i))

# --- Step 3: Data Generation ---
np.random.seed(42)
t = np.linspace(0, 20, 500)
x = np.sin(t)
y = np.cos(t)
z = np.sin(2 * t)
original_data = np.stack([x, y, z], axis=1)
print("Step 3 done")

# --- Step 4: Compute Persistence Entropies ---
original_entropy = compute_persistence_entropy(original_data)
embedded_entropy = compute_persistence_entropy(takens_embedding(original_data, m=3, tau=10))
print("Step 4 done")

# --- Step 5: Statistical Comparison ---
real_samples = [compute_persistence_entropy(original_data) for _ in range(20)]
embedded_samples = [compute_persistence_entropy(takens_embedding(original_data, m=3, tau=10)) for _ in range(20)]
t_stat, p_value = ttest_ind(real_samples, embedded_samples, equal_var=False)
print("Step 5 done")

# --- Step 6: Visualization ---
plt.figure(figsize=(10, 5))
plt.bar(['Original Data', 'Embedded Data'], [original_entropy, embedded_entropy], color=['#1f77b4', '#ff7f0e'])
plt.title(f"Persistence Entropy Comparison (p = {p_value:.5f})")
plt.ylabel("Persistence Entropy")
plt.show()

print(f"Original Entropy: {original_entropy:.4f}")
print(f"Embedded Entropy: {embedded_entropy:.4f}")
print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.5f}")
if p_value < 0.05:
    print("Significant Difference: Takens embedding affects topology.")
else:
    print("No Significant Difference: Takens embedding retains topological features.")


# from mpl_toolkits.mplot3d import Axes3D
#
# # Simulated multivariate time series data (3D)
# t = np.linspace(0, 20, 500)
# x = np.sin(t)  # Component 1
# y = np.cos(t)  # Component 2
# z = np.sin(2 * t)  # Component 3
# data = np.stack([x, y, z], axis=1)
#
# # Takens embedding
# embedded_data = takens_embedding(data, m=3, tau=10)
#
# # Visualization
# fig = plt.figure(figsize=(12, 6))
#
# # Original 3D Time Series
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot(data[:, 0], data[:, 1], data[:, 2], label="Original Data")
# ax1.set_title("Original 3D Time Series")
# ax1.set_xlabel("$x(t)$")
# ax1.set_ylabel("$y(t)$")
# ax1.set_zlabel("$z(t)$")
# ax1.legend()
#
# # Embedded Trajectory
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], color='orange', label="Embedded Data")
# ax2.set_title("Takens Embedded Trajectory")
# ax2.set_xlabel("$x(t)$")
# ax2.set_ylabel("$x(t + τ)$")
# ax2.set_zlabel("$x(t + 2τ)$")
# ax2.legend()
#
# plt.tight_layout()
# plt.show()
