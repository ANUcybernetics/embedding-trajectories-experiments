import numpy as np
import gudhi as gd
import gudhi.representations as gdr
from persim.landscapes import PersistenceLandscaper
from scipy.stats import ttest_ind, ks_2samp, permutation_test
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_random_time_series(N=1000, dim=1024, method="gaussian", real_data=None):
    """
    Generate a random 1024-dimensional time series for hypothesis testing.

    Parameters:
    - N (int): Number of time steps.
    - dim (int): Number of dimensions per time step.
    - method (str): Type of random data. Options: "gaussian", "shuffled", "ar", "random_walk".
    - real_data (np.array): Used for "shuffled" method (should be shape (N, dim)).

    Returns:
    - np.array: Generated random time series of shape (N, dim).
    """
    if method == "gaussian":
        return np.random.randn(N, dim)  # Independent Gaussian noise

    elif method == "shuffled":
        if real_data is None:
            raise ValueError("real_data must be provided for 'shuffled' method.")
        shuffled_data = real_data.copy()
        np.random.shuffle(shuffled_data)  # Shuffle time order
        return shuffled_data

    elif method == "ar":  # Autoregressive process AR(1)
        alpha = 0.8  # Determines temporal dependency strength (closer to 1 = stronger correlation)
        noise = np.random.randn(N, dim)  # White noise
        ar_series = np.zeros((N, dim))
        ar_series[0] = noise[0]  # Initialize first point
        for t in range(1, N):
            ar_series[t] = alpha * ar_series[t - 1] + noise[t]
        return ar_series

    elif method == "random_walk":  # Cumulative sum of noise
        return np.cumsum(np.random.randn(N, dim), axis=0)

    else:
        raise ValueError("Invalid method. Choose from 'gaussian', 'shuffled', 'ar', or 'random_walk'.")


def compute_persistence_entropy(time_series):
    """
    Compute persistence entropy for a given time series using the Vietoris-Rips complex.

    Parameters:
    - time_series (np.array): Shape (N, dim), representing the high-dimensional trajectory.

    Returns:
    - float: Persistence entropy of the dataset.
    """
    # Compute pairwise Euclidean distances
    from scipy.spatial.distance import pdist, squareform
    distance_matrix = squareform(pdist(time_series, metric="euclidean"))

    # Construct Vietoris-Rips complex
    rips = gd.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)

    # Compute persistence diagram
    persistence = simplex_tree.persistence()

    # Extract lifetimes
    lifetimes = []
    for birth, death in simplex_tree.persistence_intervals_in_dimension(1):  # H1 features
        if np.isfinite(death):  # Ignore infinite bars
            lifetimes.append(death - birth)

    # Normalize lifetimes into a probability distribution
    if len(lifetimes) == 0:
        return 0  # If no significant features, entropy is 0

    lifetimes = np.array(lifetimes)
    p_i = lifetimes / np.sum(lifetimes)  # Normalize
    entropy = -np.sum(p_i * np.log(p_i))  # Compute entropy

    return entropy

# Generate real and random time series
N, dim = 100, 10
real_data = generate_random_time_series(N, dim, method="ar")  # Assume AR(1) is real data
random_data = generate_random_time_series(N, dim, method="gaussian")  # Null hypothesis

# Compute persistence entropy for multiple trials
num_samples = 20  # Number of samples for better statistical robustness
real_entropy = [compute_persistence_entropy(generate_random_time_series(N, dim, method="ar")) for _ in range(num_samples)]
random_entropy = [compute_persistence_entropy(generate_random_time_series(N, dim, method="gaussian")) for _ in range(num_samples)]

# Apply Welch’s t-test
t_stat, p_value = ttest_ind(real_entropy, random_entropy, equal_var=False)

# Interpret results
alpha = 0.05  # Significance level
print(f"T-statistic: {t_stat:.3f}, p-value: {p_value:.5f}")
if p_value < alpha:
    print("Statistically significant difference (Reject Null Hypothesis)")
else:
    print("No significant difference (Fail to Reject Null Hypothesis)")


# Create a dataframe for visualization
df = pd.DataFrame({
    "Persistence Entropy": real_entropy + random_entropy,
    "Dataset": ["Real"] * num_samples + ["Random"] * num_samples
})

# Plot boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x="Dataset", y="Persistence Entropy", data=df, palette=["#2C7BB6", "#D7191C"])
plt.title(f"Persistence Entropy Comparison (p = {p_value:.5f})")
plt.xlabel("Dataset Type")
plt.ylabel("Persistence Entropy")
plt.show()

# Plot histogram
plt.figure(figsize=(8, 5))
sns.histplot(real_entropy, bins=10, kde=True, color="#2C7BB6", label="Real")
sns.histplot(random_entropy, bins=10, kde=True, color="#D7191C", label="Random", alpha=0.6)
plt.axvline(np.mean(real_entropy), color='blue', linestyle='dashed', linewidth=2, label="Real Mean")
plt.axvline(np.mean(random_entropy), color='red', linestyle='dashed', linewidth=2, label="Random Mean")
plt.legend()
plt.title("Distribution of Persistence Entropy")
plt.xlabel("Persistence Entropy")
plt.ylabel("Frequency")
plt.show()


# # Simulated 1024-dimensional time series data
# np.random.seed(42)
#
# def generate_time_series(num_samples=50, length=100, dim=1024):
#     """Generate synthetic 1024-dimensional time series as random walks."""
#     return np.cumsum(np.random.randn(num_samples, length, dim), axis=1)
#
# # Generate two groups of trajectories for comparison
# group_A = generate_time_series(num_samples=30)
# group_B = generate_time_series(num_samples=30)
#
# def compute_persistence_diagram(time_series):
#     """Compute persistence diagram from time series using Vietoris-Rips filtration."""
#     diagrams = []
#     for traj in time_series:
#         dist_matrix = np.linalg.norm(traj[:, None, :] - traj[None, :, :], axis=-1)
#         rips = gd.RipsComplex(distance_matrix=dist_matrix)
#         simplex_tree = rips.create_simplex_tree(max_dimension=2)
#         diag = simplex_tree.persistence()
#         diagrams.append(simplex_tree.persistence_intervals_in_dimension(1))  # H1 features
#     return diagrams
#
# # Compute persistence diagrams
# diagrams_A = compute_persistence_diagram(group_A)
# diagrams_B = compute_persistence_diagram(group_B)
#
# def compute_persistence_landscape(diagrams, num_landscapes=5, resolution=100):
#     """Compute persistence landscapes from diagrams."""
#     landscapes = []
#     for diag in diagrams:
#         pl = PersistenceLandscaper(resolution=resolution, num_landscapes=num_landscapes)
#         landscapes.append(pl.fit_transform([diag])[0])  # Extract first landscape
#     return np.array(landscapes)
#
# # # Compute persistence landscapes
# # landscapes_A = compute_persistence_landscape(diagrams_A)
# # landscapes_B = compute_persistence_landscape(diagrams_B)
# #
# # # Flatten landscapes for statistical comparison
# # landscape_features_A = landscapes_A.reshape(len(landscapes_A), -1)
# # landscape_features_B = landscapes_B.reshape(len(landscapes_B), -1)
#
# # Compute Persistence Entropy
# def compute_persistence_entropy(diagrams):
#     """Compute persistence entropy for each diagram."""
#     entropies = []
#     for diag in diagrams:
#         if len(diag) == 0:
#             entropies.append(0)  # Handle empty diagrams
#             continue
#         lifetimes = diag[:, 1] - diag[:, 0]
#         p = lifetimes / lifetimes.sum()
#         entropy = -np.sum(p * np.log(p))
#         entropies.append(entropy)
#     return np.array(entropies)
#
# entropy_A = compute_persistence_entropy(diagrams_A)
# entropy_B = compute_persistence_entropy(diagrams_B)
#
# # Statistical Tests
# t_stat, p_ttest = ttest_ind(entropy_A, entropy_B, equal_var=False)  # Welch’s t-test
# ks_stat, p_ks = ks_2samp(entropy_A, entropy_B)  # KS test for distribution comparison
#
# # perm_test_result = permutation_test((entropy_A, entropy_B), np.mean, n_resamples=1000, alternative="two-sided")
#
# # Print results
# print(f"T-test: p-value = {p_ttest:.4f} (test for mean entropy difference)")
# print(f"KS-test: p-value = {p_ks:.4f} (test for distribution similarity)")
# # print(f"Permutation Test: p-value = {perm_test_result.pvalue:.4f} (non-parametric test)")
#
