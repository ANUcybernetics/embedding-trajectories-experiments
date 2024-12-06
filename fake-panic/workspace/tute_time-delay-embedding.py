#!/usr/bin/env python
import os
import gudhi
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gtda.time_series import SingleTakensEmbedding  # seems like it's only available for 1d time series
from sklearn.decomposition import PCA
from gtda.plotting import plot_point_cloud
import json
import pandas as pd
import numpy as np
import pickle as pickle
from pylab import *
import matplotlib.pyplot as plt
import natsort

def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )
    return y_embedded

# dim_tde = 4
# delay = 2
# skip = 2
# time_series = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
# time_series = list(range(100))
# point_cloud = TimeDelayEmbedding(dim=dim_tde, delay=delay, skip=skip).__call__(time_series)
# print("Gudhi:", point_cloud)

x_periodic = np.linspace(0, 10, 1000)
y_periodic = np.cos(5 * x_periodic)

max_embedding_dimension = 30
max_time_delay = 30
stride = 6
embedder = SingleTakensEmbedding(parameters_type="search", time_delay=max_time_delay, dimension=max_embedding_dimension, stride=stride, )
y_periodic_embedded = fit_embedder(embedder, y_periodic)
print(y_periodic_embedded)

pca = PCA(n_components=3)
y_periodic_embedded_pca = pca.fit_transform(y_periodic_embedded)
plot_point_cloud(y_periodic_embedded_pca)
plt.show()