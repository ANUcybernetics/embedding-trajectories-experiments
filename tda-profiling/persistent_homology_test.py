# /// script
# requires-python = ">=3.8,<3.12"
# dependencies = [
#     "giotto-ph",
#     "gudhi",
#     "numpy",
# ]
# ///

# to run this file, use the following command in this directory:
#
#     uv run persistent_homology_test.py
#
import time

import gudhi
import numpy as np
from gph import ripser_parallel

MAX_DIM = 10
MAX_POINTS = 1000
TDA_LIB = "gudhi" # Options: "giotto", "gudhi"

# Print CSV header
print("ndim,npoints,time")
for ndim in range(2, MAX_DIM, 1):
    for npoints in range(100, MAX_POINTS, 100):
        pc = np.random.random((npoints, ndim))

        start_time = time.time()
        if TDA_LIB == "giotto":
            dgm = ripser_parallel(pc, maxdim=2, n_threads=-1)
        elif TDA_LIB == "gudhi":
            rips = gudhi.RipsComplex(points=pc)
            st = rips.create_simplex_tree(max_dimension=2)
            dgm = st.persistence()
        else:
            raise ValueError(f"Unsupported TDA library: {TDA_LIB}")
        end_time = time.time()

        total_time = end_time - start_time
        print(f"{ndim},{npoints},{total_time:.4f}")
