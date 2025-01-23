# /// script
# requires-python = ">=3.8,<3.12"
# dependencies = [
#     "giotto-ph",
#     "numpy",
# ]
# ///

# to run this file, use the following command in this directory:
#
#     uv run persistent_homology_test.py
#
import numpy as np
from gph import ripser_parallel

ndim = 10
npoints = 100

pc = np.random.random((npoints, ndim))
dgm = ripser_parallel(pc, maxdim=2, n_threads=-1)
