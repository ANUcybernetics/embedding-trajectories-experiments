# /// script
# requires-python = ">=3.8,<3.12"
# dependencies = [
#     "giotto-ph",
#     "numpy",
# ]
# ///

import numpy as np
from gph import ripser_parallel

ndim = 10
npoints = 100

pc = np.random.random((npoints, ndim))
dgm = ripser_parallel(pc, maxdim=2, n_threads=-1)
