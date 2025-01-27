import os
import json
import numpy as np
import matplotlib.pyplot as plt
import gudhi    # https://gudhi.inria.fr/python/latest/rips_complex_user.html
from gph import ripser_parallel # https://giotto-ai.github.io/giotto-ph/build/html/modules/ripser_parallel.html
import time


ndim = 10
npoints = 100
MAX_DIM = 2
TDA_LIB = "gudhi" # Options: "giotto", "gudhi"

# pc = np.random.random((npoints, ndim))
# np.savetxt("sample_points.txt", pc, fmt="%f")

pc = np.loadtxt("sample_points.txt", dtype=float)
start_time = time.time()
if TDA_LIB == "giotto":
    dgm_dict = ripser_parallel(pc, maxdim=MAX_DIM, n_threads=-1)
    # print(dgm_dict['dgms'][0])  # format: list of tuples (homology_dim, (birth, death))
    dgm = []
    for dim in range(MAX_DIM):
        for bd_pair in dgm_dict['dgms'][dim]:
            dgm.append((dim, (bd_pair[0], bd_pair[1])))
    # print(dgm)
    ax = gudhi.plot_persistence_diagram(dgm)
    ax.set_title(f"PH computed by {TDA_LIB}")
    ax.set_aspect("equal")
    # plt.show()
    plt.savefig(f"sample_PD_{TDA_LIB}.png", bbox_inches="tight")
elif TDA_LIB == "gudhi":
    rips = gudhi.RipsComplex(points=pc)
    st = rips.create_simplex_tree(max_dimension=MAX_DIM)
    dgm = st.persistence()
    print(dgm)  # format: list of tuples (homology_dim, (birth, death))
    ax = gudhi.plot_persistence_diagram(dgm)
    ax.set_title(f"PH computed by {TDA_LIB}")
    ax.set_aspect("equal")
    # plt.show()
    plt.savefig(f"sample_PD_{TDA_LIB}.png", bbox_inches="tight")
else:
    raise ValueError(f"Unsupported TDA library: {TDA_LIB}")
end_time = time.time()

total_time = end_time - start_time
print(f"PH computation by {TDA_LIB} taken {total_time:.4f}(s)")