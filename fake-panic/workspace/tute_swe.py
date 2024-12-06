# Sliding window embedding - Ch. 10 in Computational and Applied Topology by Paweł Dłotko
import numpy as np
import gudhi as gd
import math
import matplotlib.pyplot as plt
from random import random


# Construction of the grid points x, and the values therein, y = sin(x)
arg = np.linspace(0, 10*math.pi, 500)
values = np.sin(arg)    # 1-dim values (not a vector... hmm)

N = 200 # Size of the sliding window
swe = []
for i in range(0, len(values)-N):
    point = []
    for j in range(0, N):
        point.append(float(values[i+j]))# + random()*1))
    swe.append(point)


# Now we have the point cloud, swe, and we can compute the persistent homology in dim 1
rips = gd.RipsComplex(points=swe, max_edge_length=20)
simplex_tree = rips.create_simplex_tree(max_dimension=2)

diagram = simplex_tree.persistence()
# print(diagram)
persistence = simplex_tree.persistence_intervals_in_dimension(1)
print(persistence)
gd.plot_persistence_diagram(diagram)
plt.show()