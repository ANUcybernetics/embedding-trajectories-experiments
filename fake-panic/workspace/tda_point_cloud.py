#!/usr/bin/env python
import gudhi
import json
import pandas as pd
import numpy as np
import pickle as pickle
from pylab import *
import matplotlib.pyplot as plt


""" Data processing """
# file_name = "2024-08-24T19-59-35_a_little_boy_holding_a_pistachio_icecream_in_winter"
file_name = "2024-08-24T20-43-11_a_little_girl_holding_a_red_toy_train_in_winter"
with open(f"../data/{file_name}.json", "r") as json_file:
    json_data = json.load(json_file)
    points = []
    for i in range(len(json_data)):
        embedding_vector = json_data[i]["embedding"]
        in_data = json_data[i]["input"]
        seq_no = json_data[i]["seq_no"]
        data_type = json_data[i]["type"]
        points.append(embedding_vector)
        print(in_data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # points = np.array(points)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # for i in range(len(json_data)):
    #     ax.text(points[i, 0], points[i, 1], points[i, 2], s=i)
    # plt.show()

    rips = gudhi.RipsComplex(points=points)#, max_edge_length=10)

    simplex_tree = rips.create_simplex_tree(max_dimension=3)
    # print("Num simplices:", simplex_tree.num_simplices())

    # simplex_tree.compute_persistence()
    # print("Betti nums", simplex_tree.persistent_betti_numbers())

    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    print("diag=", diag)

    gudhi.plot_persistence_diagram(diag)
    plt.show()
