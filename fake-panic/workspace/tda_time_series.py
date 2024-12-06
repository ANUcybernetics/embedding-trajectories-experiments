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


""" Data processing """
# # file_name = "2024-08-24T19-59-35_a_little_boy_holding_a_pistachio_icecream_in_winter"
# file_name = "2024-08-24T20-43-11_a_little_girl_holding_a_red_toy_train_in_winter"

data_path = "embeddings"
data_list = os.listdir(data_path)
for file_name in data_list:
    file_name = file_name[:-5]
    with open(f"{data_path}/{file_name}.json", "r") as json_file:
        json_data = json.load(json_file)
        input_prompt = ""
        time_series = []
        for i in range(len(json_data)):
            embedding_vector = json_data[i]["embedding"]
            in_data = json_data[i]["input"]
            if i == 0:
                input_prompt = in_data
            seq_no = json_data[i]["seq_no"]
            data_type = json_data[i]["type"]
            time_series.append(embedding_vector)
            # print(in_data)
        print(input_prompt)

        dim_tde = len(time_series[0]) * 2
        delay = 1
        skip = 1
        stride = skip
        point_cloud = TimeDelayEmbedding(dim=dim_tde, delay=delay, skip=skip).transform(ts=time_series)
        # dim (int): `d` of R^d to be embedded. Optional (default=3). -> Num of columns
        # delay (int): Time-Delay embedding. Optional (default=1). -> Next elt in each row
        # skip (int): How often to skip embedded points. Optional (default=1). -> First elt in the next row
        # len(point_cloud) = Number of PANIC iterations
        # print(time_series[0])
        # print(time_series[1])
        # print(point_cloud[0])

        rips = gudhi.RipsComplex(points=point_cloud)  # , max_edge_length=10)

        simplex_tree = rips.create_simplex_tree()#max_dimension=3)
        # print("Num simplices:", simplex_tree.num_simplices())

        # simplex_tree.compute_persistence()
        # print("Betti nums", simplex_tree.persistent_betti_numbers())

        diagram = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
        # print("diag=", diagram)

        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        timestamp = timestamp.replace(":", "-")  # To avoid "Invalid argument" error by ":"
        file_name = input_prompt + "_" + timestamp

        ax = gudhi.plot_persistence_diagram(diagram)
        ax.set_title(input_prompt)
        ax.set_aspect("equal")
        # plt.show()
        plt.savefig(f"TDE/PD_{file_name}.png")
        plt.close()

        ax = gudhi.plot_persistence_barcode(diagram)
        ax.set_title(input_prompt)
        # plt.show()
        plt.savefig(f"TDE/BC_{file_name}.png")
        plt.close()