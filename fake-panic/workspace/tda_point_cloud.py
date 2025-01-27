#!/usr/bin/env python
import os
import datetime
import gudhi
import json
# import pandas as pd
import numpy as np
import pickle as pickle
from pylab import *
import matplotlib.pyplot as plt
import natsort
from panic import Panic
from gudhi.point_cloud.timedelay import TimeDelayEmbedding


""" Run PANIC """
num_iterations = 2
run_panic = Panic()
prompt_list = open("input_prompts.txt", "r")
for line in prompt_list.readlines():
    input_prompt = line
    print("Input prompt:", input_prompt)
    # Panic.run_panic(num_iterations=100, prompt=input_prompt)
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    timestamp = timestamp.replace(":", "-")  # To avoid "Invalid argument" error by ":"
    file_name = input_prompt + "_" + timestamp
    prompt = input_prompt
    point_cloud = []
    output_data = []
    for i in range(int(num_iterations / 2)):
        # print(f"Sequence number {2 * i}")
        image_url = run_panic.generate_image(prompt)
        print(f"Image {2*i}: {image_url}")

        image_embedding = run_panic.calculate_embedding("vision", image_url)
        # point_cloud.append(image_embedding)
        # output_data.append(
        #     {
        #         "seq_no": 2 * i,
        #         "type": "image",
        #         "input": prompt,
        #         # "embedding": image_embedding,
        #     }
        # )

        # print(f"Sequence number {2 * i + 1}")
        # BLIP prefixes the string with "Caption: " - remove this
        caption = run_panic.caption_image(image_url)
        if caption.startswith("Caption: "):
            caption = caption[9:]
        print(f"Caption {2*i + 1}: {caption}")

        caption_embedding = run_panic.calculate_embedding("text", caption)
        # point_cloud.append(caption_embedding)
        # output_data.append(
        #     {
        #         "seq_no": 2 * i + 1,
        #         "type": "text",
        #         "input": image_url,
        #         # "embedding": caption_embedding,
        #     }
        # )

        prompt = caption
        # print("\n")
    # with open(f"embeddings/{file_name}.json", "w") as f:
    #     json.dump(output_data, f, indent=4, sort_keys=True)

    # rips = gudhi.RipsComplex(points=point_cloud)  # , max_edge_length=10)
    #
    # simplex_tree = rips.create_simplex_tree(max_dimension=3)
    # # print("Num simplices:", simplex_tree.num_simplices())
    #
    # # simplex_tree.compute_persistence()
    # # print("Betti nums", simplex_tree.persistent_betti_numbers())
    #
    # diagram = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    # # print("diag=", diagram)
    #
    # ax = gudhi.plot_persistence_diagram(diagram)
    # ax.set_title(input_prompt)
    # ax.set_aspect("equal")
    # # plt.show()
    # plt.savefig(f"ph_results/PD_{file_name}.png")
    # plt.close()
    #
    # ax = gudhi.plot_persistence_barcode(diagram)
    # ax.set_title(input_prompt)
    # # plt.show()
    # plt.savefig(f"ph_results/BC_{file_name}.png")
    # plt.close()


    # """ Data processing """
    # # # file_name = "2024-08-24T19-59-35_a_little_boy_holding_a_pistachio_icecream_in_winter"
    # # file_name = "2024-08-24T20-43-11_a_little_girl_holding_a_red_toy_train_in_winter"
    #
    # data_path = "embeddings"
    # data_list = os.listdir(data_path)
    # for file_name in data_list:
    #     file_name = file_name[:-5]
    #     with open(f"{data_path}/{file_name}.json", "r") as json_file:
    #         json_data = json.load(json_file)
    #         # input_prompt = ""
    #         point_cloud = []
    #         for i in range(len(json_data)):
    #             embedding_vector = json_data[i]["embedding"]
    #             in_data = json_data[i]["input"]
    #             # if i == 0:
    #             #     input_prompt = in_data
    #             seq_no = json_data[i]["seq_no"]
    #             data_type = json_data[i]["type"]
    #             point_cloud.append(embedding_vector)
    #             print(in_data)
    #
    #         rips = gudhi.RipsComplex(points=point_cloud)#, max_edge_length=10)
    #
    #         simplex_tree = rips.create_simplex_tree(max_dimension=3)
    #         # print("Num simplices:", simplex_tree.num_simplices())
    #
    #         # simplex_tree.compute_persistence()
    #         # print("Betti nums", simplex_tree.persistent_betti_numbers())
    #
    #         diagram = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    #         print("diag=", diagram)
    #
    #         ax = gudhi.plot_persistence_diagram(diagram)
    #         ax.set_title(input_prompt)
    #         ax.set_aspect("equal")
    #         # plt.show()
    #         plt.savefig(f"persistence_diagrams/{file_name}.png")
    #         plt.close()
    #
    #         ax = gudhi.plot_persistence_barcode(diagram)
    #         ax.set_title(input_prompt)
    #         # plt.show()
    #         plt.savefig(f"barcodes/{file_name}.png")
    #         plt.close()
    #
