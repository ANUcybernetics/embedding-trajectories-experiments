# /// script
# requires-python = ">=3.7"
# dependencies = [
#     "giotto-ph",
#     "gudhi",
#     "numpy<2",
#     "scipy",
#     "scikit-learn"
# ]
# ///

# sdoic: sensitivity dependence on initial conditions (cf. chaos)

import os
import datetime
import json
# import pandas as pd
import numpy as np
import pickle as pickle
from pylab import *
import matplotlib.pyplot as plt
import natsort
from panic import Panic
from gudhi import RipsComplex, plot_persistence_diagram, plot_persistence_barcode
from gph import ripser_parallel

run_panic = Panic()
TDA_LIB = "gudhi"  # "gudhi", "giotto"
MAX_DIM = 2
num_iterations = 20


# init_cond = np.loadtxt("init_cond.txt", dtype=str)
init_cond = open("init_cond.txt", "r")
for line in init_cond.readlines():
    init_prompt = line.strip().split(" ")
    pos_arg = 1
    variations = ["cow", "horse", "duck", "goat", "dog", "cat", "bird"]

    for seed in range(1):
        for substitute in variations:
            timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()  # e.g. 2025-01-24T14:31:47
            # timestamp = timestamp.replace(":", "-")  # To avoid "Invalid argument" error by ":"
            init_prompt[pos_arg] = substitute
            prompt = " ".join(init_prompt)
            # print("Input prompt:", prompt)
            file_name = f"{prompt}_run{seed}"  #timestamp
            print(file_name)

            point_cloud = []
            # output_data = []
            for i in range(int(num_iterations / 2)):
                # print(f"Sequence number {2 * i}")
                image_url = run_panic.generate_image(prompt)
                print(f"Image {2 * i}: {image_url}")

                image_embedding = run_panic.calculate_embedding("vision", image_url)
                point_cloud.append(image_embedding)
                # output_data.append(
                #     {
                #         "seq_no": 2 * i,
                #         "type": "image",
                #         "input": prompt,
                #         # "embedding": image_embedding,
                #     }
                # )

                # BLIP prefixes the string with "Caption: " - remove this
                caption = run_panic.caption_image(image_url)
                if caption.startswith("Caption: "):
                    caption = caption[9:]
                print(f"Caption {2 * i + 1}: {caption}")

                caption_embedding = run_panic.calculate_embedding("text", caption)
                point_cloud.append(caption_embedding)
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

            if TDA_LIB == "giotto":
                dgm_dict = ripser_parallel(point_cloud, maxdim=2, n_threads=-1)
                dgm = []
                for dim in range(MAX_DIM):
                    for bd_pair in dgm_dict['dgms'][dim]:
                        dgm.append((dim, (bd_pair[0], bd_pair[1])))
                # print(dgm)
                ax = plot_persistence_diagram(dgm)
                ax.set_title(f"{prompt}_run{seed}")
                ax.set_aspect("equal")
                # plt.show()
                plt.savefig(f"ph_results/PD_{file_name}.png", bbox_inches="tight")

            elif TDA_LIB == "gudhi":
                rips = RipsComplex(points=point_cloud)  # , max_edge_length=10)

                simplex_tree = rips.create_simplex_tree(max_dimension=3)

                # simplex_tree.compute_persistence()
                # print("Betti nums", simplex_tree.persistent_betti_numbers())

                diagram = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

                ax = plot_persistence_diagram(diagram)
                ax.set_title(prompt)
                ax.set_aspect("equal")
                # plt.show()
                plt.savefig(f"ph_results/PD_{file_name}.png")
                plt.close()

                # ax = plot_persistence_barcode(diagram)
                # ax.set_title(prompt)
                # # plt.show()
                # plt.savefig(f"ph_results/BC_{file_name}.png")
                # plt.close()
            else:
                raise ValueError(f"Unsupported TDA library: {TDA_LIB}")