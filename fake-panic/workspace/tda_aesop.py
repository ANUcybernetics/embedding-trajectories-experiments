
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
nseed = 3

os.system("mkdir aesop aesop/persistence_dgm aesop/point_cloud")
folder = os.path.join("aesop/extract")
if os.path.isdir(folder):
    directFiles = os.listdir(folder)
    files = natsort.natsorted(directFiles)
    for f in files:
        fable_file = str(os.path.join(folder, f))
        fable_texts = open(fable_file, "r")
        count = 0
        for line in fable_texts.readlines():
            line = line.strip().split(" ")
            prompt = " ".join(line)
            init_prompt = prompt
            for seed in range(nseed):
                file_name = f"{f[:-4]}_line{count}_seed{seed}"
                print(file_name)

                # point_cloud = []
                # # output_data = []
                # for i in range(int(num_iterations / 2)):
                #     # print(f"Sequence number {2 * i}")
                #     image_url = run_panic.generate_image(prompt)
                #     print(f"Image {2 * i}: {image_url}")
                #
                #     image_embedding = run_panic.calculate_embedding("vision", image_url)
                #     point_cloud.append(image_embedding)
                #     # output_data.append(
                #     #     {
                #     #         "seq_no": 2 * i,
                #     #         "type": "image",
                #     #         "input": prompt,
                #     #         # "embedding": image_embedding,
                #     #     }
                #     # )
                #
                #     # BLIP prefixes the string with "Caption: " - remove this
                #     caption = run_panic.caption_image(image_url)
                #     if caption.startswith("Caption: "):
                #         caption = caption[9:]
                #     print(f"Caption {2 * i + 1}: {caption}")
                #
                #     caption_embedding = run_panic.calculate_embedding("text", caption)
                #     point_cloud.append(caption_embedding)
                #     prompt = caption
                # np.savetxt(f"aesop/point_cloud/{file_name}.txt", point_cloud, fmt="%f")#"%.6f")
                #
                #
                # rips = RipsComplex(points=point_cloud)  # , max_edge_length=10)
                # simplex_tree = rips.create_simplex_tree(max_dimension=3)
                # diagram = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
                #
                # ax = plot_persistence_diagram(diagram)
                # ax.set_title(prompt)
                # ax.set_aspect("equal")
                # # plt.show()
                # plt.savefig(f"aesop/persistence_dgm/PD_{file_name}.png")
                # plt.close()
                
            count += 1
