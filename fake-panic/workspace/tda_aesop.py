
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
from gudhi import RipsComplex, SimplexTree, plot_persistence_diagram, plot_persistence_barcode
from gph import ripser_parallel

def create_point_cloud(init_prompt, json=False):
    point_cloud = []
    output_data = []
    output_data.append(
        {
            "seq_no": 0,
            "type": "text",
            "input": init_prompt,
            # "embedding": text_embedding,
        }
    )

    prompt = init_prompt
    for i in range(int(num_iterations / 2)):
        # print(f"Sequence number {2 * i}")
        image_url = run_panic.generate_image(prompt)
        print(f"Image {2 * i + 1}: {image_url}")
        # prompts.write(image_url + "\n")
        image_embedding = run_panic.calculate_embedding("vision", image_url)
        point_cloud.append(image_embedding)
        output_data.append(
            {
                "seq_no": 2 * i + 1,
                "type": "image",
                "input": image_url,
                # "embedding": image_embedding,
            }
        )

        # BLIP prefixes the string with "Caption: " - remove this
        caption = run_panic.caption_image(image_url)
        if caption.startswith("Caption: "):
            caption = caption[9:]
        print(f"Caption {2 * i + 2}: {caption}")
        # prompts.write(caption + "\n")
        caption_embedding = run_panic.calculate_embedding("text", caption)
        point_cloud.append(caption_embedding)
        output_data.append(
            {
                "seq_no": 2 * i + 2,
                "type": "text",
                "input": caption,
                # "embedding": caption_embedding,
            }
        )
        prompt = caption
    # np.savetxt(f"aesop/embedding/{file_name}.txt", point_cloud, fmt="%f")#"%.6f")
    # prompts.close()
    final_prompt = prompt

    if json:
        with open(f"aesop/embedding/{file_name}.json", "w") as embedding_file:
            json.dump(output_data, embedding_file, indent=4, sort_keys=True)

        with open(f"aesop/embedding/{file_name}.json", "r") as json_file:
            json_data = json.load(json_file)
            # input_prompt = ""
            final_prompt = ""
            # point_cloud = []
            for i in range(len(json_data)):
                # embedding_vector = json_data[i]["embedding"]
                in_data = json_data[i]["input"]
                # if i == 0:
                #     input_prompt = in_data
                if i == len(json_data) - 1:
                    final_prompt = in_data
                seq_no = json_data[i]["seq_no"]
                data_type = json_data[i]["type"]
                # point_cloud.append(embedding_vector)

    return point_cloud, final_prompt


run_panic = Panic()
TDA_LIB = "gudhi"  # "gudhi", "giotto"
MAX_DIM = 2
num_iterations = 200
nseed = 10
plot = False

os.system("mkdir aesop aesop/persistence aesop/embedding aesop/diagram")
folder = os.path.join("aesop/extract")
if os.path.isdir(folder):
    directFiles = os.listdir(folder)
    files = natsort.natsorted(directFiles)
    for f1 in files:
        fable_file = str(os.path.join(folder, f1))
        fable_texts = open(fable_file, "r")
        count = 0
        for line in fable_texts.readlines():
            line = line.strip().split(" ")
            prompt = " ".join(line)
            init_prompt = prompt
            for seed in range(nseed):
                file_name = f"{f1[:-4]}_line{count}_seed{seed}"
                print(file_name)
                print("Initial prompt:", init_prompt)

                point_cloud = create_point_cloud(init_prompt)

                # PH computation by Gudhi
                rips = RipsComplex(points=point_cloud)  # , max_edge_length=10)
                simplex_tree = rips.create_simplex_tree(max_dimension=3)
                diagram = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

                # PH computation by Giotto (gph)
                # dgm_dict = ripser_parallel(point_cloud, maxdim=2, n_threads=-1)
                # diagram = []
                # to_save = []
                # for dim in range(1, MAX_DIM):
                #     for bd_pair in dgm_dict['dgms'][dim]:
                #         diagram.append((dim, (bd_pair[0], bd_pair[1])))
                #         to_save.append([dim, bd_pair[0], bd_pair[1]])
                # to_save = np.array(to_save)
                # np.savetxt(f"aesop/persistence/{file_name}.txt", to_save, fmt="%i %f %f")

                ax = plot_persistence_diagram(diagram)
                ax.set_title(final_prompt)
                ax.set_aspect("equal")
                if plot:
                    plt.savefig(f"aesop/diagram/{file_name}.png")
                else:
                    plt.show()

            count += 1
