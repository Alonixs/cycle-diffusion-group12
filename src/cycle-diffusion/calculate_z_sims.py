import matplotlib.pyplot as plt
import numpy as np
import os, os.path
from pathlib import Path
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
import argparse


def main(input, output):
    IN_PATH = Path(input)
    OUTPUT_DIR = Path(output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input Dir: {IN_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")

    # get all files
    all_npy_file_paths = [
        x for x in IN_PATH.iterdir() if x.suffix == ".npy"
    ]  # list paths to the z vectors

    all_image_names = [
        x.name[:19] for x in all_npy_file_paths
    ]  # list of only file name of the z
    unique_cats = list(set(all_image_names))  # get unique cats

    sim_data = {}
    for cat in unique_cats:
        print(f"CAT: {cat}")
        z_paths = []
        for z in all_npy_file_paths:
            if z.name.startswith(cat):
                z_paths.append(z)
        z_paths_sorted = [
            z for z in sorted(z_paths)
        ]  # ensure the z vectors are in order

        if len(z_paths_sorted) != 9:
            print(f"WARNING: {cat} has {len(z_paths_sorted)} images, skipping")
            continue

        cat_array = np.zeros((len(z_paths), 167116800), dtype=np.float32)
        for i, z in enumerate(z_paths_sorted):
            z_vec = np.load(z)
            cat_array[i] = z_vec

        # cosine similarity
        print(f"Calculating cosine similarity for {cat}")
        cos_sim = [
            np.dot(cat_array[0], row) / (norm(cat_array[0]) * norm(row))
            for row in cat_array
        ]  # cosine distance is 1.0 - cosine sim

        assert len(cos_sim) == 9, f"cos_sim should be 9 but got {len(cos_sim)}"

        # euclidean distance
        print(f"Calculating euclidean distance for {cat}")
        euc_dist = [np.linalg.norm(cat_array[0] - row) for row in cat_array]

        assert len(euc_dist) == 9, f"euc_dist should be 9 but got {len(euc_dist)}"

        # save the cosine similarity and euclidean distance
        sim_data[cat] = {
            "cos_sim": cos_sim,
            "euc_dist": euc_dist,
            "mod_order_ref": z_paths_sorted,
        }

    # save the sim_data
    with open(OUTPUT_DIR / f"{IN_PATH.name}_sim_data.pkl", "wb") as f:
        pickle.dump(sim_data, f)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    main(**kwargs)
