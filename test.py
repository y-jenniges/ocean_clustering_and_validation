import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
from glob import glob
import config
from uncertainty_experiments.uncertainty_nemi import compute_and_store_nemi_cluster_set, compute_volume

# sbatch submit.slurm


def load_shared_data():
    print("Loading shared data...")
    df = pd.read_csv(config.output_dir + "/wide_table_knn.csv")

    # Load all clustering runs
    pack = []
    for filename in tqdm(glob.glob(f"{config.output_dir_uncertainty}umap_dbscan_*.csv")):
        try:
            i = int(filename.split("_")[-1].rstrip(".csv"))

            # Load data
            df = pd.read_csv(filename)
            df.label = df.label + 1  # Make sure no label is -1 (noise in DBSCAN)
            pack.append([i, df])

        except ValueError:
            print(f"Skipping invalid file: {filename}")
            continue

    # Sort by size and compute volumes
    for i, cl in tqdm(pack):
        clusters = cl.label
        n_clusters = len(clusters.unique())
        hist, _ = np.histogram(clusters, np.arange(n_clusters + 1))
        sorted_clusters = np.argsort(hist)[::-1]  # Sort from largest to smallest (if same size, last cluster is taken)
        new_labels = np.full(clusters.shape, np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label
        cl["sorted_label"] = new_labels

        # Compute volume
        dlat, dlon = [1, 1]
        depths = np.append(np.sort(cl.LEV_M.unique()), 5000)
        cl.loc[:, "volume"] = cl.apply(compute_volume, axis=1, args=(dlat, dlon, depths))  # Careful with rounding

    return pack


def main(base_id: int, pack):
    prefix = "volume_"
    df = pack[0][1]
    compute_and_store_nemi_cluster_set(df=df, pack=pack, base_id=base_id, prefix=prefix)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <base_id>")
        sys.exit(1)

    try:
        base_id = int(sys.argv[1])
    except ValueError:
        print("base_id must be an integer.")
        sys.exit(1)

    pack = load_shared_data()
    main(base_id, pack)
