import matplotlib
import logging
from umap import UMAP
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import config
import hyp_config
from uncertainty_experiments.uncertainty_metrics import compute_uncertainty_metrics, plot_uncertainty_metrics
from uncertainty_experiments.uncertainty_nemi import compute_and_store_nemi_cluster_set, compute_volume


def run_dbscan_on_umap(df):
    """ Run UMAP-DBSCAN n times on a given dataframe, where n is taken from config.py. """
    logging.info(f"  Start computing {config.n_iterations_uncertainty} UMAP-DBSCAN cluster sets...")

    # Scale selected columns
    scaler = MinMaxScaler().fit(df[config.parameters])
    df_scaled = pd.DataFrame(scaler.transform(df[config.parameters]), columns=config.parameters)

    # Get UMAP components
    umap_cols = [f"e{i}" for i in range(hyp_config.umap_hyps["n_components"])]

    for i in tqdm(range(config.n_iterations_uncertainty)):
        # Compute embedding
        df_umap = pd.DataFrame(UMAP(**hyp_config.umap_hyps).fit_transform(df_scaled), columns=umap_cols)

        # Compute DBSCAN
        model = DBSCAN(**hyp_config.dbscan_umap_hyps).fit(df_umap)

        # Assemble output dataframe
        df_out = df.copy()
        df_out[umap_cols] = df_umap[umap_cols]
        df_out["label"] = model.labels_

        # Store results
        df_out.to_csv(f"{config.output_dir_uncertainty}/umap_dbscan_{i}.csv", index=False)

    logging.info("    Completed UMAP-DBSCAN runs.")


def run_dbscan_on_fixed_umap(df):
    """ Run DBSCAN on a fixed embedding n times on a given dataframe, where n is taken from config.py. """
    logging.info(f"Start computing {config.n_iterations_uncertainty} fixedUMAP-DBSCAN cluster sets...")

    # Scale selected columns
    scaler = MinMaxScaler().fit(df[config.parameters])
    df_scaled = pd.DataFrame(scaler.transform(df[config.parameters]), columns=config.parameters)

    # Get UMAP components
    umap_cols = [f"e{i}" for i in range(hyp_config.umap_hyps["n_components"])]

    # Compute one embedding (fixed for all DBSCAN runs)
    logging.info("    Compute embedding...")
    df_umap = pd.DataFrame(UMAP(**hyp_config.umap_hyps).fit_transform(df_scaled), columns=umap_cols)

    # Compute DBSCAN on UMAP multiple times while shuffling the input data
    logging.info("    Start computing DBSCAN runs on fixed embedding...")
    for i in tqdm(range(config.n_iterations_uncertainty)):
        # Shuffling data (shuffling needed since DBSCAN is only sensitive to sequence of input)
        idx = np.random.permutation(df.index)
        temp_df_umap = df_umap.reindex(idx)
        temp_df = df.reindex(idx)

        # Compute DBSCAN on given embedding
        model = DBSCAN(**hyp_config.dbscan_umap_hyps).fit(temp_df_umap)
        temp_df["label"] = model.labels_

        # Store results (create dir if it does not exist)
        temp_df.to_csv(f"{config.output_dir_uncertainty}/fixedUmap_dbscan_{i}.csv", index=False)

    logging.info("    Completed fixedUMAP-DBSCAN runs.")


def compute_nemi_uncertainty(df):
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

    prefix = "volume_"
    df = pack[0][1]
    for base_id in tqdm(range(config.n_iterations_uncertainty)):
        compute_and_store_nemi_cluster_set(df=df, pack=pack, base_id=base_id, prefix=prefix)


def run_uncertainty_experiments(df):
    logging.getLogger('numba').setLevel(logging.WARNING)  # Hide numba debug messages (numba is used in umap-learn)
    matplotlib.set_loglevel("warning")  # Hide matplotlib debug messages
    logging.info("Start uncertainty experiments...")

    # Run DBSCAN on fixed UMAP
    run_dbscan_on_fixed_umap(df)
    compute_uncertainty_metrics(prefix="fixedUmap_dbscan_", ignore_noise=False)
    plot_uncertainty_metrics(prefix="fixedUmap_dbscan_")

    # Run UMAP-DBSCAN multiple times
    run_dbscan_on_umap(df)
    compute_uncertainty_metrics(prefix="umap_dbscan_", ignore_noise=False)
    plot_uncertainty_metrics(prefix="umap_dbscan_")
    logging.info("Finished uncertainty experiments.")

    # Compute NEMI uncertainty
    # compute_nemi_uncertainty(df)
    # logging.info("Finished NEMI uncertainty experiments.")

    logging.info("Uncertainty experiments finished.")
