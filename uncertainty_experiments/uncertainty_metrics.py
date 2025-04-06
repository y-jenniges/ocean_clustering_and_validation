import logging
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import config


def compute_overlap_matrix(a, b):
    """ Compute overlap matrix of two cluster sets a and b, i.e. the number of grid cells each label combination from
    a and b have in common. """

    merge_columns = ["LATITUDE", "LONGITUDE", "LEV_M"]
    m = pd.merge(a[merge_columns + ["label"]],
                 b[merge_columns + ["label"]],
                 how="outer",
                 on=merge_columns, suffixes=("_a", "_b"))

    # Count grid cells for all combinations of labels from a and b
    counts = pd.DataFrame(m[["label_a", "label_b"]].value_counts(dropna=False))
    overlap_matrix = counts["count"].unstack(fill_value=0).T.drop(np.nan, axis=1, errors="ignore").drop(np.nan, axis=0,
                                                                                                        errors="ignore")

    return overlap_matrix


def compute_overlap(a, b, overlap_matrix):
    """ Compute overlap between two cluster sets a and b. Also called purity. The measure is asymmetric. Averaging both
    overlaps yields the symmetric overlap (last return value). """
    N_a = len(a)
    N_b = len(b)

    overlap_ab = 1 / N_a * overlap_matrix.max(axis=0).sum()
    overlap_ba = 1 / N_b * overlap_matrix.max(axis=1).sum()
    overlap = (overlap_ab + overlap_ba) / 2

    return overlap_ab, overlap_ba, overlap


def compute_f_clustering_accuracy(a, b, overlap_matrix):
    """ Compute F_clustering_accuracy between two cluster sets a and b. """
    N_a = len(a)
    N_b = len(b)
    N = (N_a + N_b) / 2

    column_sum = np.array(overlap_matrix.sum(axis=0), dtype=float)
    row_sum = np.array(overlap_matrix.sum(axis=1), dtype=float)

    divider = np.array([column_sum] * len(row_sum)) + np.array([row_sum] * len(column_sum)).T
    factor = np.divide(1, divider, out=np.zeros(divider.shape),
                       where=divider != 0)  # if dividing by zero, set the factor to zero instead
    f = 2 * overlap_matrix * factor

    fca_ab = 1 / N_a * np.multiply(column_sum, f.max(axis=0)).sum()
    fca_ba = 1 / N_b * np.multiply(row_sum, f.max(axis=1)).sum()
    fca = (fca_ab + fca_ba) / 2

    return fca_ab, fca_ba, fca


def compute_entropies(a, b, overlap_matrix):
    """ Compute entropy measures between clustering a and b, i.e. mutual information, variation of information and
    normalised mutual information. """
    N_a = len(a)
    N_b = len(b)
    N = (N_a + N_b) / 2

    # Entropy of each partition
    h_a = -1 * sum(
        overlap_matrix.sum(axis=0) / N_a * (overlap_matrix.sum(axis=0) / N_a).map(lambda x: np.log(x) if x != 0 else 0))
    h_b = -1 * sum(
        overlap_matrix.sum(axis=1) / N_b * (overlap_matrix.sum(axis=1) / N_b).map(lambda x: np.log(x) if x != 0 else 0))

    # Joint entropy
    h_ab = -1 * overlap_matrix.map(lambda x: x / N * np.log(x / N) if x != 0 else 0).sum().sum()

    # Entropy-related measures
    mi = h_a + h_b - h_ab  # Mutual information
    vi = h_a + h_b - 2 * mi  # Variation of information

    # Normalised mutual information
    if max(h_a, h_b) == 0:
        nmi = 0
    else:
        nmi = mi / max(h_a, h_b)

    return mi, vi, nmi


def compute_uncertainty_metrics(prefix="fixedUmap_dbscan_", ignore_noise=True):
    """ Computes uncertainty metrics: F clustering accuracy, normalised mutual information and overlap by comparing
     existing cluster sets (loaded from files).

     Args:
         prefix (str): Used to load the data as {config.output_dir_uncertainty}{prefix}uncertainty_metrics.csv and as
         prefix of the stored output.
         ignore_noise (bool): Whether to ignore or keep DBSCAN noise.
    """
    logging.info(f"Compute uncertainty metrics (prefix={prefix})...")
    start = time()

    # Compute uncertainty measures
    df_res = []
    num_files = config.n_iterations_uncertainty

    # Iterate over each pair of cluster sets
    for i in tqdm(range(num_files), desc=""):
        for j in range(num_files):
            if j < i:
                # Load cluster sets
                clustering_a = pd.read_csv(f"{config.output_dir_uncertainty}{prefix}{i}.csv")
                clustering_b = pd.read_csv(f"{config.output_dir_uncertainty}{prefix}{j}.csv")

                # Unify float representation to enable seamless merging
                merge_columns = ["LATITUDE", "LONGITUDE", "LEV_M"]
                clustering_a[merge_columns] = clustering_a[merge_columns].round(2)
                clustering_b[merge_columns] = clustering_b[merge_columns].round(2)

                # Ignore noise cluster
                if ignore_noise:
                    clustering_a = clustering_a[clustering_a.label != -1]
                    clustering_b = clustering_b[clustering_b.label != -1]

                # Compute overlap matrix
                overlap_matrix = compute_overlap_matrix(a=clustering_a, b=clustering_b)

                # Compute overlap
                overlap_ab, overlap_ba, symmetric_overlap = compute_overlap(a=clustering_a, b=clustering_b,
                                                                            overlap_matrix=overlap_matrix)
                fca_ab, fca_ba, fca = compute_f_clustering_accuracy(clustering_a, clustering_b, overlap_matrix)
                mi, vi, nmi = compute_entropies(clustering_a, clustering_b, overlap_matrix)
                df_res.append(pd.DataFrame({"clustering_a": [i], "clustering_b": [j],
                                            "overlap_ab": [overlap_ab], "overlap_ba": [overlap_ba],
                                            "overlap": [symmetric_overlap],
                                            "f_accuracy_ab": [fca_ab], "f_accuracy_ba": [fca_ba], "f_accuracy": [fca],
                                            "mutual_infomration": [mi], "variation_of_information": [vi],
                                            "normalized_mutual_information": [nmi]
                                            }))

    # Store results
    logging.info("  Done. Storing results...")
    df_res = pd.concat(df_res)
    df_res.to_csv(f"{config.output_dir_uncertainty}{prefix}uncertainty_metrics.csv", index=False)

    end = time()
    logging.info(f"  Computing uncertainty metrics took {end - start} sec.")


def plot_uncertainty_metrics(prefix="fixedUmap_dbscan_", close_plots=True):
    """ Plot uncertainty metrics: F clustering accuracy, normalised mutual information and overlap.
    Args:
        prefix (str): Used to load the data as {config.output_dir_uncertainty}{prefix}uncertainty_metrics.csv and as
        prefix to store the plots
        close_plots (bool): Whether to only store the plots and directly close them or keep them open.
    """
    logging.info("Plotting uncertainty metrics...")
    # Load uncertainty metrics
    df_res = pd.read_csv(f"{config.output_dir_uncertainty}{prefix}uncertainty_metrics.csv")

    figsize = (6, 4)

    plt.figure(figsize=figsize)
    sns.histplot(df_res.f_accuracy)
    plt.xlabel("F clustering accuracy")
    plt.tight_layout()
    plt.savefig(f"{config.output_dir_plots}{prefix}f_clustering_accuracy.png")
    if close_plots:
        plt.close()
    else:
        plt.show()  # block=True)

    plt.figure(figsize=figsize)
    sns.histplot(df_res.normalized_mutual_information)
    plt.xlabel("Normalised mutual information")
    plt.tight_layout()
    plt.savefig(f"{config.output_dir_plots}{prefix}normalised_mutual_information.png")
    if close_plots:
        plt.close()
    else:
        plt.show()  # block=True)

    plt.figure(figsize=figsize)
    sns.histplot(df_res.overlap * 100)
    plt.xlabel("Overlap [%]")
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir_plots_high_res}{prefix}overlap.png", dpi=1000)
    if close_plots:
        plt.close()
    else:
        plt.show()  # block=True)

    logging.info(
        f"  Mean F clustering accuracy is {df_res.f_accuracy.mean() * 100} +- {df_res.f_accuracy.std() * 100} %")
    logging.info(f"  Mean normalised mutual information is {df_res.normalized_mutual_information.mean() * 100} +- "
                 f"{df_res.normalized_mutual_information.std() * 100} %")
    logging.info(f"  Mean overlap is {df_res.overlap.mean() * 100} +- {df_res.overlap.std() * 100} %")
