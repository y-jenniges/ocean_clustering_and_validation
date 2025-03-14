import logging
import pandas as pd
import matplotlib
import config
from uncertainty_experiments.uncertainty_dbscan import run_dbscan_on_fixed_umap, run_dbscan_on_umap, \
    compute_uncertainty_metrics, plot_uncertaintiy_metrics


def run_uncertainty_experiments(df):
    logging.getLogger('numba').setLevel(logging.WARNING)  # Hide numba debug messages (numba is used in umap-learn)
    matplotlib.set_loglevel("warning")  # Hide matplotlib debug messages
    logging.info("Start uncertainty experiments...")

    # Run DBSCAN on fixed UMAP
    run_dbscan_on_fixed_umap(df)
    compute_uncertainty_metrics(prefix="fixedUmap_dbscan_", ignore_noise=True)
    plot_uncertaintiy_metrics(prefix="fixedUmap_dbscan_")

    # Run UMAP-DBSCAN multiple times
    run_dbscan_on_umap(df)
    compute_uncertainty_metrics(prefix="umap_dbscan_", ignore_noise=False)  # todo check if you ignored noise before!!
    plot_uncertaintiy_metrics(prefix="umap_dbscan_")

    logging.info("Finished uncertainty experiments.")

    # Run UMAP multiple times

    # Compute NEMI uncertainty

    #  return