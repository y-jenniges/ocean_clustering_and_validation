import os
# Set thread limits
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys
import logging
import pandas as pd
from time import time
from pathlib import Path
from clustering_experiments.main_clustering_experiments import run_clustering_experiments
from preparation.preparation import grid_and_impute_data, prepare_database
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
import copy

import config
from uncertainty_experiments.main_uncertainty_experiments import run_uncertainty_experiments


def create_output_directories():
    """ Create all output directories if they do not exist yet. """
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_clustering).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_plots).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_plots_high_res).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_uncertainty).mkdir(parents=True, exist_ok=True)


def add_spatially_constrained_ward_to_config(df_in,
                                             lat_col="LATITUDE", lon_col="LONGITUDE", depth_col="LEV_M",
                                             depth_scale=1000.0, n_neighbors=10,
                                             algorithm_name="spatial_ward"):
    """
    Adds a spatially constrained AgglomerativeClustering (Ward) entry to config.algorithms_and_hyps.

    Args:
        df_in: pandas DataFrame with 'lat', 'lon', 'depth' columns
        lat_col, lon_col, depth_col: names of the spatial columns
        depth_scale: scale depth to kilometers (default: 1000)
        n_neighbors: neighbors for k-NN graph
        algorithm_name: key name to add to config

    Returns:
        updated_algorithms: new dictionary including the spatial_ward algorithm
    """
    logging.info("Compute spatial connectivity matrix...")

    # Convert lat/lon to radians
    lat_rad = np.radians(df_in[lat_col].values)
    lon_rad = np.radians(df_in[lon_col].values)

    # Project to 3D Cartesian coordinates (Earth radius in km)
    R = 6371.0
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)

    # Include scaled depth as 4th dimension
    depth_scaled = df_in[depth_col].values / depth_scale
    X_spatial = np.vstack([x, y, z, depth_scaled]).T

    # Build connectivity graph
    connectivity_matrix = kneighbors_graph(X_spatial, n_neighbors=n_neighbors, include_self=False)

    # Copy the algorithm config and add new entry
    updated_algorithms = copy.deepcopy(config.algorithms_and_hyps)
    updated_algorithms[algorithm_name] = (
        AgglomerativeClustering,
        {
            "n_clusters": range(2, 31),
            "linkage": ["ward"],
            "connectivity": [connectivity_matrix]
        }
    )

    return updated_algorithms


if __name__ == "__main__":
    prepare_data = False
    run_clusterings = True
    run_uncertainties = False

    # Create all output directories
    print("Creating output directories...")
    create_output_directories()

    # Prepare data in database and load it
    if prepare_data:
        start = time()
        # Configure logging (store in logging file and in console)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler(config.output_dir + "logs_prepare.log"),
                                      logging.StreamHandler(stream=sys.stdout)])

        prepare_database(parameters=config.parameters, quality_flags=config.quality_flags,
                         temperature_to_potential=True,
                         source_db_path=config.source_db_path, dest_db_path=config.dest_db_path)
        df = grid_and_impute_data(db_path=config.dest_db_path, grid_config=config.grid_config,
                                  bathymetry_path=config.bathymetry_path,
                                  parameters=config.parameters,
                                  output_dir=config.output_dir)
        end = time()
        logging.info(f"Database preparation, gridding and imputation took {end - start} seconds.")
    else:
        df = pd.read_csv(config.output_dir + "/wide_table_knn.csv")

    # Perform clustering experiments (and internal validation via scores)
    if run_clusterings:
        start = time()
        # Configure new logging (store in logging file and in console)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler(config.output_dir + "logs_clustering.log"),
                                      logging.StreamHandler(stream=sys.stdout)])

        # Add spatially constrained Ward to config
        algorithms_with_spatial = add_spatially_constrained_ward_to_config(df, n_neighbors=21)

        # Run experiments
        run_clustering_experiments(df=df,
                                   preprocessing_steps=config.preprocessings,
                                   clustering_algorithms=algorithms_with_spatial,
                                   n_iterations=config.n_iterations,
                                   scores=config.scores,
                                   store_labels=True)
        end = time()
        logging.info(f"Running the clustering experiments took {end - start} seconds.")

    # Run uncertainty experiments
    if run_uncertainties:
        # Configure new logging (store in logging file and in console)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler(config.output_dir + "logs_uncertainty.log"),
                                      logging.StreamHandler(stream=sys.stdout)])

        run_uncertainty_experiments(df=df)
