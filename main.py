import sys
import logging
import pandas as pd
from time import time
from pathlib import Path
from clustering_experiments.clustering_experiments import run_clustering_experiments
from preparation.preparation import grid_and_impute_data, prepare_database
import config
from uncertainty_experiments.uncertainty_experiments import run_uncertainty_experiments


def create_output_directories():
    """ Create all output directories if they do not exist yet. """
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_clustering).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_plots).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_plots_high_res).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir_uncertainty).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    prepare_data = False
    run_clusterings = True
    run_uncertainties = True

    # Create all output directories
    logging.info("Creating output directories...")
    create_output_directories()

    # Prepare data in database and load it
    if prepare_data:
        start = time()
        # Configure logging (store in logging file and in console)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler(config.output_dir + "logs_prepare.log"),
                                      logging.StreamHandler(stream=sys.stdout)])

        prepare_database(parameters=config.parameters, quality_flags=config.quality_flags, temperature_to_potential=True,
                         source_db_path=config.source_db_path, dest_db_path=config.dest_db_path)
        df = grid_and_impute_data(db_path=config.dest_db_path, grid_config=config.grid_config,
                                  bathymetry_path=config.bathymetry_path,
                                  parameters=config.parameters,
                                  output_dir=config.output_dir)
        end = time()
        logging.info(f"Database preparation, gridding and imputation took {end-start} seconds.")
    else:
        df = pd.read_csv(config.output_dir + "/wide_table_knn.csv")

    # Perform clustering experiments (and internal validation via scores)
    if run_clusterings:
        start = time()
        # Configure new logging (store in logging file and in console)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler(config.output_dir + "logs_clustering.log"),
                                      logging.StreamHandler(stream=sys.stdout)])

        df_clusterings = run_clustering_experiments(data=df[config.parameters],
                                                    preprocessing_steps=config.preprocessings,
                                                    clustering_algorithms=config.algorithms_and_hyps,
                                                    n_iterations=config.n_iterations,
                                                    scores=config.scores,
                                                    store_labels=True)
        end = time()
        logging.info(f"Running the clustering experiments took {end-start} seconds.")

    # Run uncertainty experiments
    if run_uncertainties:
        # Configure new logging (store in logging file and in console)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler(config.output_dir + "logs_uncertainty.log"),
                                      logging.StreamHandler(stream=sys.stdout)])

        run_uncertainty_experiments(df=df)
