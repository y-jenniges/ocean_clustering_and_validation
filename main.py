import sys
import logging
from clustering_experiments.clustering_experiments import run_clustering_experiments
from preparation.preparation import grid_and_impute_data, prepare_database
from uncertainty_experiments.uncertainty_experiments import run_uncertainty_experiments
import config


if __name__ == "main":
    # Configure logging (store in logging file and in console)
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(config.output_dir + "logs.log"),
                                  logging.StreamHandler(stream=sys.stdout)])

    # Prepare data in database and load it
    prepare_database(parameters=config.parameters, quality_flags=config.quality_flags,
                     source_db_path=config.source_db_path, dest_db_path=config.dest_db_path)
    df = grid_and_impute_data(db_path=config.dest_db_path, grid_config=config.grid_config,
                              bathymetry_path=config.bathymetry_path, parameters=config.parameters,
                              output_dir=config.output_dir)

    # Perform clustering experiments (and internal validation via scores)
    df_clusterings = run_clustering_experiments(data=df[config.parameters],
                                                preprocessing_steps=config.preprocessings,
                                                clustering_algorithms=config.algorithms_and_hyps,
                                                n_iterations=config.n_iterations,
                                                scores=config.scores,
                                                output_dir=config.output_dir)

    # Run uncertainty experiments
    # run_uncertainty_experiments(output_directory=output_dir)
