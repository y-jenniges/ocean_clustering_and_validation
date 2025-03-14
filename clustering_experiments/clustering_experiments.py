import pandas as pd
from itertools import product
from time import time
import numpy as np
import logging
import shutil
from sklearn.pipeline import Pipeline
from pathlib import Path


def run_clustering_experiments(data, preprocessing_steps, clustering_algorithms, n_iterations, scores, output_dir, store_labels=False):
    """
    Perform clustering experiments by building Pipelines with the given models.

    Args:
        data (pandas.DataFrame): Data to run the experiments on.
        preprocessing_steps (dict): Name and model(s) to run for preprocessing.
        clustering_algorithms (dict): Name and model(s) to run for clustering.
        n_iterations (int): How often each clustering experiment with each hyperparameter combination will be repeated.
        scores (dict): Name and model to run for internal validation.
        output_dir (str): Directory where to store results.
        store_labels (bool): Whether to store clustering labels or not (default is False).
    """
    logging.info("Starting clustering experiments...")
    logging.getLogger('numba').setLevel(logging.WARNING)  # Hide numba debug messages (numba is used in umap-learn)

    # Counter and path for temp storage of results
    counter = 0

    # Temporary output directory
    temp_output_dir = Path(output_dir + "temp/")
    temp_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    # Perform each preprocessing-clustering_hyperparameters combination 10 times
    for i in range(n_iterations):
        logging.info(f"Iteration {i}")
        # Iterate over preprocessing and clustering methods
        for preproc_name, preproc_steps in preprocessing_steps.items():
            logging.info(f"  Preprocessing {preproc_name}")

            for cluster_name, (ClusterAlgo, param_grid) in clustering_algorithms.items():
                # Generate all hyperparameter combinations
                hyp_param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

                logging.info("    " + cluster_name)

                # Iterate over hyperparameter combinations
                for hyp_params in hyp_param_combinations:
                    # Check if result file already exists
                    result_file_path = f"{temp_output_dir}/internal_validation_{cluster_name}_{counter}.csv"
                    if not Path(result_file_path).is_file():
                        # Reset results
                        results = []

                        # Measure time to run the pipeline
                        start_time = time()

                        # Construct pipeline dynamically
                        steps = [(f"{preproc_name}_step_{idx}", step() if callable(step) else step) for idx, step in
                                 enumerate(preproc_steps)]  # Preprocessing steps
                        steps.append((cluster_name, ClusterAlgo(**hyp_params)))  # Clustering step
                        pipeline = Pipeline(steps)
                        logging.info(f"      {steps}")

                        # Fit and predict
                        labels = pipeline.fit_predict(data)

                        end_time = time()

                        # Apply internal validation (scores)
                        start_time_scores = time()
                        nclusters = len(np.unique(labels))
                        score_dict = {}
                        for score_name, score_model in scores.items():
                            if nclusters > 1:
                                score = score_model(data, labels)
                            else:
                                score = np.nan
                            score_dict[score_name] = score
                        end_time_scores = time()

                        # Store results
                        results.append({**{
                            "iteration": i,
                            "preprocessing": preproc_name,
                            "clustering": cluster_name,
                            "clustering_time": end_time - start_time,
                            "score_time": end_time_scores - start_time_scores,
                            "clustering_id": counter
                        }, **hyp_params, **score_dict})

                        # Convert validation results to a DataFrame and store it
                        pd.DataFrame(results).to_csv(result_file_path, index=False)

                        # Store clustering labels
                        if store_labels:
                            pd.DataFrame(labels, columns=["label"]).to_csv(f"{temp_output_dir}/external_validation_"
                                                                           f"{cluster_name}_{counter}.csv", index=False)
                    else:
                        logging.info(f"    Result file for {result_file_path} already exists. Skipping this "
                                     f"experiment.")
                    counter = counter + 1

    # Combine and store internal validation results per clustering algorithm
    logging.info("Combining clustering experiment result files...")
    for cluster_name in clustering_algorithms.keys():
        files_to_combine = [file for file in Path().glob(f"{temp_output_dir}/internal_validation_{cluster_name}_*.csv")]
        dfs = [pd.read_csv(file) for file in files_to_combine]
        dfs = pd.concat(dfs, ignore_index=True, axis=0)
        dfs.to_csv(f"{output_dir}internal_validation_{cluster_name}.csv", index=False)

    # Remove each file
    for file in temp_output_dir.glob("internal_validation*"):
        if file.is_file():  # Ensure it's a file, not a directory
            file.unlink()

    # Remove temporary output directory
    # if temp_output_dir.exists() and temp_output_dir.is_dir():
        # shutil.rmtree(temp_output_dir)
        # shutil.rmtree(f"{temp_output_dir}/internal_validation_*")
    logging.info("Clustering experiments complete.")
