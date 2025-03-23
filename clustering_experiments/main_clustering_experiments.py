import pandas as pd
from itertools import product
from time import time
import numpy as np
import logging
import shutil
from sklearn.pipeline import Pipeline
from pathlib import Path

import config


def run_clustering_experiments(data, preprocessing_steps, clustering_algorithms, n_iterations, scores, store_labels=False):
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
                    result_file_path = f"{config.output_dir_clustering}/internal_validation_iteration{i}_" \
                                       f"{preproc_name}_{cluster_name}_" \
                                       f"{'_'.join([f'{k}{v}' for k, v in hyp_params.items()])}.csv"
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

                        # For DBSCAN, also store number of noise clusters
                        if cluster_name == "dbscan":
                            score_dict["nnoise"] = (labels == -1).sum()

                        # Store results
                        results.append({**{
                            "iteration": i,
                            "preprocessing": preproc_name,
                            "clustering": cluster_name,
                            "nclusters": nclusters,
                            "clustering_time": end_time - start_time,
                            "score_time": end_time_scores - start_time_scores,
                            # "clustering_id": counter
                        }, **hyp_params, **score_dict})

                        # Convert validation results to a DataFrame and store it
                        pd.DataFrame(results).to_csv(result_file_path, index=False)

                        # Store clustering labels
                        if store_labels:
                            label_file_path = f"{config.output_dir_clustering}/labels_iteration{i}_"\
                                              f"{preproc_name}_{cluster_name}_" \
                                              f"{'_'.join([f'{k}{v}' for k, v in hyp_params.items()])}.csv"
                            pd.DataFrame(labels, columns=["label"]).to_csv(label_file_path, index=True)
                    else:
                        logging.info(f"      Result file for {result_file_path} already exists. Skipping this "
                                     f"experiment.")
                    counter = counter + 1

    # Combine and store internal validation results per clustering algorithm
    logging.info("Combining clustering experiment result files...")
    files_to_remove = []
    for cluster_name in clustering_algorithms.keys():
        files_to_combine = [file for file in Path().glob(
            f"{config.output_dir_clustering}/internal_validation_iteration*_*_{cluster_name}_*.csv")]
        files_to_remove = files_to_remove + files_to_combine
        dfs = [pd.read_csv(file) for file in files_to_combine]
        dfs = pd.concat(dfs, ignore_index=True, axis=0)
        dfs.to_csv(f"{config.output_dir_clustering}internal_validation_{cluster_name}.csv", index=False)

    # Remove previously combined files
    for file in files_to_remove:
        if file.exists():
            file.unlink()

    # Combine label files
    files_to_remove = []
    for cluster_name in clustering_algorithms.keys():
        dfs = []
        files_to_combine = [file for file in Path().glob(
            f"{config.output_dir_clustering}/labels_iteration*_*_{cluster_name}_*.csv")]

        for file in files_to_combine:
            # Determine parameters from filename
            i = str(file).split("iteration")[1].split("_")[0]
            preproc = str(file).split(f"iteration{i}_")[1].split(f"_{cluster_name}")[0]

            # Only combine iteration 0 labels for DBSCAN (otherwise the file gets too big)
            if cluster_name == "dbscan" and int(i) != 0:
                print(i)
                continue

            # Read file
            t = pd.read_csv(file)

            # Add parameters to temporary dataframe
            t["iteration"] = i
            t["preprocessing"] = preproc

            # Add the files to the ones that can be removed
            files_to_remove.append(str(file))

            # Store each hyperparameter in a separate column
            for hyp in config.algorithms_and_hyps[cluster_name][1].keys():
                t[hyp] = str(file).split(hyp)[1].split("_")[0].rstrip(".csv")

            dfs.append(t)

        dfs = pd.concat(dfs, ignore_index=True, axis=0)
        if cluster_name == "dbscan":
            out_file = f"{config.output_dir_clustering}labels_{cluster_name}_iteration0.csv"
        else:
            out_file = f"{config.output_dir_clustering}labels_{cluster_name}.csv"
        dfs.to_csv(out_file, index=False)

    # Remove previously combined files
    for file in files_to_remove:
        if file.exists():
            file.unlink()

    logging.info("Clustering experiments complete.")



