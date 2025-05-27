import pandas as pd
from itertools import product
from time import time
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from pathlib import Path

import config


def run_clustering_experiments(df, preprocessing_steps, clustering_algorithms, n_iterations, scores,
                               store_labels=False):
    """
    Perform clustering experiments by building Pipelines with the given models.

    Args:
        df (pandas.DataFrame): Data to run the experiments on (column selection is specified by config.parameters).
        preprocessing_steps (dict): Name and model(s) to run for preprocessing.
        clustering_algorithms (dict): Name and model(s) to run for clustering.
        n_iterations (int): How often each clustering experiment with each hyperparameter combination will be repeated.
        scores (dict): Name and model to run for internal validation.
        store_labels (bool): Whether to store clustering labels or not (default is False).
    """
    logging.info("Starting clustering experiments...")
    logging.getLogger('numba').setLevel(logging.WARNING)  # Hide numba debug messages (numba is used in umap-learn)

    # Counter and path for temp storage of results
    counter = 0

    # Define experiment data
    data = df[config.parameters]

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
                    # Generate output filename
                    if "connectivity" in hyp_params.keys():
                        # Do not use connectivity matrix for file name
                        temp_hyps = {k: v for k, v in hyp_params.items() if k != "connectivity"}
                        result_file_path = f"{config.output_dir_clustering}/internal_validation_iteration{i}_" \
                                           f"{preproc_name}_{cluster_name}_" \
                                           f"{'_'.join([f'{k}{v}' for k, v in temp_hyps.items()])}_connectivity.csv"
                    else:
                        result_file_path = f"{config.output_dir_clustering}/internal_validation_iteration{i}_" \
                                           f"{preproc_name}_{cluster_name}_" \
                                           f"{'_'.join([f'{k}{v}' for k, v in hyp_params.items()])}.csv"

                    # Check if result file already exists
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

                        # Extract transformed data from the pipeline, after preprocessing (before clustering)
                        transformed_data = pipeline[:-1].transform(data)

                        for score_name, score_model in scores.items():
                            logging.info(f"          {score_name}")
                            st = time()
                            if nclusters > 1:
                                score = score_model(transformed_data, labels)
                            else:
                                score = np.nan
                            logging.info(f"          Score computed in: {time() - st}")
                            score_dict[score_name] = score
                        end_time_scores = time()

                        # For DBSCAN, also store number of noise clusters
                        if cluster_name == "dbscan":
                            score_dict["nnoise"] = (labels == -1).sum()

                        # Compute proportion of confetti clusters
                        num_grid_cells = len(labels)
                        unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
                        frac_confetti = np.round((cluster_sizes/num_grid_cells*100 <= 0.1).sum()/len(unique_labels)*100, 2)

                        # Store results
                        results.append({**{
                            "iteration": i,
                            "preprocessing": preproc_name,
                            "clustering": cluster_name,
                            "nclusters": nclusters,
                            "clustering_time": end_time - start_time,
                            "score_time": end_time_scores - start_time_scores,
                            "confetti": frac_confetti,
                        }, **hyp_params, **score_dict})

                        # Convert validation results to a DataFrame and store it
                        pd.DataFrame(results).to_csv(result_file_path, index=False)

                        # Store clustering labels
                        if store_labels:
                            if "connectivity" in hyp_params.keys():
                                # Do not use connectivity matrix for file name
                                temp_path = {k: v for k, v in hyp_params.items() if k != "connectivity"}
                                label_file_path = f"{config.output_dir_clustering}/labels_iteration{i}_" \
                                                  f"{preproc_name}_{cluster_name}_" \
                                                  f"{'_'.join([f'{k}{v}' for k, v in temp_path.items()])}" \
                                                  f"_connectivity.csv"
                            else:
                                label_file_path = f"{config.output_dir_clustering}/labels_iteration{i}_"\
                                                  f"{preproc_name}_{cluster_name}_" \
                                                  f"{'_'.join([f'{k}{v}' for k, v in hyp_params.items()])}.csv"

                            # Create a dataframe containing labels and geo coordinates
                            labels_df = pd.DataFrame(labels, columns=["label"])
                            labels_df[["LATITUDE", "LONGITUDE", "LEV_M"]] = df[["LATITUDE", "LONGITUDE", "LEV_M"]]

                            # If dimensionality reduction (e.g., UMAP) was applied, add embedding coordinates
                            if transformed_data.shape[1] < data.shape[1]:  # Check if dimensionality was reduced
                                umap_cols = [f"e{i}" for i in range(transformed_data.shape[1])]
                                embedding_df = pd.DataFrame(transformed_data, columns=umap_cols)
                                labels_df = pd.concat([labels_df, embedding_df], axis=1)

                            # Save
                            labels_df.to_csv(label_file_path, index=False)

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
            files_to_remove.append(Path(file))

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

    # # Remove previously combined files
    # for file in files_to_remove:
    #     if file.exists():
    #         file.unlink()

    logging.info("Clustering experiments complete.")



