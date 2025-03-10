import pandas as pd
from itertools import product
from time import time
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from pathlib import Path


def run_clustering_experiments(data, preprocessing_steps, clustering_algorithms, n_iterations, scores, output_dir):
    logging.info("Starting clustering experiments...")
    # Counter for temp storage of results
    counter = 0

    # Temporary output directory
    temp_output_dir = Path(output_dir + "temp/")
    temp_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

    # Perform each preprocessing-clustering_hyperparameters combination 10 times
    for i in range(n_iterations):
        # Iterate over preprocessing and clustering methods
        for preproc_name, preproc_steps in preprocessing_steps.items():
            logging.info(preproc_name, preproc_steps)

            for cluster_name, (ClusterAlgo, param_grid) in clustering_algorithms.items():
                # Generate all hyperparameter combinations
                hyp_param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

                logging.info("  " + cluster_name)

                # Iterate over hyperparameter combinations
                for hyp_params in hyp_param_combinations:
                    # Reset results
                    results = []

                    # Measure time to run the pipeline
                    start_time = time()

                    # Construct pipeline dynamically
                    steps = [(f"{preproc_name}_step_{idx}", step() if callable(step) else step) for idx, step in
                             enumerate(preproc_steps)]  # Preprocessing steps
                    steps.append((cluster_name, ClusterAlgo(**hyp_params)))  # Clustering step
                    pipeline = Pipeline(steps)
                    logging.info(f"  {steps}")

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
                        "hyp_params": hyp_params,
                        "clustering_time": end_time - start_time,
                        "score_time": end_time_scores - start_time_scores
                    }, **score_dict})

                    # Convert results to a DataFrame and store it
                    pd.DataFrame(results).to_csv(
                        f"{output_dir}temp/clustering_experiments_{cluster_name}_{counter}.csv", index=False)
                    counter = counter + 1

    # Combine and store results per clustering algorithm
    for cluster_name in clustering_algorithms.keys():
        files_to_combine = [file for file in Path().glob(f"*_{cluster_name}:*.csv")]
        dfs = [pd.read_csv(file) for file in files_to_combine]
        dfs = pd.concat(dfs, ignore_index=True, axis=0)
        dfs.to_csv(f"{output_dir}internal_validation_{cluster_name}.csv")

    # Remove temporary output directory
    temp_output_dir.rmdir()
    logging.info("Clustering experiments complete.")
