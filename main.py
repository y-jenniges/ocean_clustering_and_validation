import sys
import pandas as pd
import numpy as np
import logging
from umap import UMAP
from clustering_experiments.preprocess import preprocess
from clustering_experiments.clustering_experiments import run_clustering_experiments
from preparation.preparation import load_data, prepare_database
from validation.validation import run_validation
from uncertainty_experiments.uncertainty_experiments import run_uncertainty_experiments

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS


# Define output directory
output_dir = "output/"

# Configure logging (store in logging file and in console)
logging.basicConfig(level=logging.DEBUG,  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
                    handlers=[logging.FileHandler(output_dir + "logs.log"),
                              logging.StreamHandler(stream=sys.stdout)])

# Prepare data in database and load it
parameters = ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"]
prepare_database(parameters=parameters, quality_flags=None, source_db_path="../../data/comfort.sqlite", dest_db_path="output/custom.db")
df = load_data(parameters=parameters)

# Apply MinMaxScaling to all parameters
scaler = MinMaxScaler().fit(df[parameters])
df[parameters] = scaler.transform(df[parameters])

# Perform clustering experiments
n_iterations = 10
preprocessings = {"": None, "umap": UMAP()}
algorithms_and_hyps = {"kmeans": (KMeans, {"n_clusters": list(range(2, 16)) + [20, 30, 40, 50, 60], "n_init": ["auto"]}),
                       "ward": (AgglomerativeClustering, {"n_clusters": range(2, 31), "distance_threshold": [None], "linkage": ["ward"]}),
                       "dbscan": (DBSCAN, {"eps": np.linspace(0.01, 0.2, 60), "min_samples": range(2, 12)}),
                       "optics": (OPTICS, {"min_samples": range(1, 16), "max_eps": [np.inf]})
                       }
df_clusterings = run_clustering_experiments(data=df,
                                            preprocessing_steps=preprocessings,
                                            clustering_algorithms=algorithms_and_hyps,
                                            n_iterations=n_iterations,
                                            output_directory=output_dir)

# Compute validation
scores = ["silhouette", "davies-bouldin", "calinski-harabasz"]
run_validation(scores=scores, output_directory=output_dir)

# Run uncertainty experiments
run_uncertainty_experiments(output_directory=output_dir)

