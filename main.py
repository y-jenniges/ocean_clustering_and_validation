import sys
import pandas as pd
import numpy as np
import logging
from umap import UMAP
from clustering_experiments.clustering_experiments import run_clustering_experiments
from preparation.preparation import grid_and_impute_data, prepare_database
from validation.validation import run_validation
from uncertainty_experiments.uncertainty_experiments import run_uncertainty_experiments
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Define output directory
output_dir = "output/"

# Configure logging (store in logging file and in console)
logging.basicConfig(level=logging.DEBUG,  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
                    handlers=[logging.FileHandler(output_dir + "logs.log"),
                              logging.StreamHandler(stream=sys.stdout)])

# Prepare data in database and load it
parameters = ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"]
prepare_database(parameters=parameters, quality_flags=None, source_db_path="../../data/comfort.sqlite",
                 dest_db_path="output/custom.db")
df = grid_and_impute_data(db_path="output/custom.db", parameters=parameters, output_dir=output_dir)

# df = pd.read_csv("output/wide_table_knn.csv")

# Perform clustering experiments (and internal validation via scores)
logging.info("Starting clustering experiments...")
n_iterations = 10
umap_hyps = {"n_neighbors": 20, "min_dist": 0.0, "n_components": 3}
preprocessings = {"minmax": [MinMaxScaler], "minmax_umap": [MinMaxScaler, UMAP(**umap_hyps)]}
algorithms_and_hyps = {"kmeans": (KMeans, {"n_clusters": list(range(2, 16)) + [20, 30, 40, 50, 60],
                                           "n_init": ["auto"]}),
                       "ward": (AgglomerativeClustering, {"n_clusters": range(2, 31), "distance_threshold": [None],
                                                          "linkage": ["ward"]}),
                       "dbscan": (DBSCAN, {"eps": np.linspace(0.01, 0.2, 60), "min_samples": range(2, 12)}),
                       # "optics": (OPTICS, {"min_samples": range(1, 16), "max_eps": [np.inf]})
                       }
scores = {"silhouette": silhouette_score,
          "davies_bouldin": davies_bouldin_score,
          "calinski_harabasz": calinski_harabasz_score}

df_clusterings = run_clustering_experiments(data=df[parameters],
                                            preprocessing_steps=preprocessings,
                                            clustering_algorithms=algorithms_and_hyps,
                                            n_iterations=n_iterations,
                                            scores=scores,
                                            output_dir=output_dir)

# Run uncertainty experiments
# run_uncertainty_experiments(output_directory=output_dir)
