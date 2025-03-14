import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Output directories
output_dir = "output/"
output_dir_plots = "output/plots/"
output_dir_plots_high_res = "output/plots_high_res/"

# Parameters to impute
parameters = ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"]

# Quality flags to filter for
quality_flags = [["pqf1", ">0"], ["pqf2", ">2"], ["sqf", ">=-1"]]

# Original COMFORT database
source_db_path = "../../data/comfort.sqlite"

# Name of new database that will be created
dest_db_path = "output/custom.db"

# Specification of the grid
depth_levels = [0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000]
grid_config = {
    "param_tables": ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE", "P_PHOSPHATE"],
    "lat_min": 0,
    "lat_max": 70,
    "dlat": 1,
    "lon_min": -77,
    "lon_max": 30,
    "dlon": 1,
    "z_min": None,
    "z_max": None,
    "dz": None,
    "z_array": np.array(depth_levels),
    "time_min": "1772-01-01 00:00:00",
    "time_max": "2020-07-08 04:45:00",
    "mode": "Y",
    "selection": None,
    "dtime": 300,
    "note": "Northern Atlantic, all times, 13 depth steps, 6 params"
}
bathymetry_path = "../../data/bathymetry/gebco_2022_sub_ice_topo/GEBCO_2022_sub_ice_topo.nc"

# Configuration for clustering experiments
output_dir_clustering = "output/clustering/"
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

# Configuration for uncertainty experiments
output_dir_uncertainty = "output/uncertainty/"
n_iterations_uncertainty = 100
