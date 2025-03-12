""" Final hyperparameter combinations for the six conducted experiments. """
# Final hyperparameters for UMAP
umap_hyps = {"n_neighbors": 20, "min_dist": 0.0, "n_components": 3}

# Final hyperparameters for KMeans on original data
kmeans_original_hyps = {"n_clusters": 2, "n_init": "auto"}

# Final hyperparameters for KMeans on embedded data
kmeans_umap_hyps = {"n_clusters": 10}

# Final hyperparameters for Ward on original data
ward_original_hyps = {"n_clusters": 2, "linkage": "ward", "distance_threshold": None, "compute_distances": True}

# Final hyperparameters for Ward on embedded data
ward_umap_hyps = {"n_clusters": 24, "linkage": "ward", "distance_threshold": None, "compute_distances": True}

# Final hyperparameters for DBSCAN on original data
dbscan_original_hyps = {"eps": 0.11949153, "min_samples": 11}

# Final hyperparameters for DBSCAN on embedded data
dbscan_umap_hyps = {"eps": 0.10983051, "min_samples": 4}
