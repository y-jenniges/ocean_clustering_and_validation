""" Final hyperparameter combinations for the six conducted experiments. """
# Final hyperparameters for KMeans on original data
kmeans_original_hyps = {"n_clusters": 2, "n_init": "auto"}

# Final hyperparameters for KMeans on embedded data
kmeans_umap_hyps = {"n_clusters": 10, "n_init": "auto"}

# Final hyperparameters for Ward on original data
ward_original_hyps = {"n_clusters": 2, "linkage": "ward"}

# Final hyperparameters for Ward on embedded data
ward_umap_hyps = {"n_clusters": 23, "linkage": "ward"}

# Final hyperparameters for DBSCAN on original data
dbscan_original_hyps = {"eps": 0.11949153, "min_samples": 11}

# Final hyperparameters for DBSCAN on embedded data
dbscan_umap_hyps = {"eps": 0.10661017, "min_samples": 4}

# Final hyperparameters for Spatial Ward on original data
spatial_original_hyps = {"n_clusters": 2, "linkage": "ward"}

# Final hyperparameters for Spatial Ward on embedded data
spatial_umap_hyps = {"n_clusters": 12, "linkage": "ward"}


# CH: 3
# DB: 2
# SH: 2
# k-DBCV: 60
# CVNN Hal:
# CDR: 60