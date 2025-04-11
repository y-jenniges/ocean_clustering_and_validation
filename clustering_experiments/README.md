# Scripts for clustering experiments and feature importances
Scripts in this folder use the prepared data to define UMAP hyperparameters, conduct clustering experiments and compute importance of each feature for a clustering model.  

| File                           | Description                                                            | Required data                                                                                                                              | 
|--------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| endemicity_analysis.ipynb      | Investigation of endemicity in the final cluster set                   | cluster_set.csv (final cluster set from this study), obis_20250318_parquet [1]                                                             |
| feature_importance.py          | Functions and classes to grid data                                     | wide_table_knn.csv (from data preparation), labels_kmeans.csv, labels_ward.csv, labels_dbscan_iteration0.csv (from clustering experiments) |
| main_clustering_experiments.py | Compute different clusterings based on the configurations in config.py |                                                                                                                                            |
| umap_hyperparameters.py        | Investigation and choice of UMAP hyperparameters                       | wide_table_knn.csv (from data preparation)                                                                                                 |

[1] Ocean Biodiversity Information System (OBIS) (25 March 2025) OBIS Occurrence Data https://doi.org/10.25607/obis.occurrence.b89117cd.
