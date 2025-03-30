# Scripts for clustering experiments and feature importances
Scripts in this folder use the prepared data to define UMAP hyperparameters, conduct clustering experiments and compute importance of each feature for a clustering model.  

| File                           | Description                                                            | Required data                                                                                                                              | 
|--------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| feature_importance.py          | Functions and classes to grid data                                     | wide_table_knn.csv (from data preparation), labels_kmeans.csv, labels_ward.csv, labels_dbscan_iteration0.csv (from clustering experiments) |
| main_clustering_experiments.py | Compute different clusterings based on the configurations in config.py |                                                                                                                                            |
| umap_hyperparameters.py        | Investigation and choice of UMAP hyperparameters                       | wide_table_knn.csv (from data preparation)                                                                                                 |
