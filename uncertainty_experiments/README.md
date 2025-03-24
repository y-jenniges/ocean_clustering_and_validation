# Uncertainty scripts
Scripts in this folder analyse clustering uncertainties and perform NEMI. 

| File                            | Description                                                                            | Required data                                    | 
|---------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------|
| dask_uncertainty_nemi.ipynb     | Parallel version of uncertainty_nemi.py                                                | umap_dbscan_*.csv (from uncertainty experiments) |
| main_uncertainty_experiments.py | Main functions to compute uncertainty experiments                                      | wide_table_knn.csv (from data preparation)       |
| uncertainty_metrics.py          | Helper functions to compute uncertainty metrics like overlap                           |                                                  |
| uncertainty_nemi.py             | Helper functions to compute NEMI cluster sets and uncertainties                        | umap_dbscan_*.csv (from uncertainty experiments) |
| uncertainty_umap.ipynb          | Compute multiple embeddings using UMAP (fixed hyperparameters) and analyse variability | umap_dbscan_*.csv (from uncertainty experiments) |
